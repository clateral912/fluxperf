import asyncio
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from tqdm import tqdm

from .client import OpenAIClient
from .conversation import ConversationHistory
from .loaders import DatasetLoader, RecipeLoader, SLOLoader
from .models import (
    BenchmarkConfig,
    BenchmarkMode,
    Recipe,
    RequestMetrics,
    SessionData,
    SLOConstraints,
)
from .prometheus_collector import PrometheusCollector


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, slo: Optional[SLOConstraints] = None):
        self.config = config
        self.slo = slo
        self.results: List[RequestMetrics] = []
        self.conversation_state: Dict[str, ConversationHistory] = {}

    def _reset_conversation_state(self, sessions: List[SessionData]):
        for session in sessions:
            session.assistant_messages.clear()
        self.conversation_state = {
            session.session_id: ConversationHistory(self.config.max_context_tokens)
            for session in sessions
        }

    def _sanitize_user_message(self, text: Any) -> str:
        if text is None:
            return ""
        if isinstance(text, dict):
            text = text.get("content") or text.get("value") or ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return ""
        return DatasetLoader.truncate_text(text, self.config.max_input_length)

    def _build_conversation_history(self, session: SessionData, turn_index: int) -> List[Dict[str, str]]:
        history = []
        for i in range(turn_index + 1):
            if i < len(session.user_messages):
                history.append({"role": "user", "content": session.user_messages[i]})
            if i < turn_index and i < len(session.assistant_messages):
                history.append({"role": "assistant", "content": session.assistant_messages[i]})
        return history

    def _extract_user_messages_from_entry(self, entry: Dict[str, Any]) -> List[str]:
        if isinstance(entry.get("user_messages"), list):
            return [msg for msg in entry["user_messages"] if isinstance(msg, str)]

        conversations = entry.get("conversations")
        if isinstance(conversations, list):
            collected = []
            for message in conversations:
                if not isinstance(message, dict):
                    continue
                sender = message.get("from") or message.get("role")
                if sender in {"human", "user"}:
                    value = message.get("value") or message.get("content")
                    if isinstance(value, str):
                        collected.append(value)
            if collected:
                return collected

        messages = entry.get("messages")
        if isinstance(messages, list):
            collected = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = message.get("role") or message.get("from")
                if role in {"user", "human"}:
                    content = message.get("content")
                    if isinstance(content, str):
                        collected.append(content)
                    elif isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                                parts.append(part["text"])
                        if parts:
                            collected.append("\n".join(parts))
            if collected:
                return collected

        text = self._extract_text(entry)
        if text:
            return [text]
        return []

    def _entry_to_session(self, entry: Any, index: int) -> Optional[SessionData]:
        if isinstance(entry, SessionData):
            sanitized = [self._sanitize_user_message(msg) for msg in entry.user_messages]
            sanitized = [msg for msg in sanitized if msg]
            if not sanitized:
                return None
            return SessionData(session_id=entry.session_id, user_messages=sanitized, metadata=entry.metadata)

        if isinstance(entry, dict):
            session_id = ""
            if isinstance(entry.get("session_id"), str):
                session_id = entry["session_id"].strip()
            elif isinstance(entry.get("id"), str):
                session_id = entry["id"].strip()
            if not session_id:
                session_id = f"session_{index}"
            user_messages = [self._sanitize_user_message(msg) for msg in self._extract_user_messages_from_entry(entry)]
            user_messages = [msg for msg in user_messages if msg]
            if not user_messages:
                return None
            return SessionData(session_id=session_id, user_messages=user_messages, metadata=entry)

        sanitized = self._sanitize_user_message(entry)
        if not sanitized:
            return None
        return SessionData(session_id=f"session_{index}", user_messages=[sanitized], metadata={})

    def _normalize_sessions(self, entries: List[Any]) -> List[SessionData]:
        sessions: List[SessionData] = []
        seen_ids: Dict[str, int] = {}
        for idx, entry in enumerate(entries):
            session = self._entry_to_session(entry, idx)
            if session is None:
                continue
            base_id = session.session_id or f"session_{idx}"
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                session.session_id = f"{base_id}_{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 0
                session.session_id = base_id
            sessions.append(session)
        return sessions

    def _entries_to_single_turn_sessions(self, entries: List[Any]) -> List[SessionData]:
        """Convert dataset entries to single-turn sessions (for dual_round mode)"""
        sessions = []
        for idx, entry in enumerate(entries):
            text = self._extract_text(entry)
            sanitized = self._sanitize_user_message(text)
            if not sanitized:
                continue
            
            session_id = f"session_{idx}"
            if isinstance(entry, dict) and 'id' in entry:
                session_id = str(entry['id'])
            
            sessions.append(SessionData(
                session_id=session_id,
                user_messages=[sanitized],
                metadata=entry if isinstance(entry, dict) else {}
            ))
        return sessions

    def _total_turns(self, sessions: List[SessionData]) -> int:
        return sum(len(session.user_messages) for session in sessions)

    async def reset_kvcache(self, reason: str = ""):
        """Reset vLLM KVCache"""
        if not self.config.reset_cache_url:
            return

        try:
            print(f"Resetting KVCache{f' ({reason})' if reason else ''}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.reset_cache_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        print(f"✓ KVCache reset successfully")
                    else:
                        print(f"Warning: KVCache reset returned status code {response.status}")
        except Exception as e:
            print(f"Warning: KVCache reset failed: {e}")

    async def run_round(
        self,
        round_num: int,
        concurrency: int,
        sessions: List[SessionData]
    ) -> Tuple[List[RequestMetrics], Dict[str, List[float]]]:
        self._reset_conversation_state(sessions)

        turns: List[Tuple[str, int, str]] = []
        for session in sessions:
            for turn_index, message in enumerate(session.user_messages):
                turns.append((session.session_id, turn_index, message))

        total_count = len(turns)
        if total_count == 0:
            print("Warning: No user messages found in sessions, skipping this round\n")
            return [], {}

        order_map: Dict[Tuple[str, int], int] = {
            (session_id, turn_index): idx
            for idx, (session_id, turn_index, _) in enumerate(turns)
        }

        round_start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)
        pbar = tqdm(
            total=total_count,
            desc=f"Round {round_num} (concurrency: {concurrency})",
            ncols=120,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )

        active_requests: Dict[str, float] = {}
        status_lock = asyncio.Lock()
        results: Dict[int, RequestMetrics] = {}

        async def update_status():
            while True:
                await asyncio.sleep(0.5)
                async with status_lock:
                    if active_requests:
                        oldest_req_id, oldest_start = min(active_requests.items(), key=lambda x: x[1])
                        wait_time = time.time() - oldest_start
                        postfix = f"Waiting: {oldest_req_id[:40]:40s} ({wait_time:6.1f}s)"
                        pbar.set_postfix_str(postfix, refresh=True)
                    else:
                        pbar.set_postfix_str(" " * 54, refresh=False)

        temp_config = BenchmarkConfig(
            dataset_path=self.config.dataset_path,
            endpoint_url=self.config.endpoint_url,
            num_samples=self.config.num_samples,
            concurrency_levels=[concurrency],
            max_input_length=self.config.max_input_length,
            max_output_tokens=self.config.max_output_tokens,
            model_name=self.config.model_name,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            shuffle_round2=self.config.shuffle_round2,
            slo_file=self.config.slo_file,
            output_dir=self.config.output_dir,
            prometheus_url=self.config.prometheus_url,
            prometheus_metrics=self.config.prometheus_metrics,
            save_requests=self.config.save_requests,
            reset_cache_url=self.config.reset_cache_url,
            reset_cache_between_rounds=self.config.reset_cache_between_rounds,
            reset_cache_between_concurrency=self.config.reset_cache_between_concurrency,
            debug=self.config.debug,
            debug_log_dir=self.config.debug_log_dir,
            max_context_tokens=self.config.max_context_tokens
        )

        prometheus_metrics = {}
        prom_collector = None
        prom_task = None

        if self.config.prometheus_url and self.config.prometheus_metrics:
            prom_collector = PrometheusCollector(
                self.config.prometheus_url,
                self.config.prometheus_metrics
            )

        async with OpenAIClient(temp_config, round_num, concurrency) as client:
            debug_metadata = {
                "round": round_num,
                "concurrency": concurrency,
                "total_sessions": len(sessions),
                "total_turns": total_count,
                "config": {
                    "max_context_tokens": self.config.max_context_tokens,
                    "max_input_length": self.config.max_input_length,
                    "max_output_tokens": self.config.max_output_tokens,
                    "endpoint": self.config.endpoint_url,
                    "shuffle_round2": self.config.shuffle_round2,
                    "save_requests": self.config.save_requests,
                    "debug": self.config.debug,
                }
            }
            client.set_debug_metadata(debug_metadata)

            if prom_collector:
                estimated_end_time = round_start_time + self.config.timeout * 2
                prom_task = asyncio.create_task(
                    prom_collector.collect_during_test(
                        client.session,
                        round_start_time,
                        estimated_end_time,
                        interval=0.5
                    )
                )

            status_task = asyncio.create_task(update_status())

            async def process_session(session: SessionData):
                history = self.conversation_state[session.session_id]
                for turn_index, user_text in enumerate(session.user_messages):
                    turn_key = (session.session_id, turn_index)
                    if turn_key not in order_map:
                        continue
                    turn_order = order_map[turn_key]
                    sanitized_user = self._sanitize_user_message(user_text)
                    if not sanitized_user:
                        from .models import count_tokens
                        metrics = RequestMetrics(
                            request_id=f"round{round_num}_conc{concurrency}_{session.session_id}_turn{turn_index}",
                            round_num=round_num,
                            input_text="",
                            output_text="",
                            input_tokens=0,
                            output_tokens=0,
                            time_to_first_token=0.0,
                            start_timestamp=time.time(),
                            end_timestamp=time.time(),
                            error="Empty input",
                            session_id=session.session_id,
                            turn_index=turn_index,
                            context_tokens=history._total_tokens(),
                            history_truncated=0
                        )
                        results[turn_order] = metrics
                        pbar.update(1)
                        continue

                    messages, context_tokens, truncated = history.prepare_request(sanitized_user)
                    request_id = f"round{round_num}_conc{concurrency}_{session.session_id}_turn{turn_index}"

                    async with semaphore:
                        async with status_lock:
                            active_requests[request_id] = time.time()
                        result = await client.send_completion_request(
                            request_id,
                            round_num,
                            messages,
                            session_id=session.session_id,
                            turn_index=turn_index
                        )
                        async with status_lock:
                            active_requests.pop(request_id, None)

                    assistant_truncated = 0
                    if not result.error and result.output_text:
                        assistant_truncated = history.append_assistant(result.output_text)
                    result.context_tokens = context_tokens
                    result.history_truncated = truncated + assistant_truncated
                    results[turn_order] = result
                    pbar.update(1)

            session_tasks = [asyncio.create_task(process_session(session)) for session in sessions]
            await asyncio.gather(*session_tasks)

            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass

        pbar.close()

        round_end_time = time.time()

        if prom_task:
            prom_task.cancel()
            try:
                await prom_task
            except asyncio.CancelledError:
                pass

            if prom_collector:
                prometheus_metrics = prom_collector.get_metrics_in_timerange(
                    round_start_time,
                    round_end_time
                )

        round_results = [results[i] for i in range(total_count)]
        for result in round_results:
            result.meets_slo = SLOLoader.check_slo(result, self.slo)

        round_duration = round_end_time - round_start_time
        print(f"✓ Completed, elapsed time: {round_duration:.2f} seconds\n")

        return round_results, prometheus_metrics

    def _extract_text(self, entry: Dict[str, Any]) -> str:
        if isinstance(entry, str):
            return entry
        elif isinstance(entry, dict):
            if 'text' in entry:
                return entry['text']
            elif 'prompt' in entry:
                return entry['prompt']
            elif 'content' in entry:
                return entry['content']
            elif 'messages' in entry:
                messages = entry['messages']
                if isinstance(messages, list) and len(messages) > 0:
                    return messages[-1].get('content', str(entry))
            return str(entry)
        return str(entry)

    async def run(self) -> Dict[int, Tuple[List[RequestMetrics], List[RequestMetrics], Dict[str, List[float]], Dict[str, List[float]]]]:
        dataset = DatasetLoader.load_dataset(self.config.dataset_path)
        print(f"Loaded dataset: {len(dataset)} records")
        print(f"Mode: {self.config.mode.value}\n")

        all_results = {}

        for idx, concurrency in enumerate(self.config.concurrency_levels):
            current_num_samples = self.config.num_samples[idx]

            sampled_entries = DatasetLoader.sample_entries(dataset, current_num_samples)

            if self.config.mode == BenchmarkMode.DUAL_ROUND:
                sessions = self._entries_to_single_turn_sessions(sampled_entries)
            else:
                sessions = self._normalize_sessions(sampled_entries)
            total_turns = self._total_turns(sessions)
            print(f"Concurrency level {concurrency}: Selected {len(sessions)} sessions, total turns {total_turns}\n")

            if idx > 0 and self.config.reset_cache_between_concurrency:
                await self.reset_kvcache(f"Concurrency level switch: {self.config.concurrency_levels[idx-1]} -> {concurrency}")

            print(f"{'='*60}")
            print(f"Starting test for concurrency level: {concurrency}")
            print(f"{'='*60}\n")

            round1_results, round1_prom_metrics = await self.run_round(1, concurrency, sessions)

            # Multi-turn mode only runs one round, Dual-round mode runs two rounds
            if self.config.mode == BenchmarkMode.MULTI_TURN:
                # Multi-turn mode: only one round, reuse first round results
                all_results[concurrency] = (round1_results, [], round1_prom_metrics, {})
            else:
                # Dual-round mode: run two rounds
                if self.config.reset_cache_between_rounds:
                    await self.reset_kvcache(f"Concurrency {concurrency}: Round 1 -> Round 2")

                if self.config.shuffle_round2:
                    shuffled_sessions = sessions.copy()
                    random.shuffle(shuffled_sessions)
                else:
                    shuffled_sessions = sessions

                round2_results, round2_prom_metrics = await self.run_round(2, concurrency, shuffled_sessions)
                all_results[concurrency] = (round1_results, round2_results, round1_prom_metrics, round2_prom_metrics)

        await self.reset_kvcache("All benchmark tasks completed")

        return all_results


async def run_recipe_benchmark(recipe: Recipe):
    """Run multi-stage tests configured by Recipe"""
    
    # Start Mock Server (if needed)
    mock_task = None
    shutdown_event = None
    if recipe.mock_server and recipe.mock_server.get('enabled'):
        from llm_mocker import run_server_until_cancelled
        shutdown_event = asyncio.Event()
        
        host = recipe.mock_server.get('host', '127.0.0.1')
        port = recipe.mock_server.get('port', 8765)
        
        mock_task = asyncio.create_task(
            run_server_until_cancelled(host, port, shutdown_event)
        )
        print(f"Mock server started: http://{host}:{port}")
        await asyncio.sleep(2)
        
        # If no endpoint configured, automatically set to mock server
        if not recipe.global_config.get('endpoint'):
            recipe.global_config['endpoint'] = f"http://{host}:{port}/v1/chat/completions"
    
    all_stage_results = {}
    
    try:
        for stage_idx, stage in enumerate(recipe.stages, 1):
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx}/{len(recipe.stages)}: {stage.name}")
            print(f"{'='*60}")
            
            # Save current environment variables
            original_env = {}
            for key, value in stage.env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = str(value)
                print(f"Set environment variable: {key}={value}")
            
            try:
                # Create configuration
                config = RecipeLoader.create_config_from_recipe(recipe, stage)
                
                # Load SLO
                slo = None
                if config.slo_file:
                    slo = SLOLoader.load_slo(config.slo_file)
                
                # Print configuration information
                print(f"\nDataset: {config.dataset_path}")
                print(f"Endpoint: {config.endpoint_url}")
                print(f"Mode: {config.mode.value}")
                print(f"Concurrency levels: {', '.join(map(str, config.concurrency_levels))}")
                print(f"Samples: {', '.join(map(str, config.num_samples))}")
                print(f"Model: {config.model_name}\n")
                
                # Run test
                runner = BenchmarkRunner(config, slo)
                stage_results = await runner.run()
                
                all_stage_results[f"stage_{stage_idx}_{stage.name}"] = stage_results
                
                # Analyze and print results
                from .analyzer import MetricsAnalyzer
                for concurrency, (round1_results, round2_results, round1_prom, round2_prom) in stage_results.items():
                    round1_metrics = MetricsAnalyzer.analyze_round(1, round1_results, round1_prom, stage.name, concurrency)
                    MetricsAnalyzer.print_metrics(concurrency, round1_metrics)

                    # Multi-turn mode only runs one round, don't print second round
                    if config.mode == BenchmarkMode.DUAL_ROUND and round2_results:
                        round2_metrics = MetricsAnalyzer.analyze_round(2, round2_results, round2_prom, stage.name, concurrency)
                        MetricsAnalyzer.print_metrics(concurrency, round2_metrics)
                
            finally:
                # Restore environment variables
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
                print(f"\nEnvironment variables restored")
        
        print(f"\n{'='*60}")
        print(f"All {len(recipe.stages)} stages completed!")
        print(f"{'='*60}")
        
    finally:
        # Close Mock Server
        if mock_task and shutdown_event:
            shutdown_event.set()
            await mock_task
