import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from .models import BenchmarkConfig, RequestMetrics, count_tokens


class OpenAIClient:
    def __init__(self, config: BenchmarkConfig, round_num: int, concurrency: int):
        self.config = config
        self.round_num = round_num
        self.concurrency = concurrency
        self.session: Optional[aiohttp.ClientSession] = None
        self.requests_log = []
        self.debug_entries: List[Dict[str, Any]] = []
        self.debug_metadata: Dict[str, Any] = {}

    async def __aenter__(self):
        headers = {
            "Content-Type": "application/json"
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.config.save_requests and self.requests_log:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.config.output_dir / f"requests_round{self.round_num}_conc{self.concurrency}_{timestamp}.jsonl"
            with open(log_file, 'w', encoding='utf-8') as f:
                for req in self.requests_log:
                    f.write(json.dumps(req, ensure_ascii=False) + '\n')
            file_size = log_file.stat().st_size
            print(f"  → Request log saved: {log_file} ({file_size / 1024:.2f} KiB)")

        if self.config.debug and self.debug_entries:
            log_dir = self.config.debug_log_dir or Path("debug_logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"debug_round{self.round_num}_conc{self.concurrency}_{timestamp}.json"
            debug_payload = {
                "metadata": self.debug_metadata,
                "entries": self.debug_entries
            }
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(debug_payload, f, ensure_ascii=False, indent=2)
            print(f"  → Debug log saved: {log_file}")

        if self.session:
            await self.session.close()

    async def send_completion_request(
        self,
        request_id: str,
        round_num: int,
        messages: Sequence[Dict[str, str]],
        session_id: str,
        turn_index: int,
        semaphore_wait_start: Optional[float] = None
    ) -> RequestMetrics:
        payload = {
            "model": self.config.model_name,
            "messages": list(messages),
            "stream": True,
            "metadata": {
                "session_id": session_id,
                "turn_index": turn_index,
                "request_id": request_id
            }
        }

        if self.config.max_output_tokens:
            payload["max_tokens"] = self.config.max_output_tokens

        # 保存完整请求数据（仅在save_requests或debug_verbose时）
        if self.config.save_requests or (self.config.debug and self.config.debug_verbose):
            entry = {
                "request_id": request_id,
                "round": round_num,
                "timestamp": time.time(),
                "payload": payload,
                "session_id": session_id,
                "turn_index": turn_index
            }
            if self.config.save_requests:
                self.requests_log.append(entry)
            if self.config.debug and self.config.debug_verbose:
                self.debug_entries.append(entry)

        user_text = messages[-1]["content"] if messages else ""
        metrics = RequestMetrics(
            request_id=request_id,
            round_num=round_num,
            input_text=user_text,
            output_text="",
            input_tokens=count_tokens(user_text),
            output_tokens=0,
            time_to_first_token=0.0,
            start_timestamp=time.time(),
            end_timestamp=0.0,
            session_id=session_id,
            turn_index=turn_index,
            context_tokens=sum(count_tokens(msg["content"]) for msg in messages)
        )

        # 轻量级debug日志：仅在debug模式（非verbose）时记录生命周期
        lifecycle_entry = None
        if self.config.debug and not self.config.debug_verbose:
            lifecycle_entry = {
                "request_id": request_id,
                "round": round_num,
                "session_id": session_id,
                "turn_index": turn_index,
                "created_at": time.time(),
                "input_tokens": count_tokens(user_text),
                "context_tokens": sum(count_tokens(msg["content"]) for msg in messages),
            }
            # 如果有pending时间，记录
            if semaphore_wait_start:
                lifecycle_entry["pending_duration"] = time.time() - semaphore_wait_start

        try:
            start_time = time.time()
            if lifecycle_entry:
                lifecycle_entry["request_sent_at"] = start_time

            first_token_received = False
            last_token_time = start_time
            output_chunks = []

            async with self.session.post(
                self.config.endpoint_url,
                json=payload
            ) as response:
                buffer = b''
                async for chunk_bytes in response.content.iter_any():
                    buffer += chunk_bytes
                    while b'\n' in buffer:
                        line_bytes, buffer = buffer.split(b'\n', 1)
                        current_time = time.time()
                        
                        line = line_bytes.decode('utf-8').strip()
                        if not line or line == "data: [DONE]":
                            continue
                        
                        if line.startswith("data: "):
                            line = line[6:]
                        
                        try:
                            chunk = json.loads(line)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    if not first_token_received:
                                        metrics.time_to_first_token = current_time - start_time
                                        first_token_received = True
                                        # 记录首token时间到生命周期
                                        if lifecycle_entry:
                                            lifecycle_entry["first_token_at"] = current_time
                                    else:
                                        itl = (current_time - last_token_time) * 1000
                                        metrics.inter_token_latencies.append(itl)
                                    
                                    output_chunks.append(content)
                                    last_token_time = current_time
                        except json.JSONDecodeError:
                            continue

            end_time = time.time()
            metrics.end_timestamp = end_time
            metrics.output_text = ''.join(output_chunks)
            metrics.output_tokens = count_tokens(metrics.output_text)
            metrics.total_latency = end_time - start_time

            if metrics.output_tokens > 0 and metrics.total_latency > 0:
                metrics.throughput = metrics.output_tokens / metrics.total_latency

            # 记录成功完成的生命周期
            if lifecycle_entry:
                lifecycle_entry["completed_at"] = end_time
                lifecycle_entry["output_tokens"] = metrics.output_tokens
                lifecycle_entry["ttft_ms"] = metrics.time_to_first_token * 1000
                lifecycle_entry["total_latency_ms"] = metrics.total_latency * 1000
                lifecycle_entry["throughput"] = metrics.throughput
                self.debug_entries.append(lifecycle_entry)

        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            metrics.error = error_msg
            print(f"Request {request_id} (round {round_num}) failed: {error_msg}")

            # 记录失败的生命周期
            if lifecycle_entry:
                lifecycle_entry["failed_at"] = time.time()
                lifecycle_entry["error"] = error_msg
                lifecycle_entry["error_type"] = type(e).__name__
                self.debug_entries.append(lifecycle_entry)

        return metrics

    def set_debug_metadata(self, metadata: Dict[str, Any]):
        if self.config.debug:
            self.debug_metadata = metadata
