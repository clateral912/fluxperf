import asyncio
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
        self.debug_log_dir: Optional[Path] = None
        self._debug_metadata: Dict[str, Any] = {}

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

        if self.debug_log_dir:
            await self._flush_debug_logs()

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

        if self.config.max_output_tokens is not None:
            payload["max_tokens"] = self.config.max_output_tokens

        if self.config.min_output_tokens is not None:
            payload["min_tokens"] = self.config.min_output_tokens

        # 保存完整请求数据（仅在save_requests时）
        if self.config.save_requests:
            entry = {
                "request_id": request_id,
                "round": round_num,
                "timestamp": time.time(),
                "payload": payload,
                "session_id": session_id,
                "turn_index": turn_index
            }
            self.requests_log.append(entry)

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

        if self.debug_log_dir:
            await self._log_request_start(metrics, messages, payload, semaphore_wait_start)

        try:
            start_time = time.time()

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
                                    is_first_chunk = not first_token_received
                                    if not first_token_received:
                                        metrics.time_to_first_token = current_time - start_time
                                        first_token_received = True
                                    else:
                                        itl = (current_time - last_token_time) * 1000
                                        metrics.inter_token_latencies.append(itl)

                                    output_chunks.append(content)
                                    last_token_time = current_time
                                    if self.debug_log_dir:
                                        await self._append_stream_log(
                                            metrics,
                                            chunk,
                                            semaphore_wait_start,
                                            first_chunk=is_first_chunk,
                                            current_output=''.join(output_chunks)
                                        )
                        except json.JSONDecodeError:
                            continue

            end_time = time.time()
            metrics.end_timestamp = end_time
            metrics.output_text = ''.join(output_chunks)
            metrics.output_tokens = count_tokens(metrics.output_text)
            metrics.total_latency = end_time - start_time

            if metrics.output_tokens > 0 and metrics.total_latency > 0:
                metrics.throughput = metrics.output_tokens / metrics.total_latency

        except Exception as e:
            error_msg = str(e) if str(e) else type(e).__name__
            metrics.error = error_msg
            if self.debug_log_dir:
                await self._append_stream_log(
                    metrics,
                    {"error": error_msg},
                    semaphore_wait_start,
                    first_chunk=False,
                    current_output=''.join(output_chunks),
                    is_error=True
                )
            print(f"Request {request_id} (round {round_num}) failed: {error_msg}")

        return metrics

    def set_debug_metadata(self, metadata: Dict[str, Any]):
        if not (self.config.debug and self.config.debug_verbose):
            return

        base_dir = self.config.debug_log_dir or Path("debug_logs")
        base_dir.mkdir(parents=True, exist_ok=True)
        log_dir = base_dir / f"round{self.round_num}_conc{self.concurrency}"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.debug_log_dir = log_dir
        self._debug_metadata = metadata

    async def _append_stream_log(
        self,
        metrics: RequestMetrics,
        chunk: Dict[str, Any],
        semaphore_wait_start: Optional[float],
        first_chunk: bool,
        current_output: str,
        is_error: bool = False
    ):
        if not self.debug_log_dir:
            return

        log_file = self.debug_log_dir / f"{metrics.request_id}.jsonl"
        log_entry = {
            "request_id": metrics.request_id,
            "round": metrics.round_num,
            "session_id": metrics.session_id,
            "turn_index": metrics.turn_index,
            "timestamp": time.time(),
            "chunk": chunk,
            "received_text": current_output,
            "received_tokens": count_tokens(current_output),
            "metadata": self._debug_metadata,
            "stream_position": "first" if first_chunk else "next",
            "is_error": is_error
        }

        if semaphore_wait_start is not None:
            log_entry["wait_time_ms"] = (metrics.start_timestamp - semaphore_wait_start) * 1000

        await asyncio.to_thread(self._write_json_line, log_file, log_entry)

    async def _log_request_start(
        self,
        metrics: RequestMetrics,
        messages: Sequence[Dict[str, str]],
        payload: Dict[str, Any],
        semaphore_wait_start: Optional[float]
    ):
        if not self.debug_log_dir:
            return

        log_file = self.debug_log_dir / f"{metrics.request_id}.jsonl"
        entry = {
            "request_id": metrics.request_id,
            "round": metrics.round_num,
            "session_id": metrics.session_id,
            "turn_index": metrics.turn_index,
            "timestamp": time.time(),
            "stream_position": "request",
            "metadata": self._debug_metadata,
            "payload": payload,
            "messages": list(messages)
        }

        if semaphore_wait_start is not None:
            entry["wait_time_ms"] = (metrics.start_timestamp - semaphore_wait_start) * 1000

        await asyncio.to_thread(self._write_json_line, log_file, entry)

    async def _flush_debug_logs(self):
        if not self.debug_log_dir:
            return

        manifest_data = {
            "round": self.round_num,
            "concurrency": self.concurrency,
            "generated_at": datetime.now().isoformat(),
            "metadata": self._debug_metadata
        }
        manifest_file = self.debug_log_dir / "metadata.json"
        await asyncio.to_thread(self._write_json_file, manifest_file, manifest_data)

    def _write_json_line(self, path: Path, data: Dict[str, Any]):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _write_json_file(self, path: Path, data: Dict[str, Any]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
