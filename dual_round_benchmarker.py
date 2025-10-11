#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from statistics import mean, median, stdev

import aiohttp
import yaml
from tqdm import tqdm
from prometheus_client.parser import text_string_to_metric_families


@dataclass
class SLOConstraints:
    ttft_ms: Optional[float] = None
    itl_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    output_token_throughput: Optional[float] = None


@dataclass
class BenchmarkConfig:
    dataset_path: Path
    endpoint_url: str
    num_samples: List[int]
    concurrency_levels: List[int] = field(default_factory=lambda: [10])
    max_input_length: Optional[int] = None
    max_output_tokens: Optional[int] = None
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    timeout: int = 300
    shuffle_round2: bool = True
    slo_file: Optional[Path] = None
    output_dir: Path = Path("benchmark_results")
    prometheus_url: Optional[str] = None
    prometheus_metrics: List[str] = field(default_factory=list)
    save_requests: bool = False
    reset_cache_url: Optional[str] = None
    reset_cache_between_rounds: bool = False
    reset_cache_between_concurrency: bool = False


@dataclass
class RequestMetrics:
    request_id: str
    round_num: int
    input_text: str
    output_text: str
    input_tokens: int
    output_tokens: int
    time_to_first_token: float
    inter_token_latencies: List[float] = field(default_factory=list)
    total_latency: float = 0.0
    throughput: float = 0.0
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0
    error: Optional[str] = None
    meets_slo: bool = True


@dataclass
class RoundMetrics:
    round_num: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_ttft: float
    p50_ttft: float
    p90_ttft: float
    p95_ttft: float
    p99_ttft: float
    min_ttft: float
    max_ttft: float
    stddev_ttft: float
    avg_itl: float
    p50_itl: float
    p90_itl: float
    p95_itl: float
    p99_itl: float
    min_itl: float
    max_itl: float
    stddev_itl: float
    avg_latency: float
    p50_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    stddev_latency: float
    total_throughput: float
    request_throughput: float
    avg_input_tokens: float
    p50_input_tokens: float
    p90_input_tokens: float
    p99_input_tokens: float
    min_input_tokens: float
    max_input_tokens: float
    stddev_input_tokens: float
    avg_output_tokens: float
    p50_output_tokens: float
    p90_output_tokens: float
    p99_output_tokens: float
    min_output_tokens: float
    max_output_tokens: float
    stddev_output_tokens: float
    duration: float
    goodput_requests: int = 0
    goodput_tokens: int = 0
    goodput_request_rate: float = 0.0
    goodput_token_rate: float = 0.0
    prometheus_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class DatasetLoader:
    @staticmethod
    def load_dataset(path: Path) -> List[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix == '.json':
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    raise ValueError("不支持的 JSON 格式")
            elif path.suffix == '.jsonl':
                return [json.loads(line) for line in f if line.strip()]
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")

    @staticmethod
    def sample_entries(dataset: List[Dict[str, Any]], num_samples: int) -> List[Dict[str, Any]]:
        if num_samples > len(dataset):
            raise ValueError(f"要求的样本数 ({num_samples}) 超过数据集大小 ({len(dataset)})")
        return random.sample(dataset, num_samples)

    @staticmethod
    def truncate_text(text: str, max_length: Optional[int]) -> str:
        if max_length is None:
            return text
        if len(text) > max_length:
            return text[:max_length]
        return text


class SLOLoader:
    @staticmethod
    def load_slo(slo_file: Path) -> Optional[SLOConstraints]:
        if slo_file is None or not slo_file.exists():
            return None
        
        with open(slo_file, 'r', encoding='utf-8') as f:
            slo_data = yaml.safe_load(f)
        
        if not slo_data or 'constraints' not in slo_data:
            return None
        
        constraints_data = slo_data['constraints']
        
        return SLOConstraints(
            ttft_ms=constraints_data.get('ttft_ms', {}).get('max'),
            itl_ms=constraints_data.get('itl_ms', {}).get('max'),
            latency_ms=constraints_data.get('latency_ms', {}).get('max'),
            output_token_throughput=constraints_data.get('output_token_throughput', {}).get('min')
        )
    
    @staticmethod
    def check_slo(metrics: RequestMetrics, slo: Optional[SLOConstraints]) -> bool:
        if slo is None:
            return True
        
        if metrics.error is not None:
            return False
        
        ttft_ms = metrics.time_to_first_token * 1000
        if slo.ttft_ms is not None and ttft_ms > slo.ttft_ms:
            return False
        
        if metrics.inter_token_latencies:
            avg_itl_ms = mean(metrics.inter_token_latencies)
            if slo.itl_ms is not None and avg_itl_ms > slo.itl_ms:
                return False
        
        latency_ms = metrics.total_latency * 1000
        if slo.latency_ms is not None and latency_ms > slo.latency_ms:
            return False
        
        if slo.output_token_throughput is not None and metrics.throughput < slo.output_token_throughput:
            return False
        
        return True


class PrometheusCollector:
    def __init__(self, prometheus_url: str, metric_names: List[str]):
        self.prometheus_url = prometheus_url
        self.metric_names = metric_names
        self.collected_data: Dict[str, List[tuple[float, float]]] = {name: [] for name in metric_names}
    
    async def fetch_metrics(self, session: aiohttp.ClientSession) -> Dict[str, List[float]]:
        try:
            async with session.get(self.prometheus_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    print(f"警告: Prometheus端点返回状态码 {response.status}")
                    return {}
                
                text = await response.text()
                current_time = time.time()
                metrics_data = {}
                
                for family in text_string_to_metric_families(text):
                    if family.name in self.metric_names:
                        values = []
                        for sample in family.samples:
                            if sample.name == family.name or sample.name.startswith(f"{family.name}_"):
                                timestamp = sample.timestamp if sample.timestamp else current_time
                                values.append((timestamp, sample.value))
                        
                        if values:
                            metrics_data[family.name] = values
                
                return metrics_data
        except Exception as e:
            print(f"警告: 从Prometheus获取指标失败: {e}")
            return {}
    
    async def collect_during_test(self, session: aiohttp.ClientSession, start_time: float, end_time: float, interval: float = 1.0):
        while time.time() < end_time:
            metrics_data = await self.fetch_metrics(session)
            current_time = time.time()
            
            for metric_name, values in metrics_data.items():
                for timestamp, value in values:
                    self.collected_data[metric_name].append((timestamp, value))
            
            await asyncio.sleep(interval)
    
    def get_metrics_in_timerange(self, start_time: float, end_time: float) -> Dict[str, List[float]]:
        filtered_metrics = {}
        
        for metric_name in self.metric_names:
            values = []
            for timestamp, value in self.collected_data[metric_name]:
                if start_time <= timestamp <= end_time:
                    values.append(value)
            
            if values:
                filtered_metrics[metric_name] = values
            else:
                print(f"警告: 在测试时间范围内未找到指标 '{metric_name}' 的数据")
        
        return filtered_metrics
    
    def clear_data(self):
        for metric_name in self.metric_names:
            self.collected_data[metric_name] = []


class OpenAIClient:
    def __init__(self, config: BenchmarkConfig, round_num: int, concurrency: int):
        self.config = config
        self.round_num = round_num
        self.concurrency = concurrency
        self.session: Optional[aiohttp.ClientSession] = None
        self.requests_log = []

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
            print(f"  → 请求日志已保存: {log_file} ({file_size / 1024:.2f} KiB)")

        if self.session:
            await self.session.close()

    async def send_completion_request(
        self,
        request_id: str,
        round_num: int,
        text: str
    ) -> RequestMetrics:
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": text}
            ],
            "stream": True
        }

        if self.config.max_output_tokens:
            payload["max_tokens"] = self.config.max_output_tokens

        if self.config.save_requests:
            self.requests_log.append({
                "request_id": request_id,
                "round": round_num,
                "timestamp": time.time(),
                "payload": payload
            })

        metrics = RequestMetrics(
            request_id=request_id,
            round_num=round_num,
            input_text=text,
            output_text="",
            input_tokens=len(text.split()),
            output_tokens=0,
            time_to_first_token=0.0,
            start_timestamp=time.time(),
            end_timestamp=0.0
        )

        try:
            start_time = time.time()
            first_token_received = False
            last_token_time = start_time
            output_chunks = []

            async with self.session.post(
                self.config.endpoint_url,
                json=payload
            ) as response:
                async for line in response.content:
                    current_time = time.time()
                    
                    line = line.decode('utf-8').strip()
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
            metrics.output_tokens = len(metrics.output_text.split())
            metrics.total_latency = end_time - start_time
            
            if metrics.output_tokens > 0 and metrics.total_latency > 0:
                metrics.throughput = metrics.output_tokens / metrics.total_latency

        except Exception as e:
            metrics.error = str(e)
            print(f"请求 {request_id} (第{round_num}轮) 失败: {e}")

        return metrics


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, slo: Optional[SLOConstraints] = None):
        self.config = config
        self.slo = slo
        self.results: List[RequestMetrics] = []

    async def reset_kvcache(self, reason: str = ""):
        """重置 vLLM KVCache"""
        if not self.config.reset_cache_url:
            return

        try:
            print(f"正在重置 KVCache{f' ({reason})' if reason else ''}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.reset_cache_url,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        print(f"✓ KVCache 重置成功")
                    else:
                        print(f"警告: KVCache 重置返回状态码 {response.status}")
        except Exception as e:
            print(f"警告: KVCache 重置失败: {e}")

    async def run_round(
        self,
        round_num: int,
        concurrency: int,
        entries: List[Dict[str, Any]]
    ) -> tuple[List[RequestMetrics], Dict[str, List[float]]]:
        round_start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)

        completed_count = 0
        total_count = len(entries)
        pbar = tqdm(
            total=total_count,
            desc=f"第{round_num}轮 (并发:{concurrency})",
            ncols=120,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )

        active_requests = {}
        status_lock = asyncio.Lock()

        async def update_status():
            while True:
                await asyncio.sleep(0.5)
                async with status_lock:
                    if active_requests:
                        oldest_req_id, oldest_start = min(active_requests.items(), key=lambda x: x[1])
                        wait_time = time.time() - oldest_start
                        pbar.set_postfix_str(f"等待: {oldest_req_id} ({wait_time:.1f}s)", refresh=True)
                    else:
                        pbar.set_postfix_str("", refresh=False)

        async def bounded_request(idx: int, entry: Dict[str, Any]) -> RequestMetrics:
            nonlocal completed_count
            async with semaphore:
                text = self._extract_text(entry)
                request_id = f"round{round_num}_conc{concurrency}_req{idx}"

                async with status_lock:
                    active_requests[request_id] = time.time()

                result = await client.send_completion_request(request_id, round_num, text)

                async with status_lock:
                    active_requests.pop(request_id, None)

                completed_count += 1
                pbar.update(1)
                return result

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
            reset_cache_between_concurrency=self.config.reset_cache_between_concurrency
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

            tasks = [bounded_request(i, entry) for i, entry in enumerate(entries)]
            round_results = await asyncio.gather(*tasks)

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

        for result in round_results:
            result.meets_slo = SLOLoader.check_slo(result, self.slo)

        round_duration = round_end_time - round_start_time
        print(f"✓ 完成,耗时: {round_duration:.2f} 秒\n")

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

    async def run(self) -> Dict[int, tuple[List[RequestMetrics], List[RequestMetrics], Dict[str, List[float]], Dict[str, List[float]]]]:
        dataset = DatasetLoader.load_dataset(self.config.dataset_path)
        print(f"加载数据集: {len(dataset)} 条记录\n")

        all_results = {}

        for idx, concurrency in enumerate(self.config.concurrency_levels):
            current_num_samples = self.config.num_samples[idx]

            sampled_entries = DatasetLoader.sample_entries(dataset, current_num_samples)
            print(f"并发层级 {concurrency}: 随机抽取 {len(sampled_entries)} 条记录\n")

            if idx > 0 and self.config.reset_cache_between_concurrency:
                await self.reset_kvcache(f"并发层级切换: {self.config.concurrency_levels[idx-1]} -> {concurrency}")

            print(f"{'='*60}")
            print(f"开始测试并发层级: {concurrency}")
            print(f"{'='*60}\n")

            round1_results, round1_prom_metrics = await self.run_round(1, concurrency, sampled_entries)

            if self.config.reset_cache_between_rounds:
                await self.reset_kvcache(f"并发 {concurrency}: 第1轮 -> 第2轮")

            if self.config.shuffle_round2:
                shuffled_entries = sampled_entries.copy()
                random.shuffle(shuffled_entries)
            else:
                shuffled_entries = sampled_entries

            round2_results, round2_prom_metrics = await self.run_round(2, concurrency, shuffled_entries)

            all_results[concurrency] = (round1_results, round2_results, round1_prom_metrics, round2_prom_metrics)

        await self.reset_kvcache("所有评测任务完成")

        return all_results


class MetricsAnalyzer:
    @staticmethod
    def calculate_percentile(values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    @staticmethod
    def analyze_round(round_num: int, results: List[RequestMetrics], prometheus_metrics: Dict[str, List[float]] = None) -> RoundMetrics:
        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        prom_stats = {}
        if prometheus_metrics:
            for metric_name, values in prometheus_metrics.items():
                if values:
                    prom_stats[metric_name] = {
                        'avg': mean(values),
                        'p50': median(values),
                        'p90': MetricsAnalyzer.calculate_percentile(values, 90),
                        'p99': MetricsAnalyzer.calculate_percentile(values, 99),
                        'min': min(values),
                        'max': max(values),
                        'stddev': stdev(values) if len(values) > 1 else 0.0
                    }

        if not successful:
            return RoundMetrics(
                round_num=round_num,
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(failed),
                avg_ttft=0, p50_ttft=0, p90_ttft=0, p95_ttft=0, p99_ttft=0, min_ttft=0, max_ttft=0, stddev_ttft=0,
                avg_itl=0, p50_itl=0, p90_itl=0, p95_itl=0, p99_itl=0, min_itl=0, max_itl=0, stddev_itl=0,
                avg_latency=0, p50_latency=0, p90_latency=0, p95_latency=0, p99_latency=0, min_latency=0, max_latency=0, stddev_latency=0,
                total_throughput=0,
                request_throughput=0,
                avg_input_tokens=0, p50_input_tokens=0, p90_input_tokens=0, p99_input_tokens=0, min_input_tokens=0, max_input_tokens=0, stddev_input_tokens=0,
                avg_output_tokens=0, p50_output_tokens=0, p90_output_tokens=0, p99_output_tokens=0, min_output_tokens=0, max_output_tokens=0, stddev_output_tokens=0,
                duration=0,
                goodput_requests=0,
                goodput_tokens=0,
                goodput_request_rate=0.0,
                goodput_token_rate=0.0,
                prometheus_metrics=prom_stats
            )

        ttft_values = [r.time_to_first_token * 1000 for r in successful if r.time_to_first_token > 0]
        latency_values = [r.total_latency * 1000 for r in successful]

        all_itls = []
        for r in successful:
            all_itls.extend(r.inter_token_latencies)

        start_timestamps = [r.start_timestamp for r in results]
        end_timestamps = [r.end_timestamp for r in successful if r.end_timestamp > 0]

        if start_timestamps and end_timestamps:
            duration = max(end_timestamps) - min(start_timestamps)
        else:
            duration = 0

        total_output_tokens = sum(r.output_tokens for r in successful)
        total_throughput = total_output_tokens / duration if duration > 0 else 0

        request_throughput = len(successful) / duration if duration > 0 else 0

        goodput_results = [r for r in successful if r.meets_slo]
        goodput_requests = len(goodput_results)
        goodput_tokens = sum(r.output_tokens for r in goodput_results)
        goodput_request_rate = (goodput_requests / len(successful) * 100) if successful else 0
        goodput_token_rate = (goodput_tokens / total_output_tokens * 100) if total_output_tokens > 0 else 0

        input_token_values = [r.input_tokens for r in successful]
        output_token_values = [r.output_tokens for r in successful]

        return RoundMetrics(
            round_num=round_num,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_ttft=mean(ttft_values) if ttft_values else 0,
            p50_ttft=median(ttft_values) if ttft_values else 0,
            p90_ttft=MetricsAnalyzer.calculate_percentile(ttft_values, 90),
            p95_ttft=MetricsAnalyzer.calculate_percentile(ttft_values, 95),
            p99_ttft=MetricsAnalyzer.calculate_percentile(ttft_values, 99),
            min_ttft=min(ttft_values) if ttft_values else 0,
            max_ttft=max(ttft_values) if ttft_values else 0,
            stddev_ttft=stdev(ttft_values) if len(ttft_values) > 1 else 0,
            avg_itl=mean(all_itls) if all_itls else 0,
            p50_itl=median(all_itls) if all_itls else 0,
            p90_itl=MetricsAnalyzer.calculate_percentile(all_itls, 90),
            p95_itl=MetricsAnalyzer.calculate_percentile(all_itls, 95),
            p99_itl=MetricsAnalyzer.calculate_percentile(all_itls, 99),
            min_itl=min(all_itls) if all_itls else 0,
            max_itl=max(all_itls) if all_itls else 0,
            stddev_itl=stdev(all_itls) if len(all_itls) > 1 else 0,
            avg_latency=mean(latency_values) if latency_values else 0,
            p50_latency=median(latency_values) if latency_values else 0,
            p90_latency=MetricsAnalyzer.calculate_percentile(latency_values, 90),
            p95_latency=MetricsAnalyzer.calculate_percentile(latency_values, 95),
            p99_latency=MetricsAnalyzer.calculate_percentile(latency_values, 99),
            min_latency=min(latency_values) if latency_values else 0,
            max_latency=max(latency_values) if latency_values else 0,
            stddev_latency=stdev(latency_values) if len(latency_values) > 1 else 0,
            total_throughput=total_throughput,
            request_throughput=request_throughput,
            avg_input_tokens=mean(input_token_values) if input_token_values else 0,
            p50_input_tokens=median(input_token_values) if input_token_values else 0,
            p90_input_tokens=MetricsAnalyzer.calculate_percentile(input_token_values, 90),
            p99_input_tokens=MetricsAnalyzer.calculate_percentile(input_token_values, 99),
            min_input_tokens=min(input_token_values) if input_token_values else 0,
            max_input_tokens=max(input_token_values) if input_token_values else 0,
            stddev_input_tokens=stdev(input_token_values) if len(input_token_values) > 1 else 0,
            avg_output_tokens=mean(output_token_values) if output_token_values else 0,
            p50_output_tokens=median(output_token_values) if output_token_values else 0,
            p90_output_tokens=MetricsAnalyzer.calculate_percentile(output_token_values, 90),
            p99_output_tokens=MetricsAnalyzer.calculate_percentile(output_token_values, 99),
            min_output_tokens=min(output_token_values) if output_token_values else 0,
            max_output_tokens=max(output_token_values) if output_token_values else 0,
            stddev_output_tokens=stdev(output_token_values) if len(output_token_values) > 1 else 0,
            duration=duration,
            goodput_requests=goodput_requests,
            goodput_tokens=goodput_tokens,
            goodput_request_rate=goodput_request_rate,
            goodput_token_rate=goodput_token_rate,
            prometheus_metrics=prom_stats
        )

    @staticmethod
    def _truncate_text(text: str, max_width: int) -> str:
        if len(text) <= max_width:
            return text
        return text[:max_width - 3] + "..."

    @staticmethod
    def print_metrics(concurrency: int, metrics: RoundMetrics):
        table_data = []

        table_data.append([
            "Time to first token (ms)",
            f"{metrics.avg_ttft:.2f}",
            f"{metrics.p99_ttft:.2f}",
            f"{metrics.p90_ttft:.2f}",
            f"{metrics.p50_ttft:.2f}",
            f"{metrics.min_ttft:.2f}",
            f"{metrics.max_ttft:.2f}",
            f"{metrics.stddev_ttft:.2f}"
        ])

        table_data.append([
            "Inter token latency (ms)",
            f"{metrics.avg_itl:.2f}",
            f"{metrics.p99_itl:.2f}",
            f"{metrics.p90_itl:.2f}",
            f"{metrics.p50_itl:.2f}",
            f"{metrics.min_itl:.2f}",
            f"{metrics.max_itl:.2f}",
            f"{metrics.stddev_itl:.2f}"
        ])

        table_data.append([
            "Request latency (ms)",
            f"{metrics.avg_latency:.2f}",
            f"{metrics.p99_latency:.2f}",
            f"{metrics.p90_latency:.2f}",
            f"{metrics.p50_latency:.2f}",
            f"{metrics.min_latency:.2f}",
            f"{metrics.max_latency:.2f}",
            f"{metrics.stddev_latency:.2f}"
        ])

        table_data.append([
            "Input sequence length",
            f"{metrics.avg_input_tokens:.2f}",
            f"{metrics.p99_input_tokens:.2f}",
            f"{metrics.p90_input_tokens:.2f}",
            f"{metrics.p50_input_tokens:.2f}",
            f"{metrics.min_input_tokens:.0f}",
            f"{metrics.max_input_tokens:.0f}",
            f"{metrics.stddev_input_tokens:.2f}"
        ])

        table_data.append([
            "Output sequence length",
            f"{metrics.avg_output_tokens:.2f}",
            f"{metrics.p99_output_tokens:.2f}",
            f"{metrics.p90_output_tokens:.2f}",
            f"{metrics.p50_output_tokens:.2f}",
            f"{metrics.min_output_tokens:.0f}",
            f"{metrics.max_output_tokens:.0f}",
            f"{metrics.stddev_output_tokens:.2f}"
        ])

        table_data.append([
            "Output token throughput (per sec)",
            f"{metrics.total_throughput:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Request throughput (per sec)",
            f"{metrics.request_throughput:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Goodput requests",
            f"{metrics.goodput_requests}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Goodput requests rate (%)",
            f"{metrics.goodput_request_rate:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Goodput tokens",
            f"{metrics.goodput_tokens}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        table_data.append([
            "Goodput tokens rate (%)",
            f"{metrics.goodput_token_rate:.2f}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A"
        ])

        if metrics.prometheus_metrics:
            for metric_name, stats in sorted(metrics.prometheus_metrics.items()):
                table_data.append([
                    MetricsAnalyzer._truncate_text(metric_name, 35),
                    f"{stats.get('avg', 0):.4f}",
                    f"{stats.get('p99', 0):.4f}",
                    f"{stats.get('p90', 0):.4f}",
                    f"{stats.get('p50', 0):.4f}",
                    f"{stats.get('min', 0):.4f}",
                    f"{stats.get('max', 0):.4f}",
                    f"{stats.get('stddev', 0):.4f}"
                ])

        headers = ["Statistic", "avg", "p99", "p90", "p50", "min", "max", "stddev"]
        col_widths = [35, 8, 8, 8, 8, 8, 8, 8]

        total_width = sum(col_widths) + len(col_widths) * 3 - 1

        title = f"Dual Round Benchmarker | LLM Metrics (Concurrency: {concurrency}, Round: {metrics.round_num})"
        print(f"\n{title.center(total_width)}")

        top_border = "┏" + "┳".join("━" * (w + 2) for w in col_widths) + "┓"
        print(top_border)

        header_row = "┃ " + " ┃ ".join(
            headers[i].rjust(col_widths[i]) for i in range(len(headers))
        ) + " ┃"
        print(header_row)

        header_separator = "┡" + "╇".join("━" * (w + 2) for w in col_widths) + "┩"
        print(header_separator)

        for row in table_data:
            truncated_row = [MetricsAnalyzer._truncate_text(str(cell), col_widths[i]) for i, cell in enumerate(row)]
            data_row = "│ " + " │ ".join(
                truncated_row[i].rjust(col_widths[i]) for i in range(len(truncated_row))
            ) + " │"
            print(data_row)

        bottom_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
        print(bottom_border)

        print(f"\n并发: {concurrency} | 第 {metrics.round_num} 轮 | 总请求: {metrics.total_requests} | 成功: {metrics.successful_requests} | 失败: {metrics.failed_requests} | 时长: {metrics.duration:.2f}s")
        if metrics.goodput_requests > 0 or metrics.goodput_tokens > 0:
            print(f"Goodput: {metrics.goodput_requests} 请求 ({metrics.goodput_request_rate:.1f}%) | {metrics.goodput_tokens} tokens ({metrics.goodput_token_rate:.1f}%)")

    @staticmethod
    def save_results(
        all_results: Dict[int, tuple[RoundMetrics, RoundMetrics, List[RequestMetrics], List[RequestMetrics]]],
        output_path: Path
    ):
        data = {}
        
        for concurrency, (round1_metrics, round2_metrics, round1_results, round2_results) in all_results.items():
            data[f"concurrency_{concurrency}"] = {
                "summary": {
                    "round_1": {
                        "total_requests": round1_metrics.total_requests,
                        "successful_requests": round1_metrics.successful_requests,
                        "failed_requests": round1_metrics.failed_requests,
                        "duration_seconds": round1_metrics.duration,
                        "ttft": {
                            "avg_ms": round1_metrics.avg_ttft,
                            "p50_ms": round1_metrics.p50_ttft,
                            "p95_ms": round1_metrics.p95_ttft,
                            "p99_ms": round1_metrics.p99_ttft
                        },
                        "itl": {
                            "avg_ms": round1_metrics.avg_itl,
                            "p50_ms": round1_metrics.p50_itl,
                            "p95_ms": round1_metrics.p95_itl,
                            "p99_ms": round1_metrics.p99_itl
                        },
                        "latency": {
                            "avg_ms": round1_metrics.avg_latency,
                            "p50_ms": round1_metrics.p50_latency
                        },
                        "throughput": {
                            "tokens_per_sec": round1_metrics.total_throughput,
                            "requests_per_sec": round1_metrics.request_throughput
                        },
                        "goodput": {
                            "requests": round1_metrics.goodput_requests,
                            "tokens": round1_metrics.goodput_tokens,
                            "request_rate_percent": round1_metrics.goodput_request_rate,
                            "token_rate_percent": round1_metrics.goodput_token_rate
                        },
                        "tokens": {
                            "avg_input": round1_metrics.avg_input_tokens,
                            "avg_output": round1_metrics.avg_output_tokens
                        }
                    },
                    "round_2": {
                        "total_requests": round2_metrics.total_requests,
                        "successful_requests": round2_metrics.successful_requests,
                        "failed_requests": round2_metrics.failed_requests,
                        "duration_seconds": round2_metrics.duration,
                        "ttft": {
                            "avg_ms": round2_metrics.avg_ttft,
                            "p50_ms": round2_metrics.p50_ttft,
                            "p95_ms": round2_metrics.p95_ttft,
                            "p99_ms": round2_metrics.p99_ttft
                        },
                        "itl": {
                            "avg_ms": round2_metrics.avg_itl,
                            "p50_ms": round2_metrics.p50_itl,
                            "p95_ms": round2_metrics.p95_itl,
                            "p99_ms": round2_metrics.p99_itl
                        },
                        "latency": {
                            "avg_ms": round2_metrics.avg_latency,
                            "p50_ms": round2_metrics.p50_latency
                        },
                        "throughput": {
                            "tokens_per_sec": round2_metrics.total_throughput,
                            "requests_per_sec": round2_metrics.request_throughput
                        },
                        "goodput": {
                            "requests": round2_metrics.goodput_requests,
                            "tokens": round2_metrics.goodput_tokens,
                            "request_rate_percent": round2_metrics.goodput_request_rate,
                            "token_rate_percent": round2_metrics.goodput_token_rate
                        },
                        "tokens": {
                            "avg_input": round2_metrics.avg_input_tokens,
                            "avg_output": round2_metrics.avg_output_tokens
                        }
                    }
                },
                "detailed_results": {
                    "round_1": [
                        {
                            "request_id": r.request_id,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "ttft_ms": r.time_to_first_token * 1000,
                            "avg_itl_ms": mean(r.inter_token_latencies) if r.inter_token_latencies else 0,
                            "total_latency_ms": r.total_latency * 1000,
                            "throughput": r.throughput,
                            "meets_slo": r.meets_slo,
                            "error": r.error
                        }
                        for r in round1_results
                    ],
                    "round_2": [
                        {
                            "request_id": r.request_id,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "ttft_ms": r.time_to_first_token * 1000,
                            "avg_itl_ms": mean(r.inter_token_latencies) if r.inter_token_latencies else 0,
                            "total_latency_ms": r.total_latency * 1000,
                            "throughput": r.throughput,
                            "meets_slo": r.meets_slo,
                            "error": r.error
                        }
                        for r in round2_results
                    ]
                }
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nJSON 结果已保存到: {output_path}")

    @staticmethod
    def save_csv_results(
        config: BenchmarkConfig,
        all_results: Dict[int, tuple[RoundMetrics, RoundMetrics]],
        slo: Optional[SLOConstraints],
        output_dir: Path
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"benchmark_{timestamp}.csv"
        csv_path = output_dir / csv_filename
        
        params_filename = f"benchmark_{timestamp}_params.txt"
        params_path = output_dir / params_filename
        
        with open(params_path, 'w', encoding='utf-8') as f:
            f.write("=== 压测参数 ===\\n")
            f.write(f"数据集: {config.dataset_path}\\n")
            f.write(f"Endpoint: {config.endpoint_url}\\n")
            f.write(f"样本数: {config.num_samples}\\n")
            f.write(f"并发数: {', '.join(map(str, config.concurrency_levels))}\\n")
            f.write(f"最大输入长度: {config.max_input_length if config.max_input_length else '无限制'}\\n")
            f.write(f"模型: {config.model_name}\\n")
            f.write(f"超时时间: {config.timeout}s\\n")
            f.write(f"第二轮打乱顺序: {'是' if config.shuffle_round2 else '否'}\\n")
            
            if slo:
                f.write(f"\\n=== SLO 约束 ===\\n")
                if slo.ttft_ms:
                    f.write(f"TTFT 最大值: {slo.ttft_ms} ms\\n")
                if slo.itl_ms:
                    f.write(f"ITL 最大值: {slo.itl_ms} ms\\n")
                if slo.latency_ms:
                    f.write(f"延迟最大值: {slo.latency_ms} ms\\n")
                if slo.output_token_throughput:
                    f.write(f"输出吞吐量最小值: {slo.output_token_throughput} tokens/sec\\n")
            
            f.write(f"\\n=== 执行时间 ===\\n")
            f.write(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            concurrency_levels = sorted(all_results.keys())
            
            header = ['指标/轮次']
            for conc in concurrency_levels:
                header.extend([f'并发{conc}-第1轮', f'并发{conc}-第2轮'])
            writer.writerow(header)
            writer.writerow([])
            
            writer.writerow(['基本统计'])
            
            row = ['总请求数']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([r1_metrics.total_requests, r2_metrics.total_requests])
            writer.writerow(row)
            
            row = ['成功请求数']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([r1_metrics.successful_requests, r2_metrics.successful_requests])
            writer.writerow(row)
            
            row = ['失败请求数']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([r1_metrics.failed_requests, r2_metrics.failed_requests])
            writer.writerow(row)
            
            row = ['测试时长(秒)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.duration:.2f}', f'{r2_metrics.duration:.2f}'])
            writer.writerow(row)
            writer.writerow([])
            
            writer.writerow(['TTFT (ms)'])
            row = ['平均值']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.avg_ttft:.2f}', f'{r2_metrics.avg_ttft:.2f}'])
            writer.writerow(row)

            row = ['P50']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p50_ttft:.2f}', f'{r2_metrics.p50_ttft:.2f}'])
            writer.writerow(row)

            row = ['P90']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p90_ttft:.2f}', f'{r2_metrics.p90_ttft:.2f}'])
            writer.writerow(row)

            row = ['P95']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p95_ttft:.2f}', f'{r2_metrics.p95_ttft:.2f}'])
            writer.writerow(row)

            row = ['P99']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p99_ttft:.2f}', f'{r2_metrics.p99_ttft:.2f}'])
            writer.writerow(row)

            row = ['最小值']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.min_ttft:.2f}', f'{r2_metrics.min_ttft:.2f}'])
            writer.writerow(row)

            row = ['最大值']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.max_ttft:.2f}', f'{r2_metrics.max_ttft:.2f}'])
            writer.writerow(row)

            row = ['标准差']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.stddev_ttft:.2f}', f'{r2_metrics.stddev_ttft:.2f}'])
            writer.writerow(row)
            writer.writerow([])

            writer.writerow(['ITL (ms)'])
            row = ['平均值']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.avg_itl:.2f}', f'{r2_metrics.avg_itl:.2f}'])
            writer.writerow(row)

            row = ['P50']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p50_itl:.2f}', f'{r2_metrics.p50_itl:.2f}'])
            writer.writerow(row)

            row = ['P90']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p90_itl:.2f}', f'{r2_metrics.p90_itl:.2f}'])
            writer.writerow(row)

            row = ['P95']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p95_itl:.2f}', f'{r2_metrics.p95_itl:.2f}'])
            writer.writerow(row)

            row = ['P99']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.p99_itl:.2f}', f'{r2_metrics.p99_itl:.2f}'])
            writer.writerow(row)

            row = ['最小值']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.min_itl:.2f}', f'{r2_metrics.min_itl:.2f}'])
            writer.writerow(row)

            row = ['最大值']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.max_itl:.2f}', f'{r2_metrics.max_itl:.2f}'])
            writer.writerow(row)

            row = ['标准差']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.stddev_itl:.2f}', f'{r2_metrics.stddev_itl:.2f}'])
            writer.writerow(row)
            writer.writerow([])
            
            writer.writerow(['Throughput'])
            row = ['Token 吞吐量 (tokens/sec)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.total_throughput:.2f}', f'{r2_metrics.total_throughput:.2f}'])
            writer.writerow(row)
            
            row = ['Request 吞吐量 (requests/sec)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.request_throughput:.2f}', f'{r2_metrics.request_throughput:.2f}'])
            writer.writerow(row)
            writer.writerow([])
            
            writer.writerow(['Goodput'])
            row = ['满足 SLO 的请求数']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([r1_metrics.goodput_requests, r2_metrics.goodput_requests])
            writer.writerow(row)
            
            row = ['Request Goodput 比例 (%)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.goodput_request_rate:.2f}', f'{r2_metrics.goodput_request_rate:.2f}'])
            writer.writerow(row)
            
            row = ['Token Goodput 比例 (%)']
            for conc in concurrency_levels:
                r1_metrics, r2_metrics = all_results[conc]
                row.extend([f'{r1_metrics.goodput_token_rate:.2f}', f'{r2_metrics.goodput_token_rate:.2f}'])
            writer.writerow(row)
            
            if config.prometheus_metrics:
                all_metric_names = set()
                for conc in concurrency_levels:
                    r1_metrics, r2_metrics = all_results[conc]
                    all_metric_names.update(r1_metrics.prometheus_metrics.keys())
                    all_metric_names.update(r2_metrics.prometheus_metrics.keys())

                if all_metric_names:
                    writer.writerow([])
                    writer.writerow(['Prometheus 指标'])

                    for metric_name in sorted(all_metric_names):
                        writer.writerow([f'{metric_name}'])

                        row = ['平均值']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('avg', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('avg', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['P50']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('p50', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('p50', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['P90']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('p90', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('p90', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['P99']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('p99', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('p99', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['最小值']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('min', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('min', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['最大值']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('max', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('max', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)

                        row = ['标准差']
                        for conc in concurrency_levels:
                            r1_metrics, r2_metrics = all_results[conc]
                            r1_val = r1_metrics.prometheus_metrics.get(metric_name, {}).get('stddev', '')
                            r2_val = r2_metrics.prometheus_metrics.get(metric_name, {}).get('stddev', '')
                            row.extend([
                                f'{r1_val:.4f}' if isinstance(r1_val, (int, float)) else '',
                                f'{r2_val:.4f}' if isinstance(r2_val, (int, float)) else ''
                            ])
                        writer.writerow(row)
                        writer.writerow([])
        
        print(f"CSV 结果已保存到: {csv_path}")
        print(f"参数已保存到: {params_path}")


def main():
    parser = argparse.ArgumentParser(
        description="两轮压测工具 - 向 OpenAI 格式的 API 发送两轮相同的请求并收集性能指标"
    )
    
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="数据集文件路径 (JSON 或 JSONL 格式)"
    )
    
    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="OpenAI 格式的 API endpoint URL"
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        nargs='+',
        required=True,
        help="从数据集中随机抽取的样本数量。可指定多个值对应不同并发层级。当参数数量与 --concurrency 不匹配时，自动为每个并发层级设置为 2*concurrency"
    )
    
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs='+',
        default=[10],
        help="请求并发数,可指定多个值 (默认: 10),例如: --concurrency 5 10 50 100"
    )
    
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="最大输入长度,超过此长度的输入会被截断"
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="最大输出token数量,控制生成长度 (OSL)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="模型名称 (默认: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API 密钥 (如果需要)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="请求超时时间(秒) (默认: 300)"
    )
    
    parser.add_argument(
        "--no-shuffle-round2",
        action="store_true",
        help="第二轮不打乱样本顺序"
    )
    
    parser.add_argument(
        "--slo-file",
        type=Path,
        default=None,
        help="SLO 配置文件路径 (YAML 格式)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="结果输出目录 (默认: benchmark_results)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.json"),
        help="JSON 结果输出文件路径 (默认: benchmark_results.json)"
    )
    
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default=None,
        help="Prometheus metrics endpoint URL (例如: http://localhost:3001/metrics)"
    )
    
    parser.add_argument(
        "--prometheus-metrics",
        type=str,
        nargs='+',
        default=[],
        help="要收集的Prometheus指标名称列表 (例如: lmcache_hit_rate memory_usage)"
    )

    parser.add_argument(
        "--save-requests",
        action="store_true",
        help="保存请求数据到本地JSONL文件用于调试"
    )

    parser.add_argument(
        "--reset-cache-url",
        type=str,
        default=None,
        help="vLLM KVCache 重置端点 URL (例如: http://localhost:8000/reset_prefix_cache)"
    )

    parser.add_argument(
        "--reset-cache-between-rounds",
        action="store_true",
        help="在每个并发层级的两轮测试之间重置 KVCache"
    )

    parser.add_argument(
        "--reset-cache-between-concurrency",
        action="store_true",
        help="在不同并发层级之间重置 KVCache"
    )

    args = parser.parse_args()

    num_samples_list = args.num_samples
    concurrency_levels = args.concurrency

    if len(num_samples_list) == len(concurrency_levels):
        final_num_samples = num_samples_list
        print(f"使用自定义 num_samples: {final_num_samples}")
    else:
        final_num_samples = [conc * 2 for conc in concurrency_levels]
        print(f"num_samples 参数数量与 concurrency 不匹配，使用默认规则 (2*concurrency): {final_num_samples}")

    slo = None
    if args.slo_file:
        slo = SLOLoader.load_slo(args.slo_file)
        if slo:
            print(f"加载 SLO 配置: {args.slo_file}")
        else:
            print(f"警告: 无法加载 SLO 配置文件或文件格式错误: {args.slo_file}")
    
    config = BenchmarkConfig(
        dataset_path=args.dataset,
        endpoint_url=args.endpoint,
        num_samples=final_num_samples,
        concurrency_levels=concurrency_levels,
        max_input_length=args.max_input_length,
        max_output_tokens=args.max_output_tokens,
        model_name=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        shuffle_round2=not args.no_shuffle_round2,
        slo_file=args.slo_file,
        output_dir=args.output_dir,
        prometheus_url=args.prometheus_url,
        prometheus_metrics=args.prometheus_metrics,
        save_requests=args.save_requests,
        reset_cache_url=args.reset_cache_url,
        reset_cache_between_rounds=args.reset_cache_between_rounds,
        reset_cache_between_concurrency=args.reset_cache_between_concurrency
    )
    
    print("="*60)
    print("两轮压测工具 - 多并发测试")
    print("="*60)
    print(f"数据集: {config.dataset_path}")
    print(f"Endpoint: {config.endpoint_url}")
    print(f"样本数: {config.num_samples}")
    print(f"并发数: {', '.join(map(str, config.concurrency_levels))}")
    print(f"最大输入长度: {config.max_input_length if config.max_input_length else '无限制'}")
    print(f"最大输出tokens: {config.max_output_tokens if config.max_output_tokens else '无限制'}")
    print(f"模型: {config.model_name}")
    print(f"第二轮打乱顺序: {'是' if config.shuffle_round2 else '否'}")
    if slo:
        print(f"SLO 配置: 已加载")
        if slo.ttft_ms:
            print(f"  - TTFT 最大值: {slo.ttft_ms} ms")
        if slo.itl_ms:
            print(f"  - ITL 最大值: {slo.itl_ms} ms")
        if slo.latency_ms:
            print(f"  - 延迟最大值: {slo.latency_ms} ms")
        if slo.output_token_throughput:
            print(f"  - 输出吞吐量最小值: {slo.output_token_throughput} tokens/sec")
    if config.prometheus_url:
        print(f"Prometheus URL: {config.prometheus_url}")
        if config.prometheus_metrics:
            print(f"Prometheus 指标: {', '.join(config.prometheus_metrics)}")
    if config.reset_cache_url:
        print(f"KVCache 重置: 已启用")
        print(f"  - 重置端点: {config.reset_cache_url}")
        if config.reset_cache_between_rounds:
            print(f"  - 轮次间重置: 是")
        if config.reset_cache_between_concurrency:
            print(f"  - 并发层级间重置: 是")
    print("="*60)
    
    runner = BenchmarkRunner(config, slo)
    all_raw_results = asyncio.run(runner.run())
    
    all_results = {}
    all_metrics_only = {}
    
    for concurrency, (round1_results, round2_results, round1_prom_metrics, round2_prom_metrics) in all_raw_results.items():
        round1_metrics = MetricsAnalyzer.analyze_round(1, round1_results, round1_prom_metrics)
        round2_metrics = MetricsAnalyzer.analyze_round(2, round2_results, round2_prom_metrics)
        
        MetricsAnalyzer.print_metrics(concurrency, round1_metrics)
        MetricsAnalyzer.print_metrics(concurrency, round2_metrics)
        
        all_results[concurrency] = (round1_metrics, round2_metrics, round1_results, round2_results)
        all_metrics_only[concurrency] = (round1_metrics, round2_metrics)
    
    MetricsAnalyzer.save_results(all_results, args.output)
    
    MetricsAnalyzer.save_csv_results(
        config,
        all_metrics_only,
        slo,
        config.output_dir
    )
    
    print(f"\n{'='*60}")
    print("压测完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
