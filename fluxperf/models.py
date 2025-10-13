#!/usr/bin/env python3

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum


class BenchmarkMode(Enum):
    DUAL_ROUND = "dual_round"
    MULTI_TURN = "multi_turn"


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
    mode: BenchmarkMode = BenchmarkMode.MULTI_TURN
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
    debug: bool = False
    debug_verbose: bool = False
    debug_log_dir: Optional[Path] = None
    max_context_tokens: Optional[int] = None


@dataclass
class RecipeStage:
    name: str
    concurrency_levels: List[int]
    num_samples: List[int]
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class VLLMConfig:
    auto_restart: bool = False
    startup_command: Optional[str] = None
    port: int = 8000


@dataclass
class Recipe:
    global_config: Dict[str, Any]
    stages: List[RecipeStage]
    mock_server: Optional[Dict[str, Any]] = None
    vllm: Optional[VLLMConfig] = None


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
    session_id: str = ""
    turn_index: int = 0
    context_tokens: int = 0
    history_truncated: int = 0


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
    stage_name: Optional[str] = None
    concurrency: Optional[int] = None


@dataclass
class SessionData:
    session_id: str
    user_messages: List[str]
    assistant_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text.strip().split())
