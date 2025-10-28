"""
FluxPerf - LLM Performance Benchmarking Tool

A comprehensive benchmarking tool for LLM inference servers with support for:
- Dual-round and multi-turn conversation testing
- Multi-concurrency stress testing
- SLO (Service Level Objective) validation
- Prometheus metrics collection
- vLLM KVCache management
"""

from .analyzer import MetricsAnalyzer
from .client import OpenAIClient
from .conversation import ConversationHistory
from .loaders import DatasetLoader, RecipeLoader, SLOLoader
from .models import (
    BenchmarkConfig,
    BenchmarkMode,
    Recipe,
    RecipeStage,
    RequestMetrics,
    RoundMetrics,
    SessionData,
    SLOConstraints,
    VLLMConfig,
    count_tokens,
)
from .prometheus_collector import PrometheusCollector
from .runner import BenchmarkRunner, run_recipe_benchmark
from .vllm_manager import VLLMManager

__version__ = "0.1.0"
__all__ = [
    "MetricsAnalyzer",
    "OpenAIClient",
    "ConversationHistory",
    "DatasetLoader",
    "RecipeLoader",
    "SLOLoader",
    "BenchmarkConfig",
    "BenchmarkMode",
    "Recipe",
    "RecipeStage",
    "RequestMetrics",
    "RoundMetrics",
    "SessionData",
    "SLOConstraints",
    "VLLMConfig",
    "count_tokens",
    "PrometheusCollector",
    "BenchmarkRunner",
    "run_recipe_benchmark",
    "VLLMManager",
]
