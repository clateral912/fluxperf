# FluxPerf Architecture Documentation

## Project Overview

FluxPerf is a tool designed for testing LLM API service performance, particularly suitable for evaluating cache performance (such as KV cache, prefix cache). By comparing performance metrics across two rounds of identical requests, it can quantify cache acceleration effects.

## Core Components

### 1. fluxperf.py

Main stress testing tool providing the following core functionality:

#### Core Classes

- **BenchmarkConfig**: Configuration data class containing all test parameters
- **BenchmarkRunner**: Test executor responsible for organizing and executing test flow
- **OpenAIClient**: HTTP client responsible for communicating with API endpoint
- **MetricsAnalyzer**: Metrics analyzer for calculating and displaying performance metrics
- **PrometheusCollector**: Prometheus metrics collector
- **ConversationHistory**: Multi-turn conversation history manager

#### Test Modes

1. **DUAL_ROUND Mode**: Two rounds of single requests, each dataset entry as independent session
2. **MULTI_TURN Mode**: Multi-turn conversation mode, maintaining session context

#### Key Flow

```
Load config → Load dataset → Sample → Round 1 test → Round 2 test → Analyze metrics → Output results
                                   ↓                ↓
                           Collect Prometheus    Collect Prometheus
```

### 2. llm_mocker.py

Mock LLM server for testing and development:

- Implements OpenAI Chat Completions API compatible interface
- Supports streaming output (SSE)
- Configurable response delay and content
- Used for environment variable testing and integration testing

### 3. convert_longbench.py

LongBench dataset conversion tool:

- Download datasets from HuggingFace
- Convert to benchmarker-compatible JSONL format
- Support batch conversion and sampling

### 4. process_sharegpt.py

ShareGPT dataset processing tool:

- Clean and standardize ShareGPT format data
- Extract multi-turn conversations
- Generate JSONL format output

## Data Flow

### Input

1. **Dataset**: JSON/JSONL format, supporting various structures
   - Single-turn: `{"text": "..."}`
   - Multi-turn: `{"user_messages": [...], ...}`
   - ShareGPT: `{"conversations": [...]}`

2. **Configuration Files**
   - Recipe YAML: Multi-stage test configuration
   - SLO YAML: Service Level Objective definitions

### Output

1. **Terminal Output**: Real-time progress and formatted tables
2. **CSV Files**: Summary metric comparison tables
3. **JSON Files**: Detailed request-level data
4. **JSONL Logs**: Request details (optional)
5. **Debug Logs**: JSON format debug information (optional)

## Performance Metrics

### Basic Metrics

- **TTFT (Time to First Token)**: First token latency (ms)
- **ITL (Inter-Token Latency)**: Token interval latency (ms)
- **Latency**: Total latency (ms)
- **Throughput**: Throughput (tokens/sec, requests/sec)
- **Goodput**: Effective throughput meeting SLO

### Statistics

For each metric, calculate:
- avg, p50, p90, p95, p99, min, max, stddev

### Prometheus Metrics

Supports custom Prometheus metrics collection, commonly used metrics:
- `lmcache_hit_rate`: Cache hit rate
- `vllm_gpu_cache_usage_perc`: GPU cache usage
- `vllm_num_requests_running`: Running request count

## Recipe System

Recipe files support multi-stage test configurations:

```yaml
global:
  dataset: path/to/dataset.jsonl
  endpoint: http://localhost:8000/v1/chat/completions
  mode: multi_turn
  
stages:
  - name: "Stage 1"
    concurrency_levels: [5, 10]
    num_samples: [20, 40]
    env:
      CUSTOM_VAR: "value"
```

### Environment Variable Injection

Each stage can set independent environment variables, used for:
- Controlling server behavior
- A/B testing different configurations
- Dynamically adjusting parameters

## Prometheus Integration

### Data Scraping Mechanism

In the `PrometheusCollector` class (fluxperf.py:399-462):

```python
class PrometheusCollector:
    async def collect_during_test(self, session, start_time, end_time, interval=1.0):
        while time.time() < end_time:
            metrics_data = await self.fetch_metrics(session)
            # Store metrics with timestamp
            await asyncio.sleep(interval)
```

### Scraping Frequency

Default scrape every **0.5 seconds** (fluxperf.py:899):

```python
prom_task = asyncio.create_task(
    prom_collector.collect_during_test(
        client.session,
        round_start_time,
        estimated_end_time,
        interval=0.5  # 500ms scrape interval
    )
)
```

### Time Range Filtering

After test completes, only keep data points within test time range:

```python
def get_metrics_in_timerange(self, start_time: float, end_time: float):
    # Filter data where start_time <= timestamp <= end_time
```

## Cache Control

### KVCache Reset

Supports resetting vLLM KVCache at the following times:

1. **Between Rounds Reset**: `--reset-cache-between-rounds`
2. **Between Concurrency Levels Reset**: `--reset-cache-between-concurrency`
3. **Test End**: Automatic reset

Reset implemented via POST request to `--reset-cache-url` endpoint.

## Directory Structure

```
fluxperf/
├── fluxperf.py                  # Main program
├── llm_mocker.py                # Mock server
├── convert_longbench.py         # Data conversion
├── process_sharegpt.py          # ShareGPT processing
├── requirements.txt             # Dependencies
├── README.md                    # Project description
├── examples/                    # Configuration examples
│   ├── recipe_*.yaml
│   └── slo_example.yaml
├── datasets/                    # Datasets
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # This document
│   ├── benchmarker_guide.md    # Usage guide
│   ├── RECIPE_GUIDE.md         # Recipe configuration
│   └── ENV_VAR_TESTING.md      # Environment variable testing
├── tests/                       # Test suite
└── benchmark_results/           # Output directory (generated)
```

## Extension Points

### Adding New Dataset Formats

Add parsing logic in the `_extract_user_messages_from_entry()` method.

### Adding New Performance Metrics

1. Add fields in `RequestMetrics`
2. Calculate in `OpenAIClient.send_completion_request()`
3. Aggregate in `MetricsAnalyzer.analyze_round()`
4. Display in `MetricsAnalyzer.print_metrics()`

### Integrating New Monitoring Systems

Implement new Collector class referencing `PrometheusCollector`.
