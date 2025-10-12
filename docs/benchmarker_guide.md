# FluxPerf Complete Guide

## Table of Contents
- [Overview](#overview)
- [Installation and Configuration](#installation-and-configuration)
- [Core Features](#core-features)
- [Performance Metrics](#performance-metrics)
- [Command-Line Arguments](#command-line-arguments)
- [Output Formats](#output-formats)
- [Usage Examples](#usage-examples)
- [SLO Configuration](#slo-configuration)
- [Prometheus Integration](#prometheus-integration)
- [Troubleshooting](#troubleshooting)
- [Multi-turn Conversations and Debug Mode](#multi-turn-conversations-and-debug-mode)

---

## Overview

`fluxperf.py` is a tool specifically designed for testing LLM API service performance, particularly suitable for evaluating cache performance (such as KV cache, prefix cache).

### Core Features

- **Dual-Round Testing**: Execute two rounds on the same batch of requests to compare performance differences before and after caching
- **Multi-Concurrency Support**: Simultaneously test multiple concurrency levels (e.g., 5, 10, 20, 50)
- **Streaming Output**: Support for Server-Sent Events (SSE) streaming responses
- **Rich Metrics**: TTFT, ITL, Latency, Throughput, Goodput + Prometheus custom metrics
- **Multiple Outputs**: Command-line tables, CSV, and JSON formats
- **SLO Constraints**: Calculate Goodput based on Service Level Objectives

### Dual-Round Testing Principle

**Round 1**: Cold start, cache miss
```
Request 1 → API (no cache) → Response 1  ← Higher TTFT
Request 2 → API (no cache) → Response 2
...
```

**Round 2**: Same requests, cache hit
```
Request 1 → API (cache hit) → Response 1  ← Lower TTFT
Request 2 → API (cache hit) → Response 2
...
```

By comparing performance metrics from both rounds, you can quantify the cache acceleration effect.

---

## Installation and Configuration

### System Requirements

- Python 3.8+
- LLM service supporting OpenAI Chat Completions API format

### Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:
- `aiohttp>=3.8.0` - Async HTTP client
- `pyyaml>=6.0` - YAML configuration parsing
- `tqdm>=4.65.0` - Progress bar display
- `prometheus-client>=0.16.0` - Prometheus metrics querying

---

## Core Features

### 1. Performance Metric Measurement

#### TTFT (Time to First Token)
First token latency, time from sending request to receiving the first token.

**Influencing Factors**:
- Model loading time
- Prompt processing time
- KV cache hit rate

#### ITL (Inter-Token Latency)
Token interval latency, average time interval between generating consecutive tokens.

**Calculation Method**:
```
ITL = (Total generation time - TTFT) / (Number of tokens generated - 1)
```

#### Latency (Total Latency)
Total time from sending request to receiving complete response.

```
Latency = TTFT + (ITL × Number of tokens generated)
```

### 2. Throughput

- **Token Throughput**: Number of tokens generated per second
- **Request Throughput**: Number of requests completed per second

```
Token Throughput = Total generated tokens / Total duration
Request Throughput = Total requests / Total duration
```

### 3. Goodput (Effective Throughput)

Effective throughput that meets SLO constraints.

```
Goodput = (Tokens meeting SLO) / Total duration
```

### 4. Prometheus Metrics Integration

Supports querying arbitrary Prometheus metrics, commonly used metrics:

- `lmcache_hit_rate`: Cache hit rate
- `memory_usage_bytes`: Memory usage
- `gpu_utilization`: GPU utilization
- `kv_cache_size`: KV cache size

---

## Performance Metrics

Each metric includes the following statistical values:

| Statistic | Description | Purpose |
|--------|------|------|
| **Avg** | Average | Overall performance level |
| **P50** | Median | Typical user experience |
| **P90** | 90th percentile | Most users' experience |
| **P99** | 99th percentile | Tail latency |
| **Min** | Minimum | Best performance |
| **Max** | Maximum | Worst performance |
| **Stddev** | Standard deviation | Performance stability |

---

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|------|------|------|
| `--dataset` | Dataset file path | `--dataset data/test.jsonl` |
| `--endpoint` | API endpoint URL | `--endpoint http://localhost:8000/v1/chat/completions` |

### Core Arguments

| Argument | Default | Description |
|------|--------|------|
| `--num-samples` | All | Number of samples to test |
| `--concurrency` | 1 | Concurrency (can specify multiple) |
| `--model` | Auto | Model name |
| `--timeout` | 300 | Request timeout (seconds) |

### Optional Arguments

| Argument | Description |
|------|------|
| `--max-input-length` | Truncate input length |
| `--max-output-tokens` | Limit maximum generated tokens |
| `--temperature` | Generation temperature |
| `--top-p` | Top-p sampling |
| `--output-dir` | Output directory |
| `--slo-file` | SLO configuration file |

### Prometheus Arguments

| Argument | Description |
|------|------|
| `--prometheus-url` | Prometheus metrics endpoint |
| `--prometheus-metrics` | List of metrics to query |

### Complete Example

```bash
python fluxperf.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model meta-llama/Llama-3-8B \
  --num-samples 100 \
  --concurrency 5 10 20 \
  --max-input-length 8192 \
  --max-output-tokens 512 \
  --temperature 0.7 \
  --timeout 600 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate memory_usage_bytes \
  --slo-file slo.yaml \
  --output-dir results/test_run
```

---

## Output Formats

### 1. Command-Line Table Output

```
========================================================================================================================
Concurrency: 10 | Round 1 Performance Metrics
========================================================================================================================
Total Requests: 100 | Success: 100 | Failed: 0 | Test Duration: 45.32 seconds
Average Input Tokens: 1024.5 | Average Output Tokens: 256.3
------------------------------------------------------------------------------------------------------------------------
│ Metric                       │ Avg          │ P50          │ P90          │ P99          │ Min          │ Max          │ Stddev       │
├──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ TTFT (ms)                    │ 245.32       │ 238.15       │ 298.45       │ 312.45       │ 210.12       │ 328.91       │ 25.67        │
│ ITL (ms)                     │ 18.25        │ 17.82        │ 23.12        │ 25.33        │ 12.34        │ 28.14        │ 3.45         │
│ Latency (ms)                 │ 4756.78      │ 4501.23      │ 5789.34      │ 6201.12      │ 4023.45      │ 6512.34      │ 398.76       │
│ lmcache_hit_rate             │ 0.0000       │ 0.0000       │ 0.0000       │ 0.0000       │ 0.0000       │ 0.0000       │ 0.0000       │
└──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

Throughput:
  Token Throughput: 565.14 tokens/sec
  Request Throughput: 2.21 requests/sec

Goodput (SLO):
  Requests Meeting SLO: 95 / 100 (95.00%)
  Tokens Meeting SLO: 24348 (95.12%)
```

### 2. CSV Output

File: `results/metrics_summary.csv`

```csv
Metric/Round,Concurrency5-Round1,Concurrency5-Round2,Concurrency10-Round1,Concurrency10-Round2

TTFT (ms)
Average,245.32,98.45,312.54,105.12
Median,238.15,95.20,305.23,102.11
P90,289.45,145.67,378.12,145.23
P99,312.45,156.30,398.45,156.78
Min,210.12,80.23,280.34,85.45
Max,350.67,190.34,450.12,210.23
Stddev,25.67,18.34,35.67,20.45

ITL (ms)
Average,18.25,16.34,20.45,17.67
...
```

### 3. JSON Output

File: `results/concurrency_10_round_1.json`

```json
{
  "configuration": {
    "endpoint": "http://localhost:8000/v1/chat/completions",
    "concurrency": 10,
    "num_samples": 100,
    "round": 1
  },
  "summary": {
    "total_requests": 100,
    "successful_requests": 100,
    "failed_requests": 0,
    "avg_ttft": 245.32,
    "p50_ttft": 238.15,
    ...
  },
  "requests": [
    {
      "index": 0,
      "prompt": "...",
      "ttft": 235.67,
      "itl": 18.45,
      "latency": 4523.12,
      "input_tokens": 1024,
      "output_tokens": 256,
      "success": true
    },
    ...
  ]
}
```

---

## Multi-turn Conversations and Debug Mode

### Multi-turn Conversation Support

Since v2, the tool supports session-based multi-turn conversation datasets. Input JSONL can contain the following structures:

```json
{"session_id": "s1", "user_messages": ["Question1", "Question2"]}
{"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}, {"from": "human", "value": "how are you"}]}
{"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}]}
```

The same session maintains history:
1. User question appended to corresponding conversation history
2. Request carries history + new question when called
3. Model reply appended and truncated from earliest turn based on `--max-context-tokens`

### Token Truncation Strategy

- Only takes effect when total context tokens exceed `--max-context-tokens`
- Truncation starts from earliest Q&A turns, ensuring latest context

### Debug Mode

Added `--debug` and `--debug-log-dir` for outputting request payloads, history context, session metadata.

```
python fluxperf.py \
  --dataset data/chat.jsonl \
  --endpoint http://localhost:8001/v1/chat/completions \
  --num-samples 4 \
  --concurrency 2 \
  --debug \
  --debug-log-dir debug_logs
```

Debug log JSON contains:
- `session_id` / `turn_index`
- Complete request message list
- Current context token count and truncation count

### Mock Service

Use `--mock-server` to start built-in `llm_mocker.py`:

```
python fluxperf.py \
  --dataset examples/example_dataset.json \
  --mock-server \
  --mock-host 127.0.0.1 \
  --mock-port 8765 \
  --num-samples 4 \
  --concurrency 2
```

Mock output includes session/id hash for verifying request payload.

## Usage Examples

### Example 1: Basic Test

```bash
python fluxperf.py \
  --dataset examples/example_dataset.json \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 10 \
  --concurrency 5
```

### Example 2: Cache Performance Evaluation

```bash
python fluxperf.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate \
  --output-dir results/cache_test
```

**Expected Results**:
- Round 1: `lmcache_hit_rate ≈ 0.0`, high TTFT
- Round 2: `lmcache_hit_rate > 0.8`, significantly lower TTFT

### Example 3: Multi-Concurrency Stress Test

```bash
python fluxperf.py \
  --dataset data/mixed.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 200 \
  --concurrency 1 5 10 20 50 100 \
  --output-dir results/scaling_test
```

View `results/scaling_test/metrics_summary.csv` to compare performance under different concurrency levels.

### Example 4: SLO Compliance Test

```bash
python fluxperf.py \
  --dataset data/production_queries.jsonl \
  --endpoint http://production:8000/v1/chat/completions \
  --num-samples 1000 \
  --concurrency 50 \
  --slo-file slo_production.yaml \
  --output-dir results/slo_check
```

---

## SLO Configuration

Create `slo.yaml` file to define Service Level Objectives:

```yaml
slo:
  ttft_ms: 1000        # TTFT must be < 1000ms
  itl_ms: 50           # ITL must be < 50ms
  latency_ms: 10000    # Total latency must be < 10000ms
```

**Goodput Calculation Rules**:

Only requests meeting all SLO constraints count toward Goodput:

```python
if (ttft < slo.ttft_ms) AND (itl < slo.itl_ms) AND (latency < slo.latency_ms):
    goodput_tokens += output_tokens
    goodput_requests += 1
```

**Example Output**:

```
Goodput (SLO):
  Requests Meeting SLO: 850 / 1000 (85.00%)
  Tokens Meeting SLO: 217600 / 256000 (85.00%)
```

---

## Prometheus Integration

### Configuration

```bash
python fluxperf.py \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate memory_usage_bytes gpu_utilization \
  ...
```

### Metrics Querying

The tool queries Prometheus metrics before and after tests, calculating differences or averages.

**Query Timing**:
1. Before each request: Record `metric_before`
2. After each request: Record `metric_after`
3. Calculate: `metric_delta = metric_after - metric_before`

### Commonly Used Metrics

| Metric Name | Description | Typical Value |
|----------|------|--------|
| `lmcache_hit_rate` | Cache hit rate | 0.0 - 1.0 |
| `memory_usage_bytes` | Memory usage (bytes) | 1GB - 100GB |
| `gpu_utilization` | GPU utilization | 0.0 - 1.0 |
| `kv_cache_size` | KV cache size | - |
| `request_latency` | Request latency | - |

### Output Example

```
Prometheus Metrics

lmcache_hit_rate
Average  0.0000   0.8521   0.0012   0.9234
P50     0.0000   0.8500   0.0000   0.9200
P90     0.0000   0.9100   0.0050   0.9500
P99     0.0000   0.9300   0.0100   0.9700
Min     0.0000   0.7800   0.0000   0.8900
Max     0.0000   0.9500   0.0200   0.9800
Stddev  0.0000   0.0420   0.0045   0.0280
```

---

## Troubleshooting

### Issue 1: Connection Timeout

**Error Message**: `asyncio.TimeoutError` or `Connection timeout`

**Solution**:
```bash
# Increase timeout
python fluxperf.py \
  --timeout 600 \
  ...
```

### Issue 2: All Requests Failing

**Error Message**: `Success: 0 | Failed: 100`

**Checklist**:
1. Is API endpoint correct?
2. Is service running?
3. Is model name correct?
4. Check JSON output for error details

### Issue 3: Goodput is 0%

**Reason**: SLO configuration too strict

**Solution**:
1. Review actual performance metrics
2. Adjust SLO configuration to reasonable values
3. Or optimize service performance

### Issue 4: Abnormally High TTFT

**Possible Causes**:
- Model not loaded into VRAM
- Input too long
- Server overloaded

**Check**:
```bash
# View Prometheus metrics
--prometheus-metrics gpu_memory_usage model_load_time
```

### Issue 5: No Acceleration in Round 2

**Possible Causes**:
- Cache not enabled
- Request format not completely consistent
- Cache capacity insufficient and evicted

**Check**:
```bash
# View cache hit rate
--prometheus-metrics lmcache_hit_rate kv_cache_eviction_count
```

---

## Advanced Usage

### 1. Custom Dataset

Create JSONL file:

```
{"text": "Write a poem about spring"}
{"text": "Explain the principle of quantum entanglement"}
{"text": "Recommend three science fiction novels"}
```

### 2. Long Text Testing

```bash
# Use LongBench dataset
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 50 \
  --output data/long_context.jsonl

python fluxperf.py \
  --dataset data/long_context.jsonl \
  --max-input-length 16384 \
  --endpoint http://localhost:8000/v1/chat/completions
```

### 3. Batch Testing Script

```bash
#!/bin/bash
CONCURRENCIES="1 5 10 20 50 100"

for conc in $CONCURRENCIES; do
    echo "Testing concurrency: $conc"
    python fluxperf.py \
      --dataset data/test.jsonl \
      --endpoint http://localhost:8000/v1/chat/completions \
      --num-samples 100 \
      --concurrency $conc \
      --output-dir results/conc_$conc
done
```

---

## Performance Optimization Recommendations

### 1. Concurrency Selection

- **Low Concurrency (1-5)**: Test single request performance
- **Medium Concurrency (10-20)**: Common production scenarios
- **High Concurrency (50-100)**: Stress testing

### 2. Sample Count Selection

- **Quick Test**: 10-50 samples
- **Regular Test**: 100-500 samples
- **Production Validation**: 1000+ samples

### 3. Timeout Setting

Adjust based on output length:

```
timeout = Expected_TTFT + (max_tokens × Expected_ITL) + Buffer_time
```

Examples:
- Short output (< 100 tokens): `--timeout 60`
- Medium output (100-500 tokens): `--timeout 300`
- Long output (> 500 tokens): `--timeout 600`

---

## Output Files Description

After testing completes, the `--output-dir` directory contains:

```
results/
├── metrics_summary.csv              # Summary comparison of all concurrency levels
├── concurrency_5_round_1.json       # Detailed data for concurrency 5 round 1
├── concurrency_5_round_2.json       # Detailed data for concurrency 5 round 2
├── concurrency_10_round_1.json      # Detailed data for concurrency 10 round 1
└── concurrency_10_round_2.json      # Detailed data for concurrency 10 round 2
```

### CSV File Structure

Horizontal comparison of all test configurations:

| Metric/Round | Concurrency5-Round1 | Concurrency5-Round2 | Concurrency10-Round1 | Concurrency10-Round2 |
|-----------|-------------|-------------|--------------|--------------|
| TTFT-Average | 245.32 | 98.45 | 312.54 | 105.12 |
| TTFT-P99 | 312.45 | 156.30 | 398.45 | 156.78 |
| ... | ... | ... | ... | ... |

### JSON File Structure

Contains request-level detailed data for programmatic analysis.

---

## Related Tools

- **convert_longbench.py**: Dataset conversion tool
- **test_jsonl_output.py**: JSONL format validation
- **test_prometheus.py**: Prometheus connection testing

---

## References

- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [LongBench Dataset](https://github.com/THUDM/LongBench)
