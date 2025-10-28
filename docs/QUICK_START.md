# Quick Start Guide

## Installation

```bash
# Clone repository
git clone <repository-url>
cd fluxperf

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Testing with Mock Server

The simplest way, no real LLM service required:

```bash
python fluxperf.py \
  --mock-server \
  --dataset examples/example_dataset.json \
  --num-samples 10 \
  --concurrency 5
```

### 2. Connecting to Real API

```bash
python fluxperf.py \
  --dataset datasets/MixedBench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10
```

### 3. Multi-Concurrency Testing

```bash
python fluxperf.py \
  --dataset datasets/MixedBench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 20 40 80 \
  --concurrency 5 10 20
```

### 4. Enabling Prometheus Monitoring

```bash
python fluxperf.py \
  --dataset datasets/MixedBench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate vllm_gpu_cache_usage_perc
```

### 5. Using SLO Constraints

Create `slo.yaml`:

```yaml
constraints:
  ttft_ms:
    max: 1000
  itl_ms:
    max: 50
  latency_ms:
    max: 10000
```

Run test:

```bash
python fluxperf.py \
  --dataset datasets/MixedBench.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10 \
  --slo-file slo.yaml
```

### 6. Using Recipe Configuration

Create `my_recipe.yaml`:

```yaml
global:
  dataset: datasets/MixedBench.jsonl
  endpoint: http://localhost:8000/v1/chat/completions
  mode: multi_turn
  max_output_tokens: 256

stages:
  - name: "Low Concurrency Test"
    concurrency_levels: [5]
    num_samples: [20]
  
  - name: "High Concurrency Test"
    concurrency_levels: [10, 20]
    num_samples: [40, 80]
```

Run:

```bash
python fluxperf.py --recipe my_recipe.yaml
```

## Dataset Preparation

### Converting LongBench Data

```bash
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 100 \
  --output datasets/narrativeqa.jsonl
```

### Processing ShareGPT Data

```bash
python process_sharegpt.py \
  --input datasets/ShareGPT/sg_90k_part1.json \
  --output datasets/sharegpt_clean.jsonl \
  --max-turns 3
```

## Common Parameters

| Parameter | Description | Default |
|------|------|--------|
| `--dataset` | Dataset file path | Required |
| `--endpoint` | API endpoint URL | Required (or `--mock-server`) |
| `--num-samples` | Sample count per concurrency level | Required |
| `--concurrency` | Concurrency list | `[10]` |
| `--mode` | Test mode: `dual_round` or `multi_turn` | `multi_turn` |
| `--max-output-tokens` | Maximum output token count | Unlimited |
| `--prometheus-url` | Prometheus metrics endpoint | None |
| `--slo-file` | SLO configuration file | None |
| `--output-dir` | Results output directory | `benchmark_results` |

## Output Description

### Terminal Output

Real-time progress bars and formatted tables:

```
Round 1 (Concurrency:10): 100%|████████| 50/50 [00:25<00:00]
✓ Complete, Duration: 25.43 seconds

┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Statistic         ┃ avg    ┃ p99    ┃ p50    ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ TTFT (ms)         │ 245.32 │ 312.45 │ 238.15 │
│ ITL (ms)          │  18.25 │  25.33 │  17.82 │
│ Latency (ms)      │2456.78 │2901.12 │2401.23 │
└───────────────────┴────────┴────────┴────────┘
```

### File Output

- `benchmark_results/benchmark_YYYYMMDD_HHMMSS.csv`: CSV format summary
- `benchmark_results/benchmark_YYYYMMDD_HHMMSS_params.txt`: Test parameters
- `benchmark_results.json`: JSON format detailed data (if `--output` specified)

## Next Steps

- Detailed usage instructions: [benchmarker_guide.md](benchmarker_guide.md)
- Recipe configuration: [RECIPE_GUIDE.md](RECIPE_GUIDE.md)
- Architecture documentation: [ARCHITECTURE.md](ARCHITECTURE.md)
