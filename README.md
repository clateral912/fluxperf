# FluxPerf

A dual-round load testing tool for benchmarking LLM API service performance, specifically designed to evaluate caching performance and multi-round request behavior.

## Project Structure

```
fluxperf/
├── fluxperf.py                  # Main program
├── llm_mocker.py                # Mock LLM server
├── process_sharegpt.py          # ShareGPT data processing
├── convert_longbench.py         # LongBench data conversion
├── requirements.txt             # Python dependencies
├── examples/                    # Recipe configuration examples
│   ├── README.md
│   ├── recipe_example.yaml      # Basic example
│   ├── recipe_dual_round.yaml   # Dual-round mode
│   └── recipe_env_test.yaml     # Environment variable testing
├── datasets/                    # Dataset directory
│   ├── README.md
│   ├── ShareGPT/               # ShareGPT raw data
│   ├── LongBench/              # LongBench raw data
│   ├── sharegpt_clean.jsonl    # Processed data
│   └── MixedBench.jsonl        # Mixed test data
├── docs/                        # Documentation
│   ├── benchmarker_guide.md    # Usage guide
│   ├── RECIPE_GUIDE.md         # Recipe configuration guide
│   └── ENV_VAR_TESTING.md      # Environment variable testing
└── tests/                       # Test files
    ├── test_env_variables.py
    └── test_recipe_env_integration.sh
```

## Core Features

### 1. Dual-Round Load Testing (`fluxperf.py`)

Performs two rounds of testing on the same batch of requests to evaluate the effectiveness of KV cache, prefix cache, and other optimizations.

**Key Features**:
- ✅ Multiple concurrency level support
- ✅ Streaming output (SSE) support
- ✅ Detailed performance metrics (TTFT, ITL, Latency, Throughput, Goodput)
- ✅ Prometheus metrics integration
- ✅ SLO constraint support
- ✅ Multiple output formats (CLI tables, CSV, JSON)

**Quick Start**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run basic test
python fluxperf.py \
  --dataset examples/example_dataset.json \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 10 \
  --concurrency 5

# Run full test (multi-concurrency + Prometheus)
python fluxperf.py \
  --dataset examples/example_dataset.json \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 5 10 20 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate memory_usage_bytes \
  --slo-file examples/slo_example.yaml \
  --output-dir results/test_run
```

### 2. LongBench Data Converter (`convert_longbench.py`)

Converts LongBench datasets into JSONL format compatible with the benchmarker.

**Key Features**:
- ✅ Automatic download from HuggingFace
- ✅ Local file and directory batch conversion support
- ✅ Automatic deduplication
- ✅ Intelligent format recommendation
- ✅ Sampling protection (prevents duplicate sampling)

**Quick Start**:
```bash
# Download and convert from HuggingFace
python convert_longbench.py \
  --dataset narrativeqa \
  --num-samples 100 \
  --output data/narrativeqa.jsonl

# Batch convert from local directory
python convert_longbench.py \
  --input-dir LongBench_data/ \
  --num-samples 200 \
  --output data/mixed.jsonl
```

## Installation

```bash
# Clone repository
git clone <repository-url>
cd fluxperf

# Install dependencies
pip install -r requirements.txt

# (Optional) For HuggingFace data download
pip install datasets
```

## Directory Structure

```
fluxperf/
├── fluxperf.py                  # Core load testing tool
├── convert_longbench.py         # Data conversion tool
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── docs/                        # Detailed documentation
│   ├── benchmarker_guide.md    # Complete benchmarker guide
│   └── converter_guide.md      # Complete converter guide
├── examples/                    # Example files
│   ├── example_dataset.json    # Sample dataset
│   └── slo_example.yaml        # SLO configuration example
├── tests/                       # Test scripts
│   ├── test_jsonl_output.py    # JSONL format validation
│   └── test_prometheus.py      # Prometheus integration test
└── tools/                       # Utility tools
    └── (empty)
```

## Performance Metrics

### Basic Metrics

- **TTFT (Time to First Token)**: First token latency, time from request sent to first token received
- **ITL (Inter-Token Latency)**: Inter-token latency, average interval between consecutive tokens
- **Latency**: Total latency, time from request sent to complete response received
- **Throughput**: Throughput
  - Token throughput: tokens/sec
  - Request throughput: requests/sec
- **Goodput**: Effective throughput meeting SLO constraints

### Statistical Values

Each metric includes the following statistics:
- **Avg** (Average)
- **P50** (Median)
- **P90** (90th percentile)
- **P99** (99th percentile)
- **Min** (Minimum)
- **Max** (Maximum)
- **Stddev** (Standard deviation)

### Prometheus Metrics

Supports querying arbitrary Prometheus metrics, common metrics:
- `lmcache_hit_rate`: Cache hit rate
- `memory_usage_bytes`: Memory usage
- `gpu_utilization`: GPU utilization

## Dataset Format

### JSONL Format (Recommended)

One JSON object per line:
```
{"text": "First prompt"}
{"text": "Second prompt"}
{"text": "Third prompt"}
```

### JSON Array Format (Also Supported)

```json
[
  {"text": "First prompt"},
  {"text": "Second prompt"}
]
```

## SLO Configuration

Create an `slo.yaml` file to define service level objectives:

```yaml
slo:
  ttft_ms: 1000        # TTFT < 1000ms
  itl_ms: 50           # ITL < 50ms
  latency_ms: 10000    # Total latency < 10000ms
```

Only requests meeting all SLO constraints are counted in Goodput.

## Output Results

### Command Line Output

Displays all metrics in table format:

```
========================================================================================================================
Concurrency: 5 | Round 1 Performance Metrics
========================================================================================================================
Total Requests: 50 | Success: 50 | Failed: 0 | Test Duration: 25.43 sec
Average Input Tokens: 15.2 | Average Output Tokens: 128.6
------------------------------------------------------------------------------------------------------------------------
│ Metric                       │ Avg          │ P50          │ P90          │ P99          │ Min          │ Max          │ Stddev       │
├──────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ TTFT (ms)                    │ 245.32       │ 238.15       │ 298.45       │ 312.45       │ 210.12       │ 328.91       │ 25.67        │
│ ITL (ms)                     │ 18.25        │ 17.82        │ 23.12        │ 25.33        │ 12.34        │ 28.14        │ 3.45         │
│ Latency (ms)                 │ 2456.78      │ 2401.23      │ 2789.34      │ 2901.12      │ 2123.45      │ 3012.34      │ 198.76       │
└──────────────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

Throughput:
  Token Throughput: 52.14 tokens/sec
  Request Throughput: 0.41 requests/sec

Goodput (SLO):
  Requests Meeting SLO: 45 / 50 (90.00%)
```

### CSV Output

Automatically generates CSV files for easy analysis in Excel:
- `results/metrics_summary.csv`: Summary comparison of all concurrency levels

### JSON Output

Complete raw data for programmatic analysis:
- `results/concurrency_5_round_1.json`: Detailed request-level data

## Use Cases

### 1. Cache Performance Evaluation

Evaluate cache effectiveness by comparing performance differences between two rounds:

```bash
python fluxperf.py \
  --dataset data/narrativeqa.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 50 \
  --concurrency 10 \
  --prometheus-url http://localhost:3001/metrics \
  --prometheus-metrics lmcache_hit_rate
```

**Expected Results**:
- Round 1: Cache miss, higher TTFT
- Round 2: Cache hit, significantly lower TTFT

### 2. Multi-Concurrency Load Testing

Test performance under different concurrency levels:

```bash
python fluxperf.py \
  --dataset data/mixed.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions \
  --num-samples 100 \
  --concurrency 1 5 10 20 50 100 \
  --output-dir results/scaling_test
```

### 3. SLO Compliance Testing

Verify service meets SLO requirements:

```bash
python fluxperf.py \
  --dataset data/production_queries.jsonl \
  --endpoint http://production-api:8000/v1/chat/completions \
  --num-samples 1000 \
  --concurrency 50 \
  --slo-file slo_production.yaml \
  --output-dir results/slo_check
```

## FAQ

### Q: How to set appropriate concurrency levels?

A: Recommended to test gradually from low to high: `--concurrency 1 5 10 20 50`, observe performance inflection points.

### Q: What's the difference between TTFT and ITL?

A:
- **TTFT**: First response speed, affects user-perceived latency
- **ITL**: Continuous generation speed, affects streaming experience

### Q: What if Goodput is 0%?

A: Check if SLO configuration is too strict, or if service performance needs optimization.

### Q: Does it support non-OpenAI format APIs?

A: Currently only supports OpenAI-compatible Chat Completions API.

## Detailed Documentation

- **[Complete Benchmarker Guide](docs/benchmarker_guide.md)**: Detailed usage instructions for the load testing tool
- **[Complete Converter Guide](docs/converter_guide.md)**: Detailed instructions for the data conversion tool
- **[Developer Guide](CLAUDE.md)**: Code architecture and development instructions

## Examples and Tests

### Example Files
- `examples/example_dataset.json`: 10 sample prompts
- `examples/slo_example.yaml`: SLO configuration example

### Test Tools
```bash
# Validate JSONL format
python tests/test_jsonl_output.py data/output.jsonl

# Test Prometheus integration
python tests/test_prometheus.py
```

## Contributing

Issues and Pull Requests are welcome!

## License

MIT License
