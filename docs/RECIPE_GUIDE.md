# Recipe Configuration Guide

## Overview

The Recipe feature allows you to define complex multi-stage stress test scenarios through YAML configuration files, supporting:

- **Two Test Modes**: `dual_round` (single-turn twice) and `multi_turn` (multi-turn conversation)
- **Multi-Stage Testing**: Each stage can have different concurrency parameters and environment variables
- **Environment Variable Management**: Set environment variables before each stage starts, automatically restore after completion
- **Unified Configuration**: Avoid repeatedly entering numerous command-line parameters

## Two Modes

### 1. Dual Round Mode (`dual_round`)

Suitable for **single-turn Q&A datasets** (such as LongBench, MMLU, etc.):
- Each data entry contains only one question
- Round 1: Send all questions
- Round 2: Send the same questions again (optional shuffling)
- Used to test cache effects, consistency, etc.

**Example Data Format**:
```jsonl
{"id": "1", "text": "What is the capital of France?"}
{"id": "2", "text": "Explain quantum computing"}
```

### 2. Multi Turn Mode (`multi_turn`)

Suitable for **multi-turn conversation datasets** (such as ShareGPT, OpenAssistant, etc.):
- Each data entry contains multiple conversation turns
- Round 1: Send all conversations in original order
- Round 2: Send all conversations again (optional shuffling of session order)
- Used to test context management, multi-turn consistency, etc.

**Example Data Format**:
```jsonl
{
  "id": "conv_1",
  "conversations": [
    {"from": "human", "value": "Hello"},
    {"from": "gpt", "value": "Hi there!"},
    {"from": "human", "value": "How are you?"}
  ]
}
```

## Recipe File Structure

```yaml
# Global configuration
global:
  dataset: "path/to/dataset.jsonl"
  endpoint: "http://localhost:8001/v1/chat/completions"
  model: "gpt-3.5-turbo"
  mode: "multi_turn"  # or "dual_round"
  
  # Optional configuration
  timeout: 300
  max_output_tokens: 2048
  shuffle_round2: true
  output_dir: "results"

# Mock Server configuration (optional)
mock_server:
  enabled: true
  host: "127.0.0.1"
  port: 8765

# Test suites
suites:
  - name: "Warm-up"
    stages:
      - name: "Stage 1: Low Concurrency"
        env:
          CUDA_VISIBLE_DEVICES: "0"
          CUSTOM_VAR: "value1"
        concurrency_levels: [2, 4]
        num_samples: [4, 8]
        dataset: "datasets/sharegpt_warmup.jsonl"
        max_output_tokens: 1024
        min_output_tokens: 32

      - name: "Stage 2: High Concurrency"
        env:
          CUDA_VISIBLE_DEVICES: "0,1"
          CUSTOM_VAR: "value2"
        concurrency_levels: [8, 16]
        num_samples: [16, 32]
        max_output_tokens: 2048
        min_output_tokens: 64
  
  - name: "Stress"
    stages:
      - name: "Stage 3: Burst"
        concurrency_levels: [32, 48]
        num_samples: [64, 96]
```

## Usage

### Method 1: Using Recipe File

```bash
python fluxperf.py --recipe recipe_example.yaml
```

### Method 2: Command-Line Parameters

```bash
# Multi-turn mode
python fluxperf.py \
  --dataset sharegpt.jsonl \
  --endpoint http://localhost:8001/v1/chat/completions \
  --mode multi_turn \
  --num-samples 10 \
  --concurrency 5

# Dual-round mode
python fluxperf.py \
  --dataset longbench.jsonl \
  --endpoint http://localhost:8001/v1/chat/completions \
  --mode dual_round \
  --num-samples 20 \
  --concurrency 10
```

## Configuration Parameters Details

### Global Configuration

| Parameter | Type | Required | Description |
|------|------|------|------|
| `dataset` | string | Yes | Dataset file path |
| `endpoint` | string | Conditional* | API endpoint URL |
| `mode` | string | Yes | `dual_round` or `multi_turn` |
| `model` | string | No | Model name (default: gpt-3.5-turbo) |
| `timeout` | int | No | Request timeout/seconds (default: 300) |
| `max_output_tokens` | int | No | Maximum output token count |
| `max_context_tokens` | int | No | Maximum context token count (truncate history when exceeded) |
| `shuffle_round2` | bool | No | Whether to shuffle round 2 (default: true) |
| `output_dir` | string | No | Results output directory |
| `slo_file` | string | No | SLO configuration file path |
| `prometheus_url` | string | No | Prometheus metrics URL |
| `save_requests` | bool | No | Whether to save request logs |
| `debug` | bool | No | Whether to enable debug mode |

*Can be omitted when `mock_server.enabled=true`

### Stage Configuration

| Parameter | Type | Required | Description |
|------|------|------|------|
| `name` | string | No | Stage name |
| `concurrency_levels` | list[int] | Yes | Concurrency level list |
| `num_samples` | list[int] | Yes | Sample count for each concurrency level |
| `dataset` | string | No | Override dataset for the stage |
| `max_output_tokens` | int | No | Override maximum output tokens |
| `min_output_tokens` | int | No | Override minimum output tokens |
| `env` | dict | No | Environment variable key-value pairs |

## Example Recipes

### Example 1: Multi-turn Mode (ShareGPT)

```yaml
global:
  dataset: "datasets/sharegpt_clean.jsonl"
  endpoint: "http://localhost:8001/v1/chat/completions"
  mode: "multi_turn"
  model: "llama-3-8b"
  max_context_tokens: 4096

stages:
  - name: "Warm-up"
    concurrency_levels: [2]
    num_samples: [4]
  
  - name: "Production Load"
    env:
      VLLM_ATTENTION_BACKEND: "FLASH_ATTN"
    concurrency_levels: [10, 20, 40]
    num_samples: [20, 40, 80]
```

### Example 2: Dual-round Mode (LongBench)

```yaml
global:
  dataset: "longbench_qasper.jsonl"
  mode: "dual_round"
  model: "qwen-72b"
  shuffle_round2: true

mock_server:
  enabled: true
  port: 8765

stages:
  - name: "Cache Cold"
    env:
      ENABLE_PREFIX_CACHE: "false"
    concurrency_levels: [5, 10]
    num_samples: [10, 20]
  
  - name: "Cache Warm"
    env:
      ENABLE_PREFIX_CACHE: "true"
    concurrency_levels: [5, 10]
    num_samples: [10, 20]
```

## Environment Variable Management

Each stage can set dedicated environment variables. Running process:

1. **Before Stage Starts**: Save current environment variables, set new values specified by stage
2. **During Stage Run**: Use new environment variables during testing
3. **After Stage Ends**: Restore original environment variables

This allows you to test different configurations in different stages, for example:
- Different GPU devices (`CUDA_VISIBLE_DEVICES`)
- Different backends (`VLLM_ATTENTION_BACKEND`)
- Different cache strategies
- Custom application configurations

## Best Practices

1. **Start Small**: Use smaller concurrency and sample counts in first stage
2. **Gradually Increase Load**: Gradually increase concurrency in subsequent stages
3. **Set Reasonable Sample Count**: Recommend `num_samples >= 2 * concurrency`
4. **Use Meaningful Names**: Stage names should clearly describe test purpose
5. **Isolate Environment Variable Impact**: Only set necessary environment variables per stage
6. **Enable Mock Server for Testing**: Verify configuration with mock before actual testing

## Troubleshooting

### Recipe Loading Failed
- Check YAML syntax is correct
- Ensure all required fields are filled
- Verify file paths exist

### Environment Variables Not Taking Effect
- Confirm application reads these environment variables
- Check if service restart needed to apply variables

### Connection Failed
- Confirm endpoint URL is correct
- Check if service is running
- If using mock server, ensure `enabled: true`

## Advanced Usage

### Combining Multiple Recipes

You can create multiple Recipe files to test different scenarios separately:

```bash
# Test cache effects
python fluxperf.py --recipe recipes/cache_test.yaml

# Test long context
python fluxperf.py --recipe recipes/long_context.yaml

# Test high concurrency
python fluxperf.py --recipe recipes/stress_test.yaml
```

### CI/CD Integration

Recipe files can be version controlled and run automatically in CI/CD:

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmark
  run: |
    python fluxperf.py --recipe ci_recipe.yaml
```
