# Recipe Examples

This directory contains various Recipe configuration file examples.

## File List

### 1. `recipe_example.yaml`
Basic multi-turn mode example showcasing all configuration options.

**Purpose**: Learn all Recipe configuration features

**Run**:
```bash
python fluxperf.py --recipe examples/recipe_example.yaml
```

### 2. `recipe_dual_round.yaml`
Dual-round mode example, suitable for single-turn Q&A datasets like LongBench.

**Purpose**: Test cache effectiveness, two-round consistency

**Run**:
```bash
python fluxperf.py --recipe examples/recipe_dual_round.yaml
```

### 3. `recipe_env_test.yaml`
Environment variable testing Recipe for validating environment variable management functionality.

**Purpose**: Test different environment configurations (GPU, backend, etc.)

**Run**:
```bash
python fluxperf.py --recipe examples/recipe_env_test.yaml
```

## Recipe Configuration Elements

Each Recipe file contains:

1. **global**: Global configuration
   - `dataset`: Dataset path
   - `mode`: `dual_round` or `multi_turn`
   - `endpoint`: API endpoint
   - `model`: Model name
   - Other optional settings

2. **mock_server** (optional): Mock server configuration
   - `enabled`: Whether to enable
   - `host`: Host address
   - `port`: Port number

3. **stages**: Test stage list
   - `name`: Stage name
   - `env`: Environment variables
   - `concurrency_levels`: Concurrency levels
   - `num_samples`: Sample count

## Quick Start

### Testing with Mock Server

```bash
# Use built-in mock server, no real LLM service needed
python fluxperf.py --recipe examples/recipe_env_test.yaml
```

### Connecting to Real Service

Modify the `endpoint` and `mock_server.enabled` in the recipe file:

```yaml
global:
  endpoint: "http://your-llm-service:8001/v1/chat/completions"

mock_server:
  enabled: false  # Disable mock server
```

## Custom Recipe

1. Copy an example file
2. Modify configuration parameters
3. Adjust stages to match your testing scenario
4. Run and view results

For detailed documentation, see: [RECIPE_GUIDE.md](../RECIPE_GUIDE.md)
