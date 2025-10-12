# Test Suite

Complete unit test suite covering the core functionality of the project.

## Test Files

### Core Tests

| File | Coverage | Test Count |
|------|----------|------------|
| `test_dataclasses.py` | All dataclass creation and validation | 8 |
| `test_utils.py` | Utility functions and helper methods | 6 |
| `test_benchmark_runner.py` | BenchmarkRunner core logic | 8 |
| `test_llm_client.py` | LLMClient and request handling | 4 |
| `test_env_variables.py` | Environment variable management | 2 |

### Integration Tests

| File | Coverage |
|------|----------|
| `test_llm_mocker.py` | Mock LLM server functionality |
| `test_process_sharegpt.py` | ShareGPT data processing |
| `test_conversation_history.py` | Conversation history management |
| `test_prometheus.py` | Prometheus integration |
| `test_jsonl_output.py` | JSONL output format |
| `test_recipe_env_integration.sh` | Recipe environment variable integration test |

## Running Tests

### Run All Tests

```bash
# Simple mode
bash tests/run_all_tests.sh

# Or run individually
python tests/test_dataclasses.py
python tests/test_utils.py
python tests/test_benchmark_runner.py
python tests/test_llm_client.py
python tests/test_env_variables.py
```

### Generate Coverage Report

```bash
# Install coverage first
pip install coverage

# Run coverage analysis
bash tests/coverage_report.sh

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Coverage Scope

### test_dataclasses.py

Tests creation, default values, and field validation for all data classes:

- ✓ `SLOConstraints` - empty and value creation
- ✓ `BenchmarkConfig` - all configuration parameters
- ✓ `RequestMetrics` - request metrics recording
- ✓ `RoundMetrics` - round statistics
- ✓ `SessionData` - session data management
- ✓ `RecipeStage` - Recipe stage configuration
- ✓ `Recipe` - complete Recipe structure
- ✓ `BenchmarkMode` - enum type

**Coverage**: 100% dataclass definitions

### test_utils.py

Tests utility functions and helper methods:

- ✓ `count_tokens()` - token counting logic
  - Normal text
  - Empty text and None
  - Multiple spaces handling
  - Special characters

- ✓ `MetricsAnalyzer.calculate_percentile()` - percentile calculation
  - P50, P90, P99
  - Empty list and single value

- ✓ `MetricsAnalyzer._truncate_text()` - text truncation
  - Short text, long text
  - Boundary cases

- ✓ `DatasetLoader.sample_entries()` - dataset sampling
  - Normal sampling
  - Exceeding dataset size
  - Empty dataset

- ✓ `SLOLoader.validate_slo()` - SLO validation
  - TTFT exceeds limit
  - ITL exceeds limit
  - Latency exceeds limit
  - Throughput insufficient
  - Error requests

- ✓ `RecipeLoader.load_recipe()` - Recipe loading
  - Valid recipe
  - Invalid mode

**Coverage**: 90%+ main utility functions

### test_benchmark_runner.py

Tests BenchmarkRunner core logic:

- ✓ `_sanitize_user_message()` - message sanitization
  - Normal text, empty text
  - Dictionary format
  - Strip spaces

- ✓ `_extract_text()` - text extraction
  - String, dictionary multiple formats
  - text, prompt, content, messages fields
  - Type conversion

- ✓ `_total_turns()` - turn calculation
  - Multiple sessions
  - Empty list, single session

- ✓ `_reset_conversation_state()` - state reset
  - Clear history
  - Retain user messages

- ✓ `_entries_to_single_turn_sessions()` - Dual-round mode conversion
  - Different entry formats
  - ID handling
  - Empty entry filtering

- ✓ `_normalize_sessions()` - Multi-turn mode conversion
  - ShareGPT format
  - ID deduplication

- ✓ `_build_conversation_history()` - conversation history building
  - History at different turns
  - Correct message order

- ✓ Conversation history truncation
  - max_context_tokens limit
  - Retain recent messages

**Coverage**: 85%+ BenchmarkRunner main methods

### test_llm_client.py

Tests LLMClient functionality:

- ✓ Client initialization
  - Correct configuration
  - Initial state

- ✓ Lifecycle management
  - initialize()
  - cleanup()

- ✓ Payload building
  - model_name
  - max_output_tokens
  - save_requests mode

- ✓ Metrics initialization
  - All fields correct
  - Default values

**Coverage**: 75%+ LLMClient core functionality

### test_env_variables.py

Tests environment variable management:

- ✓ Environment variable setting and restoration
  - Set new values
  - Restore original values
  - Multiple stage isolation

- ✓ Nested environment variables
  - Multiple nesting levels
  - Correct restoration

**Coverage**: 100% environment variable management logic

## Testing Best Practices

### 1. Test Naming

```python
def test_function_name():
    """Concise description of test"""
    print("Testing function_name...")
    # Test code
    print("✓ function_name tests passed")
```

### 2. Assertion Usage

```python
# Use clear assertions
assert result == expected, f"Expected {expected}, got {result}"

# Test exceptions
try:
    func_that_should_fail()
    assert False, "Should raise exception"
except ExpectedError:
    pass  # Test passed
```

### 3. Test Isolation

Each test should:
- Run independently
- Not depend on other tests
- Clean up its own state

### 4. Test Coverage Goals

- **Core functionality**: 100% coverage
- **Utility functions**: 90%+ coverage
- **Boundary cases**: focus testing
- **Error handling**: ensure testing

## Adding New Tests

1. **Create test file**: `tests/test_new_feature.py`

2. **Write test**:
```python
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxperf import YourClass

def test_your_feature():
    """Test description"""
    print("Testing your_feature...")
    # Test code
    assert condition
    print("✓ your_feature tests passed")

if __name__ == '__main__':
    try:
        test_your_feature()
        print("\nAll tests passed! ✓")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
```

3. **Add to test runner**: Edit `run_all_tests.sh`

4. **Run tests**: `bash tests/run_all_tests.sh`

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install coverage
      - name: Run tests
        run: bash tests/run_all_tests.sh
      - name: Generate coverage
        run: bash tests/coverage_report.sh
```

## Troubleshooting

### Test Failures

1. Review error messages and stack trace
2. Check for missing dependencies
3. Confirm test environment is correct

### Import Errors

```bash
# Ensure running from project root
cd /path/to/dual_round_benchmark
python tests/test_xxx.py
```

### Low Coverage

1. Run coverage report to see uncovered code
2. Add tests for critical paths
3. Test error handling branches

## Contribution Guidelines

When submitting new features, please:

1. ✓ Write corresponding unit tests
2. ✓ Ensure all tests pass
3. ✓ Maintain test coverage > 80%
4. ✓ Update test documentation
