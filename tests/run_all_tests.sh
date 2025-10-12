#!/bin/bash

# Run all unit tests

set -e

echo "=========================================="
echo "Running All Unit Tests"
echo "========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PASSED=0
FAILED=0
TOTAL=0

# Test file list
TESTS=(
    "tests/test_dataclasses.py"
    "tests/test_utils.py"
    "tests/test_benchmark_runner.py"
    "tests/test_llm_client.py"
    "tests/test_env_variables.py"
    "tests/test_llm_mocker.py"
    "tests/test_process_sharegpt.py"
    "tests/test_conversation_history.py"
)

# Run each test
for test in "${TESTS[@]}"; do
    if [ -f "$test" ]; then
        TOTAL=$((TOTAL + 1))
        echo "=========================================="
        echo "Running: $test"
        echo "========================================="
        
        if python "$test"; then
            PASSED=$((PASSED + 1))
            echo "✓ $test passed"
        else
            FAILED=$((FAILED + 1))
            echo "✗ $test failed"
        fi
        echo ""
    else
        echo "⚠️  $test does not exist, skipping"
    fi
done

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "========================================="

if [ $FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
