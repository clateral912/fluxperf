#!/bin/bash

# 运行所有单元测试

set -e

echo "=========================================="
echo "运行所有单元测试"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PASSED=0
FAILED=0
TOTAL=0

# 测试文件列表
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

# 运行每个测试
for test in "${TESTS[@]}"; do
    if [ -f "$test" ]; then
        TOTAL=$((TOTAL + 1))
        echo "=========================================="
        echo "运行: $test"
        echo "=========================================="
        
        if python "$test"; then
            PASSED=$((PASSED + 1))
            echo "✓ $test 通过"
        else
            FAILED=$((FAILED + 1))
            echo "✗ $test 失败"
        fi
        echo ""
    else
        echo "⚠️  $test 不存在，跳过"
    fi
done

echo "=========================================="
echo "测试总结"
echo "=========================================="
echo "总计: $TOTAL"
echo "通过: $PASSED"
echo "失败: $FAILED"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo "✓ 所有测试通过!"
    exit 0
else
    echo "✗ 有测试失败"
    exit 1
fi
