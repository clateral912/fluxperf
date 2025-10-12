#!/bin/bash

# 生成测试覆盖率报告
# 需要安装: pip install coverage

echo "=========================================="
echo "生成测试覆盖率报告"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 检查是否安装了 coverage
if ! python -m coverage --version &> /dev/null; then
    echo "❌ 未安装 coverage 模块"
    echo "请运行: pip install coverage"
    exit 1
fi

echo "清理旧的覆盖率数据..."
python -m coverage erase

echo ""
echo "运行测试并收集覆盖率数据..."
echo ""

# 运行每个测试文件
python -m coverage run --source=. --omit="tests/*,lib/*" tests/test_dataclasses.py
python -m coverage run -a --source=. --omit="tests/*,lib/*" tests/test_utils.py
python -m coverage run -a --source=. --omit="tests/*,lib/*" tests/test_benchmark_runner.py
python -m coverage run -a --source=. --omit="tests/*,lib/*" tests/test_llm_client.py
python -m coverage run -a --source=. --omit="tests/*,lib/*" tests/test_env_variables.py

echo ""
echo "=========================================="
echo "覆盖率报告"
echo "=========================================="
echo ""

# 生成终端报告
python -m coverage report -m

echo ""
echo "=========================================="
echo "详细 HTML 报告"
echo "=========================================="

# 生成 HTML 报告
python -m coverage html

echo ""
echo "HTML 报告已生成: htmlcov/index.html"
echo "在浏览器中打开查看详细覆盖率"
echo ""
echo "=========================================="
