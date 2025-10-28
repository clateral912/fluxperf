#!/bin/bash

# Generate test coverage report
# Required: pip install coverage

echo "=========================================="
echo "Generating Test Coverage Report"
echo "========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check if coverage is installed
if ! python -m coverage --version &> /dev/null; then
    echo "❌ coverage module not installed"
    echo "Please run: pip install coverage"
    exit 1
fi

echo "Cleaning old coverage data..."
python -m coverage erase

echo ""
echo "Running tests and collecting coverage data..."
echo ""

# Run each test file
python -m coverage run --source=. --omit="tests/*,lib/*" tests/test_dataclasses.py
python -m coverage run -a --source=. --omit="tests/*,lib/*" tests/test_utils.py
python -m coverage run -a --source=. --omit="tests/*,lib/*" tests/test_benchmark_runner.py
python -m coverage run -a --source=. --omit="tests/*,lib/*" tests/test_llm_client.py
python -m coverage run -a --source=. --omit="tests/*,lib/*" tests/test_env_variables.py

echo ""
echo "=========================================="
echo "Coverage Report"
echo "=========================================="
echo ""

# Generate terminal report
python -m coverage report -m

echo ""
echo "=========================================="
echo "Detailed HTML Report"
echo "========================================="

# Generate HTML report
python -m coverage html

echo ""
echo "HTML report generated: htmlcov/index.html"
echo "Open in browser to view detailed coverage"
echo ""
echo "=========================================="
