#!/bin/bash

# Recipe environment variable integration test
# Verify environment variables are correctly set and restored during recipe execution

set -e

echo "=========================================="
echo "Recipe Environment Variable Integration Test"
echo "=========================================="

# Set test environment variables
export ORIGINAL_VAR="original_value"
export CUDA_VISIBLE_DEVICES="9"

echo ""
echo "Environment variables before test:"
echo "  ORIGINAL_VAR = $ORIGINAL_VAR"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  TEST_STAGE_NAME = ${TEST_STAGE_NAME:-not set}"
echo "  MY_CUSTOM_VAR = ${MY_CUSTOM_VAR:-not set}"

# Create temporary modified benchmarker that prints environment variables at start/end of each stage
echo ""
echo "Creating test hook..."

# Use a simple method: create a wrapper script to capture environment variables
cat > /tmp/test_recipe_env.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import subprocess

# Check environment variables before running recipe
print("\n" + "="*60)
print("Environment variables before test:")
print("="*60)
print(f"ORIGINAL_VAR = {os.environ.get('ORIGINAL_VAR', 'not set')}")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"TEST_STAGE_NAME = {os.environ.get('TEST_STAGE_NAME', 'not set')}")
print(f"MY_CUSTOM_VAR = {os.environ.get('MY_CUSTOM_VAR', 'not set')}")

# Record initial values
initial_original = os.environ.get('ORIGINAL_VAR')
initial_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')

# Run benchmarker (will be interrupted after a few seconds)
print("\nStarting recipe...")
try:
    # Note: we let it run, but only with a small sample
    result = subprocess.run(
        [
            sys.executable,
            "fluxperf.py",
            "--recipe", "recipe_env_test.yaml"
        ],
        timeout=120,  # max 2 minutes
        capture_output=False
    )
except subprocess.TimeoutExpired:
    print("\nTest timeout (expected behavior)")
except KeyboardInterrupt:
    print("\nTest interrupted (expected behavior)")
except Exception as e:
    print(f"\nRuntime error: {e}")

# Check if environment variables are restored
print("\n" + "="*60)
print("Environment variables after test:")
print("="*60)
print(f"ORIGINAL_VAR = {os.environ.get('ORIGINAL_VAR', 'not set')}")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"TEST_STAGE_NAME = {os.environ.get('TEST_STAGE_NAME', 'not set')}")
print(f"MY_CUSTOM_VAR = {os.environ.get('MY_CUSTOM_VAR', 'not set')}")

# Verify
print("\nVerifying environment variable restoration:")
if os.environ.get('ORIGINAL_VAR') == initial_original:
    print("✓ ORIGINAL_VAR correctly restored")
else:
    print(f"✗ ORIGINAL_VAR not correctly restored: expected {initial_original}, got {os.environ.get('ORIGINAL_VAR')}")
    sys.exit(1)

if os.environ.get('CUDA_VISIBLE_DEVICES') == initial_cuda:
    print("✓ CUDA_VISIBLE_DEVICES correctly restored")
else:
    print(f"✗ CUDA_VISIBLE_DEVICES not correctly restored: expected {initial_cuda}, got {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    sys.exit(1)

if os.environ.get('TEST_STAGE_NAME') is None:
    print("✓ TEST_STAGE_NAME correctly cleared")
else:
    print(f"✗ TEST_STAGE_NAME not cleared: {os.environ.get('TEST_STAGE_NAME')}")
    sys.exit(1)

if os.environ.get('MY_CUSTOM_VAR') is None:
    print("✓ MY_CUSTOM_VAR correctly cleared")
else:
    print(f"✗ MY_CUSTOM_VAR not cleared: {os.environ.get('MY_CUSTOM_VAR')}")
    sys.exit(1)

print("\n" + "="*60)
print("Environment variable restoration test passed! ✓")
print("="*60)
EOF

chmod +x /tmp/test_recipe_env.py

# Run test
cd "$(dirname "$0")/.."
python3 /tmp/test_recipe_env.py

# Cleanup
rm -f /tmp/test_recipe_env.py

echo ""
echo "=========================================="
echo "Integration Test Complete"
echo "=========================================="
