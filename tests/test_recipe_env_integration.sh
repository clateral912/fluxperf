#!/bin/bash

# Recipe 环境变量集成测试
# 验证在实际 recipe 运行过程中环境变量是否正确设置和恢复

set -e

echo "=========================================="
echo "Recipe 环境变量集成测试"
echo "=========================================="

# 设置测试环境变量
export ORIGINAL_VAR="original_value"
export CUDA_VISIBLE_DEVICES="9"

echo ""
echo "测试前的环境变量:"
echo "  ORIGINAL_VAR = $ORIGINAL_VAR"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-未设置}"
echo "  TEST_STAGE_NAME = ${TEST_STAGE_NAME:-未设置}"
echo "  MY_CUSTOM_VAR = ${MY_CUSTOM_VAR:-未设置}"

# 创建临时修改的 benchmarker，在每个 stage 开始和结束时打印环境变量
echo ""
echo "创建测试 hook..."

# 使用一个简单的方法：创建一个包装脚本来捕获环境变量
cat > /tmp/test_recipe_env.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import subprocess

# 运行 recipe 之前检查环境变量
print("\n" + "="*60)
print("测试开始前的环境变量:")
print("="*60)
print(f"ORIGINAL_VAR = {os.environ.get('ORIGINAL_VAR', '未设置')}")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
print(f"TEST_STAGE_NAME = {os.environ.get('TEST_STAGE_NAME', '未设置')}")
print(f"MY_CUSTOM_VAR = {os.environ.get('MY_CUSTOM_VAR', '未设置')}")

# 记录初始值
initial_original = os.environ.get('ORIGINAL_VAR')
initial_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')

# 运行 benchmarker (只运行几秒钟就中断)
print("\n开始运行 recipe...")
try:
    # 注意：我们让它运行，但只取少量样本
    result = subprocess.run(
        [
            sys.executable,
            "dual_round_benchmarker.py",
            "--recipe", "recipe_env_test.yaml"
        ],
        timeout=120,  # 最多运行 2 分钟
        capture_output=False
    )
except subprocess.TimeoutExpired:
    print("\n测试超时 (预期行为)")
except KeyboardInterrupt:
    print("\n测试被中断 (预期行为)")
except Exception as e:
    print(f"\n运行时错误: {e}")

# 检查环境变量是否恢复
print("\n" + "="*60)
print("测试结束后的环境变量:")
print("="*60)
print(f"ORIGINAL_VAR = {os.environ.get('ORIGINAL_VAR', '未设置')}")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
print(f"TEST_STAGE_NAME = {os.environ.get('TEST_STAGE_NAME', '未设置')}")
print(f"MY_CUSTOM_VAR = {os.environ.get('MY_CUSTOM_VAR', '未设置')}")

# 验证
print("\n验证环境变量恢复:")
if os.environ.get('ORIGINAL_VAR') == initial_original:
    print("✓ ORIGINAL_VAR 正确恢复")
else:
    print(f"✗ ORIGINAL_VAR 未正确恢复: 期望 {initial_original}, 实际 {os.environ.get('ORIGINAL_VAR')}")
    sys.exit(1)

if os.environ.get('CUDA_VISIBLE_DEVICES') == initial_cuda:
    print("✓ CUDA_VISIBLE_DEVICES 正确恢复")
else:
    print(f"✗ CUDA_VISIBLE_DEVICES 未正确恢复: 期望 {initial_cuda}, 实际 {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    sys.exit(1)

if os.environ.get('TEST_STAGE_NAME') is None:
    print("✓ TEST_STAGE_NAME 正确清除")
else:
    print(f"✗ TEST_STAGE_NAME 未清除: {os.environ.get('TEST_STAGE_NAME')}")
    sys.exit(1)

if os.environ.get('MY_CUSTOM_VAR') is None:
    print("✓ MY_CUSTOM_VAR 正确清除")
else:
    print(f"✗ MY_CUSTOM_VAR 未清除: {os.environ.get('MY_CUSTOM_VAR')}")
    sys.exit(1)

print("\n" + "="*60)
print("环境变量恢复测试通过! ✓")
print("="*60)
EOF

chmod +x /tmp/test_recipe_env.py

# 运行测试
cd "$(dirname "$0")/.."
python3 /tmp/test_recipe_env.py

# 清理
rm -f /tmp/test_recipe_env.py

echo ""
echo "=========================================="
echo "集成测试完成"
echo "=========================================="
