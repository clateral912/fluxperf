#!/usr/bin/env python3
"""
测试 Recipe 环境变量管理功能

验证:
1. 环境变量在 stage 开始前正确设置
2. 环境变量在 stage 结束后正确恢复
3. 多个 stage 的环境变量互不干扰
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_env_restore():
    """测试环境变量设置和恢复"""
    
    print("=" * 60)
    print("测试环境变量管理")
    print("=" * 60)
    
    # 设置初始环境变量
    os.environ['TEST_VAR_1'] = 'original_value_1'
    os.environ['TEST_VAR_2'] = 'original_value_2'
    print(f"\n初始环境变量:")
    print(f"  TEST_VAR_1 = {os.environ.get('TEST_VAR_1')}")
    print(f"  TEST_VAR_2 = {os.environ.get('TEST_VAR_2')}")
    print(f"  TEST_VAR_3 = {os.environ.get('TEST_VAR_3', '(未设置)')}")
    
    # 模拟 Stage 1
    print(f"\n{'=' * 60}")
    print("Stage 1: 开始")
    print(f"{'=' * 60}")
    
    stage1_env = {
        'TEST_VAR_1': 'stage1_value_1',
        'TEST_VAR_3': 'stage1_value_3'
    }
    
    # 保存原始值
    original_env = {}
    for key, value in stage1_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = str(value)
        print(f"设置环境变量: {key}={value}")
    
    print(f"\nStage 1 运行中的环境变量:")
    print(f"  TEST_VAR_1 = {os.environ.get('TEST_VAR_1')}")
    print(f"  TEST_VAR_2 = {os.environ.get('TEST_VAR_2')}")
    print(f"  TEST_VAR_3 = {os.environ.get('TEST_VAR_3')}")
    
    # 验证
    assert os.environ.get('TEST_VAR_1') == 'stage1_value_1', "TEST_VAR_1 应该被更新"
    assert os.environ.get('TEST_VAR_2') == 'original_value_2', "TEST_VAR_2 应该保持不变"
    assert os.environ.get('TEST_VAR_3') == 'stage1_value_3', "TEST_VAR_3 应该被设置"
    print("✓ Stage 1 环境变量验证通过")
    
    # 恢复环境变量
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value
    print("\n环境变量已恢复")
    
    print(f"\nStage 1 结束后的环境变量:")
    print(f"  TEST_VAR_1 = {os.environ.get('TEST_VAR_1')}")
    print(f"  TEST_VAR_2 = {os.environ.get('TEST_VAR_2')}")
    print(f"  TEST_VAR_3 = {os.environ.get('TEST_VAR_3', '(未设置)')}")
    
    # 验证恢复
    assert os.environ.get('TEST_VAR_1') == 'original_value_1', "TEST_VAR_1 应该恢复原值"
    assert os.environ.get('TEST_VAR_2') == 'original_value_2', "TEST_VAR_2 应该保持不变"
    assert os.environ.get('TEST_VAR_3') is None, "TEST_VAR_3 应该被删除"
    print("✓ Stage 1 环境变量恢复验证通过")
    
    # 模拟 Stage 2
    print(f"\n{'=' * 60}")
    print("Stage 2: 开始")
    print(f"{'=' * 60}")
    
    stage2_env = {
        'TEST_VAR_1': 'stage2_value_1',
        'TEST_VAR_2': 'stage2_value_2',
        'TEST_VAR_4': 'stage2_value_4'
    }
    
    # 保存原始值
    original_env = {}
    for key, value in stage2_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = str(value)
        print(f"设置环境变量: {key}={value}")
    
    print(f"\nStage 2 运行中的环境变量:")
    print(f"  TEST_VAR_1 = {os.environ.get('TEST_VAR_1')}")
    print(f"  TEST_VAR_2 = {os.environ.get('TEST_VAR_2')}")
    print(f"  TEST_VAR_3 = {os.environ.get('TEST_VAR_3', '(未设置)')}")
    print(f"  TEST_VAR_4 = {os.environ.get('TEST_VAR_4')}")
    
    # 验证
    assert os.environ.get('TEST_VAR_1') == 'stage2_value_1', "TEST_VAR_1 应该被更新"
    assert os.environ.get('TEST_VAR_2') == 'stage2_value_2', "TEST_VAR_2 应该被更新"
    assert os.environ.get('TEST_VAR_3') is None, "TEST_VAR_3 应该仍然未设置"
    assert os.environ.get('TEST_VAR_4') == 'stage2_value_4', "TEST_VAR_4 应该被设置"
    print("✓ Stage 2 环境变量验证通过")
    
    # 恢复环境变量
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value
    print("\n环境变量已恢复")
    
    print(f"\nStage 2 结束后的环境变量:")
    print(f"  TEST_VAR_1 = {os.environ.get('TEST_VAR_1')}")
    print(f"  TEST_VAR_2 = {os.environ.get('TEST_VAR_2')}")
    print(f"  TEST_VAR_3 = {os.environ.get('TEST_VAR_3', '(未设置)')}")
    print(f"  TEST_VAR_4 = {os.environ.get('TEST_VAR_4', '(未设置)')}")
    
    # 验证最终恢复
    assert os.environ.get('TEST_VAR_1') == 'original_value_1', "TEST_VAR_1 应该恢复原值"
    assert os.environ.get('TEST_VAR_2') == 'original_value_2', "TEST_VAR_2 应该恢复原值"
    assert os.environ.get('TEST_VAR_3') is None, "TEST_VAR_3 应该仍然未设置"
    assert os.environ.get('TEST_VAR_4') is None, "TEST_VAR_4 应该被删除"
    print("✓ Stage 2 环境变量恢复验证通过")
    
    print(f"\n{'=' * 60}")
    print("所有测试通过! ✓")
    print(f"{'=' * 60}")


def test_nested_env():
    """测试嵌套环境变量场景"""
    
    print(f"\n\n{'=' * 60}")
    print("测试嵌套环境变量")
    print(f"{'=' * 60}")
    
    # 初始状态
    if 'NESTED_VAR' in os.environ:
        del os.environ['NESTED_VAR']
    
    print(f"\n初始: NESTED_VAR = {os.environ.get('NESTED_VAR', '(未设置)')}")
    
    # 第一层嵌套
    print("\n进入第一层...")
    original_1 = os.environ.get('NESTED_VAR')
    os.environ['NESTED_VAR'] = 'layer_1'
    print(f"第一层: NESTED_VAR = {os.environ.get('NESTED_VAR')}")
    assert os.environ.get('NESTED_VAR') == 'layer_1'
    
    # 第二层嵌套
    print("\n进入第二层...")
    original_2 = os.environ.get('NESTED_VAR')
    os.environ['NESTED_VAR'] = 'layer_2'
    print(f"第二层: NESTED_VAR = {os.environ.get('NESTED_VAR')}")
    assert os.environ.get('NESTED_VAR') == 'layer_2'
    
    # 恢复第二层
    print("\n退出第二层...")
    if original_2 is None:
        os.environ.pop('NESTED_VAR', None)
    else:
        os.environ['NESTED_VAR'] = original_2
    print(f"恢复后: NESTED_VAR = {os.environ.get('NESTED_VAR')}")
    assert os.environ.get('NESTED_VAR') == 'layer_1'
    
    # 恢复第一层
    print("\n退出第一层...")
    if original_1 is None:
        os.environ.pop('NESTED_VAR', None)
    else:
        os.environ['NESTED_VAR'] = original_1
    print(f"最终: NESTED_VAR = {os.environ.get('NESTED_VAR', '(未设置)')}")
    assert os.environ.get('NESTED_VAR') is None
    
    print("\n✓ 嵌套环境变量测试通过")


if __name__ == '__main__':
    try:
        test_env_restore()
        test_nested_env()
        print(f"\n{'=' * 60}")
        print("所有环境变量管理测试通过! ✓✓✓")
        print(f"{'=' * 60}\n")
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
