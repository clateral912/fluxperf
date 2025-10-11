#!/usr/bin/env python3
"""
测试 convert_longbench.py 的 JSONL 输出格式

验证输出文件是否符合 JSONL 标准：
- 每行一个独立的 JSON 对象
- 行之间没有逗号
- 没有包裹的方括号
"""

import json
import sys
from pathlib import Path


def test_jsonl_format(file_path: Path) -> bool:
    """
    测试文件是否为有效的 JSONL 格式

    Args:
        file_path: 要测试的文件路径

    Returns:
        True 如果格式正确，False 否则
    """
    print(f"测试文件: {file_path}")

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print("❌ 文件为空")
            return False

        # 检查是否以 [ 开头（JSON 数组格式，错误）
        first_line = lines[0].strip()
        if first_line.startswith('['):
            print("❌ 文件以 '[' 开头，这是 JSON 数组格式，不是 JSONL")
            return False

        # 检查每一行
        valid_lines = 0
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # 检查是否有逗号结尾（JSON 数组元素，错误）
            if line.endswith(','):
                print(f"❌ 第 {i} 行以逗号结尾，这是 JSON 数组格式，不是 JSONL")
                return False

            # 尝试解析为 JSON 对象
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    print(f"❌ 第 {i} 行不是 JSON 对象: {type(obj)}")
                    return False
                valid_lines += 1
            except json.JSONDecodeError as e:
                print(f"❌ 第 {i} 行 JSON 解析失败: {e}")
                return False

        print(f"✅ JSONL 格式正确!")
        print(f"   总行数: {len(lines)}")
        print(f"   有效对象数: {valid_lines}")

        # 显示第一个对象的示例
        with open(file_path, 'r', encoding='utf-8') as f:
            first_obj = json.loads(f.readline())
            print(f"\n第一个对象示例:")
            print(f"  键: {list(first_obj.keys())}")
            if 'text' in first_obj:
                text_preview = first_obj['text'][:100] + "..." if len(first_obj['text']) > 100 else first_obj['text']
                print(f"  text 预览: {text_preview}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        print("用法: python test_jsonl_output.py <jsonl_file>")
        print("\n示例:")
        print("  python test_jsonl_output.py datasets/output.jsonl")
        sys.exit(1)

    success = test_jsonl_format(file_path)
    sys.exit(0 if success else 1)
