#!/usr/bin/env python3
"""
Test JSONL output format from convert_longbench.py

Verifies output file conforms to JSONL standard:
- One independent JSON object per line
- No commas between lines
- No wrapping brackets
"""

import json
import sys
from pathlib import Path


def test_jsonl_format(file_path: Path) -> bool:
    """
    Test if file is valid JSONL format

    Args:
        file_path: Path to file to test

    Returns:
        True if format is correct, False otherwise
    """
    print(f"Testing file: {file_path}")

    if not file_path.exists():
        print(f"❌ File does not exist: {file_path}")
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            print("❌ File is empty")
            return False

        # Check if starts with [ (JSON array format, wrong)
        first_line = lines[0].strip()
        if first_line.startswith('['):
            print("❌ File starts with '[', this is JSON array format, not JSONL")
            return False

        # Check each line
        valid_lines = 0
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # Check if ends with comma (JSON array element, wrong)
            if line.endswith(','):
                print(f"❌ Line {i} ends with comma, this is JSON array format, not JSONL")
                return False

            # Try to parse as JSON object
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    print(f"❌ Line {i} is not a JSON object: {type(obj)}")
                    return False
                valid_lines += 1
            except json.JSONDecodeError as e:
                print(f"❌ Line {i} JSON parsing failed: {e}")
                return False

        print(f"✅ JSONL format is correct!")
        print(f"   Total lines: {len(lines)}")
        print(f"   Valid objects: {valid_lines}")

        # Display example of first object
        with open(file_path, 'r', encoding='utf-8') as f:
            first_obj = json.loads(f.readline())
            print(f"\nFirst object example:")
            print(f"  Keys: {list(first_obj.keys())}")
            if 'text' in first_obj:
                text_preview = first_obj['text'][:100] + "..." if len(first_obj['text']) > 100 else first_obj['text']
                print(f"  text preview: {text_preview}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        print("Usage: python test_jsonl_output.py <jsonl_file>")
        print("\nExample:")
        print("  python test_jsonl_output.py datasets/output.jsonl")
        sys.exit(1)

    success = test_jsonl_format(file_path)
    sys.exit(0 if success else 1)
