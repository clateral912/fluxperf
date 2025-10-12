#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, TextIO


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    if not input_path.is_dir():
        raise ValueError(f"输入路径无效: {input_path}")
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            yield path


def load_sessions(path: Path) -> Iterable[dict]:
    # 如果是 JSONL 格式（每行一个 JSON 对象）
    if path.suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            yield item
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSON 解析失败: {path}:{line_num}: {exc}") from exc
        except Exception as exc:
            if not isinstance(exc, ValueError):
                raise ValueError(f"读取文件失败: {path}: {exc}") from exc
            raise
        return

    # 如果是普通 JSON 格式
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {path}: {exc}") from exc
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        conversations = data.get("data")
        if isinstance(conversations, list):
            for item in conversations:
                if isinstance(item, dict):
                    yield item
    else:
        raise ValueError(f"不支持的 JSON 结构: {path}")


def extract_user_messages(session: dict) -> list[str]:
    messages: list[str] = []

    conversations = session.get("conversations")
    if isinstance(conversations, list):
        for msg in conversations:
            if not isinstance(msg, dict):
                continue
            if msg.get("from") != "human":
                continue
            value = msg.get("value")
            if not isinstance(value, str):
                continue
            value = value.strip()
            if value:
                messages.append(value)

    dialogs = session.get("messages")
    if isinstance(dialogs, list):
        for msg in dialogs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("from")
            if role not in {"user", "human"}:
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    messages.append(text)
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_val = part.get("text")
                        if isinstance(text_val, str) and text_val.strip():
                            parts.append(text_val.strip())
                if parts:
                    messages.append("\n".join(parts))

    if messages:
        return messages

    text = session.get("text") or session.get("prompt")
    if isinstance(text, str) and text.strip():
        return [text.strip()]

    return []


def write_output(records: Iterable[dict], writer: TextIO) -> int:
    count = 0
    for record in records:
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
    writer.flush()
    return count


def process_dataset(input_path: Path, output_path: Path, max_sessions: int | None) -> int:
    seen_ids: set[str] = set()
    total_written = 0
    with output_path.open("w", encoding="utf-8") as out_file:
        for source_file in iter_input_files(input_path):
            for session in load_sessions(source_file):
                session_id = session.get("id")
                if isinstance(session_id, str) and session_id:
                    key = session_id
                else:
                    key = f"{source_file.name}-{total_written}"
                if key in seen_ids:
                    continue
                messages = extract_user_messages(session)
                if not messages:
                    continue
                record = {
                    "session_id": key,
                    "user_messages": messages,
                    "source": str(source_file)
                }
                total_written += write_output([record], out_file)
                seen_ids.add(key)
                if max_sessions is not None and total_written >= max_sessions:
                    return total_written
    return total_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="process_sharegpt", description="提取 ShareGPT 数据集中所有 human 发言")
    parser.add_argument("input", type=Path, help="ShareGPT 数据集路径，支持文件或目录")
    parser.add_argument("output", type=Path, help="输出 JSONL 文件路径")
    parser.add_argument("--max-sessions", type=int, default=None, help="最多处理的会话数量")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = process_dataset(args.input, args.output, args.max_sessions)
    except Exception as exc:
        print(f"处理失败: {exc}", file=sys.stderr)
        return 1
    print(f"已写入 {written} 条会话到 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
