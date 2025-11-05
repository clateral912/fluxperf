#!/usr/bin/env python3
"""Utility for converting ShareGPT style JSON dumps into FluxPerf sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional


def _load_source(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            yield from data["data"]
        else:
            raise ValueError("不支持的 JSON 结构，期待列表或包含 data 字段的字典")
    elif path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")


def _extract_human_messages(entry: dict) -> List[str]:
    messages: List[str] = []

    conversations = entry.get("conversations")
    if isinstance(conversations, list):
        for item in conversations:
            if isinstance(item, dict) and item.get("from") == "human":
                value = item.get("value")
                if isinstance(value, str) and value.strip():
                    messages.append(value)

    chat_messages = entry.get("messages")
    if isinstance(chat_messages, list):
        for item in chat_messages:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            if role not in {"user", "human"}:
                continue
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                messages.append(content)

    user_messages = entry.get("user_messages")
    if isinstance(user_messages, list):
        for msg in user_messages:
            if isinstance(msg, str) and msg.strip():
                messages.append(msg)

    text = entry.get("text")
    if isinstance(text, str) and text.strip():
        messages.append(text)

    return messages


def process_dataset(input_path: Path, output_path: Path, max_turns: Optional[int] = None) -> int:
    """Convert ShareGPT style dataset into FluxPerf JSONL sessions.

    Args:
        input_path: 源数据路径（.json 或 .jsonl）。
        output_path: 输出 JSONL 文件路径。
        max_turns: 可选，限制最多保留的 user message 条数。

    Returns:
        写入的记录数。
    """

    entries = []
    for idx, entry in enumerate(_load_source(input_path)):
        session_id = entry.get("id") or entry.get("session_id") or f"session-{idx}"
        if not isinstance(session_id, str):
            session_id = str(session_id)

        messages = _extract_human_messages(entry)
        if max_turns is not None:
            messages = messages[:max_turns]

        if not messages:
            continue

        entries.append({
            "session_id": session_id,
            "user_messages": messages,
            "metadata": entry,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in entries:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(entries)


__all__ = ["process_dataset"]
