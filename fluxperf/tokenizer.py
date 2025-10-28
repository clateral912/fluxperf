#!/usr/bin/env python3

import functools
from typing import Optional


_tokenizer = None


def initialize_tokenizer(
    tokenizer_name: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None
):
    global _tokenizer
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers 库未安装，无法初始化 tokenizer"
        ) from exc

    if not tokenizer_name:
        raise ValueError("tokenizer_name 不能为空")

    kwargs = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision

    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def _get_tokenizer():
    if _tokenizer is None:
        raise RuntimeError("Tokenizer 尚未初始化，请先调用 initialize_tokenizer")
    return _tokenizer


@functools.lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    if not text:
        return 0

    tok = _tokenizer
    if tok is None:
        # Fallback to simple whitespace split
        return len(text.strip().split())

    if hasattr(tok, "encode"):
        return len(tok.encode(text, add_special_tokens=False))

    if hasattr(tok, "__call__"):
        outputs = tok(text, add_special_tokens=False)
        if "input_ids" in outputs:
            return len(outputs["input_ids"])

    return len(text.strip().split())
