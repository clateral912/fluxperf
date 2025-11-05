#!/usr/bin/env python3

import functools
from typing import Optional


class TokenizerNotInitializedError(RuntimeError):
    """Raised when token-dependent operations are attempted without initialization."""
    pass


_tokenizer = None
_warned_simple_split = False


def initialize_tokenizer(
    tokenizer_name: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None
):
    global _tokenizer, _warned_simple_split
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

    try:
        # Prefer fast tokenizers for performance when available.
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    except Exception as fast_exc:
        # Some vendor drops only provide a slow SentencePiece model; retry with use_fast disabled.
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["use_fast"] = False
        try:
            _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **fallback_kwargs)
        except Exception as slow_exc:
            raise RuntimeError(
                f"Failed to initialize tokenizer '{tokenizer_name}' with fast and slow loaders"
            ) from slow_exc
        else:
            # Preserve original context in logs for easier debugging.
            print(
                f"Warning: fast tokenizer load failed for '{tokenizer_name}', falling back to slow implementation: {fast_exc}"
            )

    # Reset warning flag and cached whitespace results now that tokenizer is ready.
    _warned_simple_split = False
    try:
        count_tokens.cache_clear()
    except AttributeError:
        pass


def _get_tokenizer():
    if _tokenizer is None:
        raise TokenizerNotInitializedError(
            "Tokenizer 尚未初始化，请先调用 initialize_tokenizer"
        )
    return _tokenizer


@functools.lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    if not text:
        return 0

    tok = _tokenizer
    if tok is None:
        raise TokenizerNotInitializedError(
            "Tokenizer 尚未初始化，无法计算 token 数；请先调用 initialize_tokenizer"
        )

    if hasattr(tok, "encode"):
        return len(tok.encode(text, add_special_tokens=False))

    if hasattr(tok, "__call__"):
        outputs = tok(text, add_special_tokens=False)
        if "input_ids" in outputs:
            return len(outputs["input_ids"])

    raise RuntimeError("Tokenizer 对象不支持 encode/__call__ 接口，无法计算 token 数")


def truncate_to_tokens(text: str, max_tokens: Optional[int]) -> str:
    """Trim a string so that it contains at most `max_tokens` tokens."""
    if text is None or max_tokens is None:
        return text
    if max_tokens <= 0:
        return ""

    tok = _get_tokenizer()
    encoded = tok.encode(text, add_special_tokens=False)
    if len(encoded) <= max_tokens:
        return text
    trimmed = tok.decode(encoded[:max_tokens], skip_special_tokens=True)
    return trimmed
