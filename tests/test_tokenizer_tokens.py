from pathlib import Path

import pytest

from fluxperf.tokenizer import initialize_tokenizer, count_tokens, truncate_to_tokens, TokenizerNotInitializedError


@pytest.fixture(scope="module", autouse=True)
def _init_tokenizer():
    initialize_tokenizer(str(Path("tokenizers/Qwen2.5-7B-Instruct")))


def test_count_tokens_basic_cases():
    assert count_tokens("Hello world") == 2
    assert count_tokens("Hello, world!") == 4
    assert count_tokens("Hello    world") == 3
    assert count_tokens("") == 0


def test_truncate_to_tokens_shorter_returns_original():
    text = "quick brown fox"
    assert truncate_to_tokens(text, 10) == text


def test_truncate_to_tokens_exact_limit():
    text = "Hello wonderful world"
    tokens = count_tokens(text)
    assert tokens > 1
    assert truncate_to_tokens(text, tokens) == text
    shortened = truncate_to_tokens(text, tokens - 1)
    assert count_tokens(shortened) == tokens - 1


def test_count_tokens_without_init_raises(monkeypatch):
    from fluxperf import tokenizer as mod
    monkeypatch.setattr(mod, "_tokenizer", None)
    with pytest.raises(TokenizerNotInitializedError):
        mod.count_tokens("test")
