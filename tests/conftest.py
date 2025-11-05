import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxperf.tokenizer import initialize_tokenizer

def _ensure_tokenizer():
    token_dir = Path(__file__).parent.parent / "tokenizers" / "Qwen2.5-7B-Instruct"
    if not token_dir.exists():
        raise FileNotFoundError(f"无法找到 tokenizer 目录: {token_dir}")
    initialize_tokenizer(str(token_dir))

def pytest_configure(config):
    _ensure_tokenizer()

@pytest.fixture
def tokenized_input(tmp_path):
    _ensure_tokenizer()
    return tmp_path

@pytest.fixture
def file_path(tmp_path_factory):
    return tmp_path_factory.mktemp("jsonl_tests") / "output.jsonl"
