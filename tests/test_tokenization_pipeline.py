import asyncio
import json
import socket
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import ClientSession

from fluxperf.text_dataset_generator import init_tokenizer, generate_single_turn_dataset
from fluxperf.tokenizer import initialize_tokenizer, count_tokens as flux_count_tokens
from llm_mocker import run_server_until_cancelled


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_token_counts_consistent_across_components(tmp_path):
    model_path = Path("tokenizers/Qwen2.5-7B-Instruct")
    dataset_path = Path("datasets/shakespeare.txt")

    tokenizer = init_tokenizer(str(model_path))
    initialize_tokenizer(str(model_path))

    corpus = dataset_path.read_text(encoding="utf-8")[:20000]
    output_file = tmp_path / "sample_dataset.jsonl"

    target_tokens = 32

    generate_single_turn_dataset(
        corpus=corpus,
        num_entries=1,
        entry_length=target_tokens,
        output_file=output_file,
        no_overlap=True,
        use_tokens=True,
        tokenizer=tokenizer,
    )

    with output_file.open("r", encoding="utf-8") as fh:
        entry = json.loads(fh.readline())

    sample_text = entry["text"]
    generator_count = entry["token_count"]
    direct_count = flux_count_tokens(sample_text)

    assert generator_count == target_tokens
    assert direct_count == target_tokens

    host = "127.0.0.1"
    port = _get_free_port()
    shutdown_event = asyncio.Event()

    with patch("llm_mocker.count_tokens", flux_count_tokens):
        server_task = asyncio.create_task(run_server_until_cancelled(host, port, shutdown_event))
        await asyncio.sleep(0.3)

        payload = {
            "model": "mock-model",
            "messages": [
                {"role": "user", "content": sample_text},
            ],
        }

        async with ClientSession() as session:
            async with session.post(f"http://{host}:{port}/v1/chat/completions", json=payload) as resp:
                assert resp.status == 200
                body = await resp.json()

        shutdown_event.set()
        await server_task

    usage_prompt_tokens = body["usage"]["prompt_tokens"]
    assert usage_prompt_tokens == target_tokens

    assert generator_count == direct_count == usage_prompt_tokens
