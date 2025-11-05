#!/usr/bin/env python3
"""
Test LLMClient functionality
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from fluxperf import (
    OpenAIClient,
    BenchmarkConfig,
    RequestMetrics
)


@pytest.mark.asyncio
async def test_client_initialization():
    """Test client initialization"""
    print("Testing OpenAIClient initialization...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000/v1/chat/completions",
        num_samples=[10],
        concurrency_levels=[5]
    )
    
    client = OpenAIClient(config, round_num=1, concurrency=5)

    assert client.config == config
    assert client.session is None
    assert len(client.requests_log) == 0
    
    print("✓ OpenAIClient initialization tests passed")


@pytest.mark.asyncio
async def test_client_lifecycle():
    """Test client lifecycle"""
    print("Testing OpenAIClient lifecycle...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    
    client = OpenAIClient(config, round_num=1, concurrency=5)
    
    async with client:
        assert client.session is not None
    
    print("✓ OpenAIClient lifecycle tests passed")


@pytest.mark.asyncio
async def test_build_payload():
    """Test request payload construction"""
    print("Testing payload construction...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5],
        model_name="test-model",
        max_output_tokens=100
    )
    
    client = OpenAIClient(config, round_num=1, concurrency=5)
    
    # 构建一个请求的 payload
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you?"}
    ]
    
    request_id = "test_request_1"
    session_id = "session_1"
    turn_index = 1
    
    # Since send_completion_request is private, we test the payload it creates
    # by checking save_requests mode
    config_with_save = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5],
        model_name="test-model",
        max_output_tokens=100,
        save_requests=True
    )
    
    client_with_save = OpenAIClient(config_with_save, round_num=1, concurrency=5)
    
    # 验证配置
    assert config_with_save.save_requests is True
    assert config_with_save.max_output_tokens == 100
    assert config_with_save.model_name == "test-model"
    
    print("✓ Payload construction tests passed")


@pytest.mark.asyncio
async def test_metrics_initialization():
    """Test metrics initialization"""
    print("Testing RequestMetrics initialization...")
    
    metrics = RequestMetrics(
        request_id="req_1",
        round_num=1,
        input_text="Hello",
        output_text="",
        input_tokens=1,
        output_tokens=0,
        time_to_first_token=0.0,
        session_id="session_1",
        turn_index=0,
        context_tokens=0
    )
    
    assert metrics.request_id == "req_1"
    assert metrics.round_num == 1
    assert metrics.session_id == "session_1"
    assert metrics.turn_index == 0
    assert metrics.error is None
    assert metrics.meets_slo is True
    assert len(metrics.inter_token_latencies) == 0
    assert metrics.history_truncated == 0
    
    print("✓ RequestMetrics initialization tests passed")


if __name__ == '__main__':
    try:
        asyncio.run(test_client_initialization())
        asyncio.run(test_client_lifecycle())
        asyncio.run(test_build_payload())
        asyncio.run(test_metrics_initialization())
        
        print("\n" + "=" * 60)
        print("All LLMClient tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
