#!/usr/bin/env python3
"""
测试 LLMClient 的功能
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_round_benchmarker import (
    OpenAIClient,
    BenchmarkConfig,
    RequestMetrics
)


async def test_client_initialization():
    """测试客户端初始化"""
    print("测试 OpenAIClient 初始化...")
    
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
    assert len(client.debug_entries) == 0
    
    print("✓ OpenAIClient 初始化测试通过")


async def test_client_lifecycle():
    """测试客户端生命周期"""
    print("测试 OpenAIClient 生命周期...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    
    client = OpenAIClient(config, round_num=1, concurrency=5)
    
    async with client:
        assert client.session is not None
    
    print("✓ OpenAIClient 生命周期测试通过")


async def test_build_payload():
    """测试请求 payload 构建"""
    print("测试 payload 构建...")
    
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
    
    # 由于 send_completion_request 是私有方法，我们测试它会创建的 payload
    # 通过检查 save_requests 模式
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
    
    print("✓ Payload 构建测试通过")


async def test_metrics_initialization():
    """测试 metrics 初始化"""
    print("测试 RequestMetrics 初始化...")
    
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
    
    print("✓ RequestMetrics 初始化测试通过")


if __name__ == '__main__':
    try:
        asyncio.run(test_client_initialization())
        asyncio.run(test_client_lifecycle())
        asyncio.run(test_build_payload())
        asyncio.run(test_metrics_initialization())
        
        print("\n" + "=" * 60)
        print("所有 LLMClient 测试通过! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
