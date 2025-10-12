#!/usr/bin/env python3
"""
测试所有 dataclass 的创建和验证
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_round_benchmarker import (
    SLOConstraints,
    BenchmarkConfig,
    RequestMetrics,
    RoundMetrics,
    SessionData,
    RecipeStage,
    Recipe,
    BenchmarkMode
)


def test_slo_constraints():
    """测试 SLOConstraints dataclass"""
    print("测试 SLOConstraints...")
    
    # 创建空的 SLO
    slo1 = SLOConstraints()
    assert slo1.ttft_ms is None
    assert slo1.itl_ms is None
    assert slo1.latency_ms is None
    assert slo1.output_token_throughput is None
    
    # 创建带值的 SLO
    slo2 = SLOConstraints(
        ttft_ms=100.0,
        itl_ms=50.0,
        latency_ms=1000.0,
        output_token_throughput=10.0
    )
    assert slo2.ttft_ms == 100.0
    assert slo2.itl_ms == 50.0
    assert slo2.latency_ms == 1000.0
    assert slo2.output_token_throughput == 10.0
    
    print("✓ SLOConstraints 测试通过")


def test_benchmark_config():
    """测试 BenchmarkConfig dataclass"""
    print("测试 BenchmarkConfig...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10, 20],
        concurrency_levels=[5, 10],
        mode=BenchmarkMode.MULTI_TURN
    )
    
    assert config.dataset_path == Path("test.jsonl")
    assert config.endpoint_url == "http://localhost:8000"
    assert config.num_samples == [10, 20]
    assert config.concurrency_levels == [5, 10]
    assert config.mode == BenchmarkMode.MULTI_TURN
    assert config.model_name == "gpt-3.5-turbo"  # 默认值
    assert config.timeout == 300  # 默认值
    
    print("✓ BenchmarkConfig 测试通过")


def test_request_metrics():
    """测试 RequestMetrics dataclass"""
    print("测试 RequestMetrics...")
    
    metrics = RequestMetrics(
        request_id="test_req_1",
        round_num=1,
        input_text="Hello",
        output_text="World",
        input_tokens=1,
        output_tokens=1,
        time_to_first_token=0.5,
        inter_token_latencies=[10.0, 15.0, 12.0],
        total_latency=2.0,
        throughput=0.5,
        start_timestamp=1000.0,
        end_timestamp=1002.0
    )
    
    assert metrics.request_id == "test_req_1"
    assert metrics.round_num == 1
    assert metrics.input_text == "Hello"
    assert metrics.output_text == "World"
    assert metrics.time_to_first_token == 0.5
    assert len(metrics.inter_token_latencies) == 3
    assert metrics.error is None
    assert metrics.meets_slo is True  # 默认值
    
    print("✓ RequestMetrics 测试通过")


def test_round_metrics():
    """测试 RoundMetrics dataclass"""
    print("测试 RoundMetrics...")
    
    metrics = RoundMetrics(
        round_num=1,
        total_requests=100,
        successful_requests=95,
        failed_requests=5,
        avg_ttft=50.0,
        p50_ttft=45.0,
        p90_ttft=80.0,
        p95_ttft=90.0,
        p99_ttft=100.0,
        min_ttft=20.0,
        max_ttft=120.0,
        stddev_ttft=15.0,
        avg_itl=10.0,
        p50_itl=9.0,
        p90_itl=15.0,
        p95_itl=18.0,
        p99_itl=20.0,
        min_itl=5.0,
        max_itl=25.0,
        stddev_itl=3.0,
        avg_latency=500.0,
        p50_latency=480.0,
        p90_latency=600.0,
        p95_latency=650.0,
        p99_latency=700.0,
        min_latency=400.0,
        max_latency=800.0,
        stddev_latency=50.0,
        total_throughput=100.0,
        request_throughput=50.0,
        avg_input_tokens=100.0,
        p50_input_tokens=100.0,
        p90_input_tokens=150.0,
        p99_input_tokens=200.0,
        min_input_tokens=50.0,
        max_input_tokens=250.0,
        stddev_input_tokens=30.0,
        avg_output_tokens=50.0,
        p50_output_tokens=50.0,
        p90_output_tokens=80.0,
        p99_output_tokens=100.0,
        min_output_tokens=20.0,
        max_output_tokens=120.0,
        stddev_output_tokens=15.0,
        duration=10.0,
        stage_name="Test Stage",
        concurrency=10
    )
    
    assert metrics.round_num == 1
    assert metrics.total_requests == 100
    assert metrics.successful_requests == 95
    assert metrics.failed_requests == 5
    assert metrics.stage_name == "Test Stage"
    assert metrics.concurrency == 10
    
    print("✓ RoundMetrics 测试通过")


def test_session_data():
    """测试 SessionData dataclass"""
    print("测试 SessionData...")
    
    session = SessionData(
        session_id="session_1",
        user_messages=["Hello", "How are you?"],
        metadata={"key": "value"}
    )
    
    assert session.session_id == "session_1"
    assert len(session.user_messages) == 2
    assert session.user_messages[0] == "Hello"
    assert session.metadata == {"key": "value"}
    assert len(session.assistant_messages) == 0  # 默认值
    
    print("✓ SessionData 测试通过")


def test_recipe_stage():
    """测试 RecipeStage dataclass"""
    print("测试 RecipeStage...")
    
    stage = RecipeStage(
        name="Test Stage",
        concurrency_levels=[5, 10],
        num_samples=[10, 20],
        env={"CUDA_VISIBLE_DEVICES": "0"}
    )
    
    assert stage.name == "Test Stage"
    assert stage.concurrency_levels == [5, 10]
    assert stage.num_samples == [10, 20]
    assert stage.env == {"CUDA_VISIBLE_DEVICES": "0"}
    
    # 测试空 env
    stage2 = RecipeStage(
        name="Stage 2",
        concurrency_levels=[5],
        num_samples=[10]
    )
    assert stage2.env == {}
    
    print("✓ RecipeStage 测试通过")


def test_recipe():
    """测试 Recipe dataclass"""
    print("测试 Recipe...")
    
    recipe = Recipe(
        global_config={"dataset": "test.jsonl", "mode": "multi_turn"},
        stages=[
            RecipeStage("Stage 1", [5], [10])
        ],
        mock_server={"enabled": True, "port": 8765}
    )
    
    assert recipe.global_config["dataset"] == "test.jsonl"
    assert len(recipe.stages) == 1
    assert recipe.mock_server["enabled"] is True
    
    # 测试没有 mock_server
    recipe2 = Recipe(
        global_config={},
        stages=[]
    )
    assert recipe2.mock_server is None
    
    print("✓ Recipe 测试通过")


def test_benchmark_mode_enum():
    """测试 BenchmarkMode 枚举"""
    print("测试 BenchmarkMode...")
    
    assert BenchmarkMode.DUAL_ROUND.value == "dual_round"
    assert BenchmarkMode.MULTI_TURN.value == "multi_turn"
    
    # 测试从字符串创建
    mode1 = BenchmarkMode("dual_round")
    assert mode1 == BenchmarkMode.DUAL_ROUND
    
    mode2 = BenchmarkMode("multi_turn")
    assert mode2 == BenchmarkMode.MULTI_TURN
    
    # 测试无效值
    try:
        BenchmarkMode("invalid")
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass
    
    print("✓ BenchmarkMode 测试通过")


if __name__ == '__main__':
    try:
        test_slo_constraints()
        test_benchmark_config()
        test_request_metrics()
        test_round_metrics()
        test_session_data()
        test_recipe_stage()
        test_recipe()
        test_benchmark_mode_enum()
        
        print("\n" + "=" * 60)
        print("所有 dataclass 测试通过! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
