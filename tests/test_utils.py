#!/usr/bin/env python3
"""
Test utility functions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxperf import (
    count_tokens,
    DatasetLoader,
    SLOLoader,
    RecipeLoader,
    MetricsAnalyzer
)


def test_count_tokens():
    """Test token counting function"""
    print("Testing count_tokens...")
    
    # 正常文本
    assert count_tokens("Hello world") == 2
    assert count_tokens("This is a test") == 4
    
    # 空文本
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0
    
    # 多空格
    assert count_tokens("Hello    world") == 2
    
    # 特殊字符
    assert count_tokens("Hello, world!") == 2
    
    # None 处理
    assert count_tokens(None) == 0
    
    print("✓ count_tokens tests passed")


def test_metrics_analyzer_calculate_percentile():
    """Test percentile calculation"""
    print("Testing MetricsAnalyzer.calculate_percentile...")
    
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # P50 (中位数)
    p50 = MetricsAnalyzer.calculate_percentile(values, 50)
    assert p50 == 5
    
    # P90
    p90 = MetricsAnalyzer.calculate_percentile(values, 90)
    assert p90 == 9
    
    # P99
    p99 = MetricsAnalyzer.calculate_percentile(values, 99)
    assert p99 == 10
    
    # 空列表
    assert MetricsAnalyzer.calculate_percentile([], 50) == 0.0
    
    # 单个值
    assert MetricsAnalyzer.calculate_percentile([5], 50) == 5
    
    print("✓ calculate_percentile tests passed")


def test_metrics_analyzer_truncate_text():
    """Test text truncation"""
    print("Testing MetricsAnalyzer._truncate_text...")
    
    # 短文本
    text = "Hello"
    assert MetricsAnalyzer._truncate_text(text, 10) == "Hello"
    
    # 长文本
    text = "Hello World This Is A Long Text"
    truncated = MetricsAnalyzer._truncate_text(text, 10)
    assert len(truncated) == 10
    assert truncated.endswith("...")
    
    # 刚好等于长度
    text = "1234567890"
    assert MetricsAnalyzer._truncate_text(text, 10) == "1234567890"
    
    print("✓ _truncate_text tests passed")


def test_dataset_loader_sample_entries():
    """Test dataset sampling"""
    print("Testing DatasetLoader.sample_entries...")
    
    dataset = [{"id": i, "text": f"text_{i}"} for i in range(100)]
    
    # 采样 10 个
    sampled = DatasetLoader.sample_entries(dataset, 10)
    assert len(sampled) == 10
    
    # 采样数量超过数据集大小
    sampled = DatasetLoader.sample_entries(dataset, 200)
    assert len(sampled) == 100
    
    # 采样 0 个
    sampled = DatasetLoader.sample_entries(dataset, 0)
    assert len(sampled) == 0
    
    # 空数据集
    sampled = DatasetLoader.sample_entries([], 10)
    assert len(sampled) == 0
    
    print("✓ sample_entries tests passed")


def test_slo_loader_validation():
    """Test SLO validation logic"""
    print("Testing SLOLoader validation...")
    
    from fluxperf import SLOConstraints, RequestMetrics
    
    slo = SLOConstraints(
        ttft_ms=100.0,
        itl_ms=50.0,
        latency_ms=1000.0,
        output_token_throughput=10.0
    )
    
    # 创建符合 SLO 的 metrics
    good_metrics = RequestMetrics(
        request_id="test",
        round_num=1,
        input_text="test",
        output_text="response",
        input_tokens=1,
        output_tokens=10,
        time_to_first_token=0.05,  # 50ms
        inter_token_latencies=[10.0, 15.0],  # 平均 12.5ms
        total_latency=0.5,  # 500ms
        throughput=20.0  # tokens/sec
    )
    
    assert SLOLoader.validate_slo(good_metrics, slo) is True
    
    # TTFT 超标
    bad_ttft = RequestMetrics(
        request_id="test",
        round_num=1,
        input_text="test",
        output_text="response",
        input_tokens=1,
        output_tokens=10,
        time_to_first_token=0.15,  # 150ms > 100ms
        inter_token_latencies=[10.0],
        total_latency=0.5,
        throughput=20.0
    )
    
    assert SLOLoader.validate_slo(bad_ttft, slo) is False
    
    # ITL 超标
    bad_itl = RequestMetrics(
        request_id="test",
        round_num=1,
        input_text="test",
        output_text="response",
        input_tokens=1,
        output_tokens=10,
        time_to_first_token=0.05,
        inter_token_latencies=[60.0, 70.0],  # 平均 65ms > 50ms
        total_latency=0.5,
        throughput=20.0
    )
    
    assert SLOLoader.validate_slo(bad_itl, slo) is False
    
    # Latency 超标
    bad_latency = RequestMetrics(
        request_id="test",
        round_num=1,
        input_text="test",
        output_text="response",
        input_tokens=1,
        output_tokens=10,
        time_to_first_token=0.05,
        inter_token_latencies=[10.0],
        total_latency=2.0,  # 2000ms > 1000ms
        throughput=20.0
    )
    
    assert SLOLoader.validate_slo(bad_latency, slo) is False
    
    # Throughput 不足
    bad_throughput = RequestMetrics(
        request_id="test",
        round_num=1,
        input_text="test",
        output_text="response",
        input_tokens=1,
        output_tokens=10,
        time_to_first_token=0.05,
        inter_token_latencies=[10.0],
        total_latency=0.5,
        throughput=5.0  # 5 < 10
    )
    
    assert SLOLoader.validate_slo(bad_throughput, slo) is False
    
    # 有错误的请求
    error_metrics = RequestMetrics(
        request_id="test",
        round_num=1,
        input_text="test",
        output_text="",
        input_tokens=1,
        output_tokens=0,
        time_to_first_token=0.0,
        error="Connection timeout"
    )
    
    assert SLOLoader.validate_slo(error_metrics, slo) is False
    
    print("✓ SLO validation tests passed")


def test_recipe_loader_validation():
    """Test Recipe loading validation"""
    print("Testing RecipeLoader validation...")
    
    import tempfile
    import yaml
    
    # 创建有效的 recipe
    valid_recipe = {
        "global": {
            "dataset": "test.jsonl",
            "endpoint": "http://localhost:8000",
            "mode": "multi_turn"
        },
        "suites": [
            {
                "name": "Suite 1",
                "stages": [
                    {
                        "name": "Stage 1",
                        "concurrency_levels": [5],
                        "num_samples": [10],
                        "env": {"VAR": "value"},
                        "dataset": "test_stage.jsonl",
                        "max_output_tokens": 512,
                        "min_output_tokens": 64
                    }
                ]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_recipe, f)
        temp_path = Path(f.name)
    
    try:
        recipe = RecipeLoader.load_recipe(temp_path)
        assert recipe.global_config["dataset"] == "test.jsonl"
        assert recipe.suites is not None
        assert len(recipe.suites) == 1
        suite = recipe.suites[0]
        assert suite.name == "Suite 1"
        assert len(suite.stages) == 1
        stage = suite.stages[0]
        assert stage.name == "Stage 1"
        assert stage.dataset == "test_stage.jsonl"
        assert stage.max_output_tokens == 512
        assert stage.min_output_tokens == 64
        print("  ✓ Valid recipe loaded successfully")
    finally:
        temp_path.unlink()
    
    # 测试无效模式
    try:
        invalid_recipe = {
            "global": valid_recipe["global"].copy(),
            "suites": [
                {
                    "name": "Suite 1",
                    "stages": [
                        {
                            "name": "Stage 1",
                            "concurrency_levels": [5],
                            "num_samples": [10]
                        }
                    ]
                }
            ]
        }
        invalid_recipe["global"]["mode"] = "invalid_mode"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_recipe, f)
            temp_path = Path(f.name)
        
        try:
            config = RecipeLoader.create_config_from_recipe(
                RecipeLoader.load_recipe(temp_path),
                RecipeStage("test", [5], [10])
            )
            assert False, "Should raise ValueError"
        except ValueError as e:
            assert "Unsupported mode" in str(e) or "不支持的模式" in str(e)
            print("  ✓ Invalid mode correctly rejected")
        finally:
            temp_path.unlink()
    except Exception:
        pass
    
    print("✓ RecipeLoader validation tests passed")


if __name__ == '__main__':
    try:
        test_count_tokens()
        test_metrics_analyzer_calculate_percentile()
        test_metrics_analyzer_truncate_text()
        test_dataset_loader_sample_entries()
        test_slo_loader_validation()
        test_recipe_loader_validation()
        
        print("\n" + "=" * 60)
        print("All utility function tests passed! ✓")
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
