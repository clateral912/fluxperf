#!/usr/bin/env python3
"""
Test integration between all fluxperf modules
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxperf.models import (
    BenchmarkMode,
    BenchmarkConfig,
    SLOConstraints,
    RecipeStage,
    Recipe,
    VLLMConfig,
    RequestMetrics,
    RoundMetrics,
    SessionData,
    count_tokens
)
from fluxperf.conversation import ConversationHistory
from fluxperf.loaders import DatasetLoader, SLOLoader, RecipeLoader
from fluxperf.client import OpenAIClient
from fluxperf.vllm_manager import VLLMManager


def test_models_import():
    """Test that all models can be imported"""
    print("Testing models import...")
    
    assert BenchmarkMode is not None
    assert BenchmarkConfig is not None
    assert SLOConstraints is not None
    assert RecipeStage is not None
    assert Recipe is not None
    assert VLLMConfig is not None
    assert RequestMetrics is not None
    assert RoundMetrics is not None
    assert SessionData is not None
    assert count_tokens is not None
    
    print("✓ Models import test passed")


def test_conversation_import():
    """Test conversation module import"""
    print("Testing conversation import...")
    
    assert ConversationHistory is not None
    history = ConversationHistory(max_tokens=100)
    assert history.max_tokens == 100
    
    print("✓ Conversation import test passed")


def test_loaders_import():
    """Test loaders module import"""
    print("Testing loaders import...")
    
    assert DatasetLoader is not None
    assert SLOLoader is not None
    assert RecipeLoader is not None
    
    print("✓ Loaders import test passed")


def test_client_import():
    """Test client module import"""
    print("Testing client import...")
    
    assert OpenAIClient is not None
    
    print("✓ Client import test passed")


def test_vllm_manager_import():
    """Test vLLM manager import"""
    print("Testing vLLM manager import...")
    
    assert VLLMManager is not None
    manager = VLLMManager()
    assert manager is not None
    
    print("✓ vLLM manager import test passed")


def test_vllm_config_in_recipe():
    """Test VLLMConfig integration with Recipe"""
    print("Testing VLLMConfig in Recipe...")
    
    vllm_config = VLLMConfig(
        auto_restart=True,
        startup_command="vllm serve model",
        port=8000
    )
    
    recipe = Recipe(
        global_config={"dataset": "test.jsonl"},
        stages=[],
        vllm=vllm_config
    )
    
    assert recipe.vllm is not None
    assert recipe.vllm.auto_restart is True
    assert recipe.vllm.startup_command == "vllm serve model"
    assert recipe.vllm.port == 8000
    
    print("✓ VLLMConfig in Recipe test passed")


def test_conversation_with_count_tokens():
    """Test ConversationHistory uses count_tokens from models"""
    print("Testing ConversationHistory with count_tokens...")
    
    history = ConversationHistory(max_tokens=50)
    
    messages, context_tokens, truncated = history.prepare_request("Hello world test")
    
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello world test"
    assert context_tokens > 0
    
    print("✓ ConversationHistory with count_tokens test passed")


def test_benchmark_config_creation():
    """Test BenchmarkConfig creation with all fields"""
    print("Testing BenchmarkConfig creation...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10, 20],
        concurrency_levels=[5, 10],
        mode=BenchmarkMode.MULTI_TURN,
        max_input_length=1000,
        max_output_tokens=100,
        model_name="test-model",
        api_key="test-key",
        timeout=300,
        shuffle_round2=True,
        slo_file=Path("slo.yaml"),
        output_dir=Path("results"),
        prometheus_url="http://localhost:9090",
        prometheus_metrics=["metric1", "metric2"],
        save_requests=True,
        reset_cache_url="http://localhost:8000/reset",
        reset_cache_between_rounds=True,
        reset_cache_between_concurrency=False,
        debug=True,
        debug_log_dir=Path("logs"),
        max_context_tokens=2000
    )
    
    assert config.dataset_path == Path("test.jsonl")
    assert config.endpoint_url == "http://localhost:8000"
    assert config.num_samples == [10, 20]
    assert config.concurrency_levels == [5, 10]
    assert config.mode == BenchmarkMode.MULTI_TURN
    assert config.max_input_length == 1000
    assert config.max_output_tokens == 100
    assert config.model_name == "test-model"
    assert config.api_key == "test-key"
    assert config.timeout == 300
    
    print("✓ BenchmarkConfig creation test passed")


def test_session_data_with_metadata():
    """Test SessionData with metadata"""
    print("Testing SessionData with metadata...")
    
    session = SessionData(
        session_id="test_session",
        user_messages=["Hello", "How are you?"],
        assistant_messages=["Hi there!", "I'm good!"],
        metadata={"test_key": "test_value", "round": 1}
    )
    
    assert session.session_id == "test_session"
    assert len(session.user_messages) == 2
    assert len(session.assistant_messages) == 2
    assert session.metadata["test_key"] == "test_value"
    assert session.metadata["round"] == 1
    
    print("✓ SessionData with metadata test passed")


def test_slo_constraints():
    """Test SLOConstraints creation"""
    print("Testing SLOConstraints...")
    
    slo = SLOConstraints(
        ttft_ms=100.0,
        itl_ms=50.0,
        latency_ms=1000.0,
        output_token_throughput=10.0
    )
    
    assert slo.ttft_ms == 100.0
    assert slo.itl_ms == 50.0
    assert slo.latency_ms == 1000.0
    assert slo.output_token_throughput == 10.0
    
    print("✓ SLOConstraints test passed")


if __name__ == '__main__':
    try:
        test_models_import()
        test_conversation_import()
        test_loaders_import()
        test_client_import()
        test_vllm_manager_import()
        test_vllm_config_in_recipe()
        test_conversation_with_count_tokens()
        test_benchmark_config_creation()
        test_session_data_with_metadata()
        test_slo_constraints()
        
        print("\n" + "=" * 60)
        print("All module integration tests passed! ✓")
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
