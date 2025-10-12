#!/usr/bin/env python3
"""
测试 BenchmarkRunner 的核心功能
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dual_round_benchmarker import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkMode,
    SessionData
)


def test_sanitize_user_message():
    """测试用户消息清理"""
    print("测试 _sanitize_user_message...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    # 正常文本
    assert runner._sanitize_user_message("Hello world") == "Hello world"
    
    # 空文本
    assert runner._sanitize_user_message("") == ""
    assert runner._sanitize_user_message("   ") == ""
    
    # None
    assert runner._sanitize_user_message(None) == ""
    
    # 字典格式
    assert runner._sanitize_user_message({"content": "Hello"}) == "Hello"
    assert runner._sanitize_user_message({"value": "World"}) == "World"
    
    # 去除首尾空格
    assert runner._sanitize_user_message("  Hello  ") == "Hello"
    
    print("✓ _sanitize_user_message 测试通过")


def test_extract_text():
    """测试文本提取"""
    print("测试 _extract_text...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    # 字符串
    assert runner._extract_text("Hello") == "Hello"
    
    # 字典 - text 字段
    assert runner._extract_text({"text": "Hello"}) == "Hello"
    
    # 字典 - prompt 字段
    assert runner._extract_text({"prompt": "World"}) == "World"
    
    # 字典 - content 字段
    assert runner._extract_text({"content": "Test"}) == "Test"
    
    # 字典 - messages 字段
    msg_dict = {
        "messages": [
            {"content": "First"},
            {"content": "Last"}
        ]
    }
    result = runner._extract_text(msg_dict)
    assert "Last" in result
    
    # 其他类型转字符串
    assert runner._extract_text(123) == "123"
    assert runner._extract_text([1, 2, 3]) == "[1, 2, 3]"
    
    print("✓ _extract_text 测试通过")


def test_total_turns():
    """测试总轮次计算"""
    print("测试 _total_turns...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    sessions = [
        SessionData("s1", ["msg1", "msg2", "msg3"]),
        SessionData("s2", ["msg1", "msg2"]),
        SessionData("s3", ["msg1"])
    ]
    
    assert runner._total_turns(sessions) == 6  # 3 + 2 + 1
    
    # 空列表
    assert runner._total_turns([]) == 0
    
    # 单个 session
    assert runner._total_turns([SessionData("s1", ["msg"])]) == 1
    
    print("✓ _total_turns 测试通过")


def test_reset_conversation_state():
    """测试会话状态重置"""
    print("测试 _reset_conversation_state...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    # 创建有历史的 sessions
    sessions = [
        SessionData("s1", ["msg1"], assistant_messages=["reply1"]),
        SessionData("s2", ["msg2"], assistant_messages=["reply2"])
    ]
    
    # 重置
    runner._reset_conversation_state(sessions)
    
    # 验证历史被清空
    assert len(sessions[0].assistant_messages) == 0
    assert len(sessions[1].assistant_messages) == 0
    
    # 但用户消息保留
    assert len(sessions[0].user_messages) == 1
    assert len(sessions[1].user_messages) == 1
    
    print("✓ _reset_conversation_state 测试通过")


def test_entries_to_single_turn_sessions():
    """测试单轮会话转换 (dual_round 模式)"""
    print("测试 _entries_to_single_turn_sessions...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5],
        mode=BenchmarkMode.DUAL_ROUND
    )
    runner = BenchmarkRunner(config, None)
    
    # 测试数据
    entries = [
        {"id": "1", "text": "Question 1"},
        {"id": "2", "text": "Question 2"},
        "Simple string question",
        {"text": "Question without id"}
    ]
    
    sessions = runner._entries_to_single_turn_sessions(entries)
    
    assert len(sessions) == 4
    assert sessions[0].session_id == "1"
    assert sessions[0].user_messages == ["Question 1"]
    assert sessions[1].session_id == "2"
    assert sessions[2].session_id == "session_2"  # 默认 ID
    assert sessions[3].user_messages == ["Question without id"]
    
    # 空条目会被跳过
    entries_with_empty = [
        {"text": "Valid"},
        {"text": ""},
        {"text": "   "}
    ]
    
    sessions = runner._entries_to_single_turn_sessions(entries_with_empty)
    assert len(sessions) == 1
    assert sessions[0].user_messages == ["Valid"]
    
    print("✓ _entries_to_single_turn_sessions 测试通过")


def test_normalize_sessions():
    """测试会话归一化 (multi_turn 模式)"""
    print("测试 _normalize_sessions...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5],
        mode=BenchmarkMode.MULTI_TURN
    )
    runner = BenchmarkRunner(config, None)
    
    # ShareGPT 格式
    sharegpt_entries = [
        {
            "id": "conv1",
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
                {"from": "human", "value": "How are you?"}
            ]
        }
    ]
    
    sessions = runner._normalize_sessions(sharegpt_entries)
    assert len(sessions) == 1
    assert sessions[0].session_id == "conv1"
    assert len(sessions[0].user_messages) == 2
    assert sessions[0].user_messages[0] == "Hello"
    assert sessions[0].user_messages[1] == "How are you?"
    
    # 处理重复 ID
    duplicate_entries = [
        {"id": "same", "conversations": [{"from": "human", "value": "Q1"}]},
        {"id": "same", "conversations": [{"from": "human", "value": "Q2"}]},
        {"id": "same", "conversations": [{"from": "human", "value": "Q3"}]}
    ]
    
    sessions = runner._normalize_sessions(duplicate_entries)
    assert len(sessions) == 3
    assert sessions[0].session_id == "same"
    assert sessions[1].session_id == "same_1"
    assert sessions[2].session_id == "same_2"
    
    print("✓ _normalize_sessions 测试通过")


def test_build_conversation_history():
    """测试构建对话历史"""
    print("测试 _build_conversation_history...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    session = SessionData("s1", ["Q1", "Q2", "Q3"])
    session.assistant_messages = ["A1", "A2"]
    
    # Turn 0 - 只有第一个问题
    history = runner._build_conversation_history(session, 0)
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Q1"
    
    # Turn 1 - 有一轮对话历史
    history = runner._build_conversation_history(session, 1)
    assert len(history) == 3
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Q1"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "A1"
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "Q2"
    
    # Turn 2
    history = runner._build_conversation_history(session, 2)
    assert len(history) == 5
    
    print("✓ _build_conversation_history 测试通过")


def test_build_conversation_history_with_truncation():
    """测试带截断的对话历史构建"""
    print("测试对话历史截断...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5],
        max_context_tokens=10  # 设置很小的上下文限制
    )
    runner = BenchmarkRunner(config, None)
    
    session = SessionData(
        "s1",
        ["Short Q1", "This is a very long question that will exceed the token limit"]
    )
    session.assistant_messages = ["Short A1"]
    
    # Turn 1 - 应该触发截断
    history = runner._build_conversation_history(session, 1)
    
    # 应该只包含最近的消息
    # 最少应该有当前的用户消息
    assert len(history) >= 1
    assert history[-1]["role"] == "user"
    
    print("✓ 对话历史截断测试通过")


if __name__ == '__main__':
    try:
        test_sanitize_user_message()
        test_extract_text()
        test_total_turns()
        test_reset_conversation_state()
        test_entries_to_single_turn_sessions()
        test_normalize_sessions()
        test_build_conversation_history()
        test_build_conversation_history_with_truncation()
        
        print("\n" + "=" * 60)
        print("所有 BenchmarkRunner 测试通过! ✓")
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
