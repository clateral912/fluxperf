#!/usr/bin/env python3
"""
Test BenchmarkRunner core functionality
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxperf import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkMode,
    SessionData
)


def test_sanitize_user_message():
    """Test user message sanitization"""
    print("Testing _sanitize_user_message...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    # Normal text
    assert runner._sanitize_user_message("Hello world") == "Hello world"
    
    # Empty text
    assert runner._sanitize_user_message("") == ""
    assert runner._sanitize_user_message("   ") == ""
    
    # None
    assert runner._sanitize_user_message(None) == ""
    
    # Dictionary format
    assert runner._sanitize_user_message({"content": "Hello"}) == "Hello"
    assert runner._sanitize_user_message({"value": "World"}) == "World"
    
    # Strip leading/trailing spaces
    assert runner._sanitize_user_message("  Hello  ") == "Hello"
    
    print("✓ _sanitize_user_message tests passed")


def test_extract_text():
    """Test text extraction"""
    print("Testing _extract_text...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    # String
    assert runner._extract_text("Hello") == "Hello"
    
    # Dictionary - text field
    assert runner._extract_text({"text": "Hello"}) == "Hello"
    
    # Dictionary - prompt field
    assert runner._extract_text({"prompt": "World"}) == "World"
    
    # Dictionary - content field
    assert runner._extract_text({"content": "Test"}) == "Test"
    
    # Dictionary - messages field
    msg_dict = {
        "messages": [
            {"content": "First"},
            {"content": "Last"}
        ]
    }
    result = runner._extract_text(msg_dict)
    assert "Last" in result
    
    # Convert other types to string
    assert runner._extract_text(123) == "123"
    assert runner._extract_text([1, 2, 3]) == "[1, 2, 3]"
    
    print("✓ _extract_text tests passed")


def test_total_turns():
    """Test total turns calculation"""
    print("Testing _total_turns...")
    
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
    
    # Empty list
    assert runner._total_turns([]) == 0
    
    # Single session
    assert runner._total_turns([SessionData("s1", ["msg"])]) == 1
    
    print("✓ _total_turns tests passed")


def test_reset_conversation_state():
    """Test conversation state reset"""
    print("Testing _reset_conversation_state...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    # Create sessions with history
    sessions = [
        SessionData("s1", ["msg1"], assistant_messages=["reply1"]),
        SessionData("s2", ["msg2"], assistant_messages=["reply2"])
    ]
    
    # Reset
    runner._reset_conversation_state(sessions)
    
    # Verify history is cleared
    assert len(sessions[0].assistant_messages) == 0
    assert len(sessions[1].assistant_messages) == 0
    
    # But user messages are retained
    assert len(sessions[0].user_messages) == 1
    assert len(sessions[1].user_messages) == 1
    
    print("✓ _reset_conversation_state tests passed")


def test_entries_to_single_turn_sessions():
    """Test single turn session conversion (dual_round mode)"""
    print("Testing _entries_to_single_turn_sessions...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5],
        mode=BenchmarkMode.DUAL_ROUND
    )
    runner = BenchmarkRunner(config, None)
    
    # Test data
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
    assert sessions[2].session_id == "session_2"  # default ID
    assert sessions[3].user_messages == ["Question without id"]
    
    # Empty entries are skipped
    entries_with_empty = [
        {"text": "Valid"},
        {"text": ""},
        {"text": "   "}
    ]
    
    sessions = runner._entries_to_single_turn_sessions(entries_with_empty)
    assert len(sessions) == 1
    assert sessions[0].user_messages == ["Valid"]
    
    print("✓ _entries_to_single_turn_sessions tests passed")


def test_normalize_sessions():
    """Test session normalization (multi_turn mode)"""
    print("Testing _normalize_sessions...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5],
        mode=BenchmarkMode.MULTI_TURN
    )
    runner = BenchmarkRunner(config, None)
    
    # ShareGPT format
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
    
    # Handle duplicate IDs
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
    
    print("✓ _normalize_sessions tests passed")


def test_build_conversation_history():
    """Test conversation history building"""
    print("Testing _build_conversation_history...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5]
    )
    runner = BenchmarkRunner(config, None)
    
    session = SessionData("s1", ["Q1", "Q2", "Q3"])
    session.assistant_messages = ["A1", "A2"]
    
    # Turn 0 - only first question
    history = runner._build_conversation_history(session, 0)
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Q1"
    
    # Turn 1 - has one round of conversation history
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
    
    print("✓ _build_conversation_history tests passed")


def test_build_conversation_history_with_truncation():
    """Test conversation history building with truncation"""
    print("Testing conversation history truncation...")
    
    config = BenchmarkConfig(
        dataset_path=Path("test.jsonl"),
        endpoint_url="http://localhost:8000",
        num_samples=[10],
        concurrency_levels=[5],
        max_context_tokens=10  # set very small context limit
    )
    runner = BenchmarkRunner(config, None)
    
    session = SessionData(
        "s1",
        ["Short Q1", "This is a very long question that will exceed the token limit"]
    )
    session.assistant_messages = ["Short A1"]
    
    # Turn 1 - should trigger truncation
    history = runner._build_conversation_history(session, 1)
    
    # Should only contain recent messages
    # Should have at least current user message
    assert len(history) >= 1
    assert history[-1]["role"] == "user"
    
    print("✓ Conversation history truncation tests passed")


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
        print("All BenchmarkRunner tests passed! ✓")
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
