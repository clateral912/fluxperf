import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxperf import ConversationHistory, count_tokens


class ConversationHistoryTests(unittest.TestCase):
    def test_append_and_truncate(self):
        history = ConversationHistory(max_tokens=5)
        messages, context_tokens, truncated = history.prepare_request("hello world")
        self.assertEqual(len(messages), 1)
        self.assertEqual(context_tokens, count_tokens("hello world"))
        self.assertEqual(truncated, 0)

        truncated_after = history.append_assistant("answer tokens here")
        self.assertGreaterEqual(truncated_after, 0)

    def test_truncation_logic(self):
        history = ConversationHistory(max_tokens=3)
        history.prepare_request("one two")
        history.append_assistant("reply")
        messages, context_tokens, truncated = history.prepare_request("new message")
        self.assertLessEqual(context_tokens, 3)
        self.assertGreaterEqual(truncated, 1)


if __name__ == "__main__":
    unittest.main()
