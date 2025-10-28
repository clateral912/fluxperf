import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent))

from process_sharegpt_data import process_dataset


class ShareGPTProcessingTests(unittest.TestCase):
    def test_extracts_human_messages(self):
        with TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.json"
            output_path = Path(tmpdir) / "out.jsonl"
            sample = [
                {
                    "id": "session-1",
                    "conversations": [
                        {"from": "human", "value": "hi"},
                        {"from": "gpt", "value": "hello"},
                        {"from": "human", "value": "question"}
                    ]
                },
                {
                    "id": "session-2",
                    "messages": [
                        {"role": "system", "content": "note"},
                        {"role": "user", "content": "ask"},
                        {"role": "assistant", "content": "answer"}
                    ]
                }
            ]
            input_path.write_text(json.dumps(sample), encoding="utf-8")
            written = process_dataset(input_path, output_path, None)
            self.assertEqual(written, 2)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]
            messages_by_id = {record["session_id"]: record["user_messages"] for record in records}
            self.assertEqual(messages_by_id["session-1"], ["hi", "question"])
            self.assertEqual(messages_by_id["session-2"], ["ask"])


if __name__ == "__main__":
    unittest.main()
