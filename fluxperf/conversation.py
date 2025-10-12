from collections import deque
from typing import Dict, List, Optional, Tuple

from .models import count_tokens


class ConversationHistory:
    def __init__(self, max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens
        self.messages: deque[Dict[str, str]] = deque()

    def _message_tokens(self, message: Dict[str, str]) -> int:
        return count_tokens(message.get("content", ""))

    def _total_tokens(self) -> int:
        return sum(self._message_tokens(msg) for msg in self.messages)

    def _truncate_if_needed(self) -> int:
        if self.max_tokens is None:
            return 0
        truncated = 0
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.popleft()
            truncated += 1
        return truncated

    def prepare_request(self, user_text: str) -> Tuple[List[Dict[str, str]], int, int]:
        self.messages.append({"role": "user", "content": user_text})
        truncated = self._truncate_if_needed()
        messages = list(self.messages)
        context_tokens = sum(self._message_tokens(msg) for msg in messages)
        return messages, context_tokens, truncated

    def append_assistant(self, assistant_text: str) -> int:
        self.messages.append({"role": "assistant", "content": assistant_text})
        return self._truncate_if_needed()

    def reset(self):
        self.messages.clear()
