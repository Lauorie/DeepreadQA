"""Token accounting for conversations (reuses deepread_sdk.tokens)."""
from __future__ import annotations

from deepread_sdk.tokens import count_tokens


def count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                total += count_tokens(str(part))
        total += 4  # per-message overhead
    return total
