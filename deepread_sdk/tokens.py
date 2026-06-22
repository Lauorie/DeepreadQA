"""Token counting via tiktoken with a char-based fallback."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - environment without tiktoken data
    _ENC = None
    logger.warning("tiktoken unavailable; falling back to char/4 token estimate")


def count_tokens(text: str) -> int:
    """Return the token count of *text* (0 for empty)."""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:
            logger.debug("tiktoken encode failed; using char/4 fallback")
    return max(1, len(text) // 4)
