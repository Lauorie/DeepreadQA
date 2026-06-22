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


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to at most *max_tokens* tokens (token-accurate via
    tiktoken; char/4 fallback). Returns '' for max_tokens<=0 or empty text."""
    if max_tokens <= 0 or not text:
        return ""
    if _ENC is not None:
        try:
            toks = _ENC.encode(text)
            if len(toks) <= max_tokens:
                return text
            return _ENC.decode(toks[:max_tokens])
        except Exception:
            pass
    return text[: max_tokens * 4]
