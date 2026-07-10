"""Per-key token-bucket rate limiter (in-memory, thread-safe)."""
from __future__ import annotations

import threading
import time
from typing import Callable


class TokenBucket:
    """Classic token bucket: `burst` capacity, refilled at `rate_per_min`/60 s."""

    def __init__(self, rate_per_min: float, burst: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if rate_per_min <= 0 or burst <= 0:
            raise ValueError("rate_per_min and burst must be positive")
        self._rate_s = rate_per_min / 60.0
        self._burst = float(burst)
        self._clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, ts)

    def acquire(self, key: str) -> tuple[bool, float]:
        """Try to take one token; return (allowed, retry_after_seconds)."""
        now = self._clock()
        with self._lock:
            tokens, ts = self._state.get(key, (self._burst, now))
            tokens = min(self._burst, tokens + (now - ts) * self._rate_s)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            return False, (1.0 - tokens) / self._rate_s
