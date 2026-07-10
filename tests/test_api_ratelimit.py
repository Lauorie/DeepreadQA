"""Tests for deepreadqa_api.ratelimit.TokenBucket."""
from deepreadqa_api.ratelimit import TokenBucket


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_burst_then_denied_with_retry_after():
    clock = _Clock()
    bucket = TokenBucket(rate_per_min=60, burst=3, clock=clock)
    for _ in range(3):
        ok, _ = bucket.acquire("k")
        assert ok
    ok, retry_after = bucket.acquire("k")
    assert not ok
    assert retry_after > 0
    # at 60/min one token refills per second
    assert retry_after <= 1.0 + 1e-6


def test_refill_after_time_passes():
    clock = _Clock()
    bucket = TokenBucket(rate_per_min=60, burst=1, clock=clock)
    assert bucket.acquire("k")[0]
    assert not bucket.acquire("k")[0]
    clock.now += 1.0
    assert bucket.acquire("k")[0]


def test_keys_are_isolated():
    clock = _Clock()
    bucket = TokenBucket(rate_per_min=60, burst=1, clock=clock)
    assert bucket.acquire("a")[0]
    assert not bucket.acquire("a")[0]
    assert bucket.acquire("b")[0]


def test_tokens_capped_at_burst():
    clock = _Clock()
    bucket = TokenBucket(rate_per_min=60, burst=2, clock=clock)
    clock.now += 3600  # a long idle must not accumulate beyond burst
    assert bucket.acquire("k")[0]
    assert bucket.acquire("k")[0]
    assert not bucket.acquire("k")[0]
