from deepread_sdk.tokens import count_tokens


def test_count_tokens_nonempty_positive():
    assert count_tokens("hello world, this is a test") > 0


def test_count_tokens_empty_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_monotonic():
    short = count_tokens("one two three")
    long = count_tokens("one two three four five six seven eight nine ten")
    assert long > short
