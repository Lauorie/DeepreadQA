"""Tests for deepreadqa_api.config.ApiConfig."""
import pytest

from deepreadqa_api.config import ApiConfig


def _clear_env(monkeypatch) -> None:
    # isolate from the developer's .env: from_env() must only see what the
    # test sets, not whatever keys the local deployment happens to configure
    monkeypatch.setattr("deepreadqa_api.config.load_dotenv",
                        lambda *a, **k: None)
    for var in ("DEEPREADQA_API_KEYS", "DEEPREADQA_AUTH_DISABLED",
                "DEEPREADQA_DB", "DEEPREADQA_WORKERS", "DEEPREADQA_QUEUE_MAX",
                "DEEPREADQA_SYNC_WAIT_CAP_S", "DEEPREADQA_JOB_TTL_S",
                "DEEPREADQA_RATE_LIMIT_RPM", "DEEPREADQA_RATE_LIMIT_BURST",
                "DEEPREADQA_MAX_QUESTION_CHARS"):
        monkeypatch.delenv(var, raising=False)


def test_from_env_requires_api_keys(monkeypatch):
    _clear_env(monkeypatch)
    with pytest.raises(ValueError, match="DEEPREADQA_API_KEYS"):
        ApiConfig.from_env()


def test_from_env_auth_disabled_allows_no_keys(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEEPREADQA_AUTH_DISABLED", "1")
    cfg = ApiConfig.from_env()
    assert cfg.auth_disabled is True
    assert cfg.api_keys == ()


def test_from_env_parses_keys_and_defaults(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEEPREADQA_API_KEYS", "k1, k2 ,k3")
    cfg = ApiConfig.from_env()
    assert cfg.api_keys == ("k1", "k2", "k3")
    assert cfg.auth_disabled is False
    assert cfg.db_path.endswith("store/cae_vlmocr.db")
    assert cfg.workers == 2
    assert cfg.queue_max == 16
    assert cfg.sync_wait_cap_s == 300.0
    assert cfg.job_ttl_s == 3600.0
    assert cfg.rate_limit_rpm == 10.0
    assert cfg.rate_limit_burst == 5
    assert cfg.max_question_chars == 2000


def test_from_env_overrides(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("DEEPREADQA_API_KEYS", "k")
    monkeypatch.setenv("DEEPREADQA_DB", "/tmp/other.db")
    monkeypatch.setenv("DEEPREADQA_WORKERS", "4")
    monkeypatch.setenv("DEEPREADQA_QUEUE_MAX", "8")
    monkeypatch.setenv("DEEPREADQA_SYNC_WAIT_CAP_S", "60")
    monkeypatch.setenv("DEEPREADQA_RATE_LIMIT_RPM", "30")
    cfg = ApiConfig.from_env()
    assert cfg.db_path == "/tmp/other.db"
    assert cfg.workers == 4
    assert cfg.queue_max == 8
    assert cfg.sync_wait_cap_s == 60.0
    assert cfg.rate_limit_rpm == 30.0


@pytest.mark.parametrize("field,value", [
    ("workers", 0), ("queue_max", 0), ("rate_limit_rpm", 0),
    ("rate_limit_burst", 0), ("max_question_chars", 0), ("job_ttl_s", 0),
])
def test_invalid_values_rejected(field, value):
    kwargs = {"api_keys": ("k",), field: value}
    with pytest.raises(ValueError):
        ApiConfig(**kwargs)
