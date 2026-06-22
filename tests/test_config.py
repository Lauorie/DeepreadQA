import os

from deepreadqa.config import Config, Endpoint


def test_endpoint_defaults():
    e = Endpoint(name="aiberm", base_url="https://aiberm.com/v1", api_key="k",
                 model="anthropic/claude-opus-4.8", omit_temperature=True)
    assert e.omit_temperature is True


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "sk-test")
    monkeypatch.setenv("AIBERM_BASE_URL", "https://aiberm.com/v1")
    monkeypatch.setenv("DEEPREAD_AGENT_MODEL", "anthropic/claude-opus-4.8")
    cfg = Config.from_env()
    assert cfg.endpoint.api_key == "sk-test"
    assert cfg.endpoint.omit_temperature is True  # opus on aiberm
    assert cfg.max_iterations == 15
