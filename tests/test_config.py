
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


def test_no_dead_token_warning_ratio_field():
    import dataclasses
    assert "token_warning_ratio" not in {f.name for f in dataclasses.fields(Config)}


def test_backup_endpoints_default_empty():
    cfg = Config(endpoint=Endpoint("aiberm", "u", "k", "m", True))
    assert cfg.backup_endpoints == ()


def test_from_env_parses_backup_endpoint(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "sk-primary")
    monkeypatch.setenv("DEEPREAD_BACKUP_BASE_URL", "https://backup.example/v1")
    monkeypatch.setenv("DEEPREAD_BACKUP_API_KEY", "sk-backup")
    monkeypatch.setenv("DEEPREAD_BACKUP_MODEL", "same/model")
    cfg = Config.from_env()
    assert len(cfg.backup_endpoints) == 1
    b = cfg.backup_endpoints[0]
    assert b.base_url == "https://backup.example/v1"
    assert b.api_key == "sk-backup"
    assert b.model == "same/model"


def test_disabled_tools_default_is_low_freq_trio():
    # ablation-validated production default (runs/abl5*, comparsion.md §11)
    cfg = Config(endpoint=Endpoint("aiberm", "u", "k", "m", True))
    assert cfg.disabled_tools == ("intro", "preview", "read_raw")


def test_from_env_disabled_tools_none_reenables_all(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPREAD_DISABLED_TOOLS", "none")
    assert Config.from_env().disabled_tools == ()


def test_from_env_disabled_tools_empty_string_reenables_all(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPREAD_DISABLED_TOOLS", "")
    assert Config.from_env().disabled_tools == ()


def test_from_env_parses_disabled_tools(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPREAD_DISABLED_TOOLS", "intro, preview,read_raw")
    cfg = Config.from_env()
    assert cfg.disabled_tools == ("intro", "preview", "read_raw")


def test_from_env_output_token_knobs(monkeypatch):
    """Reasoning models (qwen3.x thinking, kimi, glm) need >=6000 output
    tokens or the hidden thinking budget truncates the visible answer."""
    monkeypatch.setenv("AIBERM_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPREAD_MAX_OUTPUT_TOKENS", "8000")
    monkeypatch.setenv("DEEPREAD_COMPOSE_MAX_TOKENS", "6000")
    cfg = Config.from_env()
    assert cfg.max_output_tokens == 8000
    assert cfg.compose_max_tokens == 6000


def test_catalog_defaults_off():
    cfg = Config(endpoint=Endpoint("aiberm", "u", "k", "m", True))
    assert cfg.catalog_in_prompt is False
    assert cfg.catalog_max_docs == 400


def test_from_env_catalog_flag(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "sk-test")
    for raw, want in (("1", True), ("on", True), ("true", True), ("yes", True),
                      ("0", False), ("off", False)):
        monkeypatch.setenv("DEEPREAD_CATALOG", raw)
        assert Config.from_env().catalog_in_prompt is want, raw


def test_from_env_backup_model_defaults_to_primary(monkeypatch):
    monkeypatch.setenv("AIBERM_API_KEY", "sk-primary")
    monkeypatch.setenv("DEEPREAD_AGENT_MODEL", "primary/model")
    monkeypatch.setenv("DEEPREAD_BACKUP_BASE_URL", "https://backup.example/v1")
    monkeypatch.setenv("DEEPREAD_BACKUP_API_KEY", "sk-backup")
    monkeypatch.delenv("DEEPREAD_BACKUP_MODEL", raising=False)
    cfg = Config.from_env()
    assert cfg.backup_endpoints[0].model == "primary/model"


def test_from_env_eval_file_override(monkeypatch) -> None:
    monkeypatch.setenv("AIBERM_API_KEY", "k")
    monkeypatch.setenv("DEEPREAD_EVAL_FILE", "/tmp/other-eval.json")
    from deepreadqa.config import Config
    cfg = Config.from_env()
    assert cfg.eval_file == "/tmp/other-eval.json"


def test_from_env_answer_lang(monkeypatch) -> None:
    monkeypatch.setenv("AIBERM_API_KEY", "k")
    monkeypatch.setenv("DEEPREAD_ANSWER_LANG", "en")
    from deepreadqa.config import Config
    cfg = Config.from_env()
    assert cfg.answer_lang == "en"


def test_from_env_coverage_flag(monkeypatch) -> None:
    monkeypatch.setenv("AIBERM_API_KEY", "k")
    monkeypatch.setenv("DEEPREAD_COVERAGE", "1")
    from deepreadqa.config import Config
    cfg = Config.from_env()
    assert cfg.coverage_discipline is True
    monkeypatch.delenv("DEEPREAD_COVERAGE")
    assert Config.from_env().coverage_discipline is False
