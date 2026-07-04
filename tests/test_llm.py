"""ToolLLM retry classification and multi-endpoint failover."""
from types import SimpleNamespace

import pytest

import deepreadqa.llm as llm_mod
from deepreadqa.config import Endpoint
from deepreadqa.llm import LLMError, ToolLLM


class _FakeAPIError(Exception):
    """Mimics openai APIStatusError: carries a status_code attribute."""

    def __init__(self, msg: str, status_code: int | None = None) -> None:
        super().__init__(msg)
        if status_code is not None:
            self.status_code = status_code


def _ok_resp(content: str = "ok", total: int = 7):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice],
                           usage=SimpleNamespace(total_tokens=total))


class _ScriptedOpenAI:
    """Fake OpenAI client: pops scripted results (response or exception) per base_url."""

    scripts: dict[str, list] = {}
    calls: list[tuple[str, dict]] = []

    def __init__(self, *, base_url=None, api_key=None, timeout=None, max_retries=0):
        self._base = base_url
        create = self._create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def _create(self, **kwargs):
        _ScriptedOpenAI.calls.append((self._base, kwargs))
        item = _ScriptedOpenAI.scripts[self._base].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    _ScriptedOpenAI.scripts = {}
    _ScriptedOpenAI.calls = []
    monkeypatch.setattr(llm_mod, "OpenAI", _ScriptedOpenAI)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def _ep(name: str, base: str, *, omit_temp: bool = True) -> Endpoint:
    return Endpoint(name=name, base_url=base, api_key="k", model="m",
                    omit_temperature=omit_temp)


def test_non_retryable_error_is_not_retried():
    _ScriptedOpenAI.scripts["https://p"] = [_FakeAPIError("invalid api key",
                                                          status_code=401)]
    llm = ToolLLM(_ep("p", "https://p"))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "q"}])
    assert len(_ScriptedOpenAI.calls) == 1  # no pointless retries on 401


def test_retryable_5xx_is_retried_then_succeeds():
    _ScriptedOpenAI.scripts["https://p"] = [
        _FakeAPIError("upstream boom", status_code=500), _ok_resp()]
    llm = ToolLLM(_ep("p", "https://p"))
    resp = llm.chat([{"role": "user", "content": "q"}])
    assert resp.content == "ok"
    assert resp.total_tokens == 7
    assert len(_ScriptedOpenAI.calls) == 2


def test_rate_limit_disguised_as_400_is_retried():
    # intern-ai reports rate limits as HTTP 400 + code -20048 (comparsion.md §10.2)
    err = _FakeAPIError("Error code: 400 - {'code': -20048, 'message': '请求过于频繁'}",
                        status_code=400)
    _ScriptedOpenAI.scripts["https://p"] = [err, _ok_resp()]
    llm = ToolLLM(_ep("p", "https://p"))
    assert llm.chat([{"role": "user", "content": "q"}]).content == "ok"
    assert len(_ScriptedOpenAI.calls) == 2


def test_failover_to_backup_endpoint_on_hard_failure():
    _ScriptedOpenAI.scripts["https://p"] = [_FakeAPIError("用户额度不足",
                                                          status_code=403)]
    _ScriptedOpenAI.scripts["https://b"] = [_ok_resp("from-backup")]
    llm = ToolLLM(_ep("p", "https://p"), backups=(_ep("b", "https://b"),))
    resp = llm.chat([{"role": "user", "content": "q"}])
    assert resp.content == "from-backup"
    assert [c[0] for c in _ScriptedOpenAI.calls] == ["https://p", "https://b"]


def test_failover_is_sticky_for_subsequent_calls():
    _ScriptedOpenAI.scripts["https://p"] = [_FakeAPIError("quota", status_code=403)]
    _ScriptedOpenAI.scripts["https://b"] = [_ok_resp("b1"), _ok_resp("b2")]
    llm = ToolLLM(_ep("p", "https://p"), backups=(_ep("b", "https://b"),))
    assert llm.chat([{"role": "user", "content": "q"}]).content == "b1"
    assert llm.chat([{"role": "user", "content": "q"}]).content == "b2"
    # dead primary is not re-probed on the second call
    assert [c[0] for c in _ScriptedOpenAI.calls] == ["https://p", "https://b",
                                                     "https://b"]


def test_all_endpoints_exhausted_raises_llmerror():
    _ScriptedOpenAI.scripts["https://p"] = [_FakeAPIError("quota", status_code=403)]
    _ScriptedOpenAI.scripts["https://b"] = [_FakeAPIError("also dead",
                                                          status_code=401)]
    llm = ToolLLM(_ep("p", "https://p"), backups=(_ep("b", "https://b"),))
    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "q"}])


def test_temperature_auto_disable_survives_refactor():
    _ScriptedOpenAI.scripts["https://p"] = [
        _FakeAPIError("temperature is unsupported for this model",
                      status_code=400),
        _ok_resp()]
    llm = ToolLLM(_ep("p", "https://p", omit_temp=False))
    assert llm.chat([{"role": "user", "content": "q"}]).content == "ok"
    first, second = _ScriptedOpenAI.calls
    assert "temperature" in first[1]
    assert "temperature" not in second[1]
