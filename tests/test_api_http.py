"""End-to-end HTTP tests for the DeepreadQA API (TestClient + fake QA)."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from deepreadqa_api.app import create_app
from deepreadqa_api.config import ApiConfig
from deepreadqa_api.engine import AnswerEngine

from tests.test_api_engine import CATALOG, _FakeQA

AUTH = {"Authorization": "Bearer test-key"}
PROBLEM = "application/problem+json"


def _cfg(**over) -> ApiConfig:
    defaults = dict(api_keys=("test-key",), workers=1, queue_max=8,
                    sync_wait_cap_s=5.0, job_ttl_s=60.0,
                    rate_limit_rpm=6000.0, rate_limit_burst=1000,
                    max_question_chars=100)
    defaults.update(over)
    return ApiConfig(**defaults)


def _client(cfg=None, qa=None, engine=None) -> TestClient:
    cfg = cfg or _cfg()
    engine = engine or AnswerEngine(cfg, qa_factory=lambda: qa or _FakeQA(),
                                    catalog=CATALOG)
    app = create_app(cfg, engine=engine)
    return TestClient(app, raise_server_exceptions=False)


def _wait_ready(client: TestClient, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get("/readyz").status_code == 200:
            return
        time.sleep(0.02)
    pytest.fail("engine never became ready")


def _poll_status(client: TestClient, url: str, want: str,
                 timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(url, headers=AUTH).json()
        if body["status"] == want:
            return body
        time.sleep(0.02)
    pytest.fail(f"answer never reached status {want!r}")


# -- system ----------------------------------------------------------------

def test_healthz_needs_no_auth():
    with _client() as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_readyz_ready_and_not_ready():
    class _NeverReady(AnswerEngine):
        def start(self) -> None:  # bootstrap intentionally never runs
            pass

    cfg = _cfg()
    stalled = _client(cfg, engine=_NeverReady(cfg, qa_factory=_FakeQA,
                                              catalog=CATALOG))
    with stalled as c:
        r = c.get("/readyz")
        assert r.status_code == 503
        assert r.json()["code"] == "not_ready"
        assert r.headers["content-type"].startswith(PROBLEM)
    with _client() as c:
        _wait_ready(c)
        assert c.get("/readyz").json()["status"] == "ready"


def test_service_info_fields():
    with _client() as c:
        _wait_ready(c)
        body = c.get("/v1/service", headers=AUTH).json()
        assert body["service"] == "deepreadqa-api"
        assert body["api_version"] == "v1"
        assert body["document_count"] == 2
        assert body["workers"] == 1
        assert "uptime_s" in body and "version" in body


def test_metrics_requires_auth_and_renders_text():
    with _client() as c:
        _wait_ready(c)
        assert c.get("/metrics").status_code == 401
        c.post("/v1/answers", json={"question": "HJC?"}, headers=AUTH)
        r = c.get("/metrics", headers=AUTH)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert "deepreadqa_http_requests_total" in r.text
        assert "deepreadqa_answers_finished_total" in r.text
        assert "deepreadqa_answer_latency_seconds_bucket" in r.text
        assert "deepreadqa_queue_depth" in r.text


def test_openapi_and_docs_are_public():
    with _client() as c:
        spec = c.get("/openapi.json")
        assert spec.status_code == 200
        assert "/v1/answers" in spec.json()["paths"]
        assert c.get("/docs").status_code == 200


def test_unknown_path_renders_problem():
    with _client() as c:
        r = c.get("/nope", headers=AUTH)
        assert r.status_code == 404
        assert r.headers["content-type"].startswith(PROBLEM)
        assert r.json()["code"] == "not_found"


# -- auth ------------------------------------------------------------------

def test_missing_token_401():
    with _client() as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "q"})
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        body = r.json()
        assert body["code"] == "unauthorized"
        assert body["request_id"]


def test_wrong_token_401():
    with _client() as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "q"},
                   headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401


# -- validation --------------------------------------------------------------

def test_unknown_field_422_problem():
    with _client() as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"q": "typo"}, headers=AUTH)
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "invalid_request"
        assert body["errors"]


def test_blank_question_422():
    with _client() as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "   "}, headers=AUTH)
        assert r.status_code == 422
        assert r.json()["code"] == "invalid_request"


def test_question_too_long_422():
    with _client() as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "x" * 101}, headers=AUTH)
        assert r.status_code == 422
        assert "100" in r.json()["detail"]


def test_idempotency_key_too_long_422():
    with _client() as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "q"},
                   headers={**AUTH, "Idempotency-Key": "k" * 129})
        assert r.status_code == 422


# -- answers: sync ------------------------------------------------------------

def test_sync_answer_success_full_shape():
    with _client() as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "HJC 是什么？"}, headers=AUTH)
        assert r.status_code == 200
        assert r.headers["x-request-id"]
        body = r.json()
        assert body["id"].startswith("ans_")
        assert body["object"] == "answer"
        assert body["status"] == "succeeded"
        assert body["answer"] == "答案文本"
        assert body["sources"][0] == {"doc_id": "hjc.md", "title": "HJC 本构模型"}
        assert body["usage"]["total_tokens"] == 1234
        assert body["forced_final"] is False
        assert body["latency_ms"] is not None
        assert body["error"] is None


def test_sync_failed_answer_returns_502_and_resource_is_kept():
    with _client(qa=_FakeQA(exc=RuntimeError("llm exploded"))) as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "q"}, headers=AUTH)
        assert r.status_code == 502
        body = r.json()
        assert body["code"] == "answer_failed"
        aid = body["answer_id"]
        follow = c.get(f"/v1/answers/{aid}", headers=AUTH)
        assert follow.status_code == 200
        assert follow.json()["status"] == "failed"
        assert "llm exploded" in follow.json()["error"]["message"]


def test_sync_wait_cap_degrades_to_202():
    gate = threading.Event()
    with _client(_cfg(sync_wait_cap_s=0.2), qa=_FakeQA(gate=gate)) as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "slow"}, headers=AUTH)
        assert r.status_code == 202
        assert r.headers["location"].startswith("/v1/answers/ans_")
        assert r.json()["status"] in ("queued", "running")
        gate.set()
        _poll_status(c, r.headers["location"], "succeeded")


# -- answers: async -----------------------------------------------------------

def test_prefer_respond_async_returns_202_then_poll():
    gate = threading.Event()
    with _client(qa=_FakeQA(gate=gate)) as c:
        _wait_ready(c)
        r = c.post("/v1/answers", json={"question": "async"},
                   headers={**AUTH, "Prefer": "respond-async"})
        assert r.status_code == 202
        loc = r.headers["location"]
        assert c.get(loc, headers=AUTH).json()["status"] in ("queued", "running")
        gate.set()
        body = _poll_status(c, loc, "succeeded")
        assert body["answer"] == "答案文本"


def test_idempotency_key_replays_same_answer():
    with _client() as c:
        _wait_ready(c)
        h = {**AUTH, "Idempotency-Key": "same-key"}
        first = c.post("/v1/answers", json={"question": "q"}, headers=h)
        second = c.post("/v1/answers", json={"question": "q"}, headers=h)
        assert first.status_code == 200 and second.status_code == 200
        assert first.json()["id"] == second.json()["id"]


def test_get_unknown_answer_404():
    with _client() as c:
        _wait_ready(c)
        r = c.get("/v1/answers/ans_ffffffffffffffff", headers=AUTH)
        assert r.status_code == 404
        assert r.json()["code"] == "not_found"


# -- rate limit / queue --------------------------------------------------------

def test_rate_limited_429_with_retry_after():
    with _client(_cfg(rate_limit_rpm=1.0, rate_limit_burst=2)) as c:
        _wait_ready(c)
        for _ in range(2):
            assert c.post("/v1/answers", json={"question": "q"},
                          headers=AUTH).status_code == 200
        r = c.post("/v1/answers", json={"question": "q"}, headers=AUTH)
        assert r.status_code == 429
        assert r.json()["code"] == "rate_limited"
        assert int(r.headers["retry-after"]) >= 1


def test_queue_full_503():
    gate = threading.Event()
    with _client(_cfg(workers=1, queue_max=1), qa=_FakeQA(gate=gate)) as c:
        _wait_ready(c)
        h = {**AUTH, "Prefer": "respond-async"}
        first = c.post("/v1/answers", json={"question": "q1"}, headers=h)
        assert first.status_code == 202
        _poll_status(c, first.headers["location"], "running")
        assert c.post("/v1/answers", json={"question": "q2"},
                      headers=h).status_code == 202
        r = c.post("/v1/answers", json={"question": "q3"}, headers=h)
        assert r.status_code == 503
        assert r.json()["code"] == "queue_full"
        assert "retry-after" in r.headers
        gate.set()


# -- documents ----------------------------------------------------------------

def test_documents_list_and_pagination():
    with _client() as c:
        _wait_ready(c)
        body = c.get("/v1/documents", headers=AUTH).json()
        assert body["object"] == "list"
        assert body["total"] == 2
        assert [d["doc_id"] for d in body["data"]] == ["hjc.md", "ale.md"]
        assert body["data"][0]["section_count"] == 1
        page = c.get("/v1/documents?limit=1&offset=1", headers=AUTH).json()
        assert [d["doc_id"] for d in page["data"]] == ["ale.md"]
        assert page["total"] == 2 and page["limit"] == 1 and page["offset"] == 1


def test_document_detail_and_404():
    with _client() as c:
        _wait_ready(c)
        body = c.get("/v1/documents/hjc.md", headers=AUTH).json()
        assert body["doc_id"] == "hjc.md"
        assert body["keywords"] == ["HJC"]
        assert body["sections"][0]["name"] == "简介"
        assert "content" not in (body["sections"][0] or {})
        r = c.get("/v1/documents/nope.md", headers=AUTH)
        assert r.status_code == 404


def test_root_serves_docs_page_without_auth():
    with _client() as c:
        r = c.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "DeepreadQA API" in r.text
