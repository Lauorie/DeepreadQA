"""HTTP tests for private collections: upload markdown, then answer over it."""
import time

import pytest
from fastapi.testclient import TestClient

from deepreadqa_api.app import create_app
from deepreadqa_api.collections import CollectionManager
from deepreadqa_api.config import ApiConfig
from deepreadqa_api.engine import AnswerEngine

from tests.test_api_collections import MD, _StubEnricher
from tests.test_api_engine import CATALOG, _FakeQA

AUTH = {"Authorization": "Bearer test-key"}
OTHER = {"Authorization": "Bearer other-key"}


def _cfg(tmp_path, **over) -> ApiConfig:
    d = dict(api_keys=("test-key", "other-key"), workers=1, queue_max=8,
             sync_wait_cap_s=5.0, job_ttl_s=60.0, rate_limit_rpm=6000.0,
             rate_limit_burst=1000, max_question_chars=200,
             collections_dir=str(tmp_path / "cols"),
             max_docs_per_collection=3, max_collections_per_key=2)
    d.update(over)
    return ApiConfig(**d)


def _client(tmp_path, factory=None, **over):
    cfg = _cfg(tmp_path, **over)
    engine = AnswerEngine(cfg, qa_factory=factory or (lambda *a: _FakeQA()),
                          catalog=CATALOG)
    manager = CollectionManager(cfg, enricher_factory=_StubEnricher)
    app = create_app(cfg, engine=engine, collections_manager=manager)
    return TestClient(app, raise_server_exceptions=False)


def _mkcol(c, name="试用库", headers=AUTH) -> str:
    r = c.post("/v1/collections", json={"name": name}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(c, cid, files, headers=AUTH):
    return c.post(f"/v1/collections/{cid}/documents",
                  files=[("files", (n, b, "text/markdown")) for n, b in files],
                  headers=headers)


def _wait_doc(c, cid, doc_id, want="ready", timeout=10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = c.get(f"/v1/collections/{cid}/documents/{doc_id}", headers=AUTH)
        if r.status_code == 200 and r.json()["status"] in ("ready", "failed"):
            body = r.json()
            assert body["status"] == want, body
            return body
        time.sleep(0.05)
    pytest.fail("document never reached a final status")


def test_collection_crud_lifecycle(tmp_path):
    with _client(tmp_path) as c:
        cid = _mkcol(c)
        got = c.get(f"/v1/collections/{cid}", headers=AUTH).json()
        assert got["name"] == "试用库" and got["status"] == "empty"
        listed = c.get("/v1/collections", headers=AUTH).json()
        assert listed["total"] == 1 and listed["data"][0]["id"] == cid
        assert c.delete(f"/v1/collections/{cid}", headers=AUTH).status_code == 204
        assert c.get(f"/v1/collections/{cid}", headers=AUTH).status_code == 404


def test_collection_count_limit_http(tmp_path):
    with _client(tmp_path) as c:
        _mkcol(c, "a")
        _mkcol(c, "b")
        r = c.post("/v1/collections", json={"name": "c"}, headers=AUTH)
        assert r.status_code == 422
        assert r.json()["code"] == "collection_limit"


def test_upload_then_ready_then_catalog(tmp_path):
    with _client(tmp_path) as c:
        cid = _mkcol(c)
        r = _upload(c, cid, [("ale.md", MD)])
        assert r.status_code == 202
        assert r.json()["data"][0]["status"] == "processing"
        doc = _wait_doc(c, cid, "ale.md")
        assert doc["title"] == "ALE 方法总览" and doc["section_count"] >= 2
        docs = c.get(f"/v1/collections/{cid}/documents", headers=AUTH).json()
        assert docs["total"] == 1 and docs["data"][0]["status"] == "ready"
        col = c.get(f"/v1/collections/{cid}", headers=AUTH).json()
        assert col["status"] == "ready" and col["documents_ready"] == 1


def test_upload_rejection_renders_problem(tmp_path):
    with _client(tmp_path) as c:
        cid = _mkcol(c)
        r = _upload(c, cid, [("notes.txt", MD)])
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "upload_rejected"
        assert "notes.txt" in body["detail"]


def test_cross_key_collections_are_invisible(tmp_path):
    with _client(tmp_path) as c:
        cid = _mkcol(c)
        assert c.get(f"/v1/collections/{cid}", headers=OTHER).status_code == 404
        assert _upload(c, cid, [("a.md", MD)], headers=OTHER).status_code == 404
        assert c.get("/v1/collections", headers=OTHER).json()["total"] == 0


def test_answer_over_collection(tmp_path):
    seen_factory_args = []

    def factory(db_path=None, index=None, mode="qa"):
        seen_factory_args.append(db_path)
        return _FakeQA()

    with _client(tmp_path, factory=factory) as c:
        cid = _mkcol(c)
        _upload(c, cid, [("ale.md", MD)])
        _wait_doc(c, cid, "ale.md")
        r = c.post("/v1/answers",
                   json={"question": "PFAC 取多少？", "collection_id": cid},
                   headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.json()["collection_id"] == cid
        assert any(p and p.endswith(f"{cid}.db") for p in seen_factory_args)


def test_answer_collection_404_and_409(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/v1/answers",
                   json={"question": "q", "collection_id": "col_deadbeef"},
                   headers=AUTH)
        assert r.status_code == 404
        cid = _mkcol(c)  # empty: no ready documents
        r = c.post("/v1/answers", json={"question": "q", "collection_id": cid},
                   headers=AUTH)
        assert r.status_code == 409
        assert r.json()["code"] == "collection_not_ready"


def test_answer_without_collection_is_unchanged(tmp_path):
    seen_factory_args = []

    def factory(db_path=None, index=None, mode="qa"):
        seen_factory_args.append(db_path)
        return _FakeQA()

    with _client(tmp_path, factory=factory) as c:
        r = c.post("/v1/answers", json={"question": "HJC?"}, headers=AUTH)
        assert r.status_code == 200
        assert r.json()["collection_id"] is None
        assert seen_factory_args == [None]  # builtin path only


def test_openapi_covers_collections(tmp_path):
    with _client(tmp_path) as c:
        paths = c.get("/openapi.json").json()["paths"]
        assert "/v1/collections" in paths
        assert "/v1/collections/{cid}/documents" in paths


def test_upload_file_count_checked_before_reading(tmp_path):
    with _client(tmp_path) as c:
        cid = _mkcol(c)
        r = _upload(c, cid, [(f"f{i}.md", MD) for i in range(4)])  # limit 3
        assert r.status_code == 422
        assert r.json()["code"] == "collection_limit"


def test_choice_over_collection(tmp_path):
    from tests.test_api_engine import _FakeChoiceQA

    fake = _FakeChoiceQA(letter="D")
    seen = []

    def factory(db_path=None, index=None, mode="qa"):
        seen.append((db_path, mode))
        return fake if mode == "choice" else _FakeQA()

    with _client(tmp_path, factory=factory) as c:
        cid = _mkcol(c)
        _upload(c, cid, [("ale.md", MD)])
        _wait_doc(c, cid, "ale.md")
        r = c.post("/v1/answers",
                   json={"question": "PFAC 取多少？", "mode": "choice",
                         "collection_id": cid,
                         "options": {"A": "1", "B": "0.5", "C": "0.2", "D": "0.1"}},
                   headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.json()["choice"] == "D"
        assert r.json()["collection_id"] == cid
        assert any(db and db.endswith(f"{cid}.db") and m == "choice"
                   for db, m in seen)
