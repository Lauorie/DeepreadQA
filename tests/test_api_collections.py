"""Tests for deepreadqa_api.collections.CollectionManager (stub enricher)."""
import time

import pytest

from deepreadqa_api.collections import CollectionManager, UploadRejected
from deepreadqa_api.config import ApiConfig

MD = ("# ALE 方法总览\n\n## 引言\nALE 多物质耦合用于流固耦合模拟。\n\n"
      "## 罚函数耦合\n罚函数系数 PFAC 取 0.1，接触刚度按主从面质量估算。\n").encode()
MD2 = ("# HJC 本构\n\n## 屈服面\n损伤 D 由等效塑性应变累积。\n").encode()


class _StubEnricher:
    def enrich_document(self, title, doc, lang):
        return ("stub tldr", ["ale"], ["sec tldr"] * len(doc.sections))


class _BoomEnricher:
    def enrich_document(self, title, doc, lang):
        raise RuntimeError("enrich exploded")


def _cfg(tmp_path, **over) -> ApiConfig:
    d = dict(api_keys=("k1", "k2"), collections_dir=str(tmp_path / "cols"),
             max_docs_per_collection=3, max_collections_per_key=2,
             max_upload_bytes=500_000)
    d.update(over)
    return ApiConfig(**d)


def _mgr(tmp_path, enricher=None, **over) -> CollectionManager:
    m = CollectionManager(_cfg(tmp_path, **over),
                          enricher_factory=lambda: enricher or _StubEnricher())
    m.start()
    return m


def _wait(m, key, cid, doc_id, timeout=10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = m.document(key, cid, doc_id)
        if d and d["status"] in ("ready", "failed"):
            return d
        time.sleep(0.05)
    pytest.fail("ingest never finished")


def test_create_get_list_delete_with_owner_isolation(tmp_path):
    m = _mgr(tmp_path)
    col = m.create("k1", "我的库")
    assert col["id"].startswith("col_") and col["object"] == "collection"
    assert col["name"] == "我的库" and col["status"] == "empty"
    assert m.get("k1", col["id"])["id"] == col["id"]
    assert [c["id"] for c in m.list("k1")] == [col["id"]]
    # another key must see nothing
    assert m.get("k2", col["id"]) is None
    assert m.list("k2") == []
    assert m.delete("k2", col["id"]) is False
    assert m.delete("k1", col["id"]) is True
    assert m.get("k1", col["id"]) is None
    m.shutdown()


def test_collection_count_limit(tmp_path):
    m = _mgr(tmp_path)
    m.create("k1", "a")
    m.create("k1", "b")
    with pytest.raises(UploadRejected) as exc:
        m.create("k1", "c")
    assert exc.value.code == "collection_limit"
    m.create("k2", "other-key-unaffected")
    m.shutdown()


@pytest.mark.parametrize("filename,content,reason", [
    ("notes.txt", MD, "extension"),
    ("big.md", b"x" * 500_001, "exceeds"),
    ("bad.md", b"\xff\xfe\x00 not utf8", "UTF-8"),
    ("....md", MD, "filename"),
])
def test_upload_rejections(tmp_path, filename, content, reason):
    m = _mgr(tmp_path)
    cid = m.create("k1", "c")["id"]
    with pytest.raises(UploadRejected) as exc:
        m.upload("k1", cid, [(filename, content)])
    assert exc.value.code == "upload_rejected"
    assert reason.lower() in str(exc.value).lower()
    m.shutdown()


def test_upload_duplicate_and_doc_limit(tmp_path):
    m = _mgr(tmp_path)
    cid = m.create("k1", "c")["id"]
    with pytest.raises(UploadRejected):  # duplicate within one batch
        m.upload("k1", cid, [("a.md", MD), ("a.md", MD2)])
    m.upload("k1", cid, [("a.md", MD)])
    _wait(m, "k1", cid, "a.md")
    with pytest.raises(UploadRejected):  # duplicate vs existing
        m.upload("k1", cid, [("a.md", MD2)])
    with pytest.raises(UploadRejected) as exc:  # 1 existing + 3 > max 3
        m.upload("k1", cid, [("b.md", MD), ("c.md", MD), ("d.md", MD)])
    assert exc.value.code == "collection_limit"
    m.shutdown()


def test_ingest_to_ready_with_real_structure(tmp_path):
    m = _mgr(tmp_path)
    cid = m.create("k1", "c")["id"]
    out = m.upload("k1", cid, [("ale.md", MD)])
    assert out[0]["doc_id"] == "ale.md" and out[0]["status"] == "processing"
    doc = _wait(m, "k1", cid, "ale.md")
    assert doc["status"] == "ready"
    assert doc["title"] == "ALE 方法总览"
    assert doc["tldr"] == "stub tldr"
    assert doc["token_count"] > 0 and doc["section_count"] >= 2
    col = m.get("k1", cid)
    assert col["documents_ready"] == 1 and col["status"] == "ready"
    assert m.documents("k1", cid)[0]["doc_id"] == "ale.md"
    m.shutdown()


def test_ingest_failure_marks_failed(tmp_path):
    m = _mgr(tmp_path, enricher=_BoomEnricher())
    cid = m.create("k1", "c")["id"]
    m.upload("k1", cid, [("ale.md", MD)])
    doc = _wait(m, "k1", cid, "ale.md")
    assert doc["status"] == "failed"
    assert "enrich exploded" in doc["error"]
    assert m.get("k1", cid)["documents_failed"] == 1
    m.shutdown()


def test_bundle_search_and_index_invalidation(tmp_path):
    m = _mgr(tmp_path)
    cid = m.create("k1", "c")["id"]
    assert m.bundle("k2", cid) is None  # owner isolation
    db, index, titles, ready = m.bundle("k1", cid)
    assert ready == 0
    m.upload("k1", cid, [("ale.md", MD)])
    _wait(m, "k1", cid, "ale.md")
    db, index, titles, ready = m.bundle("k1", cid)
    assert ready == 1 and titles["ale.md"] == "ALE 方法总览"
    assert any(h.doc_id == "ale.md" for h in index.search_many(["罚函数"], top_k=5))
    # second doc must appear after cache invalidation, without manual action
    m.upload("k1", cid, [("hjc.md", MD2)])
    _wait(m, "k1", cid, "hjc.md")
    _, index2, titles2, ready2 = m.bundle("k1", cid)
    assert ready2 == 2 and "hjc.md" in titles2
    assert any(h.doc_id == "hjc.md" for h in index2.search_many(["HJC"], top_k=5))
    m.shutdown()


def test_restart_recovers_registry_and_marks_orphans(tmp_path):
    m = _mgr(tmp_path)
    cid = m.create("k1", "persisted")["id"]
    m.upload("k1", cid, [("ale.md", MD)])
    _wait(m, "k1", cid, "ale.md")
    # simulate a doc whose ingest was cut mid-flight: meta says processing,
    # but the document row never landed
    m._set_doc_meta(cid, "lost.md", {"status": "processing", "bytes": 10,
                                     "uploaded_at": 0.0})
    m.shutdown()

    m2 = CollectionManager(_cfg(tmp_path), enricher_factory=_StubEnricher)
    m2.start()
    col = m2.get("k1", cid)
    assert col is not None and col["name"] == "persisted"
    assert m2.document("k1", cid, "ale.md")["status"] == "ready"
    lost = m2.document("k1", cid, "lost.md")
    assert lost["status"] == "failed" and "interrupt" in lost["error"]
    m2.shutdown()
