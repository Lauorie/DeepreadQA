"""Tests for deepreadqa_api.querylog.QueryLog (JSONL retention + rotation)."""
import json
from types import SimpleNamespace

from deepreadqa_api.querylog import QueryLog


def _job(**over):
    base = dict(
        id="ans_1", api_key_hash="abcd1234", mode="qa", collection_id=None,
        question="HJC 是什么？", options=None, status="succeeded",
        choice=None, abstained=None, answer="HJC 是一种混凝土本构模型。",
        usage={"iterations": 3, "total_tokens": 1200},
        sources=[{"doc_id": "hjc.md", "title": "T"}], forced_final=False,
        error=None,
        finished_at_iso="2026-07-20T10:00:00.000Z",
        latency_ms=42000)
    base.update(over)
    return SimpleNamespace(
        id=base["id"], api_key_hash=base["api_key_hash"], mode=base["mode"],
        collection_id=base["collection_id"], question=base["question"],
        options=base["options"], status=base["status"], choice=base["choice"],
        abstained=base["abstained"], answer=base["answer"],
        usage=base["usage"], sources=base["sources"],
        forced_final=base["forced_final"], error=base["error"],
        finished_at=1_753_000_000.0, latency_ms=base["latency_ms"],
        _iso=base["finished_at_iso"])


def _resource(job):
    return {"question": job.question, "mode": job.mode,
            "collection_id": job.collection_id, "status": job.status,
            "choice": job.choice, "abstained": job.abstained,
            "answer": job.answer, "usage": job.usage, "sources": job.sources,
            "forced_final": job.forced_final, "error": job.error,
            "finished_at": job._iso, "latency_ms": job.latency_ms}


def test_record_writes_one_json_line(tmp_path):
    path = tmp_path / "q.jsonl"
    log = QueryLog(str(path))
    job = _job()
    log.record(job, _resource(job))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["id"] == "ans_1"
    assert rec["api_key_hash"] == "abcd1234"
    assert rec["question"] == "HJC 是什么？"
    assert rec["answer"] == "HJC 是一种混凝土本构模型。"
    assert rec["mode"] == "qa"
    assert rec["usage"]["total_tokens"] == 1200
    assert rec["sources"] == ["hjc.md"]
    assert rec["latency_ms"] == 42000
    assert rec["ts"] == "2026-07-20T10:00:00.000Z"


def test_choice_fields_are_captured(tmp_path):
    path = tmp_path / "q.jsonl"
    log = QueryLog(str(path))
    job = _job(mode="choice", options={"A": "甲", "B": "乙"}, choice="B",
               abstained=False)
    log.record(job, _resource(job))
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert rec["mode"] == "choice"
    assert rec["options"] == {"A": "甲", "B": "乙"}
    assert rec["choice"] == "B"
    assert rec["abstained"] is False


def test_appends_across_records(tmp_path):
    path = tmp_path / "q.jsonl"
    log = QueryLog(str(path))
    for i in range(3):
        job = _job(id=f"ans_{i}")
        log.record(job, _resource(job))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_size_rotation_keeps_backups(tmp_path):
    path = tmp_path / "q.jsonl"
    log = QueryLog(str(path), max_bytes=400, backups=2)
    for i in range(20):
        job = _job(id=f"ans_{i}", answer="x" * 100)
        log.record(job, _resource(job))
    # active file plus at most `backups` rotated files
    assert path.exists()
    rotated = sorted(tmp_path.glob("q.jsonl.*"))
    assert 1 <= len(rotated) <= 2
    # every surviving line is still valid JSON
    for f in [path, *rotated]:
        for line in f.read_text(encoding="utf-8").splitlines():
            json.loads(line)


def test_record_failure_is_swallowed(tmp_path):
    # writing must never crash the answer path — here the parent path is a
    # regular file, so mkdir/open fails; record() must swallow it
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    log = QueryLog(str(blocker / "q.jsonl"))
    job = _job()
    log.record(job, _resource(job))  # must not raise
