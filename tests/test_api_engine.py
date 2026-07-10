"""Tests for deepreadqa_api.engine.AnswerEngine (fake QA, no LLM, no store)."""
import threading
import time
from types import SimpleNamespace

import pytest

from deepreadqa_api.config import ApiConfig
from deepreadqa_api.engine import AnswerEngine, NotReadyError, QueueFullError
from deepreadqa_api.jobs import JobStore

CATALOG = [
    {"doc_id": "hjc.md", "title": "HJC 本构模型", "language": "zh",
     "tldr": "混凝土冲击本构", "keywords": ["HJC"], "abstract": "",
     "token_count": 1200,
     "sections": [{"name": "简介", "idx": 0, "tldr": "t", "token_count": 100}]},
    {"doc_id": "ale.md", "title": "ALE 方法", "language": "zh",
     "tldr": "流固耦合", "keywords": ["ALE"], "abstract": "", "token_count": 800,
     "sections": []},
]


def _cfg(**over) -> ApiConfig:
    defaults = dict(api_keys=("k",), workers=1, queue_max=4,
                    sync_wait_cap_s=5.0, job_ttl_s=60.0)
    defaults.update(over)
    return ApiConfig(**defaults)


def _result(answer="答案文本", error=None):
    # tool_calls mirrors the harness call_log: search hits populate seen_docs,
    # but only head/read_section/grep-style calls count as actually reading
    calls = [
        {"iter": 0, "tool": "search", "args": {"queries": ["HJC"]}},
        {"iter": 1, "tool": "head", "args": {"doc_id": "hjc.md"}},
        {"iter": 1, "tool": "read_section",
         "args": {"doc_id": "hjc.md", "name": "简介"}},
        {"iter": 2, "tool": "grep",
         "args": {"doc_id": "unknown.md", "pattern": "damage"}},
    ]
    return SimpleNamespace(answer=answer, full_answer=answer, iterations=3,
                           total_tokens=1234, compactions=0, forced_final=False,
                           error=error, tool_calls=calls,
                           seen_docs={"hjc.md", "unknown.md", "ale.md"})


class _FakeQA:
    def __init__(self, result=None, exc=None, gate=None):
        self._result = result or _result()
        self._exc = exc
        self._gate = gate

    def answer(self, question):
        if self._gate is not None:
            self._gate.wait(timeout=10)
        if self._exc is not None:
            raise self._exc
        return self._result


def _engine(cfg, qa, catalog=CATALOG) -> AnswerEngine:
    eng = AnswerEngine(cfg, qa_factory=lambda: qa, catalog=catalog)
    eng.start()
    assert eng.wait_ready(timeout=5)
    return eng


def _run(eng, store, question="HJC 是什么？"):
    job, _ = store.create(question)
    eng.submit(job)
    assert job.done.wait(timeout=5)
    return job


def test_success_maps_result_and_sources():
    eng = _engine(_cfg(), _FakeQA())
    store = JobStore(ttl_s=60)
    job = _run(eng, store)
    assert job.status == "succeeded"
    assert job.answer == "答案文本"
    assert job.usage == {"iterations": 3, "total_tokens": 1234,
                         "compactions": 0, "documents_read": 2,
                         "documents_seen": 3}
    # sources = docs actually opened (head/read/grep), NOT every search hit;
    # sorted by doc_id, titles resolved from catalog, None if unknown
    assert job.sources == [{"doc_id": "hjc.md", "title": "HJC 本构模型"},
                           {"doc_id": "unknown.md", "title": None}]
    eng.shutdown()


def test_exception_fails_job():
    eng = _engine(_cfg(), _FakeQA(exc=RuntimeError("boom")))
    store = JobStore(ttl_s=60)
    job = _run(eng, store)
    assert job.status == "failed"
    assert job.error["code"] == "answer_failed"
    assert "boom" in job.error["message"]
    eng.shutdown()


def test_empty_answer_fails_with_engine_error():
    eng = _engine(_cfg(), _FakeQA(result=_result(answer="", error="LLM down")))
    store = JobStore(ttl_s=60)
    job = _run(eng, store)
    assert job.status == "failed"
    assert job.error["message"] == "LLM down"
    eng.shutdown()


def test_submit_before_ready_raises():
    eng = AnswerEngine(_cfg(), qa_factory=lambda: _FakeQA(), catalog=CATALOG)
    store = JobStore(ttl_s=60)
    job, _ = store.create("q")
    with pytest.raises(NotReadyError):
        eng.submit(job)


def test_queue_full_raises():
    gate = threading.Event()
    eng = _engine(_cfg(workers=1, queue_max=1), _FakeQA(gate=gate))
    store = JobStore(ttl_s=60)
    j1, _ = store.create("q1")
    eng.submit(j1)
    # wait until the single worker has picked j1 up, freeing the queue slot
    deadline = time.time() + 5
    while j1.status != "running" and time.time() < deadline:
        time.sleep(0.01)
    j2, _ = store.create("q2")
    eng.submit(j2)  # fills the queue slot
    j3, _ = store.create("q3")
    with pytest.raises(QueueFullError):
        eng.submit(j3)
    gate.set()
    assert j2.done.wait(timeout=5)
    eng.shutdown()


def test_catalog_accessors():
    eng = _engine(_cfg(), _FakeQA())
    assert eng.document_count == 2
    summaries = eng.catalog_summaries()
    assert summaries[0] == {"doc_id": "hjc.md", "title": "HJC 本构模型",
                            "language": "zh", "tldr": "混凝土冲击本构",
                            "token_count": 1200, "section_count": 1}
    head = eng.catalog_head("hjc.md")
    assert head["keywords"] == ["HJC"]
    assert eng.catalog_head("nope.md") is None
    eng.shutdown()
