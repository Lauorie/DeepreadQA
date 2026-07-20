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
    eng = AnswerEngine(cfg, qa_factory=lambda *a: qa, catalog=catalog)
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
    eng = AnswerEngine(_cfg(), qa_factory=lambda *a: _FakeQA(), catalog=CATALOG)
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


def test_collection_job_routes_through_factory_with_bundle():
    calls = []

    def factory(db_path=None, index=None, mode="qa"):
        calls.append((db_path, index))
        return _FakeQA()

    eng = AnswerEngine(_cfg(), qa_factory=factory, catalog=CATALOG)
    eng.start()
    assert eng.wait_ready(timeout=5)
    store = JobStore(ttl_s=60)
    job, _ = store.create("上传库里的问题")
    sentinel = object()
    job.collection_id = "col_ab"
    job.collection_db = "/tmp/col_ab.db"
    job.collection_index = sentinel
    job.collection_titles = {"hjc.md": "上传的 HJC 文档"}
    eng.submit(job)
    assert job.done.wait(timeout=5)
    assert ("/tmp/col_ab.db", sentinel) in calls
    # titles resolve from the collection snapshot, not the builtin catalog
    assert job.sources[0] == {"doc_id": "hjc.md", "title": "上传的 HJC 文档"}
    assert job.to_resource()["collection_id"] == "col_ab"
    eng.shutdown()


class _FakeChoiceQA:
    """Mimics ChoiceQA: same tool_calls/seen_docs shape, letter answer."""

    def __init__(self, letter="B", abstained=False, error=None):
        base = _result()
        self._res = SimpleNamespace(
            answer=letter, compose_text=f"证据支持 {letter}。\n答案：{letter}",
            draft="draft", iterations=2, total_tokens=456, compactions=0,
            forced_final=False, abstained=abstained, error=error,
            tool_calls=base.tool_calls, seen_docs=base.seen_docs)

    def answer_choice(self, question, options):
        self.seen_args = (question, options)
        return self._res


def test_choice_job_maps_letter_and_calls_answer_choice():
    seen_modes = []
    fake = _FakeChoiceQA()

    def factory(db_path=None, index=None, mode="qa"):
        seen_modes.append(mode)
        return fake if mode == "choice" else _FakeQA()

    eng = AnswerEngine(_cfg(), qa_factory=factory, catalog=CATALOG)
    eng.start()
    assert eng.wait_ready(timeout=5)
    store = JobStore(ttl_s=60)
    job, _ = store.create("哪项正确？")
    job.mode = "choice"
    job.options = {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}
    eng.submit(job)
    assert job.done.wait(timeout=5)
    assert job.status == "succeeded"
    assert fake.seen_args == ("哪项正确？", job.options)
    assert job.choice == "B" and job.abstained is False
    assert "答案：B" in job.answer
    assert job.usage["total_tokens"] == 456
    res = job.to_resource()
    assert res["mode"] == "choice" and res["choice"] == "B"
    assert "choice" in seen_modes
    eng.shutdown()


def test_choice_abstain_without_error_succeeds_with_null_choice():
    fake = _FakeChoiceQA(letter="", abstained=True)
    eng = AnswerEngine(_cfg(), qa_factory=lambda *a, **k: fake, catalog=CATALOG)
    eng.start()
    assert eng.wait_ready(timeout=5)
    store = JobStore(ttl_s=60)
    job, _ = store.create("q")
    job.mode = "choice"
    job.options = {"A": "1", "B": "2", "C": "3", "D": "4"}
    eng.submit(job)
    assert job.done.wait(timeout=5)
    assert job.status == "succeeded"
    assert job.choice is None and job.abstained is True
    eng.shutdown()


def test_choice_error_with_abstain_fails():
    fake = _FakeChoiceQA(letter="", abstained=True, error="all endpoints down")
    eng = AnswerEngine(_cfg(), qa_factory=lambda *a, **k: fake, catalog=CATALOG)
    eng.start()
    assert eng.wait_ready(timeout=5)
    store = JobStore(ttl_s=60)
    job, _ = store.create("q")
    job.mode = "choice"
    job.options = {"A": "1", "B": "2", "C": "3", "D": "4"}
    eng.submit(job)
    assert job.done.wait(timeout=5)
    assert job.status == "failed"
    assert "all endpoints down" in job.error["message"]
    eng.shutdown()


def test_qa_mode_resource_has_null_choice_fields():
    eng = _engine(_cfg(), _FakeQA())
    store = JobStore(ttl_s=60)
    job = _run(eng, store)
    res = job.to_resource()
    assert res["mode"] == "qa" and res["choice"] is None
    assert res["abstained"] is None
    eng.shutdown()


def test_recorder_called_for_finished_job():
    recorded = []

    class _Rec:
        def record(self, job, resource):
            recorded.append((job.id, resource["question"], resource["answer"]))

    eng = AnswerEngine(_cfg(), qa_factory=lambda *a: _FakeQA(), catalog=CATALOG)
    eng.attach_recorder(_Rec())
    eng.start()
    assert eng.wait_ready(timeout=5)
    store = JobStore(ttl_s=60)
    job = _run(eng, store, question="记录我")
    assert recorded == [(job.id, "记录我", "答案文本")]
    eng.shutdown()


def test_recorder_called_even_when_job_fails():
    recorded = []

    class _Rec:
        def record(self, job, resource):
            recorded.append((resource["status"], resource["error"]))

    eng = AnswerEngine(_cfg(), qa_factory=lambda *a: _FakeQA(exc=RuntimeError("boom")),
                       catalog=CATALOG)
    eng.attach_recorder(_Rec())
    eng.start()
    assert eng.wait_ready(timeout=5)
    store = JobStore(ttl_s=60)
    _run(eng, store)
    assert len(recorded) == 1 and recorded[0][0] == "failed"
    eng.shutdown()
