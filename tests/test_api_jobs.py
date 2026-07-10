"""Tests for deepreadqa_api.jobs (Job state machine + JobStore TTL/idempotency)."""
from deepreadqa_api.jobs import JobStore


class _Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


def _store(ttl: float = 3600.0) -> tuple[JobStore, _Clock]:
    clock = _Clock()
    return JobStore(ttl_s=ttl, clock=clock), clock


def test_create_returns_queued_job_with_ans_id():
    store, _ = _store()
    job, created = store.create("什么是 HJC 模型？")
    assert created is True
    assert job.id.startswith("ans_")
    assert job.status == "queued"
    assert store.get(job.id) is job


def test_lifecycle_succeed_sets_fields_and_event():
    store, clock = _store()
    job, _ = store.create("q")
    clock.now += 1
    job.mark_running(clock())
    assert job.status == "running"
    clock.now += 2
    job.succeed(answer="答案", sources=[{"doc_id": "d.md", "title": "T"}],
                usage={"iterations": 3, "total_tokens": 100, "compactions": 0},
                forced_final=False, now=clock())
    assert job.status == "succeeded"
    assert job.done.is_set()
    res = job.to_resource()
    assert res["object"] == "answer"
    assert res["status"] == "succeeded"
    assert res["answer"] == "答案"
    assert res["sources"][0]["doc_id"] == "d.md"
    assert res["usage"]["total_tokens"] == 100
    assert res["latency_ms"] == 2000
    assert res["error"] is None
    assert res["created_at"].endswith("Z")


def test_lifecycle_fail_records_error():
    store, clock = _store()
    job, _ = store.create("q")
    job.mark_running(clock())
    job.fail(code="answer_failed", message="all endpoints exhausted", now=clock())
    assert job.status == "failed"
    assert job.done.is_set()
    res = job.to_resource()
    assert res["status"] == "failed"
    assert res["answer"] is None
    assert res["error"] == {"code": "answer_failed",
                            "message": "all endpoints exhausted"}


def test_idempotency_key_replays_same_job():
    store, _ = _store()
    j1, created1 = store.create("q", idempotency_key="abc")
    j2, created2 = store.create("q", idempotency_key="abc")
    assert created1 is True and created2 is False
    assert j1 is j2
    j3, created3 = store.create("q", idempotency_key="other")
    assert created3 is True and j3 is not j1


def test_ttl_purges_finished_jobs_only():
    store, clock = _store(ttl=100)
    done, _ = store.create("q1", idempotency_key="k1")
    done.mark_running(clock())
    done.succeed(answer="a", sources=[], usage=None, forced_final=False,
                 now=clock())
    running, _ = store.create("q2")
    running.mark_running(clock())
    clock.now += 101
    store.purge_expired()
    assert store.get(done.id) is None
    assert store.get(running.id) is running  # unfinished jobs never expire
    # the idempotency mapping must expire with its job
    fresh, created = store.create("q1", idempotency_key="k1")
    assert created is True and fresh is not done


def test_discard_removes_job_and_idempotency_mapping():
    store, _ = _store()
    job, _ = store.create("q", idempotency_key="k")
    store.discard(job.id)
    assert store.get(job.id) is None
    again, created = store.create("q", idempotency_key="k")
    assert created is True and again is not job
