"""In-memory answer jobs: state machine, TTL retention, idempotency replay."""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

_FINISHED = ("succeeded", "failed")


def iso_utc(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return (datetime.fromtimestamp(ts, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(ts % 1 * 1000):03d}Z")


@dataclass
class Job:
    """One answer request moving through queued → running → succeeded|failed."""

    id: str
    question: str
    created_at: float
    status: str = "queued"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    answer: Optional[str] = None
    sources: list[dict] = field(default_factory=list)
    usage: Optional[dict] = None
    forced_final: Optional[bool] = None
    error: Optional[dict] = None
    done: threading.Event = field(default_factory=threading.Event)
    # private-collection routing: the answers route resolves the collection
    # bundle at submission time and pins the snapshot here; the worker never
    # touches the CollectionManager
    collection_id: Optional[str] = None
    collection_db: Optional[str] = None
    collection_index: Any = None
    collection_titles: Optional[dict[str, str]] = None

    def mark_running(self, now: float) -> None:
        self.status = "running"
        self.started_at = now

    def succeed(self, *, answer: str, sources: list[dict],
                usage: Optional[dict], forced_final: bool, now: float) -> None:
        self.status = "succeeded"
        self.answer = answer
        self.sources = sources
        self.usage = usage
        self.forced_final = forced_final
        self.finished_at = now
        self.done.set()

    def fail(self, *, code: str, message: str, now: float) -> None:
        self.status = "failed"
        self.error = {"code": code, "message": message}
        self.finished_at = now
        self.done.set()

    def to_resource(self) -> dict:
        latency_ms = None
        if self.started_at is not None and self.finished_at is not None:
            latency_ms = int(round((self.finished_at - self.started_at) * 1000))
        return {
            "id": self.id,
            "object": "answer",
            "status": self.status,
            "question": self.question,
            "collection_id": self.collection_id,
            "answer": self.answer,
            "sources": self.sources,
            "usage": self.usage,
            "forced_final": self.forced_final,
            "created_at": iso_utc(self.created_at),
            "started_at": iso_utc(self.started_at),
            "finished_at": iso_utc(self.finished_at),
            "latency_ms": latency_ms,
            "error": self.error,
        }


class JobStore:
    """Thread-safe map of answer jobs with TTL on finished jobs.

    Single-process by design (documented API boundary); idempotency keys map
    to job ids and expire together with their job.
    """

    def __init__(self, ttl_s: float, clock: Callable[[], float] = time.time) -> None:
        self._ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._idem: dict[str, str] = {}

    def create(self, question: str,
               idempotency_key: Optional[str] = None) -> tuple[Job, bool]:
        """Return (job, created); replay an idempotency key to its live job."""
        with self._lock:
            self._purge_locked()
            if idempotency_key is not None:
                existing_id = self._idem.get(idempotency_key)
                if existing_id is not None and existing_id in self._jobs:
                    return self._jobs[existing_id], False
            job = Job(id=f"ans_{secrets.token_hex(8)}", question=question,
                      created_at=self._clock())
            self._jobs[job.id] = job
            if idempotency_key is not None:
                self._idem[idempotency_key] = job.id
            return job, True

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            self._purge_locked()
            return self._jobs.get(job_id)

    def discard(self, job_id: str) -> None:
        """Drop a job (used when queue submission fails after creation)."""
        with self._lock:
            self._jobs.pop(job_id, None)
            self._idem = {k: v for k, v in self._idem.items() if v != job_id}

    def purge_expired(self) -> None:
        with self._lock:
            self._purge_locked()

    def _purge_locked(self) -> None:
        now = self._clock()
        expired = [jid for jid, j in self._jobs.items()
                   if j.status in _FINISHED and j.finished_at is not None
                   and now - j.finished_at > self._ttl_s]
        for jid in expired:
            del self._jobs[jid]
        if expired:
            gone = set(expired)
            self._idem = {k: v for k, v in self._idem.items() if v not in gone}

    def counts(self) -> dict[str, int]:
        """Live job counts by status (for metrics/service info)."""
        with self._lock:
            out: dict[str, int] = {}
            for j in self._jobs.values():
                out[j.status] = out.get(j.status, 0) + 1
            return out
