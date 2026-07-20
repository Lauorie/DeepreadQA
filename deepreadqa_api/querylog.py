"""Append-only JSONL retention of question/answer content for audit & QA.

Enabled by config; the outward-facing docs disclose that request and response
content may be retained. Writes are best-effort — a logging failure must never
break the answer path — and size-rotated so the file cannot grow unbounded.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class QueryLog:
    """Thread-safe JSONL sink with simple size-based rotation."""

    def __init__(self, path: str, *, max_bytes: int = 50_000_000,
                 backups: int = 5) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._backups = backups
        self._lock = threading.Lock()

    def record(self, job: object, resource: dict) -> None:
        """Append one line for a finished job; never raises."""
        try:
            rec = {
                "ts": resource.get("finished_at"),
                "id": getattr(job, "id", None),
                "api_key_hash": getattr(job, "api_key_hash", None),
                "mode": resource.get("mode"),
                "collection_id": resource.get("collection_id"),
                "question": resource.get("question"),
                "options": getattr(job, "options", None),
                "status": resource.get("status"),
                "choice": resource.get("choice"),
                "abstained": resource.get("abstained"),
                "answer": resource.get("answer"),
                "usage": resource.get("usage"),
                "sources": [s.get("doc_id") for s in
                            (resource.get("sources") or [])],
                "forced_final": resource.get("forced_final"),
                "latency_ms": resource.get("latency_ms"),
                "error": resource.get("error"),
            }
            line = json.dumps(rec, ensure_ascii=False, default=str)
            with self._lock:
                self._rotate_if_needed(len(line) + 1)
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - retention is best-effort
            logger.warning("query-log write failed: %s", exc)

    def _rotate_if_needed(self, incoming: int) -> None:
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            return
        if size + incoming <= self._max_bytes:
            return
        # q.jsonl.(n-1) -> q.jsonl.n, ..., q.jsonl -> q.jsonl.1
        for i in range(self._backups - 1, 0, -1):
            src = self._path.with_suffix(self._path.suffix + f".{i}")
            if src.exists():
                os.replace(src, self._path.with_suffix(
                    self._path.suffix + f".{i + 1}"))
        if self._backups > 0:
            os.replace(self._path,
                       self._path.with_suffix(self._path.suffix + ".1"))
        else:
            self._path.unlink(missing_ok=True)
