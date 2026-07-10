"""Answer engine: a bounded queue drained by worker threads.

Each worker owns a full DeepreadQA instance (its own sqlite Reader — the
connection is thread-affine) while all workers share one in-memory BM25
SearchIndex, which is read-only after construction.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

from .config import ApiConfig
from .jobs import Job

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """The answer queue is saturated; the caller should retry later."""


class NotReadyError(Exception):
    """The engine has not finished bootstrapping (or failed to)."""


# tools whose calls constitute actually reading a document (vs. merely
# surfacing it in search results); mirrors deepreadqa.tools handler names
_READ_TOOLS = frozenset(
    {"head", "read_section", "grep", "intro", "preview", "read_raw"})


def _docs_read(tool_calls: list[dict]) -> list[str]:
    out: set[str] = set()
    for call in tool_calls:
        args = call.get("args") or {}
        if call.get("tool") in _READ_TOOLS and "doc_id" in args:
            out.add(args["doc_id"])
    return sorted(out)


def _summary(head: dict) -> dict:
    return {"doc_id": head["doc_id"], "title": head["title"],
            "language": head["language"], "tldr": head["tldr"],
            "token_count": head["token_count"],
            "section_count": len(head["sections"])}


class AnswerEngine:
    """Owns worker threads answering jobs against the knowledge base.

    Args:
        cfg: API-layer configuration.
        qa_factory: Called once per worker thread to build its QA instance;
            defaults to the real DeepreadQA over cfg.db_path. Tests inject
            fakes here (and a catalog) so no store or LLM is touched.
        catalog: Pre-built list of Reader.head()-shaped dicts; required when
            qa_factory is injected, built from the store otherwise.
        metrics: Optional metrics sink with observe_answer(job) hook.
    """

    def __init__(self, cfg: ApiConfig, *,
                 qa_factory: Optional[Callable[[], object]] = None,
                 catalog: Optional[list[dict]] = None,
                 metrics: Optional[object] = None) -> None:
        if qa_factory is not None and catalog is None:
            raise ValueError("catalog is required when qa_factory is injected")
        self._cfg = cfg
        self._qa_factory = qa_factory
        self._catalog = catalog
        self._metrics = metrics
        self._queue: queue.Queue = queue.Queue(maxsize=cfg.queue_max)
        self._threads: list[threading.Thread] = []
        self._ready = threading.Event()
        self._startup_error: Optional[str] = None
        self._titles: dict[str, str] = {}
        self._heads: dict[str, dict] = {}
        self._model_name: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Kick off bootstrap in the background; readiness is via readyz."""
        threading.Thread(target=self._bootstrap, name="engine-bootstrap",
                         daemon=True).start()

    def _bootstrap(self) -> None:
        try:
            if self._qa_factory is None:
                self._qa_factory, self._catalog = self._build_real_engine()
            self._heads = {d["doc_id"]: d for d in self._catalog}
            self._titles = {d["doc_id"]: d["title"] for d in self._catalog}
            for i in range(self._cfg.workers):
                th = threading.Thread(target=self._worker_loop,
                                      name=f"answer-worker-{i}", daemon=True)
                th.start()
                self._threads.append(th)
            self._ready.set()
            logger.info("engine ready: %d docs, %d workers",
                        len(self._catalog), self._cfg.workers)
        except Exception as exc:  # noqa: BLE001 - surfaced via readyz
            self._startup_error = f"{type(exc).__name__}: {exc}"
            logger.error("engine bootstrap failed: %s", exc, exc_info=True)

    def _build_real_engine(self) -> tuple[Callable[[], object], list[dict]]:
        from deepread_sdk import Reader
        from deepreadqa import Config as QaConfig
        from deepreadqa import DeepreadQA
        from deepreadqa.retrieval import SearchIndex

        qa_cfg = QaConfig.from_env(db_path=self._cfg.db_path)
        self._model_name = qa_cfg.endpoint.model
        boot = Reader(qa_cfg.db_path)
        index = SearchIndex(boot)  # built once, shared read-only
        catalog = [boot.head(d["doc_id"]) for d in boot.list_docs()]

        def factory() -> object:
            # Reader (and its sqlite connection) is created inside the
            # worker thread that calls this factory.
            return DeepreadQA(qa_cfg, reader=Reader(qa_cfg.db_path),
                              index=index)

        return factory, catalog

    def shutdown(self, timeout_s: float = 10.0) -> None:
        for _ in self._threads:
            self._queue.put(None)
        deadline = time.monotonic() + timeout_s
        for th in self._threads:
            th.join(timeout=max(0.0, deadline - time.monotonic()))

    # -- submission --------------------------------------------------------

    def submit(self, job: Job) -> None:
        if not self._ready.is_set():
            raise NotReadyError(self._startup_error or "engine is starting up")
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            raise QueueFullError("answer queue is full") from None

    # -- introspection -----------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    @property
    def startup_error(self) -> Optional[str]:
        return self._startup_error

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    def attach_metrics(self, metrics: object) -> None:
        self._metrics = metrics

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def document_count(self) -> int:
        return len(self._catalog or [])

    def catalog_summaries(self) -> list[dict]:
        return [_summary(d) for d in (self._catalog or [])]

    def catalog_head(self, doc_id: str) -> Optional[dict]:
        return self._heads.get(doc_id)

    # -- worker ------------------------------------------------------------

    def _worker_loop(self) -> None:
        qa = self._qa_factory()
        while True:
            job = self._queue.get()
            if job is None:
                return
            job.mark_running(time.time())
            try:
                res = qa.answer(job.question)
                if res.answer:
                    read = _docs_read(res.tool_calls)
                    sources = [{"doc_id": d, "title": self._titles.get(d)}
                               for d in read]
                    usage = {"iterations": res.iterations,
                             "total_tokens": res.total_tokens,
                             "compactions": res.compactions,
                             "documents_read": len(read),
                             "documents_seen": len(res.seen_docs)}
                    job.succeed(answer=res.answer, sources=sources,
                                usage=usage, forced_final=res.forced_final,
                                now=time.time())
                else:
                    job.fail(code="answer_failed",
                             message=res.error or
                             "engine returned an empty answer",
                             now=time.time())
            except Exception as exc:  # noqa: BLE001 - one job must not kill the worker
                logger.error("job %s failed: %s", job.id, exc, exc_info=True)
                job.fail(code="answer_failed",
                         message=f"{type(exc).__name__}: {exc}",
                         now=time.time())
            finally:
                if self._metrics is not None:
                    self._metrics.observe_answer(job)
                self._queue.task_done()
