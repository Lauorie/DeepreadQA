"""Private caller-uploaded markdown collections.

One SQLite store per collection under cfg.collections_dir; ownership and
per-document ingest status live in the store's existing meta KV table
(keys `api:*`), so deepread_sdk's schema stays untouched. A background
thread pool drains the ingest queue through deepread_sdk.build.process_one.
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from deepread_sdk import Reader, store
from deepread_sdk.build import process_one
from deepreadqa.retrieval import SearchIndex

from .config import ApiConfig
from .jobs import iso_utc

logger = logging.getLogger(__name__)

_ALLOWED_SUFFIXES = (".md", ".markdown")
_DOC_KEY = "api:doc:"
_FINISHED = ("ready", "failed")


class UploadRejected(Exception):
    """Client-side upload problem; .code maps to the problem+json code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _owner_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _sanitize_doc_id(filename: str) -> str:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    clean = re.sub(r"[^\w.\- ()]", "_", base, flags=re.UNICODE).strip()
    stem = clean.rsplit(".", 1)[0] if "." in clean else clean
    if not re.search(r"\w", stem, flags=re.UNICODE):
        raise UploadRejected("upload_rejected", f"illegal filename: {filename!r}")
    return clean


def _default_enricher_factory() -> Any:
    import os

    from deepread_sdk.enrich import Enricher
    from deepread_sdk.llm import EnrichLLM

    return Enricher(EnrichLLM(
        base_url=os.environ.get("AIBERM_BASE_URL", "https://aiberm.com/v1"),
        api_key=os.environ["AIBERM_API_KEY"],
        model=os.environ.get("DEEPREAD_ENRICH_MODEL",
                             "deepseek/deepseek-v4-flash")))


class CollectionManager:
    """Registry + ingest pipeline + per-collection index cache."""

    def __init__(self, cfg: ApiConfig, *,
                 enricher_factory: Optional[Callable[[], Any]] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._cfg = cfg
        self._clock = clock
        self._dir = Path(cfg.collections_dir)
        self._enricher_factory = enricher_factory or _default_enricher_factory
        self._enricher: Any = None  # built lazily inside the ingest thread
        self._lock = threading.Lock()
        self._registry: dict[str, dict] = {}  # cid -> {owner,name,created_at}
        self._index_lock = threading.Lock()
        self._indexes: dict[str, SearchIndex] = {}
        self._queue: queue.Queue = queue.Queue()
        self._pending: dict[tuple[str, str], str] = {}  # (cid,doc_id) -> text
        self._threads: list[threading.Thread] = []

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        for db in sorted(self._dir.glob("col_*.db")):
            cid = db.stem
            conn = None
            try:
                conn = store.connect(db)
                owner = store.get_meta(conn, "api:owner")
                if owner is None:
                    continue
                self._registry[cid] = {
                    "owner": owner,
                    "name": store.get_meta(conn, "api:name") or cid,
                    "created_at": float(
                        store.get_meta(conn, "api:created_at") or 0),
                }
                self._recover_orphans(conn)
            except Exception as exc:  # noqa: BLE001 - one bad file must never
                # crash-loop the whole service; skip it and keep serving
                logger.error("skipping malformed collection db %s: %s", db, exc)
                self._registry.pop(cid, None)
            finally:
                if conn is not None:
                    conn.close()
        for i in range(self._cfg.ingest_workers):
            th = threading.Thread(target=self._ingest_loop,
                                  name=f"ingest-{i}", daemon=True)
            th.start()
            self._threads.append(th)
        logger.info("collections ready: %d found under %s",
                    len(self._registry), self._dir)

    def _recover_orphans(self, conn) -> None:
        """A doc left 'processing' across a restart lost its in-memory text."""
        known = {r["doc_id"] for r in
                 conn.execute("SELECT doc_id FROM documents")}
        for key, meta in self._doc_metas(conn).items():
            if meta["status"] != "processing":
                continue
            if key in known:
                meta["status"] = "ready"
            else:
                meta["status"] = "failed"
                meta["error"] = "ingest interrupted by a service restart; " \
                                "re-upload the file"
            store.set_meta(conn, _DOC_KEY + key, json.dumps(meta))

    def shutdown(self, timeout_s: float = 10.0) -> None:
        for _ in self._threads:
            self._queue.put(None)
        deadline = time.monotonic() + timeout_s
        for th in self._threads:
            th.join(timeout=max(0.0, deadline - time.monotonic()))

    # -- registry ------------------------------------------------------------

    def _db_path(self, cid: str) -> str:
        return str(self._dir / f"{cid}.db")

    def _owned(self, api_key: str, cid: str) -> bool:
        entry = self._registry.get(cid)
        return entry is not None and entry["owner"] == _owner_id(api_key)

    def create(self, api_key: str, name: str) -> dict:
        owner = _owner_id(api_key)
        with self._lock:
            mine = sum(1 for e in self._registry.values()
                       if e["owner"] == owner)
            if mine >= self._cfg.max_collections_per_key:
                raise UploadRejected(
                    "collection_limit",
                    f"limit reached: {self._cfg.max_collections_per_key} "
                    "collections per API key")
            cid = f"col_{secrets.token_hex(4)}"
            conn = store.connect(self._db_path(cid))
            store.init_schema(conn)
            now = self._clock()
            store.set_meta(conn, "api:owner", owner)
            store.set_meta(conn, "api:name", name)
            store.set_meta(conn, "api:created_at", str(now))
            conn.close()
            self._registry[cid] = {"owner": owner, "name": name,
                                   "created_at": now}
        return self.get(api_key, cid)

    def list(self, api_key: str) -> list[dict]:
        owner = _owner_id(api_key)
        with self._lock:
            cids = [c for c, e in self._registry.items()
                    if e["owner"] == owner]
        return [self.get(api_key, c) for c in sorted(cids)]

    def get(self, api_key: str, cid: str) -> Optional[dict]:
        with self._lock:
            if not self._owned(api_key, cid):
                return None
            entry = dict(self._registry[cid])
        conn = store.connect(self._db_path(cid), read_only=True)
        metas = self._doc_metas(conn)
        conn.close()
        counts = {"ready": 0, "processing": 0, "failed": 0}
        for m in metas.values():
            counts[m["status"]] = counts.get(m["status"], 0) + 1
        if not metas:
            status = "empty"
        elif counts["processing"]:
            status = "ingesting"
        elif counts["ready"]:
            status = "ready"
        else:
            status = "failed"
        return {"id": cid, "object": "collection", "name": entry["name"],
                "created_at": iso_utc(entry["created_at"]),
                "document_count": len(metas),
                "documents_ready": counts["ready"],
                "documents_processing": counts["processing"],
                "documents_failed": counts["failed"], "status": status}

    def delete(self, api_key: str, cid: str) -> bool:
        with self._lock:
            if not self._owned(api_key, cid):
                return False
            del self._registry[cid]
        with self._index_lock:
            self._indexes.pop(cid, None)
        Path(self._db_path(cid)).unlink(missing_ok=True)
        return True

    # -- upload / documents ----------------------------------------------------

    def upload(self, api_key: str, cid: str,
               files: list[tuple[str, bytes]]) -> Optional[list[dict]]:
        if not self._owned(api_key, cid):
            return None
        if not files:
            raise UploadRejected("upload_rejected", "no files in the request")
        batch: list[tuple[str, str]] = []
        seen: set[str] = set()
        for filename, blob in files:
            if not filename.lower().endswith(_ALLOWED_SUFFIXES):
                raise UploadRejected(
                    "upload_rejected",
                    f"{filename!r}: unsupported extension (allowed: .md, "
                    ".markdown)")
            if len(blob) > self._cfg.max_upload_bytes:
                raise UploadRejected(
                    "upload_rejected",
                    f"{filename!r}: exceeds {self._cfg.max_upload_bytes} bytes")
            try:
                text = blob.decode("utf-8")
            except UnicodeDecodeError:
                raise UploadRejected(
                    "upload_rejected",
                    f"{filename!r}: not valid UTF-8 text") from None
            doc_id = _sanitize_doc_id(filename)
            if doc_id in seen:
                raise UploadRejected(
                    "upload_rejected", f"duplicate filename in batch: {doc_id!r}")
            seen.add(doc_id)
            batch.append((doc_id, text))

        conn = store.connect(self._db_path(cid))
        metas = self._doc_metas(conn)
        for doc_id, _ in batch:
            if doc_id in metas:
                conn.close()
                raise UploadRejected(
                    "upload_rejected",
                    f"{doc_id!r}: duplicate of an existing document")
        if len(metas) + len(batch) > self._cfg.max_docs_per_collection:
            conn.close()
            raise UploadRejected(
                "collection_limit",
                f"limit reached: {self._cfg.max_docs_per_collection} "
                "documents per collection")
        out = []
        now = self._clock()
        for doc_id, text in batch:
            meta = {"status": "processing", "bytes": len(text.encode()),
                    "uploaded_at": now}
            store.set_meta(conn, _DOC_KEY + doc_id, json.dumps(meta))
            self._pending[(cid, doc_id)] = text
            self._queue.put((cid, doc_id))
            out.append({"doc_id": doc_id, **meta,
                        "uploaded_at": iso_utc(now), "error": None})
        conn.close()
        return out

    def documents(self, api_key: str, cid: str) -> Optional[list[dict]]:
        if not self._owned(api_key, cid):
            return None
        return [self.document(api_key, cid, d) for d in
                sorted(self._doc_meta_map(cid))]

    def document(self, api_key: str, cid: str,
                 doc_id: str) -> Optional[dict]:
        if not self._owned(api_key, cid):
            return None
        meta = self._doc_meta_map(cid).get(doc_id)
        if meta is None:
            return None
        out = {"doc_id": doc_id, "status": meta["status"],
               "error": meta.get("error"), "bytes": meta.get("bytes"),
               "uploaded_at": iso_utc(meta.get("uploaded_at")), "title": None,
               "tldr": None, "token_count": None, "section_count": None}
        if meta["status"] == "ready":
            conn = store.connect(self._db_path(cid), read_only=True)
            row = conn.execute(
                "SELECT title, tldr, token_count FROM documents WHERE "
                "doc_id = ?", (doc_id,)).fetchone()
            if row is not None:
                nsec = conn.execute(
                    "SELECT COUNT(*) FROM sections WHERE doc_id = ?",
                    (doc_id,)).fetchone()[0]
                out.update({"title": row["title"], "tldr": row["tldr"],
                            "token_count": row["token_count"],
                            "section_count": nsec})
            conn.close()
        return out

    # -- answering -------------------------------------------------------------

    def bundle(self, api_key: str, cid: str
               ) -> Optional[tuple[str, SearchIndex, dict[str, str], int]]:
        """(db_path, index, titles, ready_count) for the answer path."""
        if not self._owned(api_key, cid):
            return None
        db_path = self._db_path(cid)
        with self._index_lock:
            index = self._indexes.get(cid)
            if index is None:
                index = SearchIndex(Reader(db_path))
                self._indexes[cid] = index
        conn = store.connect(db_path, read_only=True)
        titles = {r["doc_id"]: r["title"] for r in
                  conn.execute("SELECT doc_id, title FROM documents")}
        ready = sum(1 for m in self._doc_metas(conn).values()
                    if m["status"] == "ready")
        conn.close()
        return db_path, index, titles, ready

    # -- ingest ------------------------------------------------------------------

    def _ingest_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            cid, doc_id = item
            text = self._pending.pop((cid, doc_id), None)
            with self._lock:
                alive = cid in self._registry
            if not alive or text is None:
                self._queue.task_done()
                continue
            try:
                if self._enricher is None:
                    self._enricher = self._enricher_factory()
                rec = process_one(text, doc_id, self._enricher)
                conn = store.connect(self._db_path(cid))
                store.write_document(conn, rec)
                self._update_doc_meta(conn, doc_id, status="ready", error=None)
                conn.close()
                with self._index_lock:
                    self._indexes.pop(cid, None)  # rebuilt on next bundle()
                logger.info("ingested %s into %s (%d sections)", doc_id, cid,
                            len(rec.sections))
            except Exception as exc:  # noqa: BLE001 - one doc must not kill ingest
                logger.error("ingest %s/%s failed: %s", cid, doc_id, exc,
                             exc_info=True)
                try:
                    conn = store.connect(self._db_path(cid))
                    self._update_doc_meta(conn, doc_id, status="failed",
                                          error=f"{type(exc).__name__}: {exc}")
                    conn.close()
                except Exception:  # noqa: BLE001
                    logger.error("could not record failure for %s/%s", cid,
                                 doc_id)
            finally:
                self._queue.task_done()

    # -- meta helpers --------------------------------------------------------------

    @staticmethod
    def _doc_metas(conn) -> dict[str, dict]:
        rows = conn.execute("SELECT key, value FROM meta WHERE key LIKE ?",
                            (_DOC_KEY + "%",))
        return {r["key"][len(_DOC_KEY):]: json.loads(r["value"])
                for r in rows}

    def _doc_meta_map(self, cid: str) -> dict[str, dict]:
        conn = store.connect(self._db_path(cid), read_only=True)
        metas = self._doc_metas(conn)
        conn.close()
        return metas

    def _update_doc_meta(self, conn, doc_id: str, **changes) -> None:
        raw = store.get_meta(conn, _DOC_KEY + doc_id)
        meta = json.loads(raw) if raw else {}
        meta.update(changes)
        store.set_meta(conn, _DOC_KEY + doc_id, json.dumps(meta))

    def _set_doc_meta(self, cid: str, doc_id: str, meta: dict) -> None:
        conn = store.connect(self._db_path(cid))
        store.set_meta(conn, _DOC_KEY + doc_id, json.dumps(meta))
        conn.close()
