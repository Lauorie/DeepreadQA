"""Reader: progressive-access views backed by the SQLite store."""
from __future__ import annotations

import re
from pathlib import Path

from . import store
from .schema import DocRecord

_INTRO_RE = re.compile(r"(introduction|引\s*言|绪\s*论)", re.IGNORECASE)


class Reader:
    def __init__(self, db_path: str | Path, *, preview_chars: int = 10000) -> None:
        self._conn = store.connect(db_path, read_only=True)
        self._preview_chars = preview_chars

    def _get(self, doc_id: str) -> DocRecord:
        rec = store.get_document(self._conn, doc_id)
        if rec is None:
            raise KeyError(f"unknown doc_id: {doc_id!r}")
        return rec

    def brief(self, doc_id: str) -> dict:
        r = self._get(doc_id)
        return {"title": r.title, "tldr": r.tldr, "keywords": r.keywords}

    def head(self, doc_id: str) -> dict:
        r = self._get(doc_id)
        return {
            "doc_id": r.doc_id, "title": r.title, "language": r.language,
            "abstract": r.abstract, "header": r.header, "tldr": r.tldr,
            "keywords": r.keywords, "token_count": r.token_count,
            "sections": [{"name": s.name, "idx": s.idx, "tldr": s.tldr,
                          "token_count": s.token_count} for s in r.sections],
        }

    def intro(self, doc_id: str) -> str:
        r = self._get(doc_id)
        if not r.sections:
            return ""
        for s in r.sections:
            if _INTRO_RE.search(s.name):
                return s.content
        return r.sections[0].content

    def preview(self, doc_id: str) -> dict:
        r = self._get(doc_id)
        return {"doc_id": r.doc_id, "preview": r.preview,
                "is_truncated": r.preview_is_truncated,
                "total_characters": r.total_characters,
                "preview_characters": len(r.preview)}

    def section(self, doc_id: str, name: str | None = None,
                idx: int | None = None) -> dict:
        r = self._get(doc_id)
        target = None
        if idx is not None:
            target = next((s for s in r.sections if s.idx == idx), None)
        if target is None and name is not None:
            low = name.strip().lower()
            target = next((s for s in r.sections if s.name.strip().lower() == low), None)
            if target is None:
                target = next((s for s in r.sections if low in s.name.lower()), None)
        if target is None:
            raise KeyError(f"section not found in {doc_id!r}: name={name!r} idx={idx!r}")
        return {"doc_id": r.doc_id, "name": target.name, "idx": target.idx,
                "tldr": target.tldr, "token_count": target.token_count,
                "content": target.content}

    def raw(self, doc_id: str) -> str:
        return self._get(doc_id).raw_md

    def json(self, doc_id: str) -> dict:
        r = self._get(doc_id)
        data = {s.name: {"content": s.content, "start_pos": s.start_pos,
                         "end_pos": s.end_pos} for s in r.sections}
        return {"doc_id": r.doc_id, "data": data}

    def list_docs(self) -> list[dict]:
        out = []
        for doc_id in store.list_doc_ids(self._conn):
            r = self._get(doc_id)
            out.append({"doc_id": r.doc_id, "title": r.title, "tldr": r.tldr,
                        "keywords": r.keywords, "abstract": r.abstract,
                        "language": r.language,
                        "sections": [{"name": s.name, "idx": s.idx, "tldr": s.tldr,
                                      "content": s.content} for s in r.sections]})
        return out
