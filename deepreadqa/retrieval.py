"""BM25 retrieval over enriched doc summaries + section-content chunks.

Section content is sub-split into overlapping character chunks before indexing.
This is deliberate: heading-less / textbook PDF dumps land in the store as one
giant single section (e.g. the 128k-token ALE/Benson volume that is the gold doc
for ~55% of the eval). A whole-section BM25 unit for such a doc is crushed by
length normalization, so rare-term queries (e.g. "Jaumann") never surface it.
Fixed-size chunking — the same recall mechanism the agenticRAG baseline uses —
keeps a buried passage's match local and high-scoring, independent of whether
heading-based structure recovery succeeded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from deepread_sdk import Reader

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]")

__all__ = ["tokenize_mixed", "SearchHit", "SearchIndex"]


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize text with regex latin/digit + jieba for CJK characters."""
    low = text.lower()
    tokens = _TOKEN_RE.findall(low)
    if _CJK_RE.search(text):
        tokens.extend(t for t in jieba.cut(text) if t.strip())
    return tokens


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character chunks (empties dropped)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step)]


@dataclass(frozen=True)
class SearchHit:
    """A retrieval result: doc-level score, best-section hint, matched snippet."""

    doc_id: str
    title: str
    tldr: str
    score: float
    section_name: str | None
    section_idx: int | None
    snippet: str = ""


@dataclass(frozen=True)
class _Unit:
    """Internal BM25 unit: a doc-summary (chunk="") or one section-content chunk."""

    doc_id: str
    section_name: str | None
    section_idx: int | None
    chunk: str


class SearchIndex:
    """BM25 index over per-doc summary units + per-section-content chunk units."""

    def __init__(self, reader: Reader, *, chunk_chars: int = 1200,
                 overlap: int = 200) -> None:
        self._units: list[_Unit] = []
        self._meta: dict[str, tuple[str, str]] = {}  # doc_id -> (title, tldr)
        corpus: list[list[str]] = []
        for d in reader.list_docs():
            self._meta[d["doc_id"]] = (d["title"], d["tldr"])
            summary = " ".join([
                d["title"],
                d["tldr"],
                " ".join(d["keywords"]),
                d.get("abstract") or "",
            ])
            corpus.append(tokenize_mixed(summary))
            self._units.append(_Unit(d["doc_id"], None, None, ""))
            for s in d["sections"]:
                for ch in _chunk(s["content"], chunk_chars, overlap):
                    # index section name + tldr alongside each chunk so structural
                    # cues help ranking, while the chunk keeps the match local
                    corpus.append(tokenize_mixed(f"{s['name']} {s['tldr']} {ch}"))
                    self._units.append(_Unit(d["doc_id"], s["name"], s["idx"], ch))
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, *, top_k: int = 8) -> list[SearchHit]:
        """BM25 search; aggregate chunk/summary scores to doc level (max).

        The best-matching content chunk supplies the section hint and a snippet.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_mixed(query))
        best_doc: dict[str, float] = {}
        # best content chunk per doc: (section_name, section_idx, chunk, score)
        best_chunk: dict[str, tuple[str | None, int | None, str, float]] = {}
        for i, u in enumerate(self._units):
            sc = float(scores[i])
            if sc > best_doc.get(u.doc_id, -1.0):
                best_doc[u.doc_id] = sc
            if u.section_idx is not None:  # a content chunk, not the summary unit
                cur = best_chunk.get(u.doc_id)
                if cur is None or sc > cur[3]:
                    best_chunk[u.doc_id] = (u.section_name, u.section_idx, u.chunk, sc)
        ranked = [d for d in sorted(best_doc, key=lambda d: best_doc[d], reverse=True)
                  if best_doc[d] > 0.0][:top_k]
        hits: list[SearchHit] = []
        for doc_id in ranked:
            title, tldr = self._meta[doc_id]
            c = best_chunk.get(doc_id)
            snippet = " ".join(c[2].split())[:400] if (c and c[2]) else ""
            hits.append(SearchHit(
                doc_id=doc_id,
                title=title,
                tldr=tldr,
                score=best_doc[doc_id],
                section_name=(c[0] if c else None),
                section_idx=(c[1] if c else None),
                snippet=snippet,
            ))
        return hits

    def search_many(self, queries: list[str], *, top_k: int = 8) -> list[SearchHit]:
        """Search with multiple queries, dedup by doc_id keeping the highest score."""
        merged: dict[str, SearchHit] = {}
        for q in queries:
            for h in self.search(q, top_k=top_k):
                cur = merged.get(h.doc_id)
                if cur is None or h.score > cur.score:
                    merged[h.doc_id] = h
        return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:top_k]
