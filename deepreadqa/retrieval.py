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

import bisect
import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from deepread_sdk import Reader

from .paragraphs import paragraph_spans

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


def _chunk(text: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Split text into overlapping character chunks with start offsets.

    Offsets index into the original *text* (empties dropped); the chunk
    strings themselves are byte-identical to the offset-less chunking, so
    the BM25 corpus and scores are unchanged.
    """
    stripped = text.strip()
    if not stripped:
        return []
    lead = len(text) - len(text.lstrip())
    if len(stripped) <= size:
        return [(lead, stripped)]
    step = max(1, size - overlap)
    return [(lead + i, stripped[i:i + size])
            for i in range(0, len(stripped), step)]


@dataclass(frozen=True)
class SearchHit:
    """A retrieval result: doc-level score, best-section hint (with a 1-based
    ~paragraph anchor for read_section paging), matched snippet."""

    doc_id: str
    title: str
    tldr: str
    score: float
    section_name: str | None
    section_idx: int | None
    snippet: str = ""
    para_idx: int | None = None


@dataclass(frozen=True)
class _Unit:
    """Internal BM25 unit: a doc-summary (chunk="") or one section-content chunk."""

    doc_id: str
    section_name: str | None
    section_idx: int | None
    chunk: str
    chunk_start: int = 0


class SearchIndex:
    """BM25 index over per-doc summary units + per-section-content chunk units."""

    def __init__(self, reader: Reader, *, chunk_chars: int = 1200,
                 overlap: int = 200) -> None:
        self._units: list[_Unit] = []
        self._meta: dict[str, tuple[str, str]] = {}  # doc_id -> (title, tldr)
        # (doc_id, section_idx) -> paragraph start offsets, for ~¶ anchors
        self._para_starts: dict[tuple[str, int], list[int]] = {}
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
                self._para_starts[(d["doc_id"], s["idx"])] = [
                    st for st, _ in paragraph_spans(s["content"])]
                for start, ch in _chunk(s["content"], chunk_chars, overlap):
                    # index the raw chunk text only. Prepending section name/tldr
                    # to every chunk injects the same metadata into hundreds of
                    # chunks of a giant single-section doc, diluting BM25; the
                    # doc-summary unit already carries title/tldr/keywords.
                    corpus.append(tokenize_mixed(ch))
                    self._units.append(
                        _Unit(d["doc_id"], s["name"], s["idx"], ch, start))
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, *, top_k: int = 8) -> list[SearchHit]:
        """BM25 search; aggregate chunk/summary scores to doc level (max).

        The best-matching content chunk supplies the section hint and a snippet.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_mixed(query))
        best_doc: dict[str, float] = {}
        # best content chunk per doc, as the winning _Unit
        best_chunk: dict[str, tuple[_Unit, float]] = {}
        for i, u in enumerate(self._units):
            sc = float(scores[i])
            if sc > best_doc.get(u.doc_id, -1.0):
                best_doc[u.doc_id] = sc
            if u.section_idx is not None:  # a content chunk, not the summary unit
                cur = best_chunk.get(u.doc_id)
                if cur is None or sc > cur[1]:
                    best_chunk[u.doc_id] = (u, sc)
        ranked = [d for d in sorted(best_doc, key=lambda d: best_doc[d], reverse=True)
                  if best_doc[d] > 0.0][:top_k]
        hits: list[SearchHit] = []
        for doc_id in ranked:
            title, tldr = self._meta[doc_id]
            c = best_chunk.get(doc_id)
            u = c[0] if c else None
            snippet = " ".join(u.chunk.split())[:400] if (u and u.chunk) else ""
            hits.append(SearchHit(
                doc_id=doc_id,
                title=title,
                tldr=tldr,
                score=best_doc[doc_id],
                section_name=(u.section_name if u else None),
                section_idx=(u.section_idx if u else None),
                snippet=snippet,
                para_idx=(self._para_at(doc_id, u.section_idx, u.chunk_start)
                          if u else None),
            ))
        return hits

    def _para_at(self, doc_id: str, section_idx: int | None,
                 offset: int) -> int | None:
        """1-based paragraph number containing *offset* in the given section."""
        if section_idx is None:
            return None
        starts = self._para_starts.get((doc_id, section_idx))
        if not starts:
            return None
        return max(1, bisect.bisect_right(starts, offset))

    def search_many(self, queries: list[str], *, top_k: int = 8) -> list[SearchHit]:
        """Search with multiple queries, dedup by doc_id keeping the highest score."""
        merged: dict[str, SearchHit] = {}
        for q in queries:
            for h in self.search(q, top_k=top_k):
                cur = merged.get(h.doc_id)
                if cur is None or h.score > cur.score:
                    merged[h.doc_id] = h
        return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:top_k]
