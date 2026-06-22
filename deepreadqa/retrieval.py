"""BM25 retrieval over enriched doc-level and section-level units."""
from __future__ import annotations

import re
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

from deepread_sdk import Reader

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize_mixed(text: str) -> list[str]:
    """Tokenize text with regex latin/digit + jieba for CJK characters.

    Args:
        text: Input text, may contain Latin, digits, and/or CJK characters.

    Returns:
        List of tokens (lowercased).
    """
    low = text.lower()
    tokens = _TOKEN_RE.findall(low)
    if _CJK_RE.search(text):
        tokens.extend(t for t in jieba.cut(text) if t.strip())
    return tokens


@dataclass(frozen=True)
class SearchHit:
    """A single retrieval result with doc-level score and best-section hint."""

    doc_id: str
    title: str
    tldr: str
    score: float
    section_name: str | None
    section_idx: int | None


@dataclass(frozen=True)
class _Unit:
    """Internal BM25 indexing unit: either a doc-summary or a section."""

    doc_id: str
    section_name: str | None
    section_idx: int | None


class SearchIndex:
    """BM25 index where each unit is a doc-summary or a section."""

    def __init__(self, reader: Reader) -> None:
        self._reader = reader
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
            self._units.append(_Unit(d["doc_id"], None, None))
            for s in d["sections"]:
                text = " ".join([s["name"], s["tldr"], s["content"]])
                corpus.append(tokenize_mixed(text))
                self._units.append(_Unit(d["doc_id"], s["name"], s["idx"]))
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, *, top_k: int = 8) -> list[SearchHit]:
        """Search for relevant documents using BM25.

        Aggregates doc-summary and section unit scores to doc level (max score).
        The best-matching section is surfaced as a hint in the SearchHit.

        Args:
            query: Search query string (may be bilingual).
            top_k: Maximum number of results to return.

        Returns:
            List of SearchHit sorted by descending score.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_mixed(query))
        best_doc: dict[str, float] = {}
        # best_sec stores (section_name, section_idx, score) per doc_id
        best_sec: dict[str, tuple[str | None, int | None, float]] = {}
        for i, u in enumerate(self._units):
            sc = float(scores[i])
            if sc > best_doc.get(u.doc_id, -1.0):
                best_doc[u.doc_id] = sc
            # track best *section* unit separately (ignore doc-summary units)
            if u.section_idx is not None:
                cur = best_sec.get(u.doc_id)
                if cur is None or sc > cur[2]:
                    best_sec[u.doc_id] = (u.section_name, u.section_idx, sc)
        ranked = sorted(best_doc, key=lambda d: best_doc[d], reverse=True)
        ranked = [d for d in ranked if best_doc[d] > 0.0][:top_k]
        hits: list[SearchHit] = []
        for doc_id in ranked:
            title, tldr = self._meta[doc_id]
            sec = best_sec.get(doc_id)
            hits.append(SearchHit(
                doc_id=doc_id,
                title=title,
                tldr=tldr,
                score=best_doc[doc_id],
                section_name=(sec[0] if sec else None),
                section_idx=(sec[1] if sec else None),
            ))
        return hits

    def search_many(self, queries: list[str], *, top_k: int = 8) -> list[SearchHit]:
        """Search with multiple queries, deduplicating by doc_id (keep highest score).

        Args:
            queries: List of query strings.
            top_k: Maximum number of results to return.

        Returns:
            Deduplicated list of SearchHit sorted by descending score.
        """
        merged: dict[str, SearchHit] = {}
        for q in queries:
            for h in self.search(q, top_k=top_k):
                cur = merged.get(h.doc_id)
                if cur is None or h.score > cur.score:
                    merged[h.doc_id] = h
        return sorted(merged.values(), key=lambda h: h.score, reverse=True)[:top_k]
