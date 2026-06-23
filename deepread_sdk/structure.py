"""Deterministic structure recovery: split markdown into sections by headings.

Primary path: ATX (``#``) markdown headings, sectioned at the shallowest level but
**recursively split** when a section is oversized and has deeper sub-headings — so a
textbook chapter (e.g. a 23k-token "# Chapter 1" containing ## 1.1 / ### 1.4.5) is
broken down into addressable subsections instead of one giant chapter blob.

Fallback for heading-less documents (PDF dumps with no ``#`` headings): detect
plain-text numbered headings like ``1.4.7 Mixture theories``. Without structure,
such a document collapses into one giant section, breaking section-level reading
and getting buried by BM25 length normalization in retrieval.
"""
from __future__ import annotations

import re

from .schema import RawSection, StructuredDoc
from .tokens import count_tokens

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_CJK_RE = re.compile(r"[一-鿿]")
_ABSTRACT_RE = re.compile(r"^(abstract|摘\s*要)\s*[.:：]?\s*$", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_NUMBERED_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^\n]{2,70})$")
_BAD_TITLE_START = ("the ", "a ", "this ", "we ", "in ", "for ", "it ", "these ",
                    "where ", "river", "street")
_MIN_NUMBERED = 5          # heading-less doc treated as numbered-structured above this
_MAX_SECTION_TOKENS = 6000  # split an ATX section deeper when it exceeds this


def detect_language(text: str) -> str:
    """Heuristic: 'zh' if CJK characters exceed 30% of alpha chars, else 'en'."""
    cjk = len(_CJK_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = cjk + latin
    if total == 0:
        return "en"
    return "zh" if cjk / total > 0.3 else "en"


def _find_headings(text: str) -> list[tuple[int, int, str]]:
    """Return (char_pos, level, name) for every ATX heading line, skipping
    headings inside fenced code blocks (``` closes only ```, ~~~ only ~~~)."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        fm = _FENCE_RE.match(stripped)
        if fm:
            marker = fm.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            pos += len(line)
            continue
        if fence is None:
            m = _HEADING_RE.match(stripped)
            if m and m.group(2).strip():
                out.append((pos, len(m.group(1)), m.group(2).strip()))
        pos += len(line)
    return out


def _find_numbered_headings(text: str) -> list[tuple[int, str]]:
    """Detect plain-text numbered section headings ('1.4.7 Mixture theories') for
    documents with no markdown headings. Returns (char_pos, 'number title')."""
    out: list[tuple[int, str]] = []
    pos = 0
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        fm = _FENCE_RE.match(stripped)
        if fm:
            marker = fm.group(1)[0]
            fence = marker if fence is None else (None if fence == marker else fence)
            pos += len(line)
            continue
        if fence is None:
            m = _NUMBERED_RE.match(stripped)
            if m:
                num, title = m.group(1), m.group(2).strip()
                if (int(num.split(".")[0]) <= 50
                        and len(title.split()) <= 10
                        and not title.lower().startswith(_BAD_TITLE_START)
                        and not (title.isupper() and len(title) > 40)):
                    out.append((pos, f"{num} {title}"))
        pos += len(line)
    return out


def _slice_offsets(text: str, lo: int, hi: int) -> tuple[str, int, int]:
    """Return (stripped_content, start, end) such that text[start:end] == stripped."""
    raw = text[lo:hi]
    stripped = raw.strip()
    lead = len(raw) - len(raw.lstrip())
    start = lo + lead
    return stripped, start, start + len(stripped)


def _line_end_after(text: str, pos: int) -> int:
    nl = text.find("\n", pos)
    return len(text) if nl < 0 else nl + 1


def _build_sections(text: str, boundaries: list[tuple[int, str]]) -> list[RawSection]:
    """Build flat sections from ordered (char_pos, name) boundaries (numbered path)."""
    sections: list[RawSection] = []
    for i, (pos, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        content_start = _line_end_after(text, pos)
        content, s_start, s_end = _slice_offsets(text, content_start, end)
        sections.append(RawSection(name=name, idx=i, content=content,
                                   start_pos=s_start, end_pos=s_end))
    return sections


def _sectionize(text: str, headings: list[tuple[int, int, str]],
                span_start: int, span_end: int,
                parent_name: str | None = None) -> list[tuple[str, str, int, int]]:
    """Recursively split [span_start, span_end) by *headings* (each a
    (pos, level, name) inside the span). Section at the shallowest level present;
    if a resulting section exceeds _MAX_SECTION_TOKENS and contains deeper
    sub-headings, recurse into it. Returns (name, content, start, end) tuples."""
    if not headings:
        content, s, e = _slice_offsets(text, span_start, span_end)
        return [(parent_name or "Section", content, s, e)] if content else []

    min_lvl = min(lvl for _, lvl, _ in headings)
    tops = [(p, n) for (p, lvl, n) in headings if lvl == min_lvl]
    out: list[tuple[str, str, int, int]] = []

    # content before the first top-level heading (a chapter intro, when recursing)
    if parent_name is not None and tops[0][0] > span_start:
        lead, ls, le = _slice_offsets(text, span_start, tops[0][0])
        if lead:
            out.append((parent_name, lead, ls, le))

    for j, (pos, name) in enumerate(tops):
        seg_end = tops[j + 1][0] if j + 1 < len(tops) else span_end
        content_start = _line_end_after(text, pos)
        content, s, e = _slice_offsets(text, content_start, seg_end)
        subs = [h for h in headings if content_start <= h[0] < seg_end and h[1] > min_lvl]
        if subs and count_tokens(content) > _MAX_SECTION_TOKENS:
            out.extend(_sectionize(text, subs, content_start, seg_end, parent_name=name))
        elif content:
            out.append((name, content, s, e))
    return out


def recover_structure(text: str, *, fallback_title: str) -> StructuredDoc:
    """Split *text* into a title, a front-matter header block, and sections.

    - ATX headings present: first heading = title; section at the shallowest
      remaining level, recursively splitting oversized sections by their deeper
      sub-headings.
    - No ATX headings but >=5 numbered headings: use those as flat sections.
    - Otherwise: a single 'Full Document' section, title = fallback_title.
    """
    headings = _find_headings(text)
    if not headings:
        numbered = _find_numbered_headings(text)
        if len(numbered) >= _MIN_NUMBERED:
            header = text[:numbered[0][0]].strip()
            return StructuredDoc(title=fallback_title, header=header,
                                 sections=_build_sections(text, numbered))
        content, s_start, s_end = _slice_offsets(text, 0, len(text))
        return StructuredDoc(
            title=fallback_title, header="",
            sections=[RawSection(name="Full Document", idx=0, content=content,
                                 start_pos=s_start, end_pos=s_end)],
        )

    title = headings[0][2]
    rest = headings[1:]
    title_line_end = _line_end_after(text, headings[0][0])

    if not rest:
        content, s_start, s_end = _slice_offsets(text, title_line_end, len(text))
        return StructuredDoc(
            title=title, header="",
            sections=[RawSection(name=title, idx=0, content=content,
                                 start_pos=s_start, end_pos=s_end)],
        )

    first_pos = rest[0][0]
    header = text[title_line_end:first_pos].strip()
    raw_secs = _sectionize(text, rest, first_pos, len(text))
    sections = [RawSection(name=n, idx=i, content=c, start_pos=s, end_pos=e)
                for i, (n, c, s, e) in enumerate(raw_secs)]
    return StructuredDoc(title=title, header=header, sections=sections)


def extract_abstract(doc: StructuredDoc) -> str | None:
    """Return abstract content from a section named Abstract/摘要 (tolerating a
    trailing . or :), else from an inline abstract line in the header, else None."""
    for s in doc.sections:
        if _ABSTRACT_RE.match(s.name.strip()):
            return s.content
    for line in doc.header.splitlines():
        m = re.match(r"^(abstract|摘\s*要)\s*[:：.]?\s*(.+)$", line.strip(), re.IGNORECASE)
        if m and m.group(2).strip():
            return m.group(2).strip()
    return None
