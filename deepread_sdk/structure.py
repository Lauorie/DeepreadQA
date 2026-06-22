"""Deterministic structure recovery: split markdown into sections by headings.

Primary path: ATX (``#``) markdown headings. Fallback for heading-less documents
(typically PDF-to-markdown dumps of textbooks/papers): detect plain-text numbered
section headings such as ``1.4.7 Mixture theories``. Without this, such a document
collapses into one giant section, which (a) breaks section-level reading and
(b) is buried by BM25 length normalization in retrieval.
"""
from __future__ import annotations

import re

from .schema import RawSection, StructuredDoc

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_CJK_RE = re.compile(r"[一-鿿]")
_ABSTRACT_RE = re.compile(r"^(abstract|摘\s*要)\s*[.:：]?\s*$", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_NUMBERED_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^\n]{2,70})$")
_BAD_TITLE_START = ("the ", "a ", "this ", "we ", "in ", "for ", "it ", "these ",
                    "where ", "river", "street")
_MIN_NUMBERED = 5  # only treat a heading-less doc as numbered-structured above this


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
    headings inside fenced code blocks. A fence opened by ``` closes only on
    ```, and ~~~ only on ~~~ (a mismatched marker inside a fence is content)."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    fence: str | None = None  # the marker char of the open fence, or None
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
    """Build sections from ordered heading boundaries [(char_pos, name), ...].

    Each section's content excludes its own heading line and runs to the next
    boundary; offsets round-trip (text[start_pos:end_pos] == content)."""
    sections: list[RawSection] = []
    for i, (pos, name) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        content_start = _line_end_after(text, pos)
        content, s_start, s_end = _slice_offsets(text, content_start, end)
        sections.append(RawSection(name=name, idx=i, content=content,
                                   start_pos=s_start, end_pos=s_end))
    return sections


def recover_structure(text: str, *, fallback_title: str) -> StructuredDoc:
    """Split *text* into a title, a front-matter header block, and sections.

    Rules:
    - ATX headings present: first heading = title; sectioning level = the minimum
      level among the remaining headings; deeper subsections stay nested.
    - No ATX headings but >=5 plain-text numbered headings: use those as sections
      (title = fallback_title; header = text before the first numbered heading).
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

    sec_level = min(lvl for _, lvl, _ in rest)
    sec_heads = [(pos, name) for (pos, lvl, name) in rest if lvl == sec_level]
    header = text[title_line_end:sec_heads[0][0]].strip()
    return StructuredDoc(title=title, header=header,
                         sections=_build_sections(text, sec_heads))


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
