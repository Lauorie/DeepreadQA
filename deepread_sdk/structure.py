"""Deterministic structure recovery: split markdown into sections by headings."""
from __future__ import annotations

import re

from .schema import RawSection, StructuredDoc

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_CJK_RE = re.compile(r"[一-鿿]")
_ABSTRACT_RE = re.compile(r"^(abstract|摘\s*要)\s*[.:：]?\s*$", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


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
    headings inside fenced code blocks."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    in_fence = False
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if _FENCE_RE.match(stripped):
            in_fence = not in_fence
            pos += len(line)
            continue
        if not in_fence:
            m = _HEADING_RE.match(stripped)
            if m and m.group(2).strip():
                out.append((pos, len(m.group(1)), m.group(2).strip()))
        pos += len(line)
    return out


def _line_end_after(text: str, pos: int) -> int:
    nl = text.find("\n", pos)
    return len(text) if nl < 0 else nl + 1


def recover_structure(text: str, *, fallback_title: str) -> StructuredDoc:
    """Split *text* into a title, a front-matter header block, and sections.

    Rules:
    - No headings -> single 'Full Document' section, title = fallback_title.
    - First heading = document title.
    - Sectioning level = the minimum heading level among the *remaining* headings.
    - A new section begins at each heading at the sectioning level; deeper
      subsections are kept inside their parent section's content.
    - Header block = text between the title line and the first section heading.
    """
    headings = _find_headings(text)
    if not headings:
        return StructuredDoc(
            title=fallback_title,
            header="",
            sections=[RawSection(name="Full Document", idx=0,
                                 content=text.strip(), start_pos=0, end_pos=len(text))],
        )

    title = headings[0][2]
    rest = headings[1:]
    title_line_end = _line_end_after(text, headings[0][0])

    if not rest:
        return StructuredDoc(
            title=title, header="",
            sections=[RawSection(name=title, idx=0,
                                 content=text[title_line_end:].strip(),
                                 start_pos=title_line_end, end_pos=len(text))],
        )

    sec_level = min(lvl for _, lvl, _ in rest)
    sec_heads = [(pos, name) for (pos, lvl, name) in rest if lvl == sec_level]

    header = text[title_line_end:sec_heads[0][0]].strip()

    sections: list[RawSection] = []
    for i, (pos, name) in enumerate(sec_heads):
        end = sec_heads[i + 1][0] if i + 1 < len(sec_heads) else len(text)
        content_start = _line_end_after(text, pos)
        sections.append(RawSection(name=name, idx=i,
                                   content=text[content_start:end].strip(),
                                   start_pos=pos, end_pos=end))
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
