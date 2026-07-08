"""Blank-line paragraph segmentation, fence-aware.

Defines the single 1-based paragraph coordinate system shared by search's
``~¶`` hit anchors (retrieval) and read_section's ``start_para``/``end_para``
paging (tools): paragraphs are blocks separated by blank lines, but a blank
line inside a fenced code block (```) never splits.
"""
from __future__ import annotations

import re

__all__ = ["paragraph_spans", "split_paragraphs"]

_FENCE_RE = re.compile(r"^\s*```")


def paragraph_spans(text: str) -> list[tuple[int, str]]:
    """Return (start_offset, paragraph) pairs for *text*.

    Offsets index into *text* itself, so callers can map any character
    position (e.g. a BM25 chunk start) to its 1-based paragraph number.
    """
    spans: list[tuple[int, str]] = []
    in_fence = False
    start: int | None = None
    end = 0  # exclusive end of the current paragraph's last content line
    pos = 0
    for line in text.splitlines(keepends=True):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        if line.strip() or in_fence:
            if start is None:
                start = pos
            end = pos + len(line)
        elif start is not None:
            spans.append((start, text[start:end].rstrip("\r\n")))
            start = None
        pos += len(line)
    if start is not None:
        spans.append((start, text[start:end].rstrip("\r\n")))
    return spans


def split_paragraphs(text: str) -> list[str]:
    """Return the paragraphs of *text* (see :func:`paragraph_spans`)."""
    return [p for _, p in paragraph_spans(text)]
