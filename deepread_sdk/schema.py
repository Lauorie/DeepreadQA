"""Immutable data models for the DeepRead store."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RawSection:
    name: str
    idx: int
    content: str
    start_pos: int
    end_pos: int


@dataclass(frozen=True)
class StructuredDoc:
    title: str
    header: str
    sections: list[RawSection]


@dataclass(frozen=True)
class SectionRecord:
    idx: int
    name: str
    tldr: str
    token_count: int
    start_pos: int
    end_pos: int
    content: str


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    title: str
    language: str
    abstract: str | None
    header: str
    tldr: str
    keywords: list[str]
    token_count: int
    total_characters: int
    preview: str
    preview_is_truncated: bool
    raw_md: str
    content_hash: str
    sections: list[SectionRecord] = field(default_factory=list)
