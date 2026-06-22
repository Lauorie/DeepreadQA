import json

from deepread_sdk import Reader
from deepreadqa.config import Config, Endpoint
from deepreadqa.retrieval import SearchIndex
from deepreadqa.tools import TOOL_SCHEMAS, ToolBox


def _cfg() -> Config:
    return Config(endpoint=Endpoint("aiberm", "x", "x", "m", True))


def _box(db) -> ToolBox:
    reader = Reader(db)
    return ToolBox(_cfg(), reader, SearchIndex(reader))


def test_schemas_cover_all_tools():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"search", "head", "read_section", "intro", "preview",
                     "grep", "read_raw", "summarize"}


def test_search_returns_cards(populated_store):
    box = _box(populated_store)
    out = box.execute("search", {"queries": ["ALE coupling scheme"]})
    assert "en_paper.md" in out
    assert "section" in out.lower()
    assert "en_paper.md" in box.seen_docs


def test_head_lists_sections(populated_store):
    box = _box(populated_store)
    out = box.execute("head", {"doc_id": "en_paper.md"})
    assert "1. Introduction" in out and "2. Method" in out
    assert "token" in out.lower()


def test_read_section_by_name(populated_store):
    box = _box(populated_store)
    out = box.execute("read_section", {"doc_id": "en_paper.md", "section": "2. Method"})
    assert "ALE coupling" in out


def test_grep_finds_term_with_context(populated_store):
    box = _box(populated_store)
    out = box.execute("grep", {"doc_id": "en_paper.md", "patterns": ["ALE"]})
    assert "ALE" in out


def test_read_raw_capped(populated_store):
    box = _box(populated_store)
    out = box.execute("read_raw", {"doc_id": "en_paper.md"})
    assert "Hydroplaning" in out


def test_unknown_doc_is_graceful(populated_store):
    box = _box(populated_store)
    out = box.execute("head", {"doc_id": "missing.md"})
    assert "not found" in out.lower() or "unknown" in out.lower()
