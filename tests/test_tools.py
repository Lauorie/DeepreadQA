import re

from deepread_sdk import Reader
from deepread_sdk.tokens import count_tokens
from deepreadqa.config import Config, Endpoint
from deepreadqa.retrieval import SearchIndex
from deepreadqa.tools import TOOL_SCHEMAS, ToolBox


def _cfg() -> Config:
    return Config(endpoint=Endpoint("aiberm", "x", "x", "m", True))


def _box(db) -> ToolBox:
    reader = Reader(db)
    return ToolBox(_cfg(), reader, SearchIndex(reader))


_PARAS = [f"P{i} alpha beta gamma delta epsilon zeta eta theta"
          for i in range(1, 7)]


class _ParaReader:
    """Stub Reader with one six-paragraph section, for paging tests."""

    def __init__(self, paras):
        self._content = "\n\n".join(paras)

    def list_docs(self):
        return [{"doc_id": "p.md", "title": "Paged Doc", "tldr": "", "keywords": [],
                 "abstract": "", "language": "en",
                 "sections": [{"name": "Body", "idx": 0, "tldr": "",
                               "content": self._content}]}]

    def section(self, doc_id, name=None, idx=None):
        return {"doc_id": doc_id, "name": "Body", "idx": 0, "tldr": "stub",
                "token_count": count_tokens(self._content),
                "content": self._content}


def _para_box(**cfg_overrides) -> ToolBox:
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True), **cfg_overrides)
    reader = _ParaReader(_PARAS)
    return ToolBox(cfg, reader, SearchIndex(reader))


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


def test_search_card_carries_idx_and_para_anchor(populated_store):
    # the best-section line must expose read_section coordinates:
    # `best section: [<idx>] <name> (~¶<para>)`
    box = _box(populated_store)
    out = box.execute("search", {"queries": ["ALE coupling scheme"]})
    assert re.search(r"best section: \[\d+\] 2\. Method \(~¶1\)", out), out


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
    # read_raw is default-disabled; enable it to test the handler itself
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 disabled_tools=())
    reader = Reader(populated_store)
    box = ToolBox(cfg, reader, SearchIndex(reader))
    out = box.execute("read_raw", {"doc_id": "en_paper.md"})
    assert "Hydroplaning" in out


def test_unknown_doc_is_graceful(populated_store):
    box = _box(populated_store)
    out = box.execute("head", {"doc_id": "missing.md"})
    assert "not found" in out.lower() or "unknown" in out.lower()


def test_read_raw_truncates_when_over_cap(populated_store):
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 raw_token_cap=1, disabled_tools=())
    reader = Reader(populated_store)
    box = ToolBox(cfg, reader, SearchIndex(reader))
    out = box.execute("read_raw", {"doc_id": "en_paper.md"})
    assert "truncated" in out.lower()


def test_grep_unknown_doc_graceful(populated_store):
    box = _box(populated_store)
    out = box.execute("grep", {"doc_id": "missing.md", "patterns": ["x"]})
    assert "not found" in out.lower() or "unknown" in out.lower()


def test_read_section_truncates_when_over_cap(populated_store):
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True), section_token_cap=1)
    reader = Reader(populated_store)
    box = ToolBox(cfg, reader, SearchIndex(reader))
    out = box.execute("read_section", {"doc_id": "en_paper.md", "section": "2. Method"})
    assert "truncated" in out.lower()


def test_read_section_defaults_to_first(populated_store):
    box = _box(populated_store)
    out = box.execute("read_section", {"doc_id": "en_paper.md"})
    assert "not found" not in out.lower()
    assert "SECTION en_paper.md" in out


def test_grep_missing_doc_not_in_seen(populated_store):
    box = _box(populated_store)
    box.execute("grep", {"doc_id": "missing.md", "patterns": ["x"]})
    assert "missing.md" not in box.seen_docs


def test_grep_includes_section_name(populated_store):
    box = _box(populated_store)
    out = box.execute("grep", {"doc_id": "en_paper.md", "patterns": ["ALE"]})
    assert "Method" in out


def test_disabled_tool_execution_rejected(populated_store):
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 disabled_tools=("intro",))
    reader = Reader(populated_store)
    box = ToolBox(cfg, reader, SearchIndex(reader))
    out = box.execute("intro", {"doc_id": "en_paper.md"})
    assert "unknown tool" in out
    assert "en_paper.md" not in box.seen_docs
    # non-disabled tools still work
    assert "SECTION" in box.execute("read_section", {"doc_id": "en_paper.md"})


def test_read_section_schema_has_paragraph_paging_params():
    rs = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "read_section")
    params = rs["function"]["parameters"]
    for p in ("start_para", "end_para"):
        assert params["properties"][p]["type"] == "integer"
        assert "1-based" in params["properties"][p]["description"]
        assert p not in params["required"]


def test_read_section_small_no_range_byte_identical(populated_store):
    # branch (a): no range and under cap -> output unchanged vs the legacy
    # format, byte for byte (no paragraph markers, old header)
    box = _box(populated_store)
    out = box.execute("read_section",
                      {"doc_id": "en_paper.md", "section": "2. Method"})
    s = Reader(populated_store).section("en_paper.md", name="2. Method")
    assert out == (f"SECTION en_paper.md :: {s['name']} ({s['token_count']} tok)\n"
                   f"tldr: {s['tldr']}\n---\n{s['content']}")


def test_read_section_over_cap_pages_by_paragraphs():
    # branch (b): no range, over cap -> whole paragraphs ¶1..¶k + continuation
    box = _para_box(section_token_cap=40)
    out = box.execute("read_section", {"doc_id": "p.md", "section": "Body"})
    assert re.search(r"SECTION p\.md :: Body \(\d+ tok, 6 paras\)", out)
    assert "[¶1]" in out and _PARAS[0] in out
    assert "[¶6]" not in out
    m = re.search(r"\.\.\.\(section has 6 paragraphs, showed ¶1–¶(\d+); "
                  r"call again with start_para=(\d+), or grep for specifics\)", out)
    assert m, out
    k = int(m.group(1))
    assert int(m.group(2)) == k + 1
    # the last shown paragraph is complete, not cut mid-text
    assert _PARAS[k - 1] in out
    assert f"[¶{k + 1}]" not in out


def test_read_section_range_marks_paragraphs():
    # branch (c): explicit range -> only ¶start..¶end, each with a [¶i] marker
    box = _para_box()
    out = box.execute("read_section", {"doc_id": "p.md", "section": "Body",
                                       "start_para": 2, "end_para": 3})
    assert re.search(r"SECTION p\.md :: Body \(\d+ tok, 6 paras\)", out)
    assert "[¶2]" in out and "[¶3]" in out
    assert "[¶1]" not in out and "[¶4]" not in out
    assert _PARAS[1] in out and _PARAS[2] in out
    assert "P1 " not in out and "P4 " not in out
    assert "call again" not in out


def test_read_section_range_clips_out_of_bounds():
    box = _para_box()
    out = box.execute("read_section", {"doc_id": "p.md", "section": "Body",
                                       "start_para": 0, "end_para": 99})
    assert "[¶1]" in out and "[¶6]" in out
    assert "error" not in out.lower()
    out2 = box.execute("read_section", {"doc_id": "p.md", "section": "Body",
                                        "start_para": 99})
    assert "[¶6]" in out2 and "[¶5]" not in out2


def test_read_section_range_over_cap_gives_continuation_hint():
    box = _para_box(section_token_cap=40)
    out = box.execute("read_section", {"doc_id": "p.md", "section": "Body",
                                       "start_para": 1, "end_para": 6})
    m = re.search(r"showed ¶1–¶(\d+); call again with start_para=(\d+)", out)
    assert m, out
    assert int(m.group(2)) == int(m.group(1)) + 1


def test_read_section_single_huge_paragraph_still_truncates(populated_store):
    # a paragraph exceeding the whole cap must be cut, keeping the grep hint
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 section_token_cap=1)
    reader = Reader(populated_store)
    box = ToolBox(cfg, reader, SearchIndex(reader))
    out = box.execute("read_section",
                      {"doc_id": "en_paper.md", "section": "2. Method"})
    assert "¶1" in out
    assert "truncated" in out.lower() and "grep" in out


def test_read_section_default_skips_front_matter(populated_store):
    # en_paper section 0 is "ABSTRACT" (front matter) -> default read should skip
    # to the first substantive section "1. Introduction"
    box = _box(populated_store)
    out = box.execute("read_section", {"doc_id": "en_paper.md"})
    assert "SECTION en_paper.md :: 1. Introduction" in out


# ---- proxy tool-name mangling resolver --------------------------------------
# 2026-07-08 incident: an aiberm distributor rewrites tool schema names to
# Compat<CamelCase><6hex> (deterministic per tool); the model then calls the
# mangled names and every call used to bounce as "unknown tool".

def test_resolve_mangled_names_observed_forms():
    from deepreadqa.tools import resolve_tool_name
    known = {"search", "head", "read_section", "grep", "summarize"}
    assert resolve_tool_name("CompatSearch50e3a5", known) == "search"
    assert resolve_tool_name("CompatHeadd9384f", known) == "head"
    assert resolve_tool_name("CompatGrep11fe7e", known) == "grep"
    assert resolve_tool_name("CompatReadSection0a1b2c", known) == "read_section"


def test_resolve_passes_through_clean_and_rejects_unknown():
    from deepreadqa.tools import resolve_tool_name
    known = {"search", "head"}
    assert resolve_tool_name("search", known) == "search"
    assert resolve_tool_name("CompatNoSuchTool123abc", known) is None
    assert resolve_tool_name("totally_bogus", known) is None


def test_toolbox_executes_mangled_name(populated_store):
    box = _box(populated_store)
    out = box.execute("CompatSearch50e3a5", {"queries": ["hydroplaning FSI"]})
    assert "candidate documents" in out or "No documents matched" in out
