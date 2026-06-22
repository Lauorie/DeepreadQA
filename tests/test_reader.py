import pytest

from deepread_sdk.reader import Reader


def test_brief(populated_store):
    r = Reader(populated_store)
    b = r.brief("en_paper.md")
    assert b["title"] == "Hydroplaning Simulation Using FSI"
    assert b["tldr"] == "stub global tldr"
    assert b["keywords"] == ["fsi", "ale"]


def test_head_has_toc(populated_store):
    r = Reader(populated_store)
    h = r.head("en_paper.md")
    names = [s["name"] for s in h["sections"]]
    assert names == ["ABSTRACT", "1. Introduction", "2. Method"]
    assert all("token_count" in s and "tldr" in s for s in h["sections"])
    assert h["abstract"] is not None
    assert h["token_count"] > 0


def test_section_by_name_and_idx(populated_store):
    r = Reader(populated_store)
    by_name = r.section("en_paper.md", name="2. Method")
    by_idx = r.section("en_paper.md", idx=2)
    assert by_name["content"] == by_idx["content"]
    assert "ALE coupling" in by_name["content"]


def test_section_name_fuzzy(populated_store):
    r = Reader(populated_store)
    s = r.section("en_paper.md", name="method")  # case-insensitive substring
    assert s["idx"] == 2


def test_intro_prefers_introduction(populated_store):
    r = Reader(populated_store)
    assert "Tires are important" in r.intro("en_paper.md")


def test_preview_and_raw(populated_store):
    r = Reader(populated_store)
    p = r.preview("en_paper.md")
    assert p["is_truncated"] is False
    assert p["total_characters"] > 0
    assert r.raw("en_paper.md").startswith("# Hydroplaning")


def test_json_full(populated_store):
    r = Reader(populated_store)
    j = r.json("en_paper.md")
    assert "ABSTRACT" in j["data"]
    assert "content" in j["data"]["ABSTRACT"]


def test_list_docs(populated_store):
    r = Reader(populated_store)
    docs = r.list_docs()
    assert len(docs) == 4
    for d in docs:
        assert {"doc_id", "title", "tldr", "keywords", "abstract",
                "language", "sections"} <= d.keys()
        for s in d["sections"]:
            assert {"name", "idx", "tldr", "token_count", "content"} <= s.keys()


def test_unknown_doc_raises(populated_store):
    r = Reader(populated_store)
    with pytest.raises(KeyError):
        r.brief("nope.md")


def test_preview_honors_preview_chars(populated_store):
    r = Reader(populated_store, preview_chars=50)
    p = r.preview("en_paper.md")
    assert p["preview_characters"] == 50
    assert p["is_truncated"] is True
    assert len(p["preview"]) == 50


def test_json_disambiguates_duplicate_section_names(tmp_path):
    from deepread_sdk import store
    from deepread_sdk.schema import DocRecord, SectionRecord
    db = tmp_path / "d.db"
    conn = store.connect(db)
    store.init_schema(conn)
    store.write_document(conn, DocRecord(
        doc_id="x.md", title="X", language="en", abstract=None, header="",
        tldr="t", keywords=[], token_count=1, total_characters=1, preview="p",
        preview_is_truncated=False, raw_md="# X", content_hash="h",
        sections=[SectionRecord(0, "Refs", "", 1, 0, 1, "a"),
                  SectionRecord(1, "Refs", "", 1, 1, 2, "b")]))
    conn.close()
    j = Reader(db).json("x.md")
    assert len(j["data"]) == 2
