from deepread_sdk import store
from deepread_sdk.schema import DocRecord, SectionRecord


def _sample() -> DocRecord:
    return DocRecord(
        doc_id="a.md", title="A", language="en", abstract="abs", header="hdr",
        tldr="global tldr", keywords=["fsi", "ale"], token_count=42,
        total_characters=100, preview="prev", preview_is_truncated=True,
        raw_md="# A\nbody", content_hash="h1",
        sections=[SectionRecord(idx=0, name="1. Intro", tldr="s tldr",
                                token_count=10, start_pos=0, end_pos=20,
                                content="intro body")],
    )


def test_roundtrip(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.write_document(conn, _sample())
    got = store.get_document(conn, "a.md")
    assert got is not None
    assert got.title == "A"
    assert got.keywords == ["fsi", "ale"]
    assert got.preview_is_truncated is True
    assert got.sections[0].name == "1. Intro"
    assert got.sections[0].content == "intro body"


def test_upsert_overwrites(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.write_document(conn, _sample())
    rec2 = _sample()
    object.__setattr__(rec2, "title", "A2")
    store.write_document(conn, rec2)
    assert store.get_document(conn, "a.md").title == "A2"
    assert len(store.list_doc_ids(conn)) == 1


def test_content_hash_and_meta(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.write_document(conn, _sample())
    assert store.get_content_hash(conn, "a.md") == "h1"
    assert store.get_content_hash(conn, "missing.md") is None
    store.set_meta(conn, "build_model", "deepseek/deepseek-v4-flash")
    assert store.get_meta(conn, "build_model") == "deepseek/deepseek-v4-flash"


def test_missing_doc_returns_none(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    assert store.get_document(conn, "nope.md") is None


def test_connect_read_only_path_with_space(tmp_path):
    d = tmp_path / "with space"
    d.mkdir()
    db = d / "c.db"
    conn = store.connect(db)
    store.init_schema(conn)
    conn.close()
    ro = store.connect(db, read_only=True)
    assert ro.execute("SELECT 1").fetchone()[0] == 1
