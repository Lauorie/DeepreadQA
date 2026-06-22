from pathlib import Path

from deepread_sdk import store
from deepread_sdk.build import build_store, process_one
from deepread_sdk.enrich import Enricher
from tests.conftest import _StubClient

FIX = Path(__file__).parent / "fixtures" / "corpus"


def test_process_one_builds_record():
    enr = Enricher(_StubClient())
    text = (FIX / "en_paper.md").read_text(encoding="utf-8")
    rec = process_one(text, "en_paper.md", enr)
    assert rec.doc_id == "en_paper.md"
    assert rec.tldr == "stub global tldr"
    assert len(rec.sections) == 3
    assert rec.content_hash
    assert rec.token_count > 0


def test_build_store_processes_all(tmp_path):
    db = tmp_path / "cae.db"
    enr = Enricher(_StubClient())
    stats = build_store(FIX, db, enr, max_workers=2)
    assert stats["processed"] == 4
    assert stats["failed"] == 0
    conn = store.connect(db, read_only=True)
    assert len(store.list_doc_ids(conn)) == 4


def test_build_store_resumable_skips_unchanged(tmp_path):
    db = tmp_path / "cae.db"
    enr = Enricher(_StubClient())
    build_store(FIX, db, enr, max_workers=2)
    stats2 = build_store(FIX, db, enr, max_workers=2)  # second run
    assert stats2["skipped"] == 4
    assert stats2["processed"] == 0


class _FailingEnricher:
    """Enricher that raises for any doc whose title contains `fail_on`."""

    def __init__(self, fail_on: str):
        self._fail_on = fail_on
        self._real = Enricher(_StubClient())

    def enrich_document(self, title, doc, language):
        if self._fail_on in title:
            raise RuntimeError("boom")
        return self._real.enrich_document(title, doc, language)


def test_build_store_isolates_single_doc_failure(tmp_path):
    db = tmp_path / "cae.db"
    enr = _FailingEnricher("Title")  # nested.md has title exactly "Title"
    stats = build_store(FIX, db, enr, max_workers=2)
    assert stats["failed"] == 1
    assert stats["processed"] == 3
    conn = store.connect(db, read_only=True)
    assert len(store.list_doc_ids(conn)) == 3


def test_build_store_isolates_unreadable_file(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.md").write_text("# Good\n## S\nbody", encoding="utf-8")
    (corpus / "bad.md").mkdir()  # a directory named *.md -> read_text raises IsADirectoryError
    db = tmp_path / "c.db"
    enr = Enricher(_StubClient())
    stats = build_store(corpus, db, enr, max_workers=2)
    assert stats["failed"] == 1
    assert stats["processed"] == 1
