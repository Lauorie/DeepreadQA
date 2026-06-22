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
