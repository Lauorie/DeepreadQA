import json
from pathlib import Path

import pytest

from deepread_sdk import store
from deepread_sdk.enrich import Enricher
from deepread_sdk.schema import DocRecord, SectionRecord
from deepread_sdk.structure import detect_language, extract_abstract, recover_structure
from deepread_sdk.tokens import count_tokens

FIX_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


class _StubClient:
    """Deterministic fake enrichment client for tests."""

    def complete(self, system: str, user: str) -> str:
        if "STRICT JSON" in system:
            return json.dumps({"tldr": "stub global tldr", "keywords": ["fsi", "ale"]})
        return "stub section tldr"


@pytest.fixture
def populated_store(tmp_path) -> Path:
    """Build a tiny SQLite store from the fixture corpus using a stub enricher."""
    db = tmp_path / "cae.db"
    conn = store.connect(db)
    store.init_schema(conn)
    enr = Enricher(_StubClient())
    for p in sorted(FIX_CORPUS.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        sdoc = recover_structure(text, fallback_title=p.stem)
        lang = detect_language(text)
        gtldr, kws, sec_tldrs = enr.enrich_document(sdoc.title, sdoc, lang)
        secs = [SectionRecord(idx=s.idx, name=s.name, tldr=sec_tldrs[i],
                              token_count=count_tokens(s.content),
                              start_pos=s.start_pos, end_pos=s.end_pos,
                              content=s.content) for i, s in enumerate(sdoc.sections)]
        preview = text[:10000]
        store.write_document(conn, DocRecord(
            doc_id=p.name, title=sdoc.title, language=lang,
            abstract=extract_abstract(sdoc), header=sdoc.header, tldr=gtldr,
            keywords=kws, token_count=count_tokens(text), total_characters=len(text),
            preview=preview, preview_is_truncated=len(text) > 10000,
            raw_md=text, content_hash="h", sections=secs))
    conn.close()
    return db
