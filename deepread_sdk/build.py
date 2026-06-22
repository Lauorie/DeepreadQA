"""Build the DeepRead SQLite store from a markdown corpus."""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from . import store
from .enrich import Enricher
from .llm import EnrichLLM
from .schema import DocRecord, SectionRecord
from .structure import detect_language, extract_abstract, recover_structure
from .tokens import count_tokens

logger = logging.getLogger(__name__)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def process_one(text: str, doc_id: str, enricher: Enricher, *,
                preview_chars: int = 10000) -> DocRecord:
    sdoc = recover_structure(text, fallback_title=Path(doc_id).stem)
    lang = detect_language(text)
    gtldr, keywords, sec_tldrs = enricher.enrich_document(sdoc.title, sdoc, lang)
    sections = [SectionRecord(idx=s.idx, name=s.name, tldr=sec_tldrs[i],
                              token_count=count_tokens(s.content),
                              start_pos=s.start_pos, end_pos=s.end_pos,
                              content=s.content)
                for i, s in enumerate(sdoc.sections)]
    return DocRecord(
        doc_id=doc_id, title=sdoc.title, language=lang,
        abstract=extract_abstract(sdoc), header=sdoc.header, tldr=gtldr,
        keywords=keywords, token_count=count_tokens(text),
        total_characters=len(text), preview=text[:preview_chars],
        preview_is_truncated=len(text) > preview_chars, raw_md=text,
        content_hash=_hash(text), sections=sections)


def build_store(kb_root, db_path, enricher: Enricher, *, max_workers: int = 8,
                force: bool = False, limit: int | None = None,
                logger: logging.Logger | None = None) -> dict:
    logger = logger or logging.getLogger(__name__)
    kb_root = Path(kb_root)
    files = sorted(kb_root.glob("*.md"))
    if limit is not None:
        files = files[:limit]
    if not files:
        logger.warning("no .md files found under %s", kb_root)
    conn = store.connect(db_path)
    store.init_schema(conn)

    todo: list[tuple[str, str]] = []
    skipped = 0
    failed = 0
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            failed += 1
            logger.error("failed reading %s: %s", p.name, exc)
            continue
        if not force and store.get_content_hash(conn, p.name) == _hash(text):
            skipped += 1
            continue
        todo.append((p.name, text))

    processed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_one, text, doc_id, enricher): doc_id
                for doc_id, text in todo}
        for fut in as_completed(futs):
            doc_id = futs[fut]
            try:
                rec = fut.result()
                store.write_document(conn, rec)
                processed += 1
                logger.info("processed %s (%d sections)", doc_id, len(rec.sections))
            except Exception as exc:  # noqa: BLE001 - one bad doc must not kill build
                failed += 1
                logger.error("failed %s: %s", doc_id, exc)

    store.set_meta(conn, "n_docs", str(len(store.list_doc_ids(conn))))
    conn.close()
    stats = {"processed": processed, "skipped": skipped, "failed": failed}
    logger.info("build done: %s", stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    ap = argparse.ArgumentParser(description="Build the DeepRead SQLite store")
    ap.add_argument("--kb-root", default="/home/juli/CAE-QA/cae-mds")
    ap.add_argument("--db", default="store/cae.db")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    client = EnrichLLM(
        base_url=os.environ.get("AIBERM_BASE_URL", "https://aiberm.com/v1"),
        api_key=os.environ["AIBERM_API_KEY"],
        model=os.environ.get("DEEPREAD_ENRICH_MODEL", "deepseek/deepseek-v4-flash"))
    enricher = Enricher(client)
    build_store(args.kb_root, args.db, enricher, max_workers=args.workers,
                force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
