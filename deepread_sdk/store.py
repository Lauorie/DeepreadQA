"""SQLite persistence for the DeepRead store."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schema import DocRecord, SectionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    language TEXT,
    abstract TEXT,
    header TEXT,
    tldr TEXT,
    keywords_json TEXT,
    token_count INTEGER,
    total_characters INTEGER,
    preview TEXT,
    preview_is_truncated INTEGER,
    raw_md TEXT NOT NULL,
    content_hash TEXT
);
CREATE TABLE IF NOT EXISTS sections (
    doc_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    name TEXT NOT NULL,
    tldr TEXT,
    token_count INTEGER,
    start_pos INTEGER,
    end_pos INTEGER,
    content TEXT NOT NULL,
    PRIMARY KEY (doc_id, idx)
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    db_path = Path(db_path)
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def write_document(conn: sqlite3.Connection, rec: DocRecord) -> None:
    conn.execute("DELETE FROM documents WHERE doc_id = ?", (rec.doc_id,))
    conn.execute("DELETE FROM sections WHERE doc_id = ?", (rec.doc_id,))
    conn.execute(
        """INSERT INTO documents
           (doc_id, title, language, abstract, header, tldr, keywords_json,
            token_count, total_characters, preview, preview_is_truncated,
            raw_md, content_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (rec.doc_id, rec.title, rec.language, rec.abstract, rec.header, rec.tldr,
         json.dumps(rec.keywords, ensure_ascii=False), rec.token_count,
         rec.total_characters, rec.preview, int(rec.preview_is_truncated),
         rec.raw_md, rec.content_hash),
    )
    conn.executemany(
        """INSERT INTO sections
           (doc_id, idx, name, tldr, token_count, start_pos, end_pos, content)
           VALUES (?,?,?,?,?,?,?,?)""",
        [(rec.doc_id, s.idx, s.name, s.tldr, s.token_count, s.start_pos,
          s.end_pos, s.content) for s in rec.sections],
    )
    conn.commit()


def _row_to_sections(conn: sqlite3.Connection, doc_id: str) -> list[SectionRecord]:
    rows = conn.execute(
        "SELECT * FROM sections WHERE doc_id = ? ORDER BY idx", (doc_id,)
    ).fetchall()
    return [SectionRecord(idx=r["idx"], name=r["name"], tldr=r["tldr"] or "",
                          token_count=r["token_count"] or 0, start_pos=r["start_pos"],
                          end_pos=r["end_pos"], content=r["content"]) for r in rows]


def get_document(conn: sqlite3.Connection, doc_id: str) -> DocRecord | None:
    r = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if r is None:
        return None
    return DocRecord(
        doc_id=r["doc_id"], title=r["title"], language=r["language"],
        abstract=r["abstract"], header=r["header"] or "", tldr=r["tldr"] or "",
        keywords=json.loads(r["keywords_json"] or "[]"),
        token_count=r["token_count"] or 0, total_characters=r["total_characters"] or 0,
        preview=r["preview"] or "", preview_is_truncated=bool(r["preview_is_truncated"]),
        raw_md=r["raw_md"], content_hash=r["content_hash"] or "",
        sections=_row_to_sections(conn, doc_id),
    )


def list_doc_ids(conn: sqlite3.Connection) -> list[str]:
    return [r["doc_id"] for r in
            conn.execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()]


def get_content_hash(conn: sqlite3.Connection, doc_id: str) -> str | None:
    r = conn.execute("SELECT content_hash FROM documents WHERE doc_id = ?",
                     (doc_id,)).fetchone()
    return r["content_hash"] if r else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    r = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return r["value"] if r else None
