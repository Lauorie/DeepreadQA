"""Agent tools: progressive-reading views + in-document grep."""
from __future__ import annotations

import logging

from deepread_sdk import Reader
from deepread_sdk.tokens import count_tokens

from .config import Config
from .retrieval import SearchIndex

logger = logging.getLogger(__name__)

TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Lexical search over the knowledge base. Pass 1-5 bilingual "
                       "queries (Chinese + English). Returns candidate documents as "
                       "brief cards (doc_id, title, tldr) with a best-matching section hint.",
        "parameters": {"type": "object", "properties": {
            "queries": {"type": "array", "items": {"type": "string"},
                        "description": "1-5 search queries, mix Chinese and English"}},
            "required": ["queries"]}}},
    {"type": "function", "function": {
        "name": "head",
        "description": "Budget-aware document header: abstract + table of contents "
                       "with each section's tldr and token_count. Read this before "
                       "deciding which sections to open.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "read_section",
        "description": "Read one full section by name (from head) or by idx.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"},
            "section": {"type": "string", "description": "section name"},
            "idx": {"type": "integer", "description": "section index (optional)"}},
            "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "intro",
        "description": "Read the document's Introduction/引言 (background & motivation).",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "preview",
        "description": "Low-cost prefix preview (first ~10k chars) for relevance check.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Find exact keywords/numbers inside one document, returning "
                       "matching passages with surrounding context. For precise evidence.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"},
            "patterns": {"type": "array", "items": {"type": "string"}}},
            "required": ["doc_id", "patterns"]}}},
    {"type": "function", "function": {
        "name": "read_raw",
        "description": "Read the full document markdown. Use only as a last resort for "
                       "strict verification; it is token-expensive.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    {"type": "function", "function": {
        "name": "summarize",
        "description": "Consolidate progress to free context. Provide a summary and the "
                       "doc_ids whose opened content must be kept.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "keep_doc_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["summary"]}}},
]


class ToolBox:
    def __init__(self, cfg: Config, reader: Reader, index: SearchIndex) -> None:
        self._cfg = cfg
        self._reader = reader
        self._index = index
        self.seen_docs: set[str] = set()

    def execute(self, name: str, args: dict) -> str:
        try:
            handler = getattr(self, f"_t_{name}")
        except AttributeError:
            return f"error: unknown tool {name!r}"
        try:
            return handler(args)
        except KeyError as exc:
            return f"not found: {exc}"
        except Exception as exc:  # noqa: BLE001
            logger.error("tool %s failed: %s", name, exc)
            return f"error executing {name}: {exc}"

    # --- handlers ---------------------------------------------------------
    def _t_search(self, args: dict) -> str:
        queries = args.get("queries") or []
        if isinstance(queries, str):
            queries = [queries]
        queries = queries[: self._cfg.max_queries_per_search]
        hits = self._index.search_many(queries, top_k=self._cfg.results_per_query)
        if not hits:
            return "No documents matched. Try different bilingual keywords."
        lines = [f"Found {len(hits)} candidate documents:"]
        for h in hits:
            self.seen_docs.add(h.doc_id)
            hint = f" | best section: {h.section_name}" if h.section_name else ""
            lines.append(f"- doc_id: {h.doc_id}\n  title: {h.title}\n  "
                         f"tldr: {h.tldr}{hint}")
        return "\n".join(lines)

    def _t_head(self, args: dict) -> str:
        h = self._reader.head(args["doc_id"])
        self.seen_docs.add(args["doc_id"])
        lines = [f"HEAD {h['doc_id']} | {h['title']} ({h['language']})",
                 f"global tldr: {h['tldr']}"]
        if h["abstract"]:
            lines.append(f"abstract: {h['abstract'][:800]}")
        lines.append("sections (name | tokens | tldr):")
        for s in h["sections"]:
            lines.append(f"  [{s['idx']}] {s['name']} | {s['token_count']} tok | {s['tldr']}")
        return "\n".join(lines)

    def _t_read_section(self, args: dict) -> str:
        s = self._reader.section(args["doc_id"], name=args.get("section"),
                                 idx=args.get("idx"))
        self.seen_docs.add(args["doc_id"])
        return (f"SECTION {args['doc_id']} :: {s['name']} ({s['token_count']} tok)\n"
                f"tldr: {s['tldr']}\n---\n{s['content']}")

    def _t_intro(self, args: dict) -> str:
        self.seen_docs.add(args["doc_id"])
        return f"INTRO {args['doc_id']}\n---\n{self._reader.intro(args['doc_id'])}"

    def _t_preview(self, args: dict) -> str:
        p = self._reader.preview(args["doc_id"])
        self.seen_docs.add(args["doc_id"])
        flag = " (truncated)" if p["is_truncated"] else ""
        return f"PREVIEW {p['doc_id']} [{p['total_characters']} chars{flag}]\n---\n{p['preview']}"

    def _t_grep(self, args: dict) -> str:
        doc_id = args["doc_id"]
        patterns = args.get("patterns") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        self.seen_docs.add(doc_id)
        lines = self._reader.raw(doc_id).splitlines()
        ctx = self._cfg.grep_ctx_lines
        out: list[str] = []
        budget = self._cfg.grep_token_cap
        for pat in patterns:
            low = pat.lower()
            found = 0
            for i, line in enumerate(lines):
                if low in line.lower():
                    lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
                    passage = "\n".join(lines[lo:hi])
                    block = f"[{doc_id} :: '{pat}' near line {i+1}]\n{passage}"
                    if count_tokens("\n".join(out) + block) > budget:
                        out.append("...(grep truncated: token cap reached)")
                        return "\n\n".join(out)
                    out.append(block)
                    found += 1
                    if found >= self._cfg.grep_passages_per_pattern:
                        break
            if found == 0:
                out.append(f"[{doc_id} :: '{pat}'] no match")
        return "\n\n".join(out) if out else "no matches"

    def _t_read_raw(self, args: dict) -> str:
        doc_id = args["doc_id"]
        self.seen_docs.add(doc_id)
        raw = self._reader.raw(doc_id)
        if count_tokens(raw) > self._cfg.raw_token_cap:
            cap_chars = self._cfg.raw_token_cap * 4
            raw = raw[:cap_chars] + "\n...(raw truncated at token cap; use read_section/grep)"
        return f"RAW {doc_id}\n---\n{raw}"

    def _t_summarize(self, args: dict) -> str:
        return "Acknowledged; context will be consolidated."
