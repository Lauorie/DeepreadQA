"""Agent tools: progressive-reading views + in-document grep."""
from __future__ import annotations

import logging

from deepread_sdk import Reader
from deepread_sdk.reader import FRONT_MATTER_RE as _FRONTMATTER_RE
from deepread_sdk.tokens import count_tokens, truncate_to_tokens

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
            "section": {"type": "string",
                        "description": "section name from head's TOC; provide this or idx (if both omitted, the first section is read)"},
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
        if name in self._cfg.disabled_tools:
            return f"error: unknown tool {name!r}"
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
            card = (f"- doc_id: {h.doc_id}\n  title: {h.title}\n  "
                    f"tldr: {h.tldr}{hint}")
            if getattr(h, "snippet", ""):
                card += f"\n  matched snippet: {h.snippet}"
            lines.append(card)
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
        name = args.get("section")
        idx = args.get("idx")
        if name is None and idx is None:
            # no target given: read the first substantive section, skipping
            # front matter (Library of Congress / 摘要 / Table of Contents …)
            idx = 0
            for sec in self._reader.head(args["doc_id"])["sections"]:
                if sec["token_count"] > 0 and not _FRONTMATTER_RE.search(sec["name"]):
                    idx = sec["idx"]
                    break
        s = self._reader.section(args["doc_id"], name=name, idx=idx)
        self.seen_docs.add(args["doc_id"])
        content = s["content"]
        if count_tokens(content) > self._cfg.section_token_cap:
            content = (truncate_to_tokens(content, self._cfg.section_token_cap)
                       + "\n...(section truncated at token cap; use grep for specifics)")
        return (f"SECTION {args['doc_id']} :: {s['name']} ({s['token_count']} tok)\n"
                f"tldr: {s['tldr']}\n---\n{content}")

    def _t_intro(self, args: dict) -> str:
        self.seen_docs.add(args["doc_id"])
        content = self._reader.intro(args["doc_id"])
        if count_tokens(content) > self._cfg.section_token_cap:
            content = (truncate_to_tokens(content, self._cfg.section_token_cap)
                       + "\n...(intro truncated at token cap)")
        return f"INTRO {args['doc_id']}\n---\n{content}"

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
        raw = self._reader.raw(doc_id)
        self.seen_docs.add(doc_id)
        lines = raw.splitlines()
        # char offset of each line start (cae-mds markdown uses \n)
        line_starts: list[int] = []
        pos = 0
        for ln in lines:
            line_starts.append(pos)
            pos += len(ln) + 1
        try:
            secmap = [(d["start_pos"], d["end_pos"], name)
                      for name, d in self._reader.json(doc_id)["data"].items()]
        except Exception:
            secmap = []

        def _sec_for(off: int) -> str | None:
            for st, en, nm in secmap:
                if st <= off < en:
                    return nm
            return None

        ctx = self._cfg.grep_ctx_lines
        out: list[str] = []
        budget = self._cfg.grep_token_cap
        used = 0
        for pat in patterns:
            low = pat.lower()
            found = 0
            for i, line in enumerate(lines):
                if low in line.lower():
                    lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
                    passage = "\n".join(lines[lo:hi])
                    sec = _sec_for(line_starts[i])
                    tag = f" :: {sec}" if sec else ""
                    block = f"[{doc_id}{tag} :: '{pat}' near line {i + 1}]\n{passage}"
                    btok = count_tokens(block)
                    if used + btok > budget:
                        out.append("...(grep truncated: token cap reached)")
                        return "\n\n".join(out)
                    out.append(block)
                    used += btok
                    found += 1
                    if found >= self._cfg.grep_passages_per_pattern:
                        break
            if found == 0:
                out.append(f"[{doc_id} :: '{pat}'] no match")
        return "\n\n".join(out) if out else "no matches"

    def _t_read_raw(self, args: dict) -> str:
        doc_id = args["doc_id"]
        raw = self._reader.raw(doc_id)
        self.seen_docs.add(doc_id)
        if count_tokens(raw) > self._cfg.raw_token_cap:
            raw = (truncate_to_tokens(raw, self._cfg.raw_token_cap)
                   + "\n...(raw truncated at token cap; use read_section/grep)")
        return f"RAW {doc_id}\n---\n{raw}"

    def _t_summarize(self, args: dict) -> str:
        return "Acknowledged; context will be consolidated."
