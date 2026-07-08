"""Agent tools: progressive-reading views + in-document grep."""
from __future__ import annotations

import logging
import re

from deepread_sdk import Reader
from deepread_sdk.reader import FRONT_MATTER_RE as _FRONTMATTER_RE
from deepread_sdk.tokens import count_tokens, truncate_to_tokens

from .config import Config
from .paragraphs import split_paragraphs
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
        "description": "Read one full section by name (from head) or by idx. "
                       "Optionally page by paragraph via start_para/end_para.",
        "parameters": {"type": "object", "properties": {
            "doc_id": {"type": "string"},
            "section": {"type": "string",
                        "description": "section name from head's TOC; provide this or idx (if both omitted, the first section is read)"},
            "idx": {"type": "integer", "description": "section index (optional)"},
            "start_para": {"type": "integer",
                           "description": "optional 1-based first paragraph to return "
                                          "(inclusive). Paragraphs are blank-line separated "
                                          "blocks; the ¶ numbers shown by search hints and "
                                          "read_section output use this coordinate system. "
                                          "Use it to page through huge sections or to read a "
                                          "precise spot; out-of-range values are clipped."},
            "end_para": {"type": "integer",
                         "description": "optional 1-based last paragraph to return "
                                        "(inclusive, same ¶ coordinates); defaults to the "
                                        "section's last paragraph, clipped if out of range."}},
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


# Some aiberm distributors rewrite tool schema names to
# Compat<CamelCase><6 hex> (deterministic per tool, 2026-07-08 incident);
# the model then calls the rewritten names. Resolve them back so a proxy-side
# rename never bounces every call as "unknown tool".
_MANGLED_RE = re.compile(r"^Compat([A-Za-z_]+)[0-9a-f]{6}$")


def resolve_tool_name(name: str, known) -> str | None:
    """Map a (possibly proxy-mangled) tool name onto a defined tool, else None."""
    if name in known:
        return name
    m = _MANGLED_RE.match(name)
    if not m:
        return None
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", m.group(1)).lower()
    return snake if snake in known else None


class ToolBox:
    def __init__(self, cfg: Config, reader: Reader, index: SearchIndex) -> None:
        self._cfg = cfg
        self._reader = reader
        self._index = index
        self.seen_docs: set[str] = set()

    def execute(self, name: str, args: dict) -> str:
        defined = {t["function"]["name"] for t in TOOL_SCHEMAS}
        resolved = resolve_tool_name(name, defined)
        if resolved is None:
            return f"error: unknown tool {name!r}"
        if resolved != name:
            logger.info("resolved proxy-mangled tool name %r -> %r", name, resolved)
        name = resolved
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
            hint = ""
            if h.section_name:
                # coordinates for a direct jump: read_section(idx=…, start_para≈¶)
                idx_part = f"[{h.section_idx}] " if h.section_idx is not None else ""
                para_part = f" (~¶{h.para_idx})" if h.para_idx is not None else ""
                hint = f" | best section: {idx_part}{h.section_name}{para_part}"
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
        start_para = args.get("start_para")
        end_para = args.get("end_para")
        if (start_para is None and end_para is None
                and count_tokens(content) <= self._cfg.section_token_cap):
            # no range requested and the section fits: legacy output, unchanged
            return (f"SECTION {args['doc_id']} :: {s['name']} ({s['token_count']} tok)\n"
                    f"tldr: {s['tldr']}\n---\n{content}")
        paras = split_paragraphs(content)
        total = len(paras)
        header = (f"SECTION {args['doc_id']} :: {s['name']} "
                  f"({s['token_count']} tok, {total} paras)\n"
                  f"tldr: {s['tldr']}\n---\n")
        if total == 0:
            return header + content
        # clip the requested range to [1, total] (paper Algorithm 1 semantics)
        start = 1 if start_para is None else min(max(1, int(start_para)), total)
        end = total if end_para is None else min(max(start, int(end_para)), total)
        body, last, cut = self._render_paras(paras, start, end)
        if last < end:
            body += (f"\n...(section has {total} paragraphs, showed ¶{start}–¶{last}; "
                     f"call again with start_para={last + 1}, or grep for specifics)")
        elif cut:
            body += f"\n...(¶{last} truncated at token cap; use grep for specifics)"
        return header + body

    def _render_paras(self, paras: list[str], start: int,
                      end: int) -> tuple[str, int, bool]:
        """Render ``[¶i]``-marked paragraphs start..end under the section token
        cap. Returns (body, last_rendered_para, last_para_was_cut); a single
        paragraph exceeding the whole cap is hard-truncated instead of dropped."""
        cap = self._cfg.section_token_cap
        blocks: list[str] = []
        used = 0
        last = start - 1
        cut = False
        for i in range(start, end + 1):
            block = f"[¶{i}]\n{paras[i - 1]}"
            t = count_tokens(block)
            if used + t > cap:
                if not blocks:
                    blocks.append(truncate_to_tokens(block, max(cap, 1)))
                    last, cut = i, True
                break
            blocks.append(block)
            used += t
            last = i
        return "\n\n".join(blocks), last, cut

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
