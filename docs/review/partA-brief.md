# Part A Review Brief — deepread_sdk (offline preprocessing + Reader)

## Goal
Turn a local markdown corpus (`/home/juli/CAE-QA/cae-mds`, 227 mixed zh/en CAE papers, ATX headings, no frontmatter) into progressive-access views stored in a single SQLite file, exposed by a `Reader`. Mirrors the DeepRead SDK three-stage data layer: Structure Recovery → LLM enrichment → token budgeting.

## Module contracts (what to verify)
- `tokens.count_tokens(text) -> int`: tiktoken cl100k_base with char/4 fallback; `count_tokens("")==0`.
- `schema`: frozen dataclasses `RawSection`, `StructuredDoc`, `SectionRecord`, `DocRecord`. Field names/types are the cross-module contract.
- `structure.recover_structure(text, *, fallback_title) -> StructuredDoc`: deterministic. No headings → single "Full Document". First heading = title. Sectioning level = MIN heading level among the REMAINING headings. Deeper subsections stay nested in their parent's content. Header = text between title line and first section heading. Ignores ATX headings inside ``` / ~~~ code fences. `extract_abstract` matches an Abstract/摘要 section (tolerating trailing `.`/`:`/`：`) or an inline abstract line in the header. `detect_language -> "zh"|"en"`.
- `store` (SQLite): `connect(path, *, read_only=False)`, `init_schema`, `write_document` (explicit transaction; UPSERT delete-then-insert across documents+sections), `get_document` (None if missing), `list_doc_ids`, `get_content_hash`, `set_meta`/`get_meta`. Roundtrip fidelity: keywords JSON; `preview_is_truncated` int↔bool; sections ordered by idx. Parameterized SQL only.
- `llm.EnrichLLM.complete(system, user) -> str`: OpenAI-compatible (aiberm); NEVER passes `temperature`; retries with backoff (no sleep after final attempt); returns "" on exhaustion; max_tokens=768.
- `enrich`: `parse_global_response(raw) -> (tldr, keywords)` must be robust to deepseek-v4-flash's flaky output (strict JSON, ```json fences, trailing commas, truncated JSON, keywords as list OR comma/semicolon string, pure garbage). INVARIANT: a JSON-looking blob is NEVER returned as the tldr (leave tldr empty so the caller's content fallback applies); only genuine prose passes through verbatim. `Enricher.enrich_document(title, doc, language) -> (global_tldr, keywords, section_tldrs)` with `len(section_tldrs)==len(doc.sections)`; threads language into prompts; deterministic fallbacks so one bad/empty LLM response never crashes the build.
- `reader.Reader(db_path, *, preview_chars=10000)`: `brief/head/intro/preview/section/raw/json/list_docs`. `head` TOC has per-section {name,idx,tldr,token_count}. `section` addressable by idx OR name (exact then case-insensitive substring). Unknown doc_id → KeyError. `json` mirrors the SDK canonical shape `{doc_id, data:{section_name:{content,start_pos,end_pos}}}`. `list_docs` feeds the online retrieval index.
- `build`: `process_one` builds a full DocRecord; `build_store` is resumable (skip unchanged by content_hash), concurrent (ThreadPoolExecutor, SQLite writes only on the main thread), and isolates single-doc failures (failed++, continue, never abort). CLI `main` reads AIBERM_* from `.env`.

## Global constraints
- Type hints; `logging` not `print`; specific exceptions where reasonable; files 200–400 lines.
- Secrets only via `.env` (gitignored); no hardcoded keys.
- Env var convention (by design): credentials are `AIBERM_BASE_URL` / `AIBERM_API_KEY`; model selection uses `DEEPREAD_ENRICH_MODEL` (enrich), `DEEPREAD_AGENT_MODEL` (online agent), `DEEPREAD_REVIEW_MODEL` (review) — three distinct models on the same aiberm endpoint. `DEEPREAD_ENRICH_MODEL` for the build CLI is intentional, not a mismatch.
- 43 unit tests pass; a real 5-doc aiberm smoke produced clean prose tldrs after the parsing hardening.

## What I want from you (gpt-5.5)
Find correctness bugs, cross-module contract mismatches, security issues, and robustness gaps that would degrade the downstream QA (esp. anything that corrupts section inventory, tldrs, or retrieval input). Cite file:line. End with `VERDICT: APPROVE` or `VERDICT: CHANGES REQUESTED` + a numbered list of required changes.
