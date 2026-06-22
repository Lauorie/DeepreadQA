Findings:

1. **`parse_global_response` can still return JSON/blob-looking text as the TL;DR, violating the stated invariant.**  
   - `deepread_sdk/enrich.py:68-71` returns the parsed `"tldr"` string directly from strict JSON without checking whether the value itself is a JSON-looking blob.  
   - `deepread_sdk/enrich.py:75` and `deepread_sdk/enrich.py:86-87` do the same for lenient regex extraction.  
   - Example problematic response: `{"tldr": "{\"foo\": \"bar\"}", "keywords": []}` would return `{"foo": "bar"}` as the TL;DR instead of `""`, so the caller fallback would not apply. The brief explicitly requires: “a JSON-looking blob is NEVER returned as the tldr.”

2. **Section TL;DRs accept arbitrary bad LLM output verbatim, which can corrupt `Reader.head()` / section inventory quality.**  
   - `deepread_sdk/enrich.py:132-133` stores `out` directly if non-empty. If the LLM returns fenced JSON, a prompt preamble, an error string, or another structured/blob response, it becomes the section TL;DR.  
   - The global parser has hardening for this class of DeepSeek/aiberm flakiness, but section summaries do not. This can degrade downstream QA because `Reader.head()` exposes these TL;DRs as the progressive-access TOC summaries. Add a small sanitizer: strip fences/prefixes, reject JSON-looking/structured output, and fallback deterministically when rejected.

3. **Content hashing is truncated to 64 bits, which can incorrectly skip changed documents on collision.**  
   - `deepread_sdk/build.py:23-24` returns only the first 16 hex chars of SHA-256.  
   - Since `build_store` uses this as the authoritative “unchanged” check, a collision causes stale sections/TL;DRs/retrieval content to be retained. Use the full SHA-256 hex digest unless there is a strong reason not to.

4. **CLI model environment variable does not follow the stated `AIBERM_*` convention.**  
   - `deepread_sdk/build.py:104` reads `DEEPREAD_ENRICH_MODEL`, while the brief says CLI `main` reads `AIBERM_*` from `.env`.  
   - This is not a secret issue, but it is a deployment/configuration mismatch. Prefer `AIBERM_MODEL` or support both with a documented precedence.

VERDICT: CHANGES REQUESTED

Required changes:

1. Add a TL;DR sanitizer in `parse_global_response` so parsed/regex-extracted TL;DR values that are JSON-looking, fenced, or structured are returned as `""`, preserving the fallback behavior.

2. Sanitize section-level LLM responses before storing them as `section_tldrs`; reject structured/fenced/blob output and use the deterministic fallback.

3. Store/use the full SHA-256 content hash in `build._hash`.

4. Align the CLI model environment variable with the `AIBERM_*` convention, or explicitly support `AIBERM_MODEL` in addition to the current variable.
