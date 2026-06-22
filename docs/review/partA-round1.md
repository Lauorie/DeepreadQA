## Findings

1. **Fenced-code heading detection can record headings inside code fences**  
   `deepread_sdk/structure.py:35-38` toggles `in_fence` on any line matching either ``` or `~~~`, regardless of which fence opened the block. In Markdown, a block opened by ``` must be closed by a matching backtick fence, not by `~~~` inside the code body. Example:
   ```md
   ```python
   ~~~ not a closing fence
   # This is code, not a heading
   ```
   ```
   Current behavior toggles off at `~~~`, so `# This is code` can be incorrectly treated as a real section heading. This violates the contract to ignore ATX headings inside ``` / `~~~` fences and can corrupt section inventory.

2. **`Enricher.enrich_document` is not robust to client exceptions despite fallback requirement**  
   `deepread_sdk/enrich.py:104` and `deepread_sdk/enrich.py:115` call `self._client.complete(...)` directly. If the client implementation raises — e.g. a mock, wrapper, OpenAI SDK shape issue, transport error not caught by the client — the whole document processing fails instead of using deterministic fallbacks. The brief requires “deterministic fallbacks so one bad/empty LLM response never crashes the build.” `llm.EnrichLLM` catches its own failures, but `Enricher` should not rely on every compatible client doing so.

3. **Parsed JSON `tldr` is not type-validated and can violate the “JSON-looking blob is NEVER returned as tldr” invariant**  
   `deepread_sdk/enrich.py:66` does:
   ```python
   return str(obj.get("tldr", "")).strip(), ...
   ```
   If the model returns valid JSON with a non-string `tldr`, such as:
   ```json
   {"tldr": {"summary": "..."}, "keywords": ["x"]}
   ```
   the returned TL;DR becomes `"{'summary': '...'}"`, i.e. a JSON/dict-looking blob. The stated invariant says JSON-looking blobs must never be returned as the TL;DR; the TL;DR should be left empty so caller fallback applies.

4. **Malformed/truncated JSON keyword parsing does not support keyword strings**  
   `deepread_sdk/enrich.py:68-78` only extracts malformed keywords when they appear as a JSON list via `_KW_LIST_RE`. The contract explicitly requires robustness when keywords are “list OR comma/semicolon string.” Strict JSON handles string keywords through `_coerce_keywords`, but lenient malformed/truncated JSON such as:
   ```json
   {"tldr": "foo", "keywords": "CAE, finite element, fatigue"
   ```
   returns no keywords. This is a cross-contract gap in the flaky-output hardening.

5. **Single-file read/hash failures abort `build_store` before per-document isolation starts**  
   `deepread_sdk/build.py:58-63` reads every file and computes skip state before submitting futures:
   ```python
   text = p.read_text(...)
   ```
   Any `OSError`/permission error/race where a file disappears aborts the entire build. The brief requires single-doc failures to be isolated as `failed++`, continue, never abort. The future-processing path handles failures at `deepread_sdk/build.py:70-76`, but pre-processing failures are outside that guard.

6. **`Reader.json` can silently drop sections with duplicate names**  
   `deepread_sdk/reader.py:74-77` builds:
   ```python
   data = {s.name: {...} for s in r.sections}
   ```
   If a document contains repeated heading names, later sections overwrite earlier ones. This corrupts the section inventory exposed to downstream retrieval. The SDK canonical shape uses section names as keys, but this implementation still needs a collision strategy or validation/error to avoid silent data loss.

7. **SQLite read-only URI construction is fragile for paths containing URI-reserved characters**  
   `deepread_sdk/store.py:42-43` uses:
   ```python
   uri = f"file:{db_path}?mode=ro"
   ```
   Paths containing `?`, `#`, `%`, or some absolute/Windows forms can be misinterpreted as URI syntax. This is a robustness issue for `Reader(db_path, read_only=True)`. Use `Path.resolve().as_uri()` plus `?mode=ro`, or quote the path appropriately.

## Positive notes

- No hardcoded API keys were found; `AIBERM_API_KEY` is read from the environment in `deepread_sdk/build.py:94-98`.
- `llm.EnrichLLM.complete` correctly avoids `temperature`, uses `max_tokens=768`, retries with backoff, and does not sleep after the final attempt (`deepread_sdk/llm.py:27-38`).
- SQLite writes are parameterized and performed on the main thread in `build_store` (`deepread_sdk/build.py:66-73`).
- `store.write_document` uses an explicit transaction via `with conn:` and roundtrips `keywords` JSON and `preview_is_truncated` int/bool conversion correctly.

VERDICT: CHANGES REQUESTED

1. Fix fenced-code parsing in `structure._find_headings` to track the opening fence marker/backtick-vs-tilde type and only close on a matching fence, so headings inside fenced code are never emitted.
2. Wrap global and per-section `client.complete` calls in `Enricher.enrich_document` with exception handling and deterministic fallbacks.
3. Harden `parse_global_response` so parsed JSON `tldr` is accepted only when it is genuine string prose, not a dict/list/JSON-looking blob.
4. Extend malformed/truncated JSON parsing to support `"keywords": "a, b; c"` string forms, not only keyword arrays.
5. Move file read/hash failures in `build_store` into per-file isolation: increment `failed`, log, and continue instead of aborting the whole build.
6. Prevent silent section loss in `Reader.json` when section names collide, either by deterministic disambiguation or by raising a clear error.
7. Quote or otherwise safely construct the SQLite read-only URI in `store.connect`.
