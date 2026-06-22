VERDICT: CHANGES REQUESTED

1. Fix `LLMResponse.total_tokens` to match the stated contract.
   - `deepreadqa/llm.py:57-64` increments `self.total_tokens`, but returns `LLMResponse.total_tokens=tok`, i.e. only the current call’s usage.
   - The review brief says `total_tokens` accumulates. Either return `self.total_tokens` in the response or rename/clarify the field. As written, callers/tests that rely on the response object will see per-call tokens while `ToolLLM.total_tokens` is cumulative.

2. Make compose evidence collection robust to a single oversized recent evidence block.
   - `deepreadqa/harness.py:177-194`, especially `harness.py:188-190`, stops collection entirely when the newest evidence message exceeds the remaining budget:
     ```python
     if used + t > budget:
         break
     ```
   - This can produce an empty or severely incomplete compose evidence block if the latest tool output is large. This is realistic because `read_raw` is capped at `raw_token_cap=40000` in `deepreadqa/tools.py:195-198`, while compose evidence is also capped at `compose_evidence_token_cap=40000`; the wrapper/header plus overhead can push the message over the compose budget.
   - This directly risks lower rubric scores: the compose head may rewrite from only the draft and lose citations/evidence. Required fix: truncate the oversized chunk to fit, or skip it and continue to earlier evidence, rather than `break`ing with no evidence.

3. Tighten the `read_section` tool schema so the model must provide a section name or index.
   - `deepreadqa/tools.py:31-38` declares only `"doc_id"` as required:
     ```python
     "required": ["doc_id"]
     ```
   - But `deepreadqa/tools.py:135-136` calls:
     ```python
     self._reader.section(args["doc_id"], name=args.get("section"), idx=args.get("idx"))
     ```
     If the model follows the schema literally and calls `read_section` with only `doc_id`, this can fail and waste iterations. The prompt says to read sections “by name or by idx,” so the schema should enforce that contract with `anyOf`/`oneOf` or split into clearer required alternatives.

4. Avoid marking nonexistent documents as seen when a doc-touching tool fails lookup.
   - `deepreadqa/tools.py:155-156` adds `doc_id` to `seen_docs` before `reader.raw(doc_id)` in `_t_grep`.
   - `deepreadqa/tools.py:193-194` does the same before `reader.raw(doc_id)` in `_t_read_raw`.
   - If `Reader.raw` raises `KeyError`, `execute()` returns `"not found"` but telemetry still records the nonexistent doc as seen. This degrades diagnostics and can mislead later analysis. Add to `seen_docs` only after the read succeeds, as `_t_head`, `_t_read_section`, and `_t_preview` effectively do.

5. Add section/source information to grep evidence where possible, or otherwise compensate in final citation handling.
   - `deepreadqa/tools.py:168-169` formats grep blocks as:
     ```python
     [{doc_id} :: '{pat}' near line {i+1}]
     ```
   - The prompts require final citations in `doc_id / section_name` form, but grep evidence does not expose section names. Since grep is likely to be used for exact numerical/formula evidence, the compose head may be unable to produce the required citation format from the strongest evidence. This can hurt rubric-aligned citation quality. Required fix: include section heading/nearest section if the SDK can provide it, or ensure the compose prompt/tool output makes acceptable citation format explicit for grep-only evidence.
