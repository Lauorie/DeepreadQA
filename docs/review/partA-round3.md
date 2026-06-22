Findings:

- `deepread_sdk/enrich.py:86-89` violates the stated invariant that “a JSON-looking blob is NEVER returned as the tldr.” The current `looks_structured` check only treats responses as structured if they start with `{`/`[`, contain fences, or contain `"tldr"`/`"keywords"`. A flaky response such as `Here is the JSON: {"foo":` or `Result: {bad json}` would be returned verbatim as a TL;DR, polluting downstream summaries/retrieval metadata. Since `_JSON_OBJ_RE.search(cleaned)` is already computed at `deepread_sdk/enrich.py:60`, the fallback should also treat any detected JSON-object-like blob as structured/non-prose.

- `deepread_sdk/reader.py:62-65` allows an empty or whitespace-only `name` to match the first section because `low == ""` and `"" in s.name.lower()` is always true. This can silently return arbitrary section content instead of raising `KeyError`, which is dangerous for retrieval callers. Empty normalized names should be rejected before substring matching.

- `deepread_sdk/structure.py:8` incorrectly strips trailing `#` characters from heading text even when they are part of the heading name, e.g. `## C#` becomes section/title name `C`. ATX closing hashes should only be removed when they are preceded by whitespace per Markdown convention. This can corrupt section inventory and name-based lookup.

VERDICT: CHANGES REQUESTED

1. Harden `parse_global_response` so any response containing a JSON-object-like blob is never returned as prose TL;DR when parsing/extraction fails.

2. Update `Reader.section()` to reject empty/whitespace-only `name` values before exact or substring matching.

3. Fix the ATX heading regex/parsing so legitimate trailing `#` characters in heading names are preserved unless they are valid closing hash markers.
