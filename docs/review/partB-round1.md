Findings:

1. **Context-budget hazard: compression is attempted by sending the already-over-threshold conversation back to the LLM.**  
   In `deepreadqa/harness.py:75-84`, once `count_messages_tokens(conversation) >= cfg.token_threshold`, `_compress()` is called; `_compress()` then sends `conversation + FORCE_SUMMARIZE_PROMPT` to the same model (`deepreadqa/harness.py:120-130`). If the conversation is already near or beyond the provider context limit because of a large tool result, the summarization call itself can fail, after which the main chat is still attempted on the oversized conversation. This can turn one large `read_section`/`read_raw` into a failed item. The design says “over token_threshold → force-compress,” but this implementation has no local fallback pruning/truncation when model compression cannot be sent.

2. **Tool outputs are not consistently budget-controlled; `read_section` can return an arbitrarily large section.**  
   `deepreadqa/tools.py:140-145` returns the full section content with no token cap. `intro` similarly returns the full intro at `deepreadqa/tools.py:147-149`. Since the loop only checks budget before the next model call, a single large section can blow the context and trigger the failure mode above. The brief explicitly calls for “Budget-controlled compact views”; grep and raw are capped, but section/intro are not.

3. **`read_raw` token cap is approximate and can substantially exceed `raw_token_cap`.**  
   In `deepreadqa/tools.py:198-205`, the code checks `count_tokens(raw) > raw_token_cap`, then truncates by `raw_token_cap * 4` characters. That does not guarantee the resulting text is within `raw_token_cap`, especially for Chinese/CJK-heavy markdown or token-dense text. This can still inject far more than 40k tokens into the conversation and degrade/crash the 94-item eval. The code should truncate by actual token count or iteratively reduce until `count_tokens(raw) <= raw_token_cap`.

4. **Final compose evidence collection can drop the most relevant/latest evidence.**  
   `_collect_evidence()` in `deepreadqa/harness.py:164-176` walks tool messages from oldest to newest and stops at the cap. In progressive reading, early messages are often broad `search`/`head` output, while later `grep`, `read_section`, or `read_raw` messages contain the decisive evidence. If the cap is reached, the compose head may miss the final verification evidence and produce weaker/no-citation answers. This is a scoring risk. Prefer collecting from the tail, prioritizing substantive/latest evidence, or explicitly preserving final verification snippets.

5. **After pruning, the compose head receives no evidence because summaries are not treated as evidence.**  
   `_prune()` keeps system, first user, and an assistant summary (`deepreadqa/harness.py:137-146`), but `_collect_evidence()` only includes messages with `role == "tool"` (`deepreadqa/harness.py:169`). After any compaction, the final compose prompt may have an empty evidence block and only a draft. That weakens citation fidelity and can hurt rubric alignment. The compressed summary should be included in compose evidence or stored as an evidence-bearing message.

6. **Shard argument lacks range validation.**  
   `run_eval.py:35-39` validates `--num-shards > 0` but does not validate `0 <= --shard < --num-shards`. A typo like `--shard 8 --num-shards 8` silently produces an empty run. This is not a model-quality bug, but it can waste an eval run and produce misleading outputs.

VERDICT: CHANGES REQUESTED

1. Add a local safe-prune/truncation fallback when conversation size exceeds `token_threshold` and model-based compression fails or cannot be safely attempted; do not continue by sending the oversized conversation unchanged.

2. Enforce real token caps on large tool outputs, especially `read_section`, `intro`, and `read_raw`; ensure post-truncation token counts are actually within configured caps.

3. Change final evidence collection to preserve/prioritize the latest and most specific evidence instead of stopping after oldest tool outputs.

4. Include compressed summaries in final compose evidence, or otherwise preserve their evidence/citation content after `_prune()`.

5. Validate `--shard` range in `run_eval.py` with `0 <= shard < num_shards`.
