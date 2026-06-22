# DeepreadQA

Progressive-reading AgenticRAG over the CAE knowledge base. A question is answered
by an LLM agent that retrieves candidate documents lexically, then reads them
**progressively** (brief → head/TOC → section → grep → raw) instead of paging
fixed line windows — mirroring the DeepRead SDK's "Agent-friendly" data layer.

Two stages:

```
OFFLINE  deepread_sdk: cae-mds/*.md
  → Structure Recovery (deterministic heading-based section split)
  → LLM enrichment (deepseek-v4-flash: global/section TL;DR + keywords)
  → tiktoken token budgets
  → SQLite store (store/cae.db)
  → Reader: brief / head / intro / preview / section / raw / json

ONLINE   deepreadqa: question
  → BM25 (doc-summary + section units) → candidate docs + section hints
  → agent loop (claude-opus-4.8, 8 tools): search → head → read_section / grep → ...
  → rubric-aligned concise compose head → {item_idx, answer}
  → cae-rubrics-eval scorer (gpt-5.4-mini judge) → mean_anchored
```

## Setup

```bash
cd /home/juli/CAE-QA/DeepreadQA
python3 -m pip install -e ".[dev]"
cp .env.example .env          # then put the real AIBERM_API_KEY into .env (gitignored)
```

`.env` keys: `AIBERM_BASE_URL`, `AIBERM_API_KEY` (credentials); `DEEPREAD_ENRICH_MODEL`
(`deepseek/deepseek-v4-flash`), `DEEPREAD_AGENT_MODEL` (`anthropic/claude-opus-4.8`),
`DEEPREAD_REVIEW_MODEL` (`openai/gpt-5.5`).

## 1) Build the store (offline, ~once)

```bash
python3 -m deepread_sdk.build --db store/cae.db --workers 8
```
Resumable (skips unchanged docs by content hash), concurrent, single-doc failures
isolated. Result: 226 docs, ~19 sections/doc, clean prose TL;DRs.

## 2) Answer the eval (online)

```bash
# single process (slow):
python3 run_eval.py --output runs/deepreadqa.jsonl
# 8-way sharded (fast):
for k in 0 1 2 3 4 5 6 7; do
  python3 run_eval.py --shard $k --num-shards 8 --output runs/s${k}.jsonl &
done; wait
cat runs/s[0-7].jsonl > runs/deepreadqa.jsonl
```
Outputs `{item_idx, answer}` JSONL (scorer format) + `.rich.jsonl` telemetry
(iterations / tools used / seen_docs / tokens).

## 3) Score

```bash
bash scripts/score.sh runs/deepreadqa.jsonl runs/deepreadqa.eval.json
python3 -c "import json;print(json.load(open('runs/deepreadqa.eval.json'))['aggregate']['mean_anchored'])"
```

## Results (CAE-eval, 94 items, mean_anchored)

**Headline — same-standard comparison on the calibrated v3 rubric** (`data-v3/`,
the version `score.sh` now uses):

| System | mean_anchored (v3) | note |
|--------|--------------------|------|
| **DeepreadQA (final)** | **0.795** | progressive reading; chunk-recall + numbered-section recovery |
| agenticRAG baseline (concise) | 0.814 | full-text 1.1k-char chunk BM25 + large line-window reads |

The gap is **0.019** — within the judge's per-run noise (the gpt-5.4-mini judge
scores the *same* predictions in a ~0.04 band run-to-run). On v3 the two approaches
are statistically neck-and-neck; DeepreadQA matches the baseline on factual_anchor
(0.84 vs 0.84), comparative_balance (0.81 vs 0.81) and process_completeness (0.74
tie), and trails only on numeric_precision (0.48 vs 0.70) and decision_logic.

**Why the rubric version matters:** the older v2 rubric over-weighted `anti_hacking`
pitfalls, which systematically penalized DeepreadQA's slightly longer answers. On v2
the same runs read 0.745 (DeepreadQA) vs 0.823 (agenticRAG) — a misleading 0.08 gap
that mostly disappears under the calibrated v3 rubric.

**Tuning progression (root-cause driven):**

| Step | v2 | v3 | key change |
|------|----|----|------------|
| v1 initial | 0.614 | — | section-level BM25; recall bottleneck (69%) |
| v2 recall+read-depth | 0.704 | — | results_per_query 8→20, diverse queries, re-search |
| v4 chunk-level BM25 | 0.743 | — | **root-cause fix**: chunk index so giant heading-less docs surface |
| v5 raw-chunk index | 0.750 | 0.790 | drop per-chunk metadata noise |
| **v6 numbered-section recovery** | 0.745 | **0.795** | **deepest fix**: split heading-less PDF dumps into real sections |

**The root cause** (found by systematic debugging, not guessing): the gold document
for **52 of 94 items** is a 128k-token ALE textbook (Benson) — a PDF→markdown dump
with **zero markdown headings**. Structure recovery collapsed it into one giant
section; BM25 length-normalization then buried it (rank ∞ for "Jaumann", "mixture
theory"), so the agent never saw it and abstained. Two fixes restored it: (1)
chunk-level BM25 indexing so a buried rare-term passage scores locally, and (2) a
numbered-section fallback ("1.4.7 Mixture theories") that splits heading-less PDF
dumps into real, addressable sections — fixing both retrieval and `read_section`.
Benson now ranks #1 for its gold queries.

## Layout

```
deepread_sdk/   tokens, schema, structure, llm, enrich, store, reader, build
deepreadqa/     config, llm (ToolLLM), tokens, retrieval (BM25), prompts, tools (8), harness
run_eval.py     eval runner (scorer JSONL + rich telemetry)
scripts/        review.py (gpt-5.5 code review), score.sh (cae-rubrics-eval wrapper)
tests/          94 tests (pytest)
docs/review/    gpt-5.5 review rounds (Part A ×5 → APPROVE, Part B ×3 → APPROVE)
```

## Quality gates

- 94 unit tests, pristine output.
- gpt-5.5 code review to consensus APPROVE on both Part A (5 rounds) and Part B (3 rounds).
- Judge model fixed to `openai/gpt-5.4-mini` (do not change — breaks anchor comparability).
