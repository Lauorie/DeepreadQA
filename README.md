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

| Run | mean_anchored | retrieval hit-rate | note |
|-----|---------------|--------------------|------|
| v1 (initial) | 0.614 | 69.1% | search→grep heavy; thin evidence |
| v2 (recall + read-depth tuning) | 0.704 | 86.2% | results_per_query 8→20, 4–6 diverse queries, re-search, read_section over grep |
| v3 (numeric-precision balance) | 0.677 (reverted) | — | lifted 数值提取 (0.58→0.75) but regressed 数值关系/对比/主观 net-negative; **v2 retained as final config** |
| **DeepreadQA final (v2)** | **0.704** | 86.2% | progressive reading; current code state |
| **agenticRAG baseline (concise)** | **0.823** | — | full-text chunk BM25 + large line-window reads |

**Diagnosis that drove tuning:** when the agent reaches a gold document its mean score
is ~0.74; when it misses, ~0.50. v1's bottleneck was retrieval recall (69%). Raising
candidate breadth + encouraging re-search and full-section reading lifted recall to
86% and the aggregate from 0.614 → 0.704. The residual gap to the line-based
agenticRAG baseline (0.823) is mostly answer completeness on retrieved documents.

## Layout

```
deepread_sdk/   tokens, schema, structure, llm, enrich, store, reader, build
deepreadqa/     config, llm (ToolLLM), tokens, retrieval (BM25), prompts, tools (8), harness
run_eval.py     eval runner (scorer JSONL + rich telemetry)
scripts/        review.py (gpt-5.5 code review), score.sh (cae-rubrics-eval wrapper)
tests/          90 tests (pytest)
docs/review/    gpt-5.5 review rounds (Part A ×5 → APPROVE, Part B ×3 → APPROVE)
```

## Quality gates

- 90 unit tests, pristine output.
- gpt-5.5 code review to consensus APPROVE on both Part A (5 rounds) and Part B (3 rounds).
- Judge model fixed to `openai/gpt-5.4-mini` (do not change — breaks anchor comparability).
