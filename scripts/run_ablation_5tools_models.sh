#!/usr/bin/env bash
# 5-tool ablation across the 5 comparison models (single round each, vs their
# 8-tool single-round cmp_* baselines; opus already validated in runs/abl5*).
# Models run SEQUENTIALLY to smooth aiberm spend — the 2026-06-25 multi-model
# concurrency spike drained the balance mid-run. Per model: 1-item smoke
# (skip model if channel dead), 8-shard run, --resume self-heal, empty guard,
# v3 scoring. qwen3.7-max goes through wismodel-proxy, the endpoint its
# 0.7893 baseline was measured on.
set -uo pipefail

ROOT="/home/juli/CAE-QA/DeepreadQA"
RUNS="$ROOT/runs"
cd "$ROOT"
export DEEPREAD_DISABLED_TOOLS="intro,preview,read_raw"

run_one() {
  local MODEL="$1" LABEL="$2"
  echo "================================================================"
  echo "[$(date '+%F %T')] abl5_${LABEL}: smoke (item 88) MODEL=$MODEL"
  if ! timeout 1200 env DEEPREAD_AGENT_MODEL="$MODEL" python3 run_eval.py \
        --ids 88 --output "$RUNS/abl5_${LABEL}_smoke.jsonl" \
        > "$RUNS/abl5_${LABEL}_smoke.log" 2>&1; then
    echo "[SKIP] abl5_${LABEL}: smoke crashed/timed out — channel unhealthy"
    return 0
  fi
  if grep -q '"answer": ""' "$RUNS/abl5_${LABEL}_smoke.jsonl"; then
    echo "[SKIP] abl5_${LABEL}: smoke empty answer — channel likely unfunded"
    return 0
  fi

  echo "[$(date '+%F %T')] abl5_${LABEL}: answering 94 items (8 shards)"
  local pids=() k p
  for k in 0 1 2 3 4 5 6 7; do
    DEEPREAD_AGENT_MODEL="$MODEL" python3 run_eval.py --shard "$k" --num-shards 8 \
        --output "$RUNS/abl5_${LABEL}_s${k}.jsonl" \
        > "$RUNS/abl5_${LABEL}_s${k}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p" || echo "[WARN] abl5_${LABEL}: shard pid $p non-zero"
  done

  echo "[$(date '+%F %T')] abl5_${LABEL}: resume self-heal pass"
  for k in 0 1 2 3 4 5 6 7; do
    DEEPREAD_AGENT_MODEL="$MODEL" python3 run_eval.py --shard "$k" --num-shards 8 \
        --resume --output "$RUNS/abl5_${LABEL}_s${k}.jsonl" \
        >> "$RUNS/abl5_${LABEL}_s${k}.log" 2>&1
  done

  cat "$RUNS"/abl5_${LABEL}_s[0-7].jsonl > "$RUNS/abl5_${LABEL}.jsonl"
  local N EMPTY
  N=$(wc -l < "$RUNS/abl5_${LABEL}.jsonl")
  EMPTY=$(grep -c '"answer": ""' "$RUNS/abl5_${LABEL}.jsonl" || true)
  echo "[$(date '+%F %T')] abl5_${LABEL}: $N preds, $EMPTY empty"
  if [ "$EMPTY" -gt 5 ]; then
    echo "[FAIL] abl5_${LABEL}: too many empties — skip scoring (channel died mid-run?)"
    return 0
  fi

  echo "[$(date '+%F %T')] abl5_${LABEL}: scoring (v3 rubric, judge=gpt-5.4-mini)"
  bash scripts/score.sh "$RUNS/abl5_${LABEL}.jsonl" "$RUNS/abl5_${LABEL}.eval.json"
  python3 -c "
import json
a = json.load(open('$RUNS/abl5_${LABEL}.eval.json'))['aggregate']
print('[abl5_${LABEL}] mean_anchored=%.4f n_ok=%s n_err=%s' % (
    a['mean_anchored'], a['n_scored_ok'], a['n_errors']))
"
}

run_one "glm-5.2"                    glm52
run_one "deepseek/deepseek-v4-flash" dsv4flash
run_one "gemini-3.5-flash"           gemini35flash
run_one "kimi-k2.7-code"             kimik27code

# qwen last, via wismodel-proxy (key pulled at runtime; never hardcoded).
(
  WK=$(grep -iE "WISMODEL_API_KEY|PROXY_API_KEY" /home/juli/interp/.env 2>/dev/null \
       | grep -oE "sk-[A-Za-z0-9]+" | head -1)
  if [ -z "$WK" ]; then
    echo "[SKIP] abl5_qwen37max: no wismodel key found"
    exit 0
  fi
  export AIBERM_BASE_URL="https://wismodel-proxy-dev.atominnolab.com/api/v1"
  export AIBERM_API_KEY="$WK"
  run_one "qwen3.7-max" qwen37max
)

echo "[$(date '+%F %T')] MODEL ABLATION BATCH DONE"
