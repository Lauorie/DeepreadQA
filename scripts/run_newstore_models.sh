#!/usr/bin/env bash
# Re-run the 5 comparison models on the VLM-OCR repaired store (single round
# each, 5-tool default face) — same knobs as their abl5_* old-store runs so
# the store is the only variable. Sequential to smooth aiberm spend.
# Labels: nlib_<model>. qwen3.7-max answers via wismodel-proxy in a scoped
# subshell; its scoring runs OUTSIDE the subshell with clean judge creds.
set -uo pipefail

ROOT="/home/juli/CAE-QA/DeepreadQA"
RUNS="$ROOT/runs"
cd "$ROOT"
export DEEPREAD_DB="store/cae_vlmocr.db"

answer_one() {
  local MODEL="$1" LABEL="$2"
  echo "================================================================"
  echo "[$(date '+%F %T')] nlib_${LABEL}: smoke (item 88) MODEL=$MODEL"
  if ! timeout 1200 env DEEPREAD_AGENT_MODEL="$MODEL" python3 run_eval.py \
        --ids 88 --output "$RUNS/nlib_${LABEL}_smoke.jsonl" \
        > "$RUNS/nlib_${LABEL}_smoke.log" 2>&1; then
    echo "[SKIP] nlib_${LABEL}: smoke crashed/timed out — channel unhealthy"
    return 1
  fi
  if grep -q '"answer": ""' "$RUNS/nlib_${LABEL}_smoke.jsonl"; then
    echo "[SKIP] nlib_${LABEL}: smoke empty answer — channel likely unfunded"
    return 1
  fi

  echo "[$(date '+%F %T')] nlib_${LABEL}: answering 94 items (8 shards)"
  local pids=() k p
  for k in 0 1 2 3 4 5 6 7; do
    DEEPREAD_AGENT_MODEL="$MODEL" python3 run_eval.py --shard "$k" --num-shards 8 \
        --output "$RUNS/nlib_${LABEL}_s${k}.jsonl" \
        > "$RUNS/nlib_${LABEL}_s${k}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p" || echo "[WARN] nlib_${LABEL}: shard pid $p non-zero"
  done

  echo "[$(date '+%F %T')] nlib_${LABEL}: resume self-heal pass"
  for k in 0 1 2 3 4 5 6 7; do
    DEEPREAD_AGENT_MODEL="$MODEL" python3 run_eval.py --shard "$k" --num-shards 8 \
        --resume --output "$RUNS/nlib_${LABEL}_s${k}.jsonl" \
        >> "$RUNS/nlib_${LABEL}_s${k}.log" 2>&1
  done

  cat "$RUNS"/nlib_${LABEL}_s[0-7].jsonl > "$RUNS/nlib_${LABEL}.jsonl"
  local N EMPTY
  N=$(wc -l < "$RUNS/nlib_${LABEL}.jsonl")
  EMPTY=$(grep -c '"answer": ""' "$RUNS/nlib_${LABEL}.jsonl" || true)
  echo "[$(date '+%F %T')] nlib_${LABEL}: $N preds, $EMPTY empty"
  if [ "$EMPTY" -gt 5 ]; then
    echo "[FAIL] nlib_${LABEL}: too many empties — skip scoring"
    return 1
  fi
  return 0
}

score_one() {
  local LABEL="$1"
  echo "[$(date '+%F %T')] nlib_${LABEL}: scoring (v3 rubric, judge=gpt-5.4-mini)"
  bash scripts/score.sh "$RUNS/nlib_${LABEL}.jsonl" "$RUNS/nlib_${LABEL}.eval.json"
  python3 -c "
import json
a = json.load(open('$RUNS/nlib_${LABEL}.eval.json'))['aggregate']
print('[nlib_${LABEL}] mean_score=%.4f mean_anchored=%.4f n_ok=%s n_err=%s' % (
    a['mean_score'], a['mean_anchored'], a['n_scored_ok'], a['n_errors']))
"
}

for spec in "glm-5.2|glm52" "deepseek/deepseek-v4-flash|dsv4flash" \
            "gemini-3.5-flash|gemini35flash" "kimi-k2.7-code|kimik27code"; do
  MODEL="${spec%%|*}"; LABEL="${spec##*|}"
  if answer_one "$MODEL" "$LABEL"; then
    score_one "$LABEL"
  fi
done

# qwen3.7-max: answer via wismodel-proxy (scoped creds), score with clean env.
QWEN_OK=0
(
  WK=$(grep -iE "WISMODEL_API_KEY|PROXY_API_KEY" /home/juli/interp/.env 2>/dev/null \
       | grep -oE "sk-[A-Za-z0-9]+" | head -1)
  if [ -z "$WK" ]; then
    echo "[SKIP] nlib_qwen37max: no wismodel key found"
    exit 1
  fi
  export AIBERM_BASE_URL="https://wismodel-proxy-dev.atominnolab.com/api/v1"
  export AIBERM_API_KEY="$WK"
  answer_one "qwen3.7-max" qwen37max
) && QWEN_OK=1
if [ "$QWEN_OK" = "1" ]; then
  score_one qwen37max
fi

echo "[$(date '+%F %T')] NEW-STORE MODEL BATCH DONE"
