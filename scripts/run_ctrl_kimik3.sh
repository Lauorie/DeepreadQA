#!/usr/bin/env bash
# Control-variable eval for kimi-k3 (comparsion.md §15/§16 protocol):
# cae_vlmocr.db + 5-tool default face + 8000/6000 thinking budgets, default
# reasoning effort, 94 items, v3 rubric scoring (judge=gpt-5.4-mini).
#
# Usage: run_ctrl_kimik3.sh <label> [num_shards]
#   e.g. run_ctrl_kimik3.sh kimik3   8     # round 1
#        run_ctrl_kimik3.sh kimik3b  8     # round 2
#
# Channel injection (kimi-k3 is not on the default aiberm channel yet):
#   KIMI3_BASE_URL / KIMI3_API_KEY / KIMI3_MODEL — exported over AIBERM_* for
#   the answering phase only; scoring always runs with clean .env judge creds.
set -uo pipefail

LABEL="${1:?usage: run_ctrl_kimik3.sh <label> [num_shards]}"
SHARDS="${2:-8}"
ROOT="/home/juli/CAE-QA/DeepreadQA"
RUNS="$ROOT/runs"
cd "$ROOT"

# Pull KIMI3_* channel creds from .env (gitignored) unless already exported.
if [ -z "${KIMI3_API_KEY:-}" ] && [ -f "$ROOT/.env" ]; then
  set -a; source "$ROOT/.env"; set +a
fi
MODEL="${KIMI3_MODEL:-moonshotai/kimi-k3}"

export DEEPREAD_DB="store/cae_vlmocr.db"
export DEEPREAD_MAX_OUTPUT_TOKENS=8000
export DEEPREAD_COMPOSE_MAX_TOKENS=6000
export DEEPREAD_REQUEST_TIMEOUT_S=300
# Day-1 OpenRouter congestion: patient same-endpoint retries (8 × 1.5s·n
# backoff ≈ 54s ridethrough); backup endpoint disabled — wismodel has no
# kimi-k3, failover would only burn a doomed model_not_found call per chat.
export DEEPREAD_MAX_RETRIES=8

answer_env=(DEEPREAD_AGENT_MODEL="$MODEL" DEEPREAD_BACKUP_BASE_URL= DEEPREAD_BACKUP_API_KEY=)
if [ -n "${KIMI3_BASE_URL:-}" ] && [ -n "${KIMI3_API_KEY:-}" ]; then
  answer_env+=(AIBERM_BASE_URL="$KIMI3_BASE_URL" AIBERM_API_KEY="$KIMI3_API_KEY")
  echo "[info] answering via override channel: $KIMI3_BASE_URL"
fi

echo "================================================================"
echo "[$(date '+%F %T')] ctrl_${LABEL}: smoke (item 88) MODEL=$MODEL"
if ! timeout 1800 env "${answer_env[@]}" python3 run_eval.py \
      --ids 88 --output "$RUNS/ctrl_${LABEL}_smoke.jsonl" \
      > "$RUNS/ctrl_${LABEL}_smoke.log" 2>&1; then
  echo "[FAIL] ctrl_${LABEL}: smoke crashed/timed out — channel unhealthy"
  exit 1
fi
if grep -q '"answer": ""' "$RUNS/ctrl_${LABEL}_smoke.jsonl"; then
  echo "[FAIL] ctrl_${LABEL}: smoke empty answer — channel unfunded/broken"
  exit 1
fi
echo "[ok] smoke passed"

echo "[$(date '+%F %T')] ctrl_${LABEL}: answering 94 items ($SHARDS shards)"
pids=()
for ((k=0; k<SHARDS; k++)); do
  env "${answer_env[@]}" python3 run_eval.py --shard "$k" --num-shards "$SHARDS" \
      --output "$RUNS/ctrl_${LABEL}_s${k}.jsonl" \
      > "$RUNS/ctrl_${LABEL}_s${k}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do
  wait "$p" || echo "[WARN] ctrl_${LABEL}: shard pid $p non-zero"
done

echo "[$(date '+%F %T')] ctrl_${LABEL}: resume self-heal pass"
for ((k=0; k<SHARDS; k++)); do
  env "${answer_env[@]}" python3 run_eval.py --shard "$k" --num-shards "$SHARDS" \
      --resume --output "$RUNS/ctrl_${LABEL}_s${k}.jsonl" \
      >> "$RUNS/ctrl_${LABEL}_s${k}.log" 2>&1
done

# Explicit-shard merge (lesson from §16: never glob `_s*.jsonl` — a stray
# .rich.jsonl once polluted scoring).
: > "$RUNS/ctrl_${LABEL}.jsonl"
for ((k=0; k<SHARDS; k++)); do
  cat "$RUNS/ctrl_${LABEL}_s${k}.jsonl" >> "$RUNS/ctrl_${LABEL}.jsonl"
done
N=$(wc -l < "$RUNS/ctrl_${LABEL}.jsonl")
EMPTY=$(grep -c '"answer": ""' "$RUNS/ctrl_${LABEL}.jsonl" || true)
echo "[$(date '+%F %T')] ctrl_${LABEL}: $N preds, $EMPTY empty"
if [ "$N" -lt 94 ]; then
  echo "[FAIL] ctrl_${LABEL}: only $N/94 predictions — inspect shard logs"
  exit 1
fi
if [ "$EMPTY" -gt 5 ]; then
  echo "[FAIL] ctrl_${LABEL}: $EMPTY empties (>5) — circuit break, skip scoring"
  exit 1
fi

# KIMI3_SKIP_SCORING=1: stop after answering (e.g. judge channel unfunded;
# score later with score.sh once it is recharged).
if [ "${KIMI3_SKIP_SCORING:-0}" = "1" ]; then
  echo "[$(date '+%F %T')] ctrl_${LABEL}: answering done, scoring skipped by flag"
  exit 0
fi

# Judge-balance probe before scoring (lesson from §14: an unfunded judge
# silently scores met=false across the board). Uses the scorer's own creds.
echo "[$(date '+%F %T')] ctrl_${LABEL}: probing judge channel balance"
set -a; source /home/juli/RLM/cae-rubrics-eval/.env; set +a
PROBE=$(curl -sS -m 60 "$LLM_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $LLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-5.4-mini","messages":[{"role":"user","content":"ping"}],"max_tokens":16}')
if ! echo "$PROBE" | grep -q '"choices"'; then
  echo "[FAIL] judge probe failed: $(echo "$PROBE" | head -c 200)"
  exit 1
fi
echo "[ok] judge channel healthy"

echo "[$(date '+%F %T')] ctrl_${LABEL}: scoring (v3 rubric, judge=gpt-5.4-mini)"
bash scripts/score.sh "$RUNS/ctrl_${LABEL}.jsonl" "$RUNS/ctrl_${LABEL}.eval.json"
python3 -c "
import json
a = json.load(open('$RUNS/ctrl_${LABEL}.eval.json'))['aggregate']
print('[ctrl_${LABEL}] mean_score=%.4f mean_anchored=%.4f n_ok=%s n_err=%s' % (
    a['mean_score'], a['mean_anchored'], a['n_scored_ok'], a['n_errors']))
"
echo "[$(date '+%F %T')] DONE ctrl_${LABEL}"
