#!/usr/bin/env bash
# Re-run qwen3.7-max via the wismodel-proxy endpoint (aiberm's qwen channel is out of credit).
# Same underlying Alibaba model; only the billing/routing proxy differs. Judge stays on aiberm.
set -uo pipefail
ROOT="/home/juli/CAE-QA/DeepreadQA"
RUNS="$ROOT/runs"
LABEL="qwen37max"
cd "$ROOT"

# Pull the funded wismodel key at runtime (never hardcode secrets in tracked files).
WK=$(grep -iE "WISMODEL_API_KEY|PROXY_API_KEY" /home/juli/interp/.env 2>/dev/null | grep -oE "sk-[A-Za-z0-9]+" | head -1)
if [ -z "$WK" ]; then echo "FATAL: no wismodel key found"; exit 1; fi

export AIBERM_BASE_URL="https://wismodel-proxy-dev.atominnolab.com/api/v1"
export AIBERM_API_KEY="$WK"
export DEEPREAD_AGENT_MODEL="qwen3.7-max"

echo "[$(date '+%H:%M:%S')] qwen3.7-max RETRY via wismodel-proxy — 8 shards"
pids=()
for k in 0 1 2 3 4 5 6 7; do
  python3 run_eval.py --shard "$k" --num-shards 8 \
      --output "$RUNS/cmp_${LABEL}_s${k}.jsonl" \
      > "$RUNS/cmp_${LABEL}_s${k}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "[WARN] shard pid $p non-zero"; fail=1; }; done

cat "$RUNS"/cmp_${LABEL}_s[0-7].jsonl > "$RUNS/cmp_${LABEL}.jsonl"
N=$(wc -l < "$RUNS/cmp_${LABEL}.jsonl")
EMPTY=$(grep -c '"answer": ""' "$RUNS/cmp_${LABEL}.jsonl" || true)
echo "[$(date '+%H:%M:%S')] qwen retry answering done: $N preds, $EMPTY empty (shard_fail=$fail)"

# Judge scoring stays on aiberm (gpt-5.4-mini channel still funded).
echo "[$(date '+%H:%M:%S')] scoring qwen retry (v3 rubric, judge on aiberm)..."
bash scripts/score.sh "$RUNS/cmp_${LABEL}.jsonl" "$RUNS/cmp_${LABEL}.eval.json"
python3 -c "
import json
a=json.load(open('$RUNS/cmp_${LABEL}.eval.json'))['aggregate']
print('[qwen RETRY] mean_anchored=%.4f mean_score=%.4f n_ok=%s n_err=%s' % (a['mean_anchored'],a['mean_score'],a['n_scored_ok'],a['n_errors']))
"
echo "########## QWEN RETRY END $(date '+%F %H:%M:%S') ##########"
