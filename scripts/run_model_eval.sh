#!/usr/bin/env bash
# Run the DeepreadQA agent over all 94 CAE items for ONE model, then score with v3 rubric.
# Usage: run_model_eval.sh <model_id> <label>
#   e.g. run_model_eval.sh "glm-5.2" glm52
set -uo pipefail

MODEL="${1:?usage: run_model_eval.sh <model_id> <label>}"
LABEL="${2:?usage: run_model_eval.sh <model_id> <label>}"
ROOT="/home/juli/CAE-QA/DeepreadQA"
RUNS="$ROOT/runs"
cd "$ROOT"

echo "================================================================"
echo "[$(date '+%H:%M:%S')] MODEL=$MODEL LABEL=$LABEL  — launching 8 shards"
echo "================================================================"

pids=()
for k in 0 1 2 3 4 5 6 7; do
  DEEPREAD_AGENT_MODEL="$MODEL" python3 run_eval.py \
      --shard "$k" --num-shards 8 \
      --output "$RUNS/cmp_${LABEL}_s${k}.jsonl" \
      > "$RUNS/cmp_${LABEL}_s${k}.log" 2>&1 &
  pids+=($!)
done

# wait for all shards, capture any failure
fail=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "[WARN] shard pid $p exited non-zero"; fail=1; }
done

cat "$RUNS"/cmp_${LABEL}_s[0-7].jsonl > "$RUNS/cmp_${LABEL}.jsonl"
N=$(wc -l < "$RUNS/cmp_${LABEL}.jsonl")
EMPTY=$(grep -c '"answer": ""' "$RUNS/cmp_${LABEL}.jsonl" || true)
echo "[$(date '+%H:%M:%S')] $LABEL answering done: $N predictions, $EMPTY empty answers (shard_fail=$fail)"

echo "[$(date '+%H:%M:%S')] scoring $LABEL with v3 rubric (judge=gpt-5.4-mini)..."
bash scripts/score.sh "$RUNS/cmp_${LABEL}.jsonl" "$RUNS/cmp_${LABEL}.eval.json"

python3 -c "
import json
a=json.load(open('$RUNS/cmp_${LABEL}.eval.json'))['aggregate']
print('[$LABEL] mean_anchored=%.4f mean_score=%.4f n_ok=%s n_err=%s' % (
    a['mean_anchored'], a['mean_score'], a['n_scored_ok'], a['n_errors']))
"
echo "[$(date '+%H:%M:%S')] DONE $LABEL"
