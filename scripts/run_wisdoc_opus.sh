#!/usr/bin/env bash
# DeepreadQA × wisdoc-parsed corpus × opus-4.8 (v11 concise). Isolates parse effect vs the
# mineru opus baseline (v11 = 0.816 / 3-run). Same code/model, store swapped to cae_wisdoc.db.
set -uo pipefail
ROOT="/home/juli/CAE-QA/DeepreadQA"; RUNS="$ROOT/runs"
cd "$ROOT"
LABEL="opus_wisdoc"
export DEEPREAD_DB="store/cae_wisdoc.db"
export DEEPREAD_AGENT_MODEL="anthropic/claude-opus-4.8"

echo "[$(date '+%H:%M:%S')] DeepreadQA opus×wisdoc (v11, DB=cae_wisdoc.db) — 8 shards"
pids=()
for k in 0 1 2 3 4 5 6 7; do
  python3 run_eval.py --shard "$k" --num-shards 8 \
     --output "$RUNS/${LABEL}_s${k}.jsonl" > "$RUNS/${LABEL}_s${k}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
cat "$RUNS"/${LABEL}_s[0-7].jsonl > "$RUNS/${LABEL}.jsonl"
N=$(wc -l < "$RUNS/${LABEL}.jsonl"); E=$(grep -c '"answer": ""' "$RUNS/${LABEL}.jsonl" || true)
echo "[$(date '+%H:%M:%S')] answering done: $N preds, $E empty"
bash scripts/score.sh "$RUNS/${LABEL}.jsonl" "$RUNS/${LABEL}.eval.json"
python3 -c "import json;a=json.load(open('$RUNS/${LABEL}.eval.json'))['aggregate'];print('[DeepreadQA opus×wisdoc] anchored=%.4f score=%.4f n_err=%s'%(a['mean_anchored'],a['mean_score'],a['n_errors']))"
echo "########## DEEPREAD OPUS-WISDOC END $(date '+%F %H:%M:%S') ##########"
