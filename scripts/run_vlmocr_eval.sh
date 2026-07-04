#!/usr/bin/env bash
# VLM-OCR corpus repair eval: same agent/model/tools as the abl5 baseline
# (opus-4.8, default 5-tool face), only the store differs -> cae_vlmocr.db
# (8 gold docs repaired with page-level VLM transcriptions, noise unchanged).
# 3 rounds x 94 items vs. abl5 a/b/c. Guards mirror run_ablation_5tools.sh.
set -uo pipefail

ROOT="/home/juli/CAE-QA/DeepreadQA"
RUNS="$ROOT/runs"
cd "$ROOT"
export DEEPREAD_DB="store/cae_vlmocr.db"
PREFIX="${1:-vlm}"

for tag in a b c; do
  echo "================================================================"
  echo "[$(date '+%F %T')] ${PREFIX}${tag}: answering 94 items (8 shards)"
  pids=()
  for k in 0 1 2 3 4 5 6 7; do
    python3 run_eval.py --shard "$k" --num-shards 8 \
        --output "$RUNS/${PREFIX}${tag}_s${k}.jsonl" \
        > "$RUNS/${PREFIX}${tag}_s${k}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p" || echo "[WARN] ${PREFIX}${tag}: shard pid $p exited non-zero"
  done

  echo "[$(date '+%F %T')] ${PREFIX}${tag}: resume self-heal pass (re-run empty/missing)"
  for k in 0 1 2 3 4 5 6 7; do
    python3 run_eval.py --shard "$k" --num-shards 8 --resume \
        --output "$RUNS/${PREFIX}${tag}_s${k}.jsonl" \
        >> "$RUNS/${PREFIX}${tag}_s${k}.log" 2>&1
  done

  cat "$RUNS"/${PREFIX}${tag}_s[0-7].jsonl > "$RUNS/${PREFIX}${tag}.jsonl"
  N=$(wc -l < "$RUNS/${PREFIX}${tag}.jsonl")
  EMPTY=$(grep -c '"answer": ""' "$RUNS/${PREFIX}${tag}.jsonl" || true)
  echo "[$(date '+%F %T')] ${PREFIX}${tag}: $N predictions, $EMPTY empty"
  if [ "$EMPTY" -gt 5 ]; then
    echo "[ABORT] ${PREFIX}${tag}: too many empty answers — endpoint likely unhealthy"
    exit 1
  fi

  echo "[$(date '+%F %T')] ${PREFIX}${tag}: scoring (v3 rubric, judge=gpt-5.4-mini)"
  bash scripts/score.sh "$RUNS/${PREFIX}${tag}.jsonl" "$RUNS/${PREFIX}${tag}.eval.json"
  python3 -c "
import json
a = json.load(open('$RUNS/${PREFIX}${tag}.eval.json'))['aggregate']
print('[${PREFIX}${tag}] mean_anchored=%.4f mean_score=%.4f n_ok=%s n_err=%s' % (
    a['mean_anchored'], a['mean_score'], a['n_scored_ok'], a['n_errors']))
"
done
echo "[$(date '+%F %T')] VLM-OCR EVAL DONE (3 rounds)"
