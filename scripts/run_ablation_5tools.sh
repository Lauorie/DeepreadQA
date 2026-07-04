#!/usr/bin/env bash
# Tool ablation: drop the near-dead intro/preview/read_raw (0.6%/0/0 of opus
# v11 calls), keep search/head/read_section/grep/summarize. 3 rounds x 94
# items vs. the v11 8-tool baseline (0.8246/0.8243/0.7984, mean 0.816).
# Guards: one --resume self-heal pass per round; abort if >5 empty answers.
set -uo pipefail

ROOT="/home/juli/CAE-QA/DeepreadQA"
RUNS="$ROOT/runs"
cd "$ROOT"
export DEEPREAD_DISABLED_TOOLS="intro,preview,read_raw"

for tag in a b c; do
  echo "================================================================"
  echo "[$(date '+%F %T')] abl5${tag}: answering 94 items (8 shards)"
  pids=()
  for k in 0 1 2 3 4 5 6 7; do
    python3 run_eval.py --shard "$k" --num-shards 8 \
        --output "$RUNS/abl5${tag}_s${k}.jsonl" \
        > "$RUNS/abl5${tag}_s${k}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p" || echo "[WARN] abl5${tag}: shard pid $p exited non-zero"
  done

  echo "[$(date '+%F %T')] abl5${tag}: resume self-heal pass (re-run empty/missing)"
  for k in 0 1 2 3 4 5 6 7; do
    python3 run_eval.py --shard "$k" --num-shards 8 --resume \
        --output "$RUNS/abl5${tag}_s${k}.jsonl" \
        >> "$RUNS/abl5${tag}_s${k}.log" 2>&1
  done

  cat "$RUNS"/abl5${tag}_s[0-7].jsonl > "$RUNS/abl5${tag}.jsonl"
  N=$(wc -l < "$RUNS/abl5${tag}.jsonl")
  EMPTY=$(grep -c '"answer": ""' "$RUNS/abl5${tag}.jsonl" || true)
  echo "[$(date '+%F %T')] abl5${tag}: $N predictions, $EMPTY empty"
  if [ "$EMPTY" -gt 5 ]; then
    echo "[ABORT] abl5${tag}: too many empty answers — endpoint likely unhealthy"
    exit 1
  fi

  echo "[$(date '+%F %T')] abl5${tag}: scoring (v3 rubric, judge=gpt-5.4-mini)"
  bash scripts/score.sh "$RUNS/abl5${tag}.jsonl" "$RUNS/abl5${tag}.eval.json"
  python3 -c "
import json
a = json.load(open('$RUNS/abl5${tag}.eval.json'))['aggregate']
print('[abl5${tag}] mean_anchored=%.4f mean_score=%.4f n_ok=%s n_err=%s' % (
    a['mean_anchored'], a['mean_score'], a['n_scored_ok'], a['n_errors']))
"
done
echo "[$(date '+%F %T')] ABLATION DONE (3 rounds)"
