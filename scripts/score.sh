#!/usr/bin/env bash
set -euo pipefail
PRED="${1:?usage: score.sh <predictions.jsonl> <out_eval.json>}"
OUT="${2:?usage: score.sh <predictions.jsonl> <out_eval.json>}"
EVAL_DIR="/home/juli/RLM/cae-rubrics-eval"

PRED_ABS="$([[ "$PRED" = /* ]] && echo "$PRED" || realpath "$PRED")"
OUT_ABS="$([[ "$OUT" = /* ]] && echo "$OUT" || realpath -m "$OUT")"

cd "$EVAL_DIR"
python3 score.py \
  --predictions "$PRED_ABS" \
  --out "$OUT_ABS" \
  --concurrency 16 \
  --rubrics data/CAE-v2.0-1-rubrics.json \
  --anchors data/CAE-anchor-scores.json
