#!/usr/bin/env bash
set -euo pipefail
PRED="${1:?usage: score.sh <predictions.jsonl> <out_eval.json>}"
OUT="${2:?usage: score.sh <predictions.jsonl> <out_eval.json>}"
EVAL_DIR="/home/juli/RLM/cae-rubrics-eval"
cd "$EVAL_DIR"
python3 score.py \
  --predictions "$(realpath "$OLDPWD/$PRED" 2>/dev/null || echo "$PRED")" \
  --out "$(realpath -m "$OLDPWD/$OUT" 2>/dev/null || echo "$OUT")" \
  --concurrency 16 \
  --rubrics data/CAE-v2.0-1-rubrics.json \
  --anchors data/CAE-anchor-scores.json
