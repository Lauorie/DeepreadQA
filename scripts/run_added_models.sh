#!/usr/bin/env bash
# Sequentially evaluate the two ADDED comparison models on the full 94-item CAE set (v3 rubric).
set -uo pipefail
ROOT="/home/juli/CAE-QA/DeepreadQA"
cd "$ROOT"

echo "########## ADDED-MODELS RUN START $(date '+%F %H:%M:%S') ##########"

bash scripts/run_model_eval.sh "gemini-3.5-flash"  gemini35flash
bash scripts/run_model_eval.sh "kimi-k2.7-code"    kimik27code

echo "########## ADDED-MODELS RUN END $(date '+%F %H:%M:%S') ##########"
echo "=== ADDED SUMMARY ==="
for L in gemini35flash kimik27code; do
  python3 -c "
import json
try:
    a=json.load(open('runs/cmp_${L}.eval.json'))['aggregate']
    print('${L}: mean_anchored=%.4f mean_score=%.4f n_ok=%s n_err=%s' % (a['mean_anchored'],a['mean_score'],a['n_scored_ok'],a['n_errors']))
except Exception as e:
    print('${L}: NOT READY (%s)' % e)
"
done
