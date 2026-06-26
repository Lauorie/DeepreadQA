#!/usr/bin/env bash
# Sequentially evaluate all three comparison models on the full 94-item CAE set (v3 rubric).
set -uo pipefail
ROOT="/home/juli/CAE-QA/DeepreadQA"
cd "$ROOT"

echo "########## COMPARISON RUN START $(date '+%F %H:%M:%S') ##########"

bash scripts/run_model_eval.sh "glm-5.2"                     glm52
bash scripts/run_model_eval.sh "qwen3.7-max"                 qwen37max
bash scripts/run_model_eval.sh "deepseek/deepseek-v4-flash"  dsv4flash

echo "########## COMPARISON RUN END $(date '+%F %H:%M:%S') ##########"
echo "=== FINAL SUMMARY ==="
for L in glm52 qwen37max dsv4flash; do
  python3 -c "
import json
try:
    a=json.load(open('runs/cmp_${L}.eval.json'))['aggregate']
    print('${L}: mean_anchored=%.4f mean_score=%.4f n_ok=%s n_err=%s' % (a['mean_anchored'],a['mean_score'],a['n_scored_ok'],a['n_errors']))
except Exception as e:
    print('${L}: NOT READY (%s)' % e)
"
done
