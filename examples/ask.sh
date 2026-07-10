#!/usr/bin/env bash
# DeepreadQA API 最小调用示例（curl）。
# 用法：DEEPREADQA_API_KEY=<key> [BASE_URL=http://host:8000] ./ask.sh "你的问题"
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
KEY="${DEEPREADQA_API_KEY:?set DEEPREADQA_API_KEY}"
QUESTION="${1:?usage: ask.sh \"你的问题\"}"

# --- 同步模式：一条命令拿到答案（服务端最长等待 300s，请把客户端读超时设长） ---
curl -sS --fail-with-body -m 360 "$BASE_URL/v1/answers" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}, ensure_ascii=False))' "$QUESTION")" \
  | python3 -m json.tool --no-ensure-ascii

# --- 异步模式（长问题/不想长连接时）：
# 1) 提交：加 -H "Prefer: respond-async"，从响应头 Location 拿到 /v1/answers/{id}
# 2) 轮询：GET $BASE_URL/v1/answers/{id}，建议间隔 5s 起、上限 30s 的指数退避
# 3) status 变为 succeeded / failed 即为终态
