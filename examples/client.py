"""DeepreadQA API Python 客户端示例（httpx，生产级重试/轮询模式）。

用法：
    DEEPREADQA_API_KEY=<key> python3 examples/client.py "你的问题"

演示了对外集成时推荐的全部客户端纪律：
- 异步提交（Prefer: respond-async）+ 指数退避轮询，不占长连接
- Idempotency-Key：网络重试不会重复计费
- 只对 429/503（带 Retry-After）与网络错误重试；4xx 不重试
"""
from __future__ import annotations

import os
import sys
import time
import uuid

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ["DEEPREADQA_API_KEY"]

_RETRYABLE = {429, 503}


def _request_with_retry(client: httpx.Client, method: str, url: str,
                        max_attempts: int = 5, **kwargs) -> httpx.Response:
    """Retry only on 429/503 (honouring Retry-After) and transport errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            if attempt == max_attempts:
                raise
            print(f"  transport error ({exc}); retry {attempt}", file=sys.stderr)
            time.sleep(min(2 ** attempt, 30))
            continue
        if resp.status_code in _RETRYABLE and attempt < max_attempts:
            wait = int(resp.headers.get("Retry-After", 2 ** attempt))
            print(f"  got {resp.status_code}; retrying in {wait}s", file=sys.stderr)
            time.sleep(min(wait, 60))
            continue
        return resp
    raise RuntimeError("unreachable")


def ask(question: str) -> dict:
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Prefer": "respond-async",
               # 同一逻辑请求固定一个 key：重试/重放都不会重复扣费
               "Idempotency-Key": f"example-{uuid.uuid4()}"}
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        resp = _request_with_retry(client, "POST", "/v1/answers",
                                   headers=headers,
                                   json={"question": question})
        resp.raise_for_status()
        answer_url = resp.headers.get("Location") or f"/v1/answers/{resp.json()['id']}"
        print(f"submitted: {answer_url} (request_id={resp.headers['x-request-id']})")

        delay = 5.0
        while True:
            poll = _request_with_retry(
                client, "GET", answer_url,
                headers={"Authorization": f"Bearer {API_KEY}"})
            poll.raise_for_status()
            body = poll.json()
            if body["status"] in ("succeeded", "failed"):
                return body
            print(f"  status={body['status']}; next poll in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 1.5, 30.0)  # 指数退避，上限 30s


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else \
        "HJC 本构模型模拟混凝土受冲击时主要考虑哪些效应？"
    body = ask(question)
    if body["status"] == "failed":
        print(f"FAILED: {body['error']}")
        sys.exit(1)
    usage = body["usage"]
    print(f"\n=== answer (latency {body['latency_ms']} ms, "
          f"{usage['iterations']} iterations, "
          f"{usage['total_tokens']} tokens, "
          f"read {usage['documents_read']} docs) ===\n")
    print(body["answer"])
    print("\n--- sources ---")
    for s in body["sources"]:
        print(f"  {s['doc_id']}  ({s['title']})")


if __name__ == "__main__":
    main()
