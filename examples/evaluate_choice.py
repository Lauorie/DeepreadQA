"""选择题批量评测脚本（调用方持有 gold 答案时的参考实现）。

用法：
    DEEPREADQA_API_KEY=<key> python3 examples/evaluate_choice.py \
        --questions my_questions.json --collection col_xxx \
        --output results.jsonl [--workers 2]

题库格式（JSON 数组）：
    [{"id": "q1", "question": "……上限是多少？",
      "options": {"A": "2.8%", "B": "4.2%", "C": "0.8%", "D": "5.0%"},
      "answer": "B"}, ...]

行为：
- 同步调用 `POST /v1/answers`（mode=choice），默认 2 并发（与服务端 worker 数一致，
  更高并发只会排队并消耗限流配额）；429/503 按 Retry-After 自动退避。
- 结果逐行落盘（JSONL，含判定字母/是否正确/理由/耗时），可续跑：已在输出中的 id 跳过。
- 结束时输出 accuracy 与按 gold 字母的分桶，弃答（choice=null）计错。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://8.216.129.125:8000")
API_KEY = os.environ["DEEPREADQA_API_KEY"]

_RETRYABLE = {429, 503}


def _ask_choice(client: httpx.Client, item: dict,
                collection_id: str | None) -> dict:
    body = {"mode": "choice", "question": item["question"],
            "options": item["options"]}
    if collection_id:
        body["collection_id"] = collection_id
    for attempt in range(1, 6):
        try:
            resp = client.post("/v1/answers", json=body,
                               headers={"Authorization": f"Bearer {API_KEY}"})
        except httpx.TransportError:
            time.sleep(min(2 ** attempt, 30))
            continue
        if resp.status_code in _RETRYABLE:
            time.sleep(min(int(resp.headers.get("Retry-After", 2 ** attempt)), 60))
            continue
        resp.raise_for_status()
        r = resp.json()
        return {"id": item["id"], "gold": item["answer"],
                "predicted": r.get("choice"),
                "correct": r.get("choice") == item["answer"],
                "abstained": r.get("abstained"),
                "latency_ms": r.get("latency_ms"),
                "total_tokens": (r.get("usage") or {}).get("total_tokens"),
                "reason": r.get("answer")}
    return {"id": item["id"], "gold": item["answer"], "predicted": None,
            "correct": False, "abstained": None, "latency_ms": None,
            "total_tokens": None, "reason": "ERROR: retries exhausted"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--collection", default=None, help="私有知识库 id（col_…）")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=2,
                    help="并发数；建议不超过服务端作答 worker 数（默认 2）")
    args = ap.parse_args()

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    out = Path(args.output)
    done: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [x for x in items if str(x["id"]) not in done]
    print(f"{len(items)} questions, {len(done)} already done, {len(todo)} to run")

    with httpx.Client(base_url=BASE_URL, timeout=390.0) as client, \
            out.open("a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_ask_choice, client, x, args.collection): x
                   for x in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            mark = "✓" if rec["correct"] else "✗"
            print(f"  {mark} {rec['id']}: gold={rec['gold']} "
                  f"pred={rec['predicted']} ({rec['latency_ms']} ms)")

    results = [json.loads(line) for line
               in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(results)
    acc = sum(r["correct"] for r in results) / n if n else 0.0
    by_letter: Counter = Counter()
    ok_letter: Counter = Counter()
    for r in results:
        by_letter[r["gold"]] += 1
        ok_letter[r["gold"]] += int(r["correct"])
    print(f"\naccuracy: {acc:.4f} ({sum(r['correct'] for r in results)}/{n})")
    for k in sorted(by_letter):
        print(f"  gold {k}: {ok_letter[k]}/{by_letter[k]}")
    abstains = sum(1 for r in results if r.get("abstained"))
    if abstains:
        print(f"  abstained (counted wrong): {abstains}")
    sys.exit(0 if n else 1)


if __name__ == "__main__":
    main()
