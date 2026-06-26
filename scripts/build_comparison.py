"""Aggregate the 3-model comparison eval + rich stats into a single JSON for reporting."""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

ROOT = Path("/home/juli/CAE-QA/DeepreadQA")
RUNS = ROOT / "runs"

MODELS = [
    ("glm52", "glm-5.2"),
    ("qwen37max", "qwen3.7-max"),
    ("dsv4flash", "deepseek/deepseek-v4-flash"),
    ("gemini35flash", "gemini-3.5-flash"),
    ("kimik27code", "kimi-k2.7-code"),
]


def load_eval(label: str) -> dict | None:
    p = RUNS / f"cmp_{label}.eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def merge_rich(label: str) -> list[dict]:
    recs: list[dict] = []
    for f in sorted(glob.glob(str(RUNS / f"cmp_{label}_s*.rich.jsonl"))):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def rich_stats(recs: list[dict]) -> dict:
    if not recs:
        return {}
    iters = [r.get("iterations", 0) for r in recs]
    toks = [r.get("total_tokens", 0) for r in recs]
    alen = [len(r.get("answer", "") or "") for r in recs]
    forced = sum(1 for r in recs if r.get("forced_final"))
    errs = sum(1 for r in recs if r.get("error"))
    empty = sum(1 for r in recs if not (r.get("answer") or "").strip())
    comp = [r.get("compactions", 0) for r in recs]
    ndocs = [len(r.get("seen_docs", [])) for r in recs]
    return {
        "n": len(recs),
        "iters_mean": round(statistics.mean(iters), 2),
        "iters_max": max(iters),
        "tokens_mean": int(statistics.mean(toks)),
        "tokens_median": int(statistics.median(toks)),
        "ans_len_median": int(statistics.median(alen)),
        "ans_len_mean": int(statistics.mean(alen)),
        "forced_final": forced,
        "errors": errs,
        "empty_answers": empty,
        "compactions_total": sum(comp),
        "seen_docs_median": int(statistics.median(ndocs)),
    }


def main() -> None:
    out: dict = {"models": {}}

    # opus v11 baseline (3 runs)
    opus = []
    for r in ["v11a", "v11b", "v11c"]:
        p = RUNS / f"deepreadqa_opus_{r}_V3.eval.json"
        if p.exists():
            opus.append(json.loads(p.read_text())["aggregate"]["mean_anchored"])
    out["opus_v11_baseline"] = {
        "runs": [round(x, 4) for x in opus],
        "mean": round(statistics.mean(opus), 4) if opus else None,
    }

    for label, model_id in MODELS:
        ev = load_eval(label)
        entry: dict = {"model_id": model_id, "label": label}
        if ev is None:
            entry["status"] = "NOT_READY"
            out["models"][label] = entry
            continue
        agg = ev["aggregate"]
        entry["status"] = "ok"
        entry["mean_anchored"] = round(agg["mean_anchored"], 4)
        entry["mean_score"] = round(agg["mean_score"], 4)
        entry["n_scored_ok"] = agg["n_scored_ok"]
        entry["n_errors"] = agg["n_errors"]
        entry["elapsed_seconds"] = agg.get("elapsed_seconds")
        entry["by_criterion_type"] = {
            k: round(v["met_rate"], 4) for k, v in agg["by_criterion_type"].items()
        }
        entry["by_question_type"] = {
            k: {"n": v["n"], "anchored": round(v["mean_anchored"], 4)}
            for k, v in agg["by_question_type"].items()
        }
        entry["by_difficulty"] = {
            k: {"n": v["n"], "anchored": round(v["mean_anchored"], 4)}
            for k, v in agg["by_difficulty"].items()
        }
        entry["rich"] = rich_stats(merge_rich(label))
        out["models"][label] = entry

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
