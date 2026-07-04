"""Run the DeepreadQA agent over CAE-eval.json and write scorer-format predictions."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from deepreadqa import Config, DeepreadQA

logger = logging.getLogger(__name__)


def _load_cases(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_answered(path: Path) -> dict[int, str]:
    """Map item_idx -> raw prediction line for already-answered (non-empty) items."""
    answered: dict[int, str] = {}
    if not path.exists():
        return answered
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("answer"):
            answered[rec["item_idx"]] = line
    return answered


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--ids", default=None, help="comma-separated item_idx subset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--no-concise", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep items already answered (non-empty) in --output; "
                         "re-run only empty/missing ones")
    args = ap.parse_args(argv)

    cfg = Config.from_env(concise_compose=not args.no_concise)
    cases = _load_cases(cfg.eval_file)
    if args.ids:
        want = {int(x) for x in args.ids.split(",")}
        cases = [c for c in cases if c["item_idx"] in want]
    if args.shard is not None:
        if not args.num_shards or args.num_shards < 1:
            ap.error("--num-shards must be a positive integer when --shard is used")
        if not (0 <= args.shard < args.num_shards):
            ap.error("--shard must satisfy 0 <= shard < num_shards")
        cases = [c for c in cases if c["item_idx"] % args.num_shards == args.shard]
    if args.limit is not None:
        cases = cases[: args.limit]

    qa = DeepreadQA(cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rich = out.with_suffix(".rich.jsonl")
    answered = _load_answered(out) if args.resume else {}
    if answered:
        logger.info("resume: keeping %d already-answered items", len(answered))
    # predictions are rewritten in case order (kept lines verbatim); the rich
    # log is appended so earlier trajectories are preserved
    rich_mode = "a" if args.resume else "w"
    with out.open("w", encoding="utf-8") as sf, rich.open(rich_mode,
                                                          encoding="utf-8") as rf:
        for c in cases:
            idx = c["item_idx"]
            if idx in answered:
                sf.write(answered[idx] + "\n")
                sf.flush()
                continue
            logger.info("answering item %s", idx)
            try:
                res = qa.answer(c["question"])
                answer = res.answer
            except Exception as exc:  # noqa: BLE001 - one bad item must not abort run
                logger.error("item %s crashed: %s", idx, exc, exc_info=True)
                res = None
                answer = ""
            sf.write(json.dumps({"item_idx": idx, "answer": answer},
                                ensure_ascii=False) + "\n")
            sf.flush()
            rec = {"item_idx": idx, "question": c["question"], "answer": answer}
            if res is not None:
                rec.update({"full_answer": res.full_answer, "iterations": res.iterations,
                            "total_tokens": res.total_tokens,
                            "compactions": res.compactions,
                            "forced_final": res.forced_final, "error": res.error,
                            "seen_docs": sorted(res.seen_docs),
                            "tool_calls": res.tool_calls})
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rf.flush()
        # answered items outside the current selection are preserved, not dropped
        # (e.g. resuming with a narrower --ids / different shard filter)
        selected = {c["item_idx"] for c in cases}
        for i, ln in answered.items():
            if i not in selected:
                sf.write(ln + "\n")
    logger.info("wrote %s and %s", out, rich)


if __name__ == "__main__":
    main()
