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
    args = ap.parse_args(argv)

    cfg = Config.from_env(concise_compose=not args.no_concise)
    cases = _load_cases(cfg.eval_file)
    if args.ids:
        want = {int(x) for x in args.ids.split(",")}
        cases = [c for c in cases if c["item_idx"] in want]
    if args.shard is not None and args.num_shards:
        cases = [c for c in cases if c["item_idx"] % args.num_shards == args.shard]
    if args.limit:
        cases = cases[: args.limit]

    qa = DeepreadQA(cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rich = out.with_suffix(".rich.jsonl")
    with out.open("w", encoding="utf-8") as sf, rich.open("w", encoding="utf-8") as rf:
        for c in cases:
            idx = c["item_idx"]
            logger.info("answering item %s", idx)
            try:
                res = qa.answer(c["question"])
                answer = res.answer
            except Exception as exc:  # noqa: BLE001 - one bad item must not abort run
                logger.error("item %s crashed: %s", idx, exc)
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
    logger.info("wrote %s and %s", out, rich)


if __name__ == "__main__":
    main()
