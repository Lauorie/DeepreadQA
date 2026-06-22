"""Send selected source files to openai/gpt-5.5 for a code review."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_SYS = (
    "You are a meticulous senior Python reviewer. Review the provided files for "
    "correctness, edge cases, security (no hardcoded secrets), error handling, "
    "and adherence to the stated design. Be concrete and cite file:line. End with "
    "a verdict line: 'VERDICT: APPROVE' or 'VERDICT: CHANGES REQUESTED' followed "
    "by a numbered list of required changes (empty if APPROVE)."
)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", required=True, help="path to a markdown review brief")
    ap.add_argument("files", nargs="+", help="source files to review")
    args = ap.parse_args(argv)

    parts = ["# Review brief\n", Path(args.brief).read_text(encoding="utf-8"), "\n\n# Files\n"]
    for f in args.files:
        parts.append(f"\n## {f}\n```python\n{Path(f).read_text(encoding='utf-8')}\n```\n")
    user = "".join(parts)

    client = OpenAI(base_url=os.environ.get("AIBERM_BASE_URL", "https://aiberm.com/v1"),
                    api_key=os.environ["AIBERM_API_KEY"], timeout=180.0)
    resp = client.chat.completions.create(
        model=os.environ.get("DEEPREAD_REVIEW_MODEL", "openai/gpt-5.5"),
        messages=[{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
        max_tokens=3000)
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
