"""Compose-side verify-repair loop (axis ②).

After the one-shot compose head produces an answer, a *verify* call reviews it
against the collected evidence (coverage / numeric explicitness / unsupported
claims) and may propose up to two retrieval probes phrased in domain terms —
the query-mismatch rescue for failures like item 46, where the question's
wording (位移衰减/阻尼) never reaches the evidence's wording (侧壁缝隙/对称边界).
Probes run against the same ToolBox; a *repair* call then revises the answer.
Any failure in the loop falls back to the composed answer — it can only add.

This module holds the pure, testable half: the review-protocol parser and the
probe runner. Prompt templates live in ``prompts.py``; wiring in ``harness``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_PROBE_SEARCH_RE = re.compile(r"^-\s*search:\s*(.+)$")
_PROBE_GREP_RE = re.compile(r"^-\s*grep:\s*(.+?)\s*::\s*(.+)$")
_MISSING_HEADER_RE = re.compile(r"缺失要点")
_SECTION_HEADER_RE = re.compile(r"^(缺失要点|补充检索|结论)")

PROBE_OUTPUT_CHAR_CAP = 6000


@dataclass
class VerifyReport:
    """Parsed review: what to add, where to look, and the verdict."""

    missing: list[str] = field(default_factory=list)
    probes: list[tuple[str, str]] = field(default_factory=list)
    verdict: str = "PASS"


def parse_verify(text: str) -> VerifyReport:
    """Parse the line-protocol review; unparseable input degrades to PASS.

    PASS-on-garbage is the safe default: the repair step only runs when the
    reviewer produced something actionable, so a malformed review can never
    make the answer worse.
    """
    missing: list[str] = []
    probes: list[tuple[str, str]] = []
    in_missing = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if _SECTION_HEADER_RE.match(line):
            in_missing = bool(_MISSING_HEADER_RE.match(line))
            continue
        m = _PROBE_SEARCH_RE.match(line)
        if m:
            probes.append(("search", m.group(1).strip()))
            continue
        m = _PROBE_GREP_RE.match(line)
        if m:
            probes.append(("grep", f"{m.group(1).strip()} :: {m.group(2).strip()}"))
            continue
        if in_missing and line.startswith("-"):
            item = line.lstrip("-").strip()
            if item and item != "无":
                missing.append(item)
    verdict = "REVISE" if (missing or probes) else "PASS"
    if "PASS" in text and not missing and not probes:
        verdict = "PASS"
    return VerifyReport(missing=missing, probes=probes, verdict=verdict)


def run_probes(box, probes: list[tuple[str, str]], *, max_probes: int = 2,
               char_cap: int = PROBE_OUTPUT_CHAR_CAP) -> str:
    """Execute up to ``max_probes`` reviewer probes against the ToolBox.

    Returns the concatenated, per-probe-capped outputs ready to embed in the
    repair prompt. Probe failures are logged and skipped — the loop must
    never take the answer down with it.
    """
    blocks: list[str] = []
    for kind, spec in probes[:max_probes]:
        try:
            if kind == "search":
                args = {"queries": [spec]}
            else:
                doc_id, _, patterns = spec.partition(" :: ")
                args = {"doc_id": doc_id.strip(),
                        "patterns": [p for p in patterns.split("|") if p.strip()]}
            result = str(box.execute(kind, args))[:char_cap]
            blocks.append(f"[probe {kind}: {spec}]\n{result}")
        except Exception as exc:  # noqa: BLE001 - probes are best-effort
            logger.warning("verify probe %s %r failed: %s", kind, spec, exc)
    return "\n\n".join(blocks)
