"""Tests for scripts/behavior_analysis.py (DeepRead behavior metrics, read-only).

Synthetic trajectories only; no real runs/ data required.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import behavior_analysis as ba  # noqa: E402


def _traj(
    idx: int,
    tools: List[str],
    iterations: int = 3,
    tokens: int = 1000,
    compactions: int = 0,
    forced_final: bool = False,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "item_idx": idx,
        "question": "q",
        "answer": "a",
        "full_answer": "fa",
        "iterations": iterations,
        "total_tokens": tokens,
        "compactions": compactions,
        "forced_final": forced_final,
        "error": error,
        "seen_docs": [],
        "tool_calls": [{"iter": i, "tool": t, "args": {}} for i, t in enumerate(tools)],
    }


@pytest.fixture
def trajs() -> List[Dict[str, Any]]:
    return [
        _traj(0, ["search", "head", "read_section", "read_section"], iterations=4, tokens=2000),
        _traj(1, ["head", "search", "read_section"], iterations=3, tokens=1000),
        _traj(2, ["search", "grep"], iterations=2, tokens=500, forced_final=True),
        _traj(3, [], iterations=1, tokens=100, error="boom", compactions=2),
    ]


@pytest.fixture
def scores() -> Dict[int, float]:
    return {0: 0.9, 1: 0.6, 2: 0.4, 3: 0.2}


# ---------- rank / spearman ----------

def test_average_ranks_with_ties() -> None:
    assert ba.average_ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]
    assert ba.average_ranks([5.0, 5.0, 5.0]) == [2.0, 2.0, 2.0]


def test_spearman_monotonic() -> None:
    assert ba.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ba.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_with_ties_hand_computed() -> None:
    # ranks x: [1, 2.5, 2.5, 4]; ranks y: [1, 3, 2, 4] -> r = sqrt(0.9)
    r = ba.spearman([1, 2, 2, 4], [1, 3, 2, 4])
    assert r == pytest.approx(math.sqrt(0.9), abs=1e-9)


def test_spearman_degenerate_returns_none() -> None:
    assert ba.spearman([1, 1, 1], [1, 2, 3]) is None
    assert ba.spearman([1], [2]) is None


# ---------- load_scores (parsed eval.json dict -> maps) ----------

def test_load_scores_maps_item_idx_to_score_and_anchored() -> None:
    eval_data = {
        "per_candidate": [
            {"item_idx": 0, "score": 0.71875,
             "score_anchored": {"ref_score": 1.0, "weak_score": 0.0, "normalized": 0.7}},
            {"item_idx": 1, "score": 1.0, "score_anchored": None},
        ],
        "aggregate": {"n_predictions": 2},
    }
    scores, anchored = ba.load_scores(eval_data)
    assert scores == {0: 0.71875, 1: 1.0}
    assert anchored == {0: 0.7}


# ---------- group metrics ----------

def test_s2r_and_first_tool_dist(trajs: List[Dict[str, Any]], scores: Dict[int, float]) -> None:
    m = ba.compute_group_metrics("g", trajs, scores)
    assert m["s2r"]["rate"] == pytest.approx(0.25)  # only item 0
    assert m["s2r"]["n_qualified"] == 1
    assert m["s2r"]["first_tool_dist"] == {"search": 2, "head": 1, "none": 1}


def test_cs_over_cr(trajs: List[Dict[str, Any]], scores: Dict[int, float]) -> None:
    m = ba.compute_group_metrics("g", trajs, scores)
    cc = m["cs_over_cr"]
    assert cc["search_total"] == 3
    assert cc["read_section_total"] == 3
    assert cc["group_ratio"] == pytest.approx(1.0)
    # per-item ratios only where read_section > 0: item0 1/2, item1 1/1
    assert cc["per_item_mean"] == pytest.approx(0.75)
    assert cc["n_items_with_read"] == 2


def test_tool_histogram_and_per_item_means(trajs: List[Dict[str, Any]], scores: Dict[int, float]) -> None:
    m = ba.compute_group_metrics("g", trajs, scores)
    assert m["tool_totals"] == {"search": 3, "head": 2, "read_section": 3, "grep": 1}
    assert m["tool_per_item_mean"]["search"] == pytest.approx(0.75)
    pim = m["per_item_means"]
    assert pim["iterations"] == pytest.approx(2.5)
    assert pim["tool_calls"] == pytest.approx(2.25)
    assert pim["total_tokens"] == pytest.approx(900.0)


def test_buckets_and_boundaries(trajs: List[Dict[str, Any]], scores: Dict[int, float]) -> None:
    m = ba.compute_group_metrics("g", trajs, scores)
    b = m["buckets"]
    assert b["high"]["n"] == 1 and b["mid"]["n"] == 1 and b["low"]["n"] == 2
    assert b["high"]["mean_tool_calls"] == pytest.approx(4.0)
    assert b["high"]["mean_tokens"] == pytest.approx(2000.0)
    assert b["low"]["mean_tool_calls"] == pytest.approx(1.0)
    assert b["low"]["mean_tokens"] == pytest.approx(300.0)
    assert b["low"]["mean_iterations"] == pytest.approx(1.5)
    # boundary: 0.85 -> high, 0.5 -> low
    m2 = ba.compute_group_metrics("g", trajs, {0: 0.85, 1: 0.5, 2: 0.84, 3: 0.51})
    assert m2["buckets"]["high"]["n"] == 1
    assert m2["buckets"]["low"]["n"] == 1
    assert m2["buckets"]["mid"]["n"] == 2


def test_spearman_in_metrics(trajs: List[Dict[str, Any]], scores: Dict[int, float]) -> None:
    m = ba.compute_group_metrics("g", trajs, scores)
    sp = m["spearman"]
    assert sp["score_vs_tool_calls"] == pytest.approx(1.0)
    assert sp["score_vs_tokens"] == pytest.approx(1.0)
    assert sp["score_vs_iterations"] == pytest.approx(1.0)


def test_flags_counts(trajs: List[Dict[str, Any]], scores: Dict[int, float]) -> None:
    m = ba.compute_group_metrics("g", trajs, scores)
    f = m["flags"]
    assert f["forced_final"] == 1
    assert f["errors"] == 1
    assert f["compactions_total"] == 2
    assert f["items_with_compactions"] == 1


def test_unscored_items_excluded_from_buckets(trajs: List[Dict[str, Any]]) -> None:
    m = ba.compute_group_metrics("g", trajs, {0: 0.9, 1: 0.6, 2: 0.4})  # item 3 unscored
    assert m["n_scored"] == 3
    assert m["n_unscored"] == 1
    b = m["buckets"]
    assert b["high"]["n"] + b["mid"]["n"] + b["low"]["n"] == 3


def test_shard_completeness_flags(trajs: List[Dict[str, Any]], scores: Dict[int, float]) -> None:
    m = ba.compute_group_metrics("g", trajs, scores, shards_found=[0, 2])
    assert m["missing_shards"] == [1, 3, 4, 5, 6, 7]
    assert m["complete"] is False
    m2 = ba.compute_group_metrics("g", trajs, scores, shards_found=list(range(8)))
    assert m2["missing_shards"] == []
    assert m2["complete"] is True


# ---------- discovery ----------

def test_discover_groups(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "g1.eval.json").write_text("{}")
    (runs / "g1_s0.rich.jsonl").write_text("")
    (runs / "g1_s2.rich.jsonl").write_text("")
    (runs / "g1_smoke.rich.jsonl").write_text("")  # not a shard
    (runs / "g2.eval.json").write_text("{}")  # no shards
    (runs / "g3_s0.rich.jsonl").write_text("")  # no eval
    groups = ba.discover_groups(runs)
    assert set(groups) == {"g1"}
    assert sorted(groups["g1"]["shards"]) == [0, 2]
    assert groups["g1"]["eval_path"].name == "g1.eval.json"


# ---------- report smoke ----------

def test_render_report_contains_key_sections(trajs: List[Dict[str, Any]], scores: Dict[int, float]) -> None:
    m = ba.compute_group_metrics("vlm2a", trajs, scores)
    text = ba.render_report({"vlm2a": m}, runs_dir="runs", date="2026-07-07")
    assert "vlm2a" in text
    assert "2026-07-07" in text
    assert "S_s" in text  # S_s→r column present
    assert "Spearman" in text


def test_render_unknown_tools_do_not_break_table() -> None:
    # Real runs contain malformed tool names (e.g. "greep", one with an embedded
    # newline). They must be folded into an "other" column, never raw headers.
    weird = _traj(0, ["search", "greep", "Arbitrary\n</parameter", "read_section"])
    m = ba.compute_group_metrics("g1", [weird], {0: 0.9})
    text = ba.render_report({"g1": m}, runs_dir="runs", date="2026-07-07")
    assert "其他" in text
    assert "Arbitrary\n</parameter" not in text  # newline-bearing name sanitized
    header = next(l for l in text.splitlines() if l.startswith("| group | search"))
    assert "greep" not in header
