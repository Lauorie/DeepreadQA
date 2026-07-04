"""run_eval CLI: --resume skips answered items, re-runs empty/missing ones."""
import json
from types import SimpleNamespace

import run_eval
from deepreadqa.config import Config, Endpoint
from deepreadqa.harness import AgentResult


class _FakeQA:
    def __init__(self):
        self.asked = []

    def answer(self, question):
        self.asked.append(question)
        return AgentResult(answer=f"ans:{question}", full_answer="", iterations=1,
                           total_tokens=0, compactions=0, forced_final=False,
                           error=None)


def _setup(monkeypatch, tmp_path):
    cases = [{"item_idx": 0, "question": "q0"},
             {"item_idx": 1, "question": "q1"},
             {"item_idx": 2, "question": "q2"}]
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    cfg = Config(endpoint=Endpoint("t", "u", "k", "m", True),
                 eval_file=str(eval_file))
    monkeypatch.setattr(run_eval, "Config",
                        SimpleNamespace(from_env=lambda **kw: cfg))
    fake = _FakeQA()
    monkeypatch.setattr(run_eval, "DeepreadQA", lambda c: fake)
    return fake


def test_resume_skips_answered_and_reruns_empty(tmp_path, monkeypatch):
    fake = _setup(monkeypatch, tmp_path)
    out = tmp_path / "pred.jsonl"
    out.write_text(
        json.dumps({"item_idx": 0, "answer": "old-answer-0"}) + "\n"
        + json.dumps({"item_idx": 1, "answer": ""}) + "\n",
        encoding="utf-8")
    run_eval.main(["--output", str(out), "--resume"])
    assert fake.asked == ["q1", "q2"]  # 0 kept; empty 1 re-run; missing 2 run
    lines = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    by_idx = {r["item_idx"]: r["answer"] for r in lines}
    assert len(lines) == 3
    assert by_idx == {0: "old-answer-0", 1: "ans:q1", 2: "ans:q2"}
    # rich log gains records only for the re-run items
    rich = out.with_suffix(".rich.jsonl")
    rich_idx = [json.loads(ln)["item_idx"]
                for ln in rich.read_text(encoding="utf-8").splitlines()]
    assert rich_idx == [1, 2]


def test_without_resume_reruns_everything(tmp_path, monkeypatch):
    fake = _setup(monkeypatch, tmp_path)
    out = tmp_path / "pred.jsonl"
    out.write_text(json.dumps({"item_idx": 0, "answer": "old"}) + "\n",
                   encoding="utf-8")
    run_eval.main(["--output", str(out)])
    assert fake.asked == ["q0", "q1", "q2"]


def test_resume_preserves_answers_outside_selection(tmp_path, monkeypatch):
    # resuming with a narrower selection must not drop unrelated answered items
    _setup(monkeypatch, tmp_path)
    out = tmp_path / "pred.jsonl"
    out.write_text(json.dumps({"item_idx": 99, "answer": "keep-me"}) + "\n",
                   encoding="utf-8")
    run_eval.main(["--output", str(out), "--resume", "--ids", "0"])
    lines = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()]
    by_idx = {r["item_idx"]: r["answer"] for r in lines}
    assert by_idx == {0: "ans:q0", 99: "keep-me"}


def test_resume_with_no_existing_output(tmp_path, monkeypatch):
    fake = _setup(monkeypatch, tmp_path)
    out = tmp_path / "fresh.jsonl"
    run_eval.main(["--output", str(out), "--resume"])
    assert fake.asked == ["q0", "q1", "q2"]
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3
