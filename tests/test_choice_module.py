"""Tests for the ported deepreadqa.choice module (no LLM, no store)."""
from types import SimpleNamespace

from deepreadqa import ChoiceQA
from deepreadqa.choice import parse_letter
from deepreadqa.choice_prompts import format_options


class _StubLLM:
    """Terminates immediately: no tool calls, then a compose answer."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        content = ("根据证据，B 项与原文一致。\n答案：B" if self.calls > 1
                   else "初步判断选 B")
        return SimpleNamespace(content=content, tool_calls=[],
                               finish_reason="stop", total_tokens=100,
                               raw_message=None)


class _StubReader:
    def list_docs(self):
        return []


def test_parse_letter_priorities():
    assert parse_letter("分析……\n答案：C") == "C"
    assert parse_letter("答案：A …… 最终答案：D") == "D"  # last marker wins
    assert parse_letter("正确的是（B）") == "B"
    assert parse_letter("所以选 A") == "A"
    assert parse_letter("Option C matches the evidence") == "C"
    assert parse_letter("完全无字母的文本") is None
    assert parse_letter("") is None


def test_format_options_renders_in_order():
    block = format_options({"B": "乙", "A": "甲", "D": "丁", "C": "丙"})
    assert block == "A. 甲\nB. 乙\nC. 丙\nD. 丁"


def test_answer_choice_fast_path_counts_tokens(monkeypatch):
    from deepreadqa.config import Config, Endpoint

    cfg = Config(endpoint=Endpoint("t", "http://x", "k", "m", True))
    llm = _StubLLM()
    from deepreadqa.retrieval import SearchIndex

    qa = ChoiceQA(cfg, llm=llm, reader=_StubReader(),
                  index=SearchIndex(_StubReader()))
    res = qa.answer_choice("哪个正确？", {"A": "甲", "B": "乙", "C": "丙", "D": "丁"})
    assert res.answer == "B"
    assert res.abstained is False
    assert res.total_tokens == 200  # agent turn + compose turn, local tally
    assert res.iterations == 1
    assert res.error is None
