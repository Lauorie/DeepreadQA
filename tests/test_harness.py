from deepread_sdk import Reader
from deepreadqa.config import Config, Endpoint
from deepreadqa.harness import DeepreadQA
from deepreadqa.llm import LLMResponse
from deepreadqa.retrieval import SearchIndex


class _FakeToolCall:
    def __init__(self, cid, name, arguments):
        self.id = cid
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _FakeLLM:
    """Scripts: turn 0 -> search; turn 1 -> read_section; turn 2 -> final answer."""

    def __init__(self):
        self.total_tokens = 0
        self._turn = 0

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self._turn += 1
        if self._turn == 1:
            tc = _FakeToolCall("c1", "search", '{"queries": ["ALE coupling"]}')
            return LLMResponse("", [tc], "tool_calls", 5, raw_message=_Msg([tc]))
        if self._turn == 2:
            tc = _FakeToolCall("c2", "read_section",
                               '{"doc_id": "en_paper.md", "section": "2. Method"}')
            return LLMResponse("", [tc], "tool_calls", 5, raw_message=_Msg([tc]))
        return LLMResponse("最终答案：使用 ALE 耦合方案。 (en_paper.md / 2. Method)",
                           [], "stop", 5, raw_message=_Msg([]))


class _Msg:
    def __init__(self, tool_calls):
        self.role = "assistant"
        self.content = ""
        self.tool_calls = tool_calls


def _cfg() -> Config:
    return Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                  concise_compose=False)


def test_loop_terminates_with_answer(populated_store):
    reader = Reader(populated_store)
    qa = DeepreadQA(_cfg(), llm=_FakeLLM(), reader=reader, index=SearchIndex(reader))
    res = qa.answer("FSI 仿真用什么耦合方案？")
    assert "ALE" in res.answer
    assert res.iterations == 3
    assert res.forced_final is False
    assert "en_paper.md" in res.seen_docs
    assert any(c["tool"] == "read_section" for c in res.tool_calls)
