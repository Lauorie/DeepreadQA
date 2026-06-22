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


class _RaisingLLM:
    total_tokens = 0

    def chat(self, *args, **kwargs):
        from deepreadqa.llm import LLMError
        raise LLMError("boom")


def test_llmerror_forces_finish_with_iteration_count(populated_store):
    reader = Reader(populated_store)
    qa = DeepreadQA(_cfg(), llm=_RaisingLLM(), reader=reader, index=SearchIndex(reader))
    res = qa.answer("Q")
    assert res.forced_final is True
    assert res.error is not None
    assert res.iterations == 1


def test_prune_preserves_original_user_question(populated_store):
    reader = Reader(populated_store)
    qa = DeepreadQA(_cfg(), llm=_FakeLLM(), reader=reader, index=SearchIndex(reader))
    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问题：原始问题"},
        {"role": "assistant", "content": "draft"},
        {"role": "tool", "tool_call_id": "t1", "content": "evidence"},
    ]
    pruned = qa._prune(conv, "进度概要")
    assert pruned[0]["role"] == "system"
    assert any(m["role"] == "user" and "原始问题" in m["content"] for m in pruned)
    assert "进度概要" in pruned[-1]["content"]


def test_collect_evidence_prioritizes_latest_and_summary(populated_store):
    reader = Reader(populated_store)
    qa = DeepreadQA(_cfg(), llm=_FakeLLM(), reader=reader, index=SearchIndex(reader))
    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问题：Q"},
        {"role": "tool", "tool_call_id": "a", "content": "OLD broad search results"},
        {"role": "assistant", "content": "进度小结（已压缩上下文）：earlier progress"},
        {"role": "tool", "tool_call_id": "b", "content": "LATEST grep decisive evidence"},
    ]
    ev = qa._collect_evidence(conv)
    assert "LATEST grep decisive evidence" in ev
    assert "进度小结" in ev


def test_collect_evidence_drops_oldest_when_over_budget(populated_store):
    reader = Reader(populated_store)
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 concise_compose=False, compose_evidence_token_cap=20)
    qa = DeepreadQA(cfg, llm=_FakeLLM(), reader=reader, index=SearchIndex(reader))
    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问题：Q"},
        {"role": "tool", "tool_call_id": "a", "content": "OLD " * 50},
        {"role": "tool", "tool_call_id": "b", "content": "LATEST decisive evidence here"},
    ]
    ev = qa._collect_evidence(conv)
    assert "LATEST decisive evidence here" in ev
    assert "OLD" not in ev


def test_collect_evidence_truncates_oversized_latest(populated_store):
    from deepreadqa.config import Config, Endpoint
    from deepread_sdk.tokens import count_tokens
    reader = Reader(populated_store)
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 concise_compose=False, compose_evidence_token_cap=30)
    qa = DeepreadQA(cfg, llm=_FakeLLM(), reader=reader, index=SearchIndex(reader))
    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问题：Q"},
        {"role": "tool", "tool_call_id": "b", "content": "DECISIVE " * 200},
    ]
    ev = qa._collect_evidence(conv)
    assert ev.strip() != ""
    assert "DECISIVE" in ev
    assert count_tokens(ev) <= 40


def test_local_prune_clean_and_bounded(populated_store):
    reader = Reader(populated_store)
    qa = DeepreadQA(_cfg(), llm=_FakeLLM(), reader=reader, index=SearchIndex(reader))
    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问题：原始问题"},
        {"role": "assistant", "content": "a",
         "tool_calls": [{"id": "x", "type": "function",
                         "function": {"name": "grep", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "x", "content": "evidence chunk"},
    ]
    pruned = qa._local_prune(conv)
    assert all(m["role"] != "tool" for m in pruned)
    assert pruned[0]["role"] == "system"
    assert any(m["role"] == "user" and "原始问题" in m["content"] for m in pruned)
    assert "evidence chunk" in pruned[-1]["content"]
