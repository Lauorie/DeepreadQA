import pytest

from deepread_sdk import Reader
from deepreadqa.config import Config, Endpoint
from deepreadqa.harness import DeepreadQA
from deepreadqa.llm import LLMResponse
from deepreadqa.prompts import SYSTEM_PROMPT
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


def test_prune_keeps_evidence_for_named_docs(populated_store):
    reader = Reader(populated_store)
    qa = DeepreadQA(_cfg(), llm=_FakeLLM(), reader=reader, index=SearchIndex(reader))
    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "问题：Q"},
        {"role": "tool", "tool_call_id": "a",
         "content": "SECTION keep.md :: 2. Method (10 tok)\ntldr: t\n---\nKEEP THIS EVIDENCE"},
        {"role": "tool", "tool_call_id": "b",
         "content": "SECTION drop.md :: 1. Intro (10 tok)\ntldr: t\n---\nDROP THIS"},
    ]
    pruned = qa._prune(conv, "小结", ["keep.md"])
    joined = "\n".join(str(m.get("content", "")) for m in pruned)
    assert "KEEP THIS EVIDENCE" in joined
    assert "DROP THIS" not in joined
    # pruned conversation must stay API-valid: no orphan tool messages
    assert all(m["role"] != "tool" for m in pruned)


class _SummarizingLLM:
    """turn1: read_section; turn2: summarize keeping en_paper.md; turn3: final."""

    def __init__(self):
        self._turn = 0
        self.msgs_at_final = None

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self._turn += 1
        if self._turn == 1:
            tc = _FakeToolCall("c1", "read_section",
                               '{"doc_id": "en_paper.md", "section": "2. Method"}')
            return LLMResponse("", [tc], "tool_calls", 5, raw_message=_Msg([tc]))
        if self._turn == 2:
            tc = _FakeToolCall(
                "c2", "summarize",
                '{"summary": "已读方法章", "keep_doc_ids": ["en_paper.md"]}')
            return LLMResponse("", [tc], "tool_calls", 5, raw_message=_Msg([tc]))
        self.msgs_at_final = [dict(m) for m in messages]
        return LLMResponse("done", [], "stop", 5, raw_message=_Msg([]))


def test_summarize_keep_doc_ids_survive_compaction(populated_store):
    reader = Reader(populated_store)
    llm = _SummarizingLLM()
    qa = DeepreadQA(_cfg(), llm=llm, reader=reader, index=SearchIndex(reader))
    res = qa.answer("Q")
    assert res.compactions == 1
    joined = "\n".join(str(m.get("content", "")) for m in llm.msgs_at_final)
    # the read section content of the kept doc survives the compaction
    assert "ALE coupling" in joined


class _BadArgsLLM:
    """turn1: search with truncated JSON args; turn2: final (records feedback)."""

    def __init__(self):
        self._turn = 0
        self.feedback = None

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self._turn += 1
        if self._turn == 1:
            tc = _FakeToolCall("c1", "search", '{"queries": ["ALE"')
            return LLMResponse("", [tc], "tool_calls", 5, raw_message=_Msg([tc]))
        self.feedback = next(m["content"] for m in reversed(messages)
                             if m.get("role") == "tool")
        return LLMResponse("final", [], "stop", 5, raw_message=_Msg([]))


def test_malformed_tool_json_gets_explicit_error_feedback(populated_store):
    reader = Reader(populated_store)
    llm = _BadArgsLLM()
    qa = DeepreadQA(_cfg(), llm=llm, reader=reader, index=SearchIndex(reader))
    res = qa.answer("Q")
    assert res.answer == "final"
    assert "not valid JSON" in llm.feedback


class _NoCounterLLM:
    """Reports per-response usage but exposes NO total_tokens attribute:
    AgentResult.total_tokens must be derived from responses, not client state."""

    def __init__(self):
        self._calls = 0

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self._calls += 1
        if self._calls % 2 == 1:
            tc = _FakeToolCall("c1", "search", '{"queries": ["ALE coupling"]}')
            return LLMResponse("", [tc], "tool_calls", 11, raw_message=_Msg([tc]))
        return LLMResponse("done", [], "stop", 7, raw_message=_Msg([]))


def test_total_tokens_derived_from_responses_per_answer(populated_store):
    reader = Reader(populated_store)
    qa = DeepreadQA(_cfg(), llm=_NoCounterLLM(), reader=reader,
                    index=SearchIndex(reader))
    r1 = qa.answer("Q1")
    r2 = qa.answer("Q2")
    assert r1.total_tokens == 18  # 11 (search turn) + 7 (final turn)
    assert r2.total_tokens == 18  # not cumulative across answers


class _SchemaRecordingLLM:
    """Records the tool schema list the harness passes to chat."""

    def __init__(self):
        self.tools_seen = None

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.tools_seen = tools
        return LLMResponse("done", [], "stop", 5, raw_message=_Msg([]))


def test_default_tool_surface_is_five_tools(populated_store):
    reader = Reader(populated_store)
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 concise_compose=False)  # default disabled_tools
    llm = _SchemaRecordingLLM()
    qa = DeepreadQA(cfg, llm=llm, reader=reader, index=SearchIndex(reader))
    qa.answer("Q")
    names = {t["function"]["name"] for t in llm.tools_seen}
    assert names == {"search", "head", "read_section", "grep", "summarize"}


def test_default_prompts_do_not_reference_disabled_tools():
    from deepreadqa.prompts import COMPOSE_USER_TEMPLATE, SYSTEM_PROMPT
    for name in ("intro", "preview", "read_raw"):
        assert name not in SYSTEM_PROMPT
        assert name not in COMPOSE_USER_TEMPLATE


def test_disabled_tools_removed_from_llm_schemas(populated_store):
    reader = Reader(populated_store)
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 concise_compose=False,
                 disabled_tools=("intro", "preview", "read_raw"))
    llm = _SchemaRecordingLLM()
    qa = DeepreadQA(cfg, llm=llm, reader=reader, index=SearchIndex(reader))
    qa.answer("Q")
    names = {t["function"]["name"] for t in llm.tools_seen}
    assert names == {"search", "head", "read_section", "grep", "summarize"}


class _MsgRecordingLLM:
    """Records the first messages list the harness passes to chat."""

    def __init__(self):
        self.first_messages = None

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        if self.first_messages is None:
            self.first_messages = [dict(m) for m in messages]
        return LLMResponse("done", [], "stop", 5, raw_message=_Msg([]))


def test_system_prompt_byte_identical_when_catalog_off(populated_store):
    reader = Reader(populated_store)
    llm = _MsgRecordingLLM()
    qa = DeepreadQA(_cfg(), llm=llm, reader=reader, index=SearchIndex(reader))
    qa.answer("Q")
    assert llm.first_messages[0]["role"] == "system"
    assert llm.first_messages[0]["content"] == SYSTEM_PROMPT


def test_catalog_mode_appends_full_directory(populated_store):
    reader = Reader(populated_store)
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 concise_compose=False, catalog_in_prompt=True)
    llm = _MsgRecordingLLM()
    qa = DeepreadQA(cfg, llm=llm, reader=reader, index=SearchIndex(reader))
    qa.answer("Q")
    sp = llm.first_messages[0]["content"]
    assert sp.startswith(SYSTEM_PROMPT)
    assert "- en_paper.md | Hydroplaning Simulation Using FSI | stub global tldr" in sp
    for doc_id in ("nested.md", "no_heading.md", "zh_paper.md"):
        assert f"- {doc_id} | " in sp
    # fallback instruction for search-off ablations
    assert "head/read_section" in sp


def test_catalog_mode_raises_when_store_exceeds_max_docs(populated_store):
    reader = Reader(populated_store)
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 concise_compose=False, catalog_in_prompt=True,
                 catalog_max_docs=2)  # fixture store has 4 docs
    with pytest.raises(ValueError, match="catalog"):
        DeepreadQA(cfg, llm=_MsgRecordingLLM(), reader=reader,
                   index=SearchIndex(reader))


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


class _AllMsgRecordingLLM:
    """Records every messages list the harness passes to chat."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.calls.append([dict(m) for m in messages])
        return LLMResponse("done", [], "stop", 5, raw_message=_Msg([]))


def test_answer_lang_en_appends_english_instruction(populated_store):
    from deepreadqa.prompts import ANSWER_LANG_EN_LINE

    reader = Reader(populated_store)
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 concise_compose=True, answer_lang="en")
    llm = _AllMsgRecordingLLM()
    qa = DeepreadQA(cfg, llm=llm, reader=reader, index=SearchIndex(reader))
    qa.answer("Q")
    agent_system = llm.calls[0][0]["content"]
    assert agent_system.startswith(SYSTEM_PROMPT)
    assert ANSWER_LANG_EN_LINE in agent_system
    compose_systems = [c[0]["content"] for c in llm.calls[1:]
                       if c and c[0]["role"] == "system"]
    assert compose_systems, "compose stage should have run"
    assert all(ANSWER_LANG_EN_LINE in s for s in compose_systems)
