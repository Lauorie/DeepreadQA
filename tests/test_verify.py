"""Axis-② compose verify-repair loop: parser, probes, and harness wiring."""


from deepread_sdk import Reader
from deepreadqa import verify as vf
from deepreadqa.config import Config, Endpoint
from deepreadqa.harness import DeepreadQA
from deepreadqa.llm import LLMError, LLMResponse
from deepreadqa.retrieval import SearchIndex

REVIEW = """缺失要点：
- 未给出拉伸应变阈值的具体数值
- 未说明该阈值属于应变而非应力
补充检索：
- search: HJC 拉伸应变阈值
- grep: 水下爆炸论文.md :: 0.002|拉伸应变
- grep: 缺分隔符的坏行
结论：REVISE"""


class TestParseVerify:
    def test_full_review_parsed(self):
        r = vf.parse_verify(REVIEW)
        assert r.verdict == "REVISE"
        assert len(r.missing) == 2 and "阈值" in r.missing[0]
        assert ("search", "HJC 拉伸应变阈值") in r.probes
        assert ("grep", "水下爆炸论文.md :: 0.002|拉伸应变") in r.probes
        assert len(r.probes) == 2  # malformed grep line dropped

    def test_pass_with_none_marker(self):
        r = vf.parse_verify("缺失要点：\n- 无\n补充检索：\n结论：PASS")
        assert r.verdict == "PASS" and r.missing == [] and r.probes == []

    def test_garbage_defaults_to_pass(self):
        r = vf.parse_verify("这答案看起来还行吧。")
        assert r.verdict == "PASS" and r.missing == [] and r.probes == []


class _SpyBox:
    def __init__(self):
        self.calls = []

    def execute(self, name, args):
        self.calls.append((name, args))
        return f"[{name}] result for {args}"


class TestRunProbes:
    def test_probe_execution_and_arg_shapes(self):
        box = _SpyBox()
        out = vf.run_probes(box, [("search", "HJC 阈值"),
                                  ("grep", "dam.md :: 0.002|拉伸")], max_probes=2)
        assert box.calls[0] == ("search", {"queries": ["HJC 阈值"]})
        assert box.calls[1] == ("grep", {"doc_id": "dam.md",
                                         "patterns": ["0.002", "拉伸"]})
        assert "[search] result" in out and "[grep] result" in out

    def test_probe_budget_cap(self):
        box = _SpyBox()
        vf.run_probes(box, [("search", f"q{i}") for i in range(5)], max_probes=2)
        assert len(box.calls) == 2

    def test_output_char_cap(self):
        class BigBox:
            def execute(self, name, args):
                return "x" * 50000
        out = vf.run_probes(BigBox(), [("search", "q")], max_probes=2,
                            char_cap=1000)
        assert len(out) < 1400


class TestConfigKnob:
    def test_default_off_and_env_on(self, monkeypatch):
        monkeypatch.setenv("AIBERM_API_KEY", "k")
        monkeypatch.delenv("DEEPREAD_VERIFY", raising=False)
        assert Config.from_env().verify_loop is False
        monkeypatch.setenv("DEEPREAD_VERIFY", "1")
        assert Config.from_env().verify_loop is True
        monkeypatch.setenv("DEEPREAD_VERIFY", "off")
        assert Config.from_env().verify_loop is False


# --- harness wiring ---------------------------------------------------------

class _Msg:
    def __init__(self):
        self.role = "assistant"
        self.content = ""
        self.tool_calls = []


class _ScriptedLLM:
    """turn1 agent-final, turn2 compose, turn3 verify, turn4 addendum."""

    def __init__(self, review, fail_verify=False,
                 repair_text="- 拉伸应变阈值为 0.002（dam.md / 失效准则）"):
        self.turns = []
        self.review = review
        self.fail_verify = fail_verify
        self.repair_text = repair_text
        self.total_tokens = 0

    def chat(self, messages, *, tools=None, tool_choice="auto", max_tokens=None):
        self.turns.append(messages)
        n = len(self.turns)
        if n == 1:
            return LLMResponse("草稿答案", [], "stop", 5, raw_message=_Msg())
        if n == 2:
            return LLMResponse("合成答案：阈值未知。", [], "stop", 5, raw_message=_Msg())
        if n == 3:
            if self.fail_verify:
                raise LLMError("verify endpoint down")
            return LLMResponse(self.review, [], "stop", 5, raw_message=_Msg())
        return LLMResponse(self.repair_text, [], "stop", 5, raw_message=_Msg())


def _vcfg(**kw) -> Config:
    return Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                  concise_compose=True, verify_loop=True,
                  compose_evidence_token_cap=2000, **kw)


def _qa(store, llm):
    reader = Reader(store)
    return DeepreadQA(_vcfg(), llm=llm, reader=reader, index=SearchIndex(reader))


def test_revise_path_appends_addendum_never_rewrites(populated_store):
    review = ("缺失要点：\n- 缺具体数值\n补充检索：\n- search: ALE coupling\n"
              "结论：REVISE")
    llm = _ScriptedLLM(review)
    res = _qa(populated_store, llm).answer("耦合方案的刚度阈值是多少？")
    # composed answer preserved VERBATIM as prefix; additions appended after it
    assert res.answer.startswith("合成答案：阈值未知。")
    assert "补充要点" in res.answer and "0.002" in res.answer
    assert len(llm.turns) == 4
    repair_user = llm.turns[3][-1]["content"]
    assert "缺具体数值" in repair_user          # review findings forwarded
    assert "补充证据" in repair_user            # probe results section present


def test_addendum_none_marker_keeps_composed(populated_store):
    review = "缺失要点：\n- 缺数值\n结论：REVISE"
    llm = _ScriptedLLM(review, repair_text="无")
    res = _qa(populated_store, llm).answer("Q?")
    assert res.answer == "合成答案：阈值未知。"


def test_pass_short_circuits_repair(populated_store):
    llm = _ScriptedLLM("缺失要点：\n- 无\n结论：PASS")
    res = _qa(populated_store, llm).answer("Q?")
    assert res.answer == "合成答案：阈值未知。"
    assert len(llm.turns) == 3                  # no repair call

def test_verify_failure_falls_back_to_composed(populated_store):
    llm = _ScriptedLLM("", fail_verify=True)
    res = _qa(populated_store, llm).answer("Q?")
    assert res.answer == "合成答案：阈值未知。"


def test_empty_repair_keeps_composed(populated_store):
    review = "缺失要点：\n- 缺数值\n结论：REVISE"
    llm = _ScriptedLLM(review, repair_text="")
    res = _qa(populated_store, llm).answer("Q?")
    assert res.answer == "合成答案：阈值未知。"


def test_verify_off_is_two_call_legacy_path(populated_store):
    llm = _ScriptedLLM("")
    reader = Reader(populated_store)
    cfg = Config(endpoint=Endpoint("aiberm", "x", "x", "m", True),
                 concise_compose=True, verify_loop=False,
                 compose_evidence_token_cap=2000)
    res = DeepreadQA(cfg, llm=llm, reader=reader,
                     index=SearchIndex(reader)).answer("Q?")
    assert res.answer == "合成答案：阈值未知。"
    assert len(llm.turns) == 2
