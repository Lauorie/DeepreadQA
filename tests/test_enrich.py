import json

from deepread_sdk.enrich import Enricher, parse_global_response
from deepread_sdk.schema import RawSection, StructuredDoc


class FakeClient:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0) if self._responses else ""


def _doc() -> StructuredDoc:
    return StructuredDoc(title="T", header="", sections=[
        RawSection(name="1. Intro", idx=0, content="intro about FSI", start_pos=0, end_pos=10),
        RawSection(name="2. Method", idx=1, content="ALE method details", start_pos=10, end_pos=20),
    ])


def test_parse_global_response_strict_json():
    raw = '{"tldr": "a summary", "keywords": ["fsi", "ale", "added mass"]}'
    tldr, kws = parse_global_response(raw)
    assert tldr == "a summary"
    assert kws == ["fsi", "ale", "added mass"]


def test_parse_global_response_embedded_json():
    raw = 'Sure! Here it is:\n```json\n{"tldr":"x","keywords":["k1","k2"]}\n```\nDone.'
    tldr, kws = parse_global_response(raw)
    assert tldr == "x" and kws == ["k1", "k2"]


def test_parse_global_response_garbage_falls_back():
    tldr, kws = parse_global_response("totally not json")
    assert tldr == "totally not json"
    assert kws == []


def test_enrich_document_happy_path():
    client = FakeClient([
        json.dumps({"tldr": "global summary", "keywords": ["fsi", "ale"]}),
        "intro one-liner",
        "method one-liner",
    ])
    enr = Enricher(client)
    g, kws, secs = enr.enrich_document("T", _doc(), "en")
    assert g == "global summary"
    assert kws == ["fsi", "ale"]
    assert secs == ["intro one-liner", "method one-liner"]
    assert len(client.calls) == 3


def test_enrich_document_resilient_to_empty_llm():
    client = FakeClient(["", "", ""])  # LLM returns nothing
    enr = Enricher(client)
    g, kws, secs = enr.enrich_document("T", _doc(), "en")
    # falls back to deterministic content-derived tldr; never crashes
    assert isinstance(g, str) and g
    assert len(secs) == 2
    assert all(isinstance(s, str) and s for s in secs)


def test_parse_global_response_keywords_as_string():
    tldr, kws = parse_global_response('{"tldr":"t","keywords":"a, b; c"}')
    assert tldr == "t"
    assert kws == ["a", "b", "c"]


def test_enrich_document_empty_sections():
    client = FakeClient(['{"tldr":"g","keywords":["k"]}'])
    enr = Enricher(client)
    g, kws, secs = enr.enrich_document(
        "T", StructuredDoc(title="T", header="", sections=[]), "en")
    assert g == "g" and kws == ["k"] and secs == []


def test_parse_global_response_truncated_json_no_raw_blob():
    raw = '{"tldr": "Concise summary here", "keywords": ["a", "b"'
    tldr, kws = parse_global_response(raw)
    assert tldr == "Concise summary here"
    assert "{" not in tldr


def test_parse_global_response_trailing_comma():
    raw = '{"tldr": "S", "keywords": ["a", "b",],}'
    tldr, kws = parse_global_response(raw)
    assert tldr == "S"
    assert kws == ["a", "b"]


def test_parse_global_response_fenced_with_trailing_prose():
    raw = 'Here you go:\n```json\n{"tldr": "T", "keywords": ["k"]}\n```\nHope that helps!'
    tldr, kws = parse_global_response(raw)
    assert tldr == "T"
    assert kws == ["k"]


class _RaisingClient:
    def complete(self, system, user):
        raise RuntimeError("network down")


def test_enrich_document_survives_raising_client():
    enr = Enricher(_RaisingClient())
    doc = StructuredDoc(title="T", header="", sections=[
        RawSection(name="S1", idx=0, content="content one", start_pos=0, end_pos=1)])
    g, kws, secs = enr.enrich_document("T", doc, "en")
    assert isinstance(g, str) and g
    assert secs and all(s for s in secs)


def test_parse_global_response_nonstring_tldr_yields_empty():
    tldr, kws = parse_global_response('{"tldr": {"summary": "x"}, "keywords": ["k"]}')
    assert tldr == ""
    assert kws == ["k"]


def test_parse_global_response_truncated_keywords_string():
    raw = '{"tldr": "foo", "keywords": "CAE, finite element; fatigue"'
    tldr, kws = parse_global_response(raw)
    assert tldr == "foo"
    assert kws == ["CAE", "finite element", "fatigue"]


def test_parse_global_response_unparseable_braces_not_blob():
    tldr, kws = parse_global_response("{not valid json at all}")
    assert tldr == ""
    assert kws == []


def test_parse_global_response_plain_prose_passthrough():
    tldr, kws = parse_global_response("This paper proposes a new FSI coupling scheme.")
    assert tldr == "This paper proposes a new FSI coupling scheme."
    assert kws == []


def test_parse_global_response_prose_prefixed_open_brace_not_blob():
    tldr, kws = parse_global_response('Here is the JSON: {"foo":')
    assert tldr == ""
    assert kws == []


def test_parse_global_response_prose_with_citation_brackets_kept():
    tldr, kws = parse_global_response("MemoRAG [1] improves long-context RAG.")
    assert tldr == "MemoRAG [1] improves long-context RAG."
    assert kws == []
