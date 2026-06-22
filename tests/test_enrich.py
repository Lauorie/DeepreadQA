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
