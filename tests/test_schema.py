import dataclasses

import pytest

from deepread_sdk.schema import DocRecord, RawSection, SectionRecord, StructuredDoc


def test_raw_section_is_frozen():
    s = RawSection(name="1. Intro", idx=0, content="x", start_pos=0, end_pos=1)
    assert s.name == "1. Intro"
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.name = "changed"  # type: ignore[misc]


def test_doc_record_holds_sections():
    sec = SectionRecord(idx=0, name="1. Intro", tldr="t", token_count=3,
                        start_pos=0, end_pos=10, content="hello")
    doc = DocRecord(
        doc_id="a.md", title="A", language="en", abstract=None, header="",
        tldr="g", keywords=["k1", "k2"], token_count=3, total_characters=5,
        preview="hello", preview_is_truncated=False, raw_md="# A\nhello",
        content_hash="abc", sections=[sec],
    )
    assert doc.sections[0].name == "1. Intro"
    assert doc.keywords == ["k1", "k2"]


def test_structured_doc_shape():
    d = StructuredDoc(title="A", header="hdr", sections=[
        RawSection(name="S", idx=0, content="c", start_pos=0, end_pos=1)])
    assert d.title == "A" and len(d.sections) == 1
