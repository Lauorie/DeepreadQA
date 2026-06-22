from pathlib import Path

from deepread_sdk.structure import detect_language, extract_abstract, recover_structure

FIX = Path(__file__).parent / "fixtures" / "corpus"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_en_paper_title_and_sections():
    doc = recover_structure(_read("en_paper.md"), fallback_title="x")
    assert doc.title == "Hydroplaning Simulation Using FSI"
    names = [s.name for s in doc.sections]
    assert names == ["ABSTRACT", "1. Introduction", "2. Method"]
    assert "ALE coupling" in doc.sections[2].content
    assert "Koishi" in doc.header


def test_zh_paper_level1_sections():
    doc = recover_structure(_read("zh_paper.md"), fallback_title="x")
    assert doc.title == "家族企业创始控制与企业创新投入"
    names = [s.name for s in doc.sections]
    assert names == ["引言", "一、文献回顾"]
    assert "摘要" in doc.header


def test_no_heading_single_section():
    doc = recover_structure(_read("no_heading.md"), fallback_title="plain")
    assert doc.title == "plain"
    assert len(doc.sections) == 1
    assert doc.sections[0].name == "Full Document"
    assert "LS-DYNA" in doc.sections[0].content


def test_nested_subsection_stays_inside_parent():
    doc = recover_structure(_read("nested.md"), fallback_title="x")
    names = [s.name for s in doc.sections]
    assert names == ["Section One", "Section Two"]
    assert "Subsection A" in doc.sections[0].content
    assert "nested text" in doc.sections[0].content


def test_extract_abstract_from_named_section():
    doc = recover_structure(_read("en_paper.md"), fallback_title="x")
    abs = extract_abstract(doc)
    assert abs is not None and "hydroplaning" in abs.lower()


def test_detect_language():
    assert detect_language("这是一段中文文本，关于流固耦合。") == "zh"
    assert detect_language("This is English about FSI.") == "en"


def test_heading_inside_code_fence_ignored():
    text = "# Title\n## Real Section\n```\n## not a heading\ncode\n```\nmore text\n## Second\nx"
    doc = recover_structure(text, fallback_title="x")
    names = [s.name for s in doc.sections]
    assert names == ["Real Section", "Second"]
    assert "not a heading" in doc.sections[0].content


def test_extract_abstract_trailing_colon():
    text = "# T\nhdr\n## Abstract:\nbody text here\n## 1. Intro\ni"
    doc = recover_structure(text, fallback_title="x")
    assert extract_abstract(doc) == "body text here"


def test_extract_abstract_from_header_inline_zh():
    doc = recover_structure(_read("zh_paper.md"), fallback_title="x")
    abs_text = extract_abstract(doc)
    assert abs_text is not None and "创新对家族企业" in abs_text


def test_mismatched_fence_marker_stays_closed():
    text = "# T\n## Real\n```\n~~~ inner\n## not heading\n```\n## Second\nx"
    doc = recover_structure(text, fallback_title="x")
    names = [s.name for s in doc.sections]
    assert names == ["Real", "Second"]
    assert "not heading" in doc.sections[0].content


def test_heading_preserves_internal_trailing_hash():
    doc = recover_structure("# Title\n## C#\nbody\n## Other ##\nx", fallback_title="z")
    names = [s.name for s in doc.sections]
    assert names == ["C#", "Other"]


def test_section_offsets_roundtrip():
    for fname in ["en_paper.md", "zh_paper.md", "nested.md", "no_heading.md"]:
        text = _read(fname)
        doc = recover_structure(text, fallback_title="x")
        for s in doc.sections:
            assert text[s.start_pos:s.end_pos] == s.content


def test_numbered_section_fallback_splits_pdf_dump():
    text = (
        "ALE and FSI Numerical Simulation\nFront matter copyright ISTE Ltd.\n\n"
        "1.1 Introduction\nintro body about ALE methods.\n\n"
        "1.2 Governing equations\ngoverning equation body.\n\n"
        "1.4.5 Stress rates\nThe Jaumann stress rate is an objective rate.\n\n"
        "1.4.7 Mixture theories\nMixture theories handle multi-material cells.\n\n"
        "1.4.7.1 Mean strain rate mixture theory\nmean strain rate conserves energy.\n\n"
        "1.4.7.2 Mean stress mixture theory\nmean stress does not conserve energy.\n"
    )
    doc = recover_structure(text, fallback_title="benson")
    names = [s.name for s in doc.sections]
    assert "1.4.5 Stress rates" in names
    assert "1.4.7 Mixture theories" in names
    assert len(doc.sections) >= 6
    for s in doc.sections:
        assert text[s.start_pos:s.end_pos] == s.content
    sr = next(s for s in doc.sections if s.name == "1.4.5 Stress rates")
    assert "Jaumann" in sr.content


def test_numbered_fallback_not_triggered_below_threshold():
    text = "Plain prose discussing 1.5 times the load over 2.0 m spans. No headings at all."
    doc = recover_structure(text, fallback_title="p")
    assert len(doc.sections) == 1
    assert doc.sections[0].name == "Full Document"


def test_atx_doc_ignores_numbered_fallback():
    text = "# Title\n1.2 Not A Section\nbody\n## Real Section\nx"
    doc = recover_structure(text, fallback_title="z")
    names = [s.name for s in doc.sections]
    assert names == ["Real Section"]
