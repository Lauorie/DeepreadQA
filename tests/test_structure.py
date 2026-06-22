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
