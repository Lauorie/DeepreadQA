"""Tests for blank-line paragraph segmentation (fence-aware).

This is the single coordinate system shared by search's ~¶ anchors and
read_section's start_para/end_para paging.
"""
from deepreadqa.paragraphs import paragraph_spans, split_paragraphs


def test_split_on_blank_lines():
    text = "one\n\ntwo\n   \nthree"
    assert split_paragraphs(text) == ["one", "two", "three"]


def test_multiline_paragraph_not_split_on_single_newline():
    text = "line a\nline b\n\nnext para"
    assert split_paragraphs(text) == ["line a\nline b", "next para"]


def test_no_split_inside_fenced_code_block():
    text = "intro para\n\n```python\nx = 1\n\ny = 2\n```\n\ntail para"
    paras = split_paragraphs(text)
    assert paras == ["intro para", "```python\nx = 1\n\ny = 2\n```", "tail para"]


def test_indented_fence_also_protected():
    text = "a\n\n  ```\ncode\n\nmore code\n  ```\n\nb"
    paras = split_paragraphs(text)
    assert len(paras) == 3
    assert "code\n\nmore code" in paras[1]


def test_unclosed_fence_swallows_rest():
    text = "a\n\n```\ncode\n\nstill code"
    assert split_paragraphs(text) == ["a", "```\ncode\n\nstill code"]


def test_spans_offsets_index_into_original_text():
    text = "  lead ws para\n\nmid\npara\n\n```\nc\n\nc\n```\n\nend"
    spans = paragraph_spans(text)
    assert [p for _, p in spans] == split_paragraphs(text)
    for start, para in spans:
        assert text[start:start + len(para)] == para
    starts = [s for s, _ in spans]
    assert starts == sorted(starts)


def test_empty_and_whitespace_only_yield_no_paragraphs():
    assert split_paragraphs("") == []
    assert split_paragraphs("\n   \n\t\n") == []
