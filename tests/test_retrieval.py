from deepread_sdk import Reader
from deepreadqa.retrieval import SearchIndex, tokenize_mixed


def test_tokenize_mixed_bilingual():
    toks = tokenize_mixed("ALE 流固耦合 added-mass")
    assert "ale" in toks
    assert "added" in toks
    assert any("流" in t or t == "流固" for t in toks)


def test_search_finds_relevant_doc(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search("ALE coupling fluid structure interface", top_k=3)
    assert hits
    assert hits[0].doc_id == "en_paper.md"
    assert hits[0].section_name is not None


def test_search_section_hint_points_to_method(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search("ALE coupling scheme", top_k=3)
    top = hits[0]
    assert "Method" in top.section_name


def test_search_many_dedupes(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search_many(["ALE coupling", "fluid structure interaction"], top_k=3)
    ids = [h.doc_id for h in hits]
    assert len(ids) == len(set(ids))  # no duplicate docs


def test_search_chinese(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search("家族企业 创新", top_k=3)
    assert hits[0].doc_id == "zh_paper.md"
