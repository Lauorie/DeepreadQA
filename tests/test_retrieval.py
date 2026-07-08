from deepread_sdk import Reader
from deepreadqa.retrieval import SearchIndex, _chunk, tokenize_mixed


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
    assert len(ids) == len(set(ids))
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_chinese(populated_store):
    idx = SearchIndex(Reader(populated_store))
    hits = idx.search("家族企业 创新", top_k=3)
    assert hits[0].doc_id == "zh_paper.md"


class _StubReader:
    """Minimal Reader stand-in exposing only list_docs(), for index tests."""

    def __init__(self, docs):
        self._docs = docs

    def list_docs(self):
        return self._docs


def _doc(doc_id, content, title="t", tldr="", keywords=None):
    return {"doc_id": doc_id, "title": title, "tldr": tldr,
            "keywords": keywords or [], "abstract": "", "language": "en",
            "sections": [{"name": "Full Document", "idx": 0, "tldr": "", "content": content}]}


def test_rare_term_in_large_single_section_is_retrievable_with_snippet():
    # A long, heading-less doc (a textbook PDF dump) with a rare term buried deep
    # inside must still be retrievable with a focused snippet — chunk indexing
    # keeps the match local instead of letting BM25 length-norm bury the section.
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. " * 3000
    big = filler + " Jaumann objective stress rate corrected via the spin tensor. " + filler
    docs = [_doc("big.md", big, title="ALE textbook")]
    for i in range(10):
        docs.append(_doc(f"d{i}.md", "unrelated finance pension market content here"))
    idx = SearchIndex(_StubReader(docs))
    hits = idx.search("Jaumann objective stress rate spin tensor", top_k=5)
    ids = [h.doc_id for h in hits]
    assert "big.md" in ids, f"buried-term doc should be retrieved, got {ids}"
    big_hit = next(h for h in hits if h.doc_id == "big.md")
    assert "jaumann" in big_hit.snippet.lower(), "snippet must carry the matched passage"


def test_chunk_returns_start_offsets():
    text = "  abcd efgh ijkl"
    out = _chunk(text, 6, 2)
    assert out and isinstance(out[0], tuple)
    for start, ch in out:
        assert text[start:start + len(ch)] == ch
    # chunk strings byte-identical to the offset-less chunking (BM25 invariance)
    stripped = text.strip()
    expected = [stripped[i:i + 6] for i in range(0, len(stripped), 4)]
    assert [ch for _, ch in out] == expected


def test_chunk_short_and_blank_text():
    assert _chunk("  hi  ", 100, 10) == [(2, "hi")]
    assert _chunk("   \n ", 100, 10) == []


def test_search_hit_carries_para_anchor():
    # engineer the chunking so the best chunk starts exactly at paragraph 2
    p1 = "alpha beta gamma delta epsilon zeta"
    p2 = "the Jaumann objective stress rate is corrected via the spin tensor"
    p3 = "omega filler tail paragraph with unrelated words entirely"
    content = f"{p1}\n\n{p2}\n\n{p3}"
    step = len(p1) + 2  # second chunk boundary lands on p2's first char
    idx = SearchIndex(_StubReader([_doc("d.md", content)]),
                      chunk_chars=step + 10, overlap=10)
    hits = idx.search("Jaumann objective stress rate spin tensor", top_k=1)
    assert hits
    assert hits[0].para_idx == 2


def test_search_hit_para_idx_none_without_content_chunk():
    doc = {"doc_id": "meta.md", "title": "pension market study",
           "tldr": "pension finance overview", "keywords": ["pension"],
           "abstract": "", "language": "en",
           "sections": [{"name": "S", "idx": 0, "tldr": "", "content": ""}]}
    filler = _doc("other.md", "alpha beta gamma entirely unrelated words")
    hits = SearchIndex(_StubReader([doc, filler])).search("pension market", top_k=1)
    assert hits and hits[0].doc_id == "meta.md"
    assert hits[0].para_idx is None
