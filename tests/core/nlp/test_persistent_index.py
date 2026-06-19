"""Persistent-index parity: BM25/TFIDF serialize + incremental == fresh build."""

from tenets.core.nlp.bm25 import BM25Calculator
from tenets.core.nlp.keyword_extractor import TFIDFCalculator

DOCS = [
    ("a.py", "import os\nclass FileHandler:\n    def read_file(self, path): return open(path)"),
    ("b.py", "from pathlib import Path\ndef write_file(p, data): p.write_text(data)"),
    (
        "c.py",
        "class FileHandler:\n    cache = {}\n    def read_file(self): pass\n    def reset(self): ...",
    ),
]
QUERY = "FileHandler read file"


def test_bm25_roundtrip_score_parity():
    calc = BM25Calculator()
    calc.build_corpus(DOCS)
    before = calc.get_scores(QUERY)
    calc2 = BM25Calculator.from_state(calc.to_state())
    assert calc2.get_scores(QUERY) == before
    assert calc2.document_count == calc.document_count
    assert dict(calc2.document_frequency) == dict(calc.document_frequency)
    assert calc2.document_lengths == calc.document_lengths
    assert calc2.average_doc_length == calc.average_doc_length
    assert calc2.vocabulary == calc.vocabulary


def test_bm25_total_length_invariant():
    calc = BM25Calculator()
    calc.build_corpus(DOCS)
    assert calc._total_length == sum(calc.document_lengths.values())
    calc.add_document("d.py", "another reader read file handler module")
    assert calc._total_length == sum(calc.document_lengths.values())
    assert calc.average_doc_length == calc._total_length / max(1, calc.document_count)


def test_bm25_incremental_equals_build():
    built = BM25Calculator()
    built.build_corpus(DOCS)
    incr = BM25Calculator()
    for d, t in DOCS:
        incr.add_document(d, t)
    assert incr.document_count == built.document_count
    assert dict(incr.document_frequency) == dict(built.document_frequency)
    assert incr.document_lengths == built.document_lengths
    assert incr.average_doc_length == built.average_doc_length
    assert incr.get_scores(QUERY) == built.get_scores(QUERY)


def test_bm25_remove_equals_build_without():
    full = BM25Calculator()
    full.build_corpus(DOCS)
    full._remove_document("b.py")
    without = BM25Calculator()
    without.build_corpus([d for d in DOCS if d[0] != "b.py"])
    assert full.document_count == without.document_count
    assert dict(full.document_frequency) == dict(without.document_frequency)
    assert full.document_lengths == without.document_lengths
    assert full.vocabulary == without.vocabulary
    assert full.average_doc_length == without.average_doc_length
    assert full.get_scores(QUERY) == without.get_scores(QUERY)


def test_tfidf_roundtrip_similarity_parity():
    calc = TFIDFCalculator()
    calc.build_corpus(DOCS)
    before = [calc.compute_similarity(QUERY, d) for d, _ in DOCS]
    calc2 = TFIDFCalculator.from_state(calc.to_state())
    after = [calc2.compute_similarity(QUERY, d) for d, _ in DOCS]
    assert after == before
    assert calc2.document_count == calc.document_count
    assert dict(calc2.document_frequency) == dict(calc.document_frequency)
    assert calc2.vocabulary == calc.vocabulary


def test_tfidf_remove_updates_corpus_stats():
    full = TFIDFCalculator()
    full.build_corpus(DOCS)
    full.remove_document("b.py")
    without = TFIDFCalculator()
    without.build_corpus([d for d in DOCS if d[0] != "b.py"])
    assert full.document_count == without.document_count
    assert dict(full.document_frequency) == dict(without.document_frequency)
    assert full.vocabulary == without.vocabulary
    assert "b.py" not in full.document_vectors
