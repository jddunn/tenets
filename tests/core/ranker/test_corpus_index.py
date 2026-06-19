"""CorpusIndex: build-once, then re-tokenize only changed/deleted files."""

from tenets.core.nlp.bm25 import BM25Calculator
from tenets.core.nlp.keyword_extractor import TFIDFCalculator
from tenets.core.ranking.corpus_index import CorpusIndex

DOCS = [
    ("a.py", "class FileHandler:\n    def read_file(self): return open(self.path)"),
    ("b.py", "def write_file(p, data): p.write_text(data)"),
]


def _factories():
    return (lambda: BM25Calculator(), lambda: TFIDFCalculator())


def _spy_add_document():
    """Patch BM25Calculator.add_document to record which docs get (re)tokenized."""
    seen = []
    orig = BM25Calculator.add_document

    def spy(self, doc_id, text):
        seen.append(doc_id)
        return orig(self, doc_id, text)

    return seen, orig, spy


def test_first_build_populates_and_ranks(tmp_path):
    mk_b, mk_t = _factories()
    bm25, tfidf = CorpusIndex(cache_dir=tmp_path).build(
        "root", DOCS, make_bm25=mk_b, make_tfidf=mk_t
    )
    assert bm25.document_count == 2 and tfidf.document_count == 2
    top = bm25.get_scores("FileHandler read file")
    assert top and top[0][0] == "a.py"


def test_unchanged_rebuild_skips_tokenization(tmp_path):
    idx = CorpusIndex(cache_dir=tmp_path)
    mk_b, mk_t = _factories()
    idx.build("root", DOCS, make_bm25=mk_b, make_tfidf=mk_t)
    seen, orig, spy = _spy_add_document()
    BM25Calculator.add_document = spy
    try:
        bm25, _ = idx.build("root", DOCS, make_bm25=mk_b, make_tfidf=mk_t)
    finally:
        BM25Calculator.add_document = orig
    assert seen == []  # nothing changed → zero re-tokenization
    assert bm25.document_count == 2


def test_changed_file_only_reindexes_it(tmp_path):
    idx = CorpusIndex(cache_dir=tmp_path)
    mk_b, mk_t = _factories()
    idx.build("root", DOCS, make_bm25=mk_b, make_tfidf=mk_t)
    edited = [("a.py", "class FileHandler:\n    def read_file(self): pass  # edited"), DOCS[1]]
    seen, orig, spy = _spy_add_document()
    BM25Calculator.add_document = spy
    try:
        bm25, _ = idx.build("root", edited, make_bm25=mk_b, make_tfidf=mk_t)
    finally:
        BM25Calculator.add_document = orig
    assert seen == ["a.py"]  # only the changed file re-tokenized
    assert bm25.document_count == 2


def test_deleted_file_removed(tmp_path):
    idx = CorpusIndex(cache_dir=tmp_path)
    mk_b, mk_t = _factories()
    idx.build("root", DOCS, make_bm25=mk_b, make_tfidf=mk_t)
    bm25, _ = idx.build("root", [DOCS[0]], make_bm25=mk_b, make_tfidf=mk_t)
    assert bm25.document_count == 1
    assert "b.py" not in bm25.document_tokens


def test_cross_process_disk_reload_skips_tokenization(tmp_path):
    mk_b, mk_t = _factories()
    CorpusIndex(cache_dir=tmp_path).build("root", DOCS, make_bm25=mk_b, make_tfidf=mk_t)
    fresh = CorpusIndex(cache_dir=tmp_path)  # new process: empty warm layer, loads disk
    seen, orig, spy = _spy_add_document()
    BM25Calculator.add_document = spy
    try:
        bm25, _ = fresh.build("root", DOCS, make_bm25=mk_b, make_tfidf=mk_t)
    finally:
        BM25Calculator.add_document = orig
    assert seen == []  # reloaded from disk, nothing re-tokenized
    assert bm25.document_count == 2


def test_index_matches_fresh_build_corpus(tmp_path):
    """Parity: indexed BM25 scores == a plain build_corpus on the same docs."""
    mk_b, mk_t = _factories()
    bm25_idx, _ = CorpusIndex(cache_dir=tmp_path).build(
        "root", DOCS, make_bm25=mk_b, make_tfidf=mk_t
    )
    fresh = BM25Calculator()
    fresh.build_corpus(DOCS)
    q = "FileHandler read file write"
    assert bm25_idx.get_scores(q) == fresh.get_scores(q)
