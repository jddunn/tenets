"""Tests for TF-IDF and BM25 calculators used in ranking / NLP."""

import pytest
from tenets.core.nlp.tfidf import TFIDFCalculator, BM25Calculator


class TestTFIDFCalculator:
    """Test suite for TF-IDF calculator ranking API."""

    def test_initialization(self):
        """Test TF-IDF calculator initialization."""
        calc = TFIDFCalculator(use_stopwords=False)
        assert calc.use_stopwords is False
        assert calc.document_count == 0

        calc_with_stopwords = TFIDFCalculator(use_stopwords=True)
        assert calc_with_stopwords.use_stopwords is True
        assert len(calc_with_stopwords.stopwords) > 0

    def test_tokenize(self):
        """Test tokenization."""
        calc = TFIDFCalculator()
        tokens = calc.tokenize("Hello World Python Programming")

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_add_document(self):
        """Test adding document."""
        calc = TFIDFCalculator()

        vector = calc.add_document("doc1", "Python is a great programming language")

        assert isinstance(vector, dict)
        assert calc.document_count == 1
        assert len(calc.vocabulary) > 0

    def test_compute_similarity(self):
        """Test computing similarity."""
        calc = TFIDFCalculator()

        calc.add_document("doc1", "Python programming")
        calc.add_document("doc2", "Java programming")

        sim = calc.compute_similarity("Python code", "doc1")

        assert isinstance(sim, float)
        assert 0 <= sim <= 1

    def test_get_top_terms(self):
        """Test getting top terms."""
        calc = TFIDFCalculator()

        calc.add_document("doc1", "Python Python Python Java")
        top_terms = calc.get_top_terms("doc1", n=2)

        assert isinstance(top_terms, list)
        assert len(top_terms) <= 2
        if top_terms:
            assert isinstance(top_terms[0], tuple)
            assert len(top_terms[0]) == 2

    def test_build_corpus(self):
        """Test building corpus."""
        calc = TFIDFCalculator()

        documents = [("doc1", "Python programming"), ("doc2", "Java programming")]

        calc.build_corpus(documents)

        assert calc.document_count == 2
        assert "doc1" in calc.document_vectors
        assert "doc2" in calc.document_vectors


class TestBM25Calculator:
    """Test suite for BM25 calculator ranking API."""

    def test_initialization(self):
        """Test BM25 initialization."""
        calc = BM25Calculator(k1=1.5, b=0.8)

        assert calc._calculator.k1 == 1.5
        assert calc._calculator.b == 0.8

    def test_add_document(self):
        """Test adding document."""
        calc = BM25Calculator()

        calc.add_document("doc1", "Information retrieval")

        assert "doc1" in calc.document_tokens

    def test_search(self):
        """Test BM25 search."""
        calc = BM25Calculator()

        calc.add_document("doc1", "Python tutorial")
        calc.add_document("doc2", "Java guide")

        results = calc.search("Python", top_k=1)

        assert len(results) <= 1
        if results:
            assert results[0][0] == "doc1"
