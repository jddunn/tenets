"""Tests for keyword extraction utilities.

This module provides comprehensive tests for the keyword extraction functionality,
including RAKE, YAKE, TF-IDF, and BM25 algorithms. Tests cover basic extraction,
scoring, edge cases, and Python version compatibility.
"""

from unittest.mock import Mock, patch

from tenets.core.nlp.keyword_extractor import (
    BM25Calculator,
    KeywordExtractor,
    TFIDFCalculator,
    TFIDFExtractor,
)


class TestKeywordExtractor:
    """Test suite for KeywordExtractor."""

    def test_initialization(self):
        """Test KeywordExtractor initialization with default settings."""
        extractor = KeywordExtractor(use_rake=True, use_yake=True, use_stopwords=True)
        assert extractor.use_stopwords is True
        assert extractor.stopword_set == "prompt"
        assert extractor.language == "en"

    def test_initialization_without_rake(self):
        """Test KeywordExtractor initialization with RAKE disabled."""
        extractor = KeywordExtractor(use_rake=False, use_yake=False)
        assert extractor.use_rake is False
        assert extractor.use_yake is False
        # Should still work with fallback methods

    def test_extract_basic(self):
        """Test basic keyword extraction."""
        extractor = KeywordExtractor(use_yake=False)  # Use fallback
        text = (
            "Python programming is great. Python is powerful for data science and machine learning."
        )

        keywords = extractor.extract(text, max_keywords=5)
        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        assert any("python" in kw.lower() for kw in keywords)

    def test_extract_with_scores(self):
        """Test keyword extraction with scores."""
        extractor = KeywordExtractor(use_yake=False)
        text = "Machine learning algorithms process data efficiently."

        keywords = extractor.extract(text, max_keywords=3, include_scores=True)
        assert isinstance(keywords, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in keywords)
        assert all(isinstance(item[1], float) for item in keywords)

    def test_extract_empty_text(self):
        """Test extraction with empty text."""
        extractor = KeywordExtractor()
        keywords = extractor.extract("", max_keywords=5)
        assert keywords == []

    @patch("tenets.core.nlp.keyword_extractor.yake")
    def test_yake_extraction(self, mock_yake):
        """Test YAKE extraction when available."""
        mock_extractor = Mock()
        mock_extractor.extract_keywords.return_value = [("python", 0.1), ("programming", 0.2)]
        mock_yake.KeywordExtractor.return_value = mock_extractor

        extractor = KeywordExtractor(use_yake=True, use_rake=False)  # Disable RAKE to test YAKE
        keywords = extractor.extract("Python programming", max_keywords=2)

        # YAKE scores are inverted (lower is better)
        assert len(keywords) > 0

    @patch("tenets.core.nlp.keyword_extractor.Rake")
    def test_rake_extraction(self, mock_rake_class):
        """Test RAKE extraction when available."""
        mock_rake = Mock()
        mock_rake.extract_keywords_from_text.return_value = None
        mock_rake.get_ranked_phrases_with_scores.return_value = [
            (8.5, "machine learning"),
            (4.0, "python programming"),
            (2.0, "data"),
        ]
        mock_rake_class.return_value = mock_rake

        with patch("tenets.core.nlp.keyword_extractor.RAKE_AVAILABLE", True):
            extractor = KeywordExtractor(use_rake=True, use_yake=False)
            keywords = extractor.extract(
                "Machine learning with Python programming and data", max_keywords=3
            )

            assert len(keywords) == 3
            assert "machine learning" in keywords
            assert "python programming" in keywords
            mock_rake.extract_keywords_from_text.assert_called_once()

    @patch("tenets.core.nlp.keyword_extractor.Rake")
    def test_rake_extraction_with_scores(self, mock_rake_class):
        """Test RAKE extraction with scores."""
        mock_rake = Mock()
        mock_rake.extract_keywords_from_text.return_value = None
        mock_rake.get_ranked_phrases_with_scores.return_value = [
            (10.0, "natural language processing"),
            (7.5, "deep learning"),
            (5.0, "neural networks"),
        ]
        mock_rake_class.return_value = mock_rake

        with patch("tenets.core.nlp.keyword_extractor.RAKE_AVAILABLE", True):
            extractor = KeywordExtractor(use_rake=True, use_yake=False)
            keywords = extractor.extract(
                "Natural language processing with deep learning and neural networks",
                max_keywords=3,
                include_scores=True,
            )

            assert len(keywords) == 3
            assert all(isinstance(k, tuple) and len(k) == 2 for k in keywords)
            assert keywords[0][0] == "natural language processing"
            assert keywords[0][1] == 1.0  # Should be normalized to 1.0 for highest score
            assert 0 <= keywords[2][1] <= 1.0  # All scores should be normalized

    def test_rake_fallback_to_yake(self):
        """Test fallback from RAKE to YAKE when RAKE fails."""
        with patch("tenets.core.nlp.keyword_extractor.RAKE_AVAILABLE", True):
            with patch("tenets.core.nlp.keyword_extractor.YAKE_AVAILABLE", True):
                with patch("tenets.core.nlp.keyword_extractor.Rake") as mock_rake:
                    # Make RAKE raise an exception
                    mock_rake.return_value.extract_keywords_from_text.side_effect = Exception(
                        "RAKE failed"
                    )

                    with patch("tenets.core.nlp.keyword_extractor.yake") as mock_yake:
                        mock_extractor = Mock()
                        mock_extractor.extract_keywords.return_value = [("fallback", 0.1)]
                        mock_yake.KeywordExtractor.return_value = mock_extractor

                        extractor = KeywordExtractor(use_rake=True, use_yake=True)
                        keywords = extractor.extract("Test text", max_keywords=1)

                        # Should have attempted RAKE and fallen back to YAKE
                        assert len(keywords) > 0

    def test_python_313_compatibility(self):
        """Test that YAKE is disabled on Python 3.13+."""
        with patch("tenets.core.nlp.keyword_extractor.sys.version_info", (3, 13, 0)):
            with patch("tenets.core.nlp.keyword_extractor.YAKE_AVAILABLE", False):
                extractor = KeywordExtractor(use_yake=True)
                # YAKE should be disabled even if requested
                assert extractor.use_yake is False
                assert extractor.yake_extractor is None

    def test_rake_with_empty_results(self):
        """Test RAKE handling when no keywords are found."""
        with patch("tenets.core.nlp.keyword_extractor.Rake") as mock_rake_class:
            mock_rake = Mock()
            mock_rake.extract_keywords_from_text.return_value = None
            mock_rake.get_ranked_phrases_with_scores.return_value = []
            mock_rake_class.return_value = mock_rake

            with patch("tenets.core.nlp.keyword_extractor.RAKE_AVAILABLE", True):
                extractor = KeywordExtractor(use_rake=True, use_yake=False)
                keywords = extractor.extract("a the of", max_keywords=5)

                # Should fallback to TF-IDF/frequency methods
                assert isinstance(keywords, list)


class TestTFIDFCalculator:
    """Test suite for TF-IDF calculator."""

    def test_initialization(self):
        """Test TF-IDF calculator initialization."""
        calc = TFIDFCalculator(use_stopwords=False)
        assert calc.document_count == 0
        assert len(calc.vocabulary) == 0

    def test_tokenize(self):
        """Test tokenization."""
        calc = TFIDFCalculator()
        tokens = calc.tokenize("Hello world, this is a test!")
        assert isinstance(tokens, list)
        assert "hello" in [t.lower() for t in tokens]
        assert "world" in [t.lower() for t in tokens]

    def test_add_document(self):
        """Test adding documents."""
        calc = TFIDFCalculator()

        vector = calc.add_document("doc1", "Python programming is fun")
        assert isinstance(vector, dict)
        assert calc.document_count == 1
        assert "python" in calc.vocabulary or "programming" in calc.vocabulary

    def test_compute_similarity(self):
        """Test similarity computation."""
        calc = TFIDFCalculator()

        calc.add_document("doc1", "Python programming language")
        calc.add_document("doc2", "Java programming language")
        calc.add_document("doc3", "Natural language processing")

        sim1 = calc.compute_similarity("Python code", "doc1")
        sim2 = calc.compute_similarity("Python code", "doc2")

        # doc1 should be more similar to "Python code" than doc2
        assert sim1 > sim2

    def test_build_corpus(self):
        """Test building corpus from documents."""
        calc = TFIDFCalculator()

        documents = [
            ("doc1", "Machine learning algorithms"),
            ("doc2", "Deep learning neural networks"),
            ("doc3", "Data science and analytics"),
        ]

        calc.build_corpus(documents)

        assert calc.document_count == 3
        assert len(calc.vocabulary) > 0
        assert "doc1" in calc.document_vectors

    def test_get_top_terms(self):
        """Test getting top terms from document."""
        calc = TFIDFCalculator()

        calc.add_document("doc1", "Python Python Python Java")
        top_terms = calc.get_top_terms("doc1", n=2)

        assert isinstance(top_terms, list)
        assert len(top_terms) <= 2
        # Python should be the top term due to frequency
        if top_terms:
            assert "python" in top_terms[0][0].lower()


class TestBM25Calculator:
    """Test suite for BM25 calculator."""

    def test_initialization(self):
        """Test BM25 initialization."""
        calc = BM25Calculator(k1=1.5, b=0.8)
        assert calc.k1 == 1.5
        assert calc.b == 0.8
        assert calc.document_count == 0

    def test_add_document(self):
        """Test adding documents to BM25."""
        calc = BM25Calculator()

        calc.add_document("doc1", "Information retrieval system")

        assert calc.document_count == 1
        assert "doc1" in calc.document_tokens
        assert calc.average_doc_length > 0

    def test_search(self):
        """Test BM25 search."""
        calc = BM25Calculator()

        # Add documents
        calc.add_document("doc1", "Python programming tutorial")
        calc.add_document("doc2", "Java programming guide")
        calc.add_document("doc3", "Python data science")

        # Search
        results = calc.search("Python tutorial", top_k=2)

        assert isinstance(results, list)
        assert len(results) <= 2
        if results:
            # doc1 should rank high for "Python tutorial"
            assert results[0][0] in ["doc1", "doc3"]

    def test_score_document(self):
        """Test document scoring."""
        calc = BM25Calculator()

        calc.add_document("doc1", "Machine learning algorithms")
        calc.add_document("doc2", "Deep learning networks")

        query_tokens = calc.tokenize("machine learning")
        score1 = calc.score_document(query_tokens, "doc1")
        score2 = calc.score_document(query_tokens, "doc2")

        assert isinstance(score1, float)
        assert isinstance(score2, float)
        # doc1 should score higher for "machine learning"
        assert score1 > score2


class TestTFIDFExtractor:
    """Test suite for TF-IDF extractor."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = TFIDFExtractor(use_stopwords=True)
        assert extractor.use_stopwords is True
        assert extractor._fitted is False

    def test_fit(self):
        """Test fitting on documents."""
        extractor = TFIDFExtractor()

        documents = ["Python is great", "Java is good", "Python and Java"]

        extractor.fit(documents)

        assert extractor._fitted is True
        assert len(extractor._vocabulary) > 0
        assert "python" in extractor._vocabulary or "Python" in extractor._vocabulary

    def test_transform(self):
        """Test transforming documents to vectors."""
        extractor = TFIDFExtractor()

        documents = ["Python programming", "Java programming"]
        extractor.fit(documents)

        vectors = extractor.transform(["Python code"])

        assert isinstance(vectors, list)
        assert len(vectors) == 1
        assert isinstance(vectors[0], list)
        assert len(vectors[0]) == len(extractor._vocabulary)

    def test_fit_transform(self):
        """Test fit and transform together."""
        extractor = TFIDFExtractor()

        documents = ["Text analysis", "Data mining"]
        vectors = extractor.fit_transform(documents)

        assert len(vectors) == 2
        assert extractor._fitted is True

    def test_get_feature_names(self):
        """Test getting feature names."""
        extractor = TFIDFExtractor()

        documents = ["feature extraction", "text features"]
        extractor.fit(documents)

        features = extractor.get_feature_names()

        assert isinstance(features, list)
        assert len(features) == len(extractor._vocabulary)
        assert all(isinstance(f, str) for f in features)
