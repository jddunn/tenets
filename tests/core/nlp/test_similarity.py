"""Tests for similarity computation utilities."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from tenets.core.nlp.similarity import (
    SemanticSimilarity,
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)


class TestSimilarityFunctions:
    """Test suite for similarity functions."""

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])

        # Orthogonal vectors
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.01  # Should be ~0

        # Identical vectors
        sim = cosine_similarity(vec1, vec3)
        assert abs(sim - 1.0) < 0.01  # Should be ~1

        # Opposite vectors
        vec4 = np.array([-1, 0, 0])
        sim = cosine_similarity(vec1, vec4)
        assert abs(sim + 1.0) < 0.01  # Should be ~-1

    def test_cosine_similarity_normalization(self):
        """Test cosine similarity with different magnitudes."""
        vec1 = np.array([1, 1, 1])
        vec2 = np.array([2, 2, 2])  # Same direction, different magnitude

        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.01  # Should be 1 (same direction)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([0, 0, 0])

        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0

    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([3, 4, 0])

        dist = euclidean_distance(vec1, vec2)
        assert abs(dist - 5.0) < 0.01  # 3-4-5 triangle

        # Same vectors
        dist = euclidean_distance(vec1, vec1)
        assert dist == 0.0

    def test_manhattan_distance(self):
        """Test Manhattan distance computation."""
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 2, 3])

        dist = manhattan_distance(vec1, vec2)
        assert dist == 6  # 1 + 2 + 3

        # Same vectors
        dist = manhattan_distance(vec1, vec1)
        assert dist == 0.0

    def test_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2])

        with pytest.raises(ValueError, match="same shape"):
            cosine_similarity(vec1, vec2)

        with pytest.raises(ValueError, match="same shape"):
            euclidean_distance(vec1, vec2)

        with pytest.raises(ValueError, match="same shape"):
            manhattan_distance(vec1, vec2)


class TestSemanticSimilarity:
    """Test suite for semantic similarity."""

    @patch("tenets.core.nlp.similarity.create_embedding_model")
    def test_initialization(self, mock_create_model):
        """Test SemanticSimilarity initialization."""
        mock_model = Mock()
        mock_create_model.return_value = mock_model

        sim = SemanticSimilarity(cache_embeddings=True)

        assert sim.model == mock_model
        assert sim.cache_embeddings is True
        assert sim._cache is not None

    @patch("tenets.core.nlp.similarity.create_embedding_model")
    def test_compute(self, mock_create_model):
        """Test computing semantic similarity."""
        mock_model = Mock()
        mock_model.encode.side_effect = lambda text: {
            "hello": np.array([1, 0, 0]),
            "hi": np.array([0.9, 0.1, 0]),
            "goodbye": np.array([0, 1, 0]),
        }.get(text, np.array([0, 0, 1]))
        mock_create_model.return_value = mock_model

        sim = SemanticSimilarity()

        # Similar texts
        score = sim.compute("hello", "hi")
        assert score > 0.8  # Should be similar

        # Different texts
        score = sim.compute("hello", "goodbye")
        assert score < 0.3  # Should be dissimilar

    @patch("tenets.core.nlp.similarity.create_embedding_model")
    def test_compute_batch(self, mock_create_model):
        """Test batch similarity computation."""
        mock_model = Mock()
        # Mock batch encoding
        mock_model.encode.side_effect = lambda texts: [
            np.array([1, 0, 0]) if "python" in t.lower() else np.array([0, 1, 0])
            for t in (texts if isinstance(texts, list) else [texts])
        ]
        mock_create_model.return_value = mock_model

        sim = SemanticSimilarity()

        query = "python programming"
        documents = ["Python is great", "Java programming", "Python code examples"]

        results = sim.compute_batch(query, documents)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(isinstance(r[0], int) and isinstance(r[1], float) for r in results)

        # Results should be sorted by similarity
        assert results[0][1] >= results[1][1]
        assert results[1][1] >= results[2][1]

    @patch("tenets.core.nlp.similarity.create_embedding_model")
    def test_compute_batch_top_k(self, mock_create_model):
        """Test batch computation with top_k limit."""
        mock_model = Mock()
        mock_model.encode.return_value = [
            np.array([1, 0, 0]),
            np.array([0.5, 0.5, 0]),
            np.array([0, 1, 0]),
            np.array([0.9, 0.1, 0]),
        ]
        mock_create_model.return_value = mock_model

        sim = SemanticSimilarity()

        documents = ["doc1", "doc2", "doc3", "doc4"]
        results = sim.compute_batch("query", documents, top_k=2)

        assert len(results) == 2

    @patch("tenets.core.nlp.similarity.create_embedding_model")
    def test_find_similar(self, mock_create_model):
        """Test finding similar documents."""
        mock_model = Mock()
        mock_model.encode.side_effect = lambda texts: [
            np.array([1, 0, 0]) if "python" in t.lower() else np.array([0, 1, 0])
            for t in (texts if isinstance(texts, list) else [texts])
        ]
        mock_create_model.return_value = mock_model

        sim = SemanticSimilarity()

        query = "python"
        documents = ["Python programming", "Java code", "Python tutorial", "C++ guide"]

        results = sim.find_similar(query, documents, threshold=0.7)

        # Should find Python-related documents
        assert len(results) > 0
        assert all(r[1] >= 0.7 for r in results)

    @patch("tenets.core.nlp.similarity.create_embedding_model")
    def test_caching(self, mock_create_model):
        """Test embedding caching."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([1, 2, 3])
        mock_create_model.return_value = mock_model

        sim = SemanticSimilarity(cache_embeddings=True)

        # First call
        emb1 = sim._get_embedding("test text")
        assert mock_model.encode.call_count == 1

        # Second call should use cache
        emb2 = sim._get_embedding("test text")
        assert mock_model.encode.call_count == 1  # Not called again
        assert np.array_equal(emb1, emb2)

    @patch("tenets.core.nlp.similarity.create_embedding_model")
    def test_clear_cache(self, mock_create_model):
        """Test clearing embedding cache."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([1, 2, 3])
        mock_create_model.return_value = mock_model

        sim = SemanticSimilarity(cache_embeddings=True)

        # Add to cache
        sim._get_embedding("test")
        assert len(sim._cache) == 1

        # Clear cache
        sim.clear_cache()
        assert len(sim._cache) == 0

    @patch("tenets.core.nlp.similarity.create_embedding_model")
    def test_different_metrics(self, mock_create_model):
        """Test different similarity metrics."""
        mock_model = Mock()
        mock_model.encode.side_effect = lambda text: {
            "a": np.array([1, 0, 0]),
            "b": np.array([0, 1, 0]),
        }.get(text, np.array([0, 0, 1]))
        mock_create_model.return_value = mock_model

        sim = SemanticSimilarity()

        # Cosine similarity
        cos_sim = sim.compute("a", "b", metric="cosine")
        assert cos_sim < 0.1

        # Euclidean similarity
        euc_sim = sim.compute("a", "b", metric="euclidean")
        assert 0 < euc_sim < 1

        # Manhattan similarity
        man_sim = sim.compute("a", "b", metric="manhattan")
        assert 0 < man_sim < 1

        # Invalid metric
        with pytest.raises(ValueError):
            sim.compute("a", "b", metric="invalid")
