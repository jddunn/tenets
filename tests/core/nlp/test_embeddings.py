"""Tests for embedding generation."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from tenets.core.nlp.embeddings import (
    EmbeddingModel,
    LocalEmbeddings,
    FallbackEmbeddings,
    create_embedding_model,
)


class TestEmbeddingModel:
    """Test suite for base EmbeddingModel."""

    def test_initialization(self):
        """Test EmbeddingModel initialization."""
        model = EmbeddingModel(model_name="test-model")

        assert model.model_name == "test-model"
        assert model.model is None
        assert model.embedding_dim == 384

    def test_encode_not_implemented(self):
        """Test that encode raises NotImplementedError."""
        model = EmbeddingModel()

        with pytest.raises(NotImplementedError):
            model.encode("test text")

    def test_get_embedding_dim(self):
        """Test getting embedding dimension."""
        model = EmbeddingModel()
        assert model.get_embedding_dim() == 384


@patch("tenets.core.nlp.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE", True)
@patch("tenets.core.nlp.embeddings.SentenceTransformer")
class TestLocalEmbeddings:
    """Test suite for LocalEmbeddings."""

    def test_initialization(self, mock_st):
        """Test LocalEmbeddings initialization."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model

        embeddings = LocalEmbeddings(model_name="all-mpnet-base-v2", device="cpu")

        assert embeddings.model_name == "all-mpnet-base-v2"
        assert embeddings.device == "cpu"
        assert embeddings.embedding_dim == 768
        mock_st.assert_called_once()

    def test_encode_single(self, mock_st):
        """Test encoding single text."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1, 2, 3]])
        mock_st.return_value = mock_model

        embeddings = LocalEmbeddings()
        result = embeddings.encode("test text")

        # Should return first element for single text
        assert isinstance(result, np.ndarray)
        mock_model.encode.assert_called_once()

    def test_encode_batch(self, mock_st):
        """Test encoding multiple texts."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mock_st.return_value = mock_model

        embeddings = LocalEmbeddings()
        texts = ["text1", "text2", "text3"]
        result = embeddings.encode(texts, batch_size=2)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3

    def test_encode_file(self, mock_st, tmp_path):
        """Test encoding a file."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_st.return_value = mock_model

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("A" * 2000)  # Long content for chunking

        embeddings = LocalEmbeddings()
        result = embeddings.encode_file(test_file, chunk_size=1000, overlap=100)

        # Should return mean pooled embedding
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)  # Single embedding

    def test_encode_empty_file(self, mock_st, tmp_path):
        """Test encoding empty file."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        embeddings = LocalEmbeddings()
        result = embeddings.encode_file(test_file)

        # Should return zero vector
        assert np.all(result == 0)
        assert result.shape == (384,)

    def test_device_selection(self, mock_st):
        """Test automatic device selection."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=True):
            embeddings = LocalEmbeddings()
            assert embeddings.device == "cuda"

        with patch("torch.cuda.is_available", return_value=False):
            embeddings = LocalEmbeddings()
            assert embeddings.device == "cpu"


class TestFallbackEmbeddings:
    """Test suite for FallbackEmbeddings."""

    def test_initialization(self):
        """Test FallbackEmbeddings initialization."""
        embeddings = FallbackEmbeddings(embedding_dim=512)

        assert embeddings.model_name == "tfidf-fallback"
        assert embeddings.embedding_dim == 512
        assert embeddings.tfidf is not None

    def test_encode_single(self):
        """Test encoding single text with fallback."""
        embeddings = FallbackEmbeddings(embedding_dim=10)

        result = embeddings.encode("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (10,)

    def test_encode_batch(self):
        """Test encoding batch with fallback."""
        embeddings = FallbackEmbeddings(embedding_dim=10)

        texts = ["text one", "text two", "text three"]
        result = embeddings.encode(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 10)

    def test_padding(self):
        """Test padding to embedding dimension."""
        embeddings = FallbackEmbeddings(embedding_dim=100)

        # With small vocabulary, should pad
        result = embeddings.encode("simple text")
        assert result.shape == (100,)

    def test_truncation(self):
        """Test truncation to embedding dimension."""
        embeddings = FallbackEmbeddings(embedding_dim=5)

        # With large vocabulary, should truncate
        long_text = " ".join([f"word{i}" for i in range(100)])
        result = embeddings.encode(long_text)
        assert result.shape == (5,)


class TestCreateEmbeddingModel:
    """Test suite for create_embedding_model factory."""

    @patch("tenets.core.nlp.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("tenets.core.nlp.embeddings.LocalEmbeddings")
    def test_create_local(self, mock_local):
        """Test creating local embeddings."""
        mock_instance = Mock()
        mock_local.return_value = mock_instance

        model = create_embedding_model(prefer_local=True)

        assert model == mock_instance
        mock_local.assert_called_once()

    @patch("tenets.core.nlp.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    def test_fallback_when_no_ml(self):
        """Test fallback when ML not available."""
        model = create_embedding_model(prefer_local=True)

        assert isinstance(model, FallbackEmbeddings)

    @patch("tenets.core.nlp.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("tenets.core.nlp.embeddings.LocalEmbeddings")
    def test_create_with_model_name(self, mock_local):
        """Test creating with specific model name."""
        mock_instance = Mock()
        mock_local.return_value = mock_instance

        model = create_embedding_model(model_name="custom-model", cache_dir="/tmp/cache")

        mock_local.assert_called_with("custom-model", cache_dir="/tmp/cache")

    @patch("tenets.core.nlp.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("tenets.core.nlp.embeddings.LocalEmbeddings")
    def test_fallback_on_error(self, mock_local):
        """Test fallback when local embeddings fail."""
        mock_local.side_effect = Exception("Model load failed")

        model = create_embedding_model()

        assert isinstance(model, FallbackEmbeddings)
