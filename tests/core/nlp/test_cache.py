"""Tests for embedding cache."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tenets.core.nlp.cache import EmbeddingCache


class TestEmbeddingCache:
    """Test suite for EmbeddingCache."""

    def test_initialization(self, tmp_path):
        """Test cache initialization."""
        cache = EmbeddingCache(cache_dir=tmp_path, max_memory_items=100, ttl_days=7)

        assert cache.cache_dir == tmp_path
        assert cache.max_memory_items == 100
        assert cache.ttl_seconds == 7 * 24 * 3600
        assert len(cache._memory_cache) == 0

    def test_put_and_get(self, tmp_path):
        """Test putting and getting embeddings."""
        cache = EmbeddingCache(tmp_path)

        text = "test text"
        embedding = np.array([1, 2, 3, 4])

        # Put embedding
        cache.put(text, embedding, model_name="test-model")

        # Get embedding
        retrieved = cache.get(text, model_name="test-model")

        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)

    def test_get_nonexistent(self, tmp_path):
        """Test getting non-existent embedding."""
        cache = EmbeddingCache(tmp_path)

        result = cache.get("nonexistent", "model")
        assert result is None

    def test_memory_cache_lru(self, tmp_path):
        """Test LRU eviction in memory cache."""
        cache = EmbeddingCache(tmp_path, max_memory_items=3)

        # Add items
        cache.put("text1", np.array([1]), "model")
        cache.put("text2", np.array([2]), "model")
        cache.put("text3", np.array([3]), "model")

        # Access text1 to make it recently used
        cache.get("text1", "model")

        # Add new item (should evict text2)
        cache.put("text4", np.array([4]), "model")

        # Check memory cache
        assert len(cache._memory_cache) == 3
        key1 = cache._make_key("text1", "model")
        key3 = cache._make_key("text3", "model")
        key4 = cache._make_key("text4", "model")

        assert key1 in cache._memory_cache
        assert key3 in cache._memory_cache
        assert key4 in cache._memory_cache

    def test_batch_operations(self, tmp_path):
        """Test batch put and get."""
        cache = EmbeddingCache(tmp_path)

        embeddings = {
            "text1": np.array([1, 2]),
            "text2": np.array([3, 4]),
            "text3": np.array([5, 6]),
        }

        # Batch put
        cache.put_batch(embeddings, "model")

        # Batch get
        texts = ["text1", "text2", "text3", "text4"]
        results = cache.get_batch(texts, "model")

        assert results["text1"] is not None
        assert results["text2"] is not None
        assert results["text3"] is not None
        assert results["text4"] is None

        assert np.array_equal(results["text1"], embeddings["text1"])

    def test_key_generation(self, tmp_path):
        """Test cache key generation."""
        cache = EmbeddingCache(tmp_path)

        key1 = cache._make_key("test text", "model1")
        key2 = cache._make_key("test text", "model2")
        key3 = cache._make_key("different text", "model1")

        # Different models should have different keys
        assert key1 != key2
        # Different texts should have different keys
        assert key1 != key3

        # Same input should produce same key
        key4 = cache._make_key("test text", "model1")
        assert key1 == key4

    def test_clear_memory(self, tmp_path):
        """Test clearing memory cache."""
        cache = EmbeddingCache(tmp_path)

        # Add items
        cache.put("text1", np.array([1]), "model")
        cache.put("text2", np.array([2]), "model")

        assert len(cache._memory_cache) > 0

        # Clear memory
        cache.clear_memory()

        assert len(cache._memory_cache) == 0
        assert len(cache._access_order) == 0

    def test_clear_all(self, tmp_path):
        """Test clearing all caches."""
        cache = EmbeddingCache(tmp_path)

        # Add items
        cache.put("text1", np.array([1]), "model")
        cache.put("text2", np.array([2]), "model")

        # Clear all
        cache.clear_all()

        assert len(cache._memory_cache) == 0

        # Disk cache should also be cleared
        result = cache.get("text1", "model")
        assert result is None

    def test_stats(self, tmp_path):
        """Test getting cache statistics."""
        cache = EmbeddingCache(tmp_path)

        # Add items
        cache.put("text1", np.array([1, 2, 3, 4]), "model")
        cache.put("text2", np.array([5, 6, 7, 8]), "model")

        stats = cache.stats()

        assert stats["memory_items"] == 2
        assert stats["memory_size_mb"] > 0
        assert stats["access_order_length"] == 2

    @patch("tenets.core.nlp.cache.DiskCache")
    def test_disk_cache_integration(self, mock_disk_cache, tmp_path):
        """Test disk cache integration."""
        mock_disk = Mock()
        mock_disk_cache.return_value = mock_disk

        cache = EmbeddingCache(tmp_path)

        # Put should write to disk
        embedding = np.array([1, 2, 3])
        cache.put("text", embedding, "model")

        mock_disk.put.assert_called_once()
        call_args = mock_disk.put.call_args
        assert call_args[0][1] is embedding  # Embedding passed
        assert "ttl" in call_args[1]  # TTL specified
        assert "metadata" in call_args[1]  # Metadata included

    def test_invalid_cached_data(self, tmp_path):
        """Test handling invalid cached data."""
        cache = EmbeddingCache(tmp_path)

        # Mock disk cache returning invalid data
        with patch.object(cache.disk_cache, "get", return_value="invalid"):
            result = cache.get("text", "model")
            assert result is None

    def test_cleanup(self, tmp_path):
        """Test cleanup of old entries."""
        cache = EmbeddingCache(tmp_path, ttl_days=1)

        with patch.object(cache.disk_cache, "cleanup", return_value=5):
            deleted = cache.cleanup()
            assert deleted == 5

            cache.disk_cache.cleanup.assert_called_with(max_age_days=1)
