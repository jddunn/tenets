"""Tests for cache utilities."""

import tempfile
import time
from pathlib import Path

import pytest

from tenets.utils.cache import (
    CacheEntry,
    EmbeddingCache,
    FileContentCache,
    LRUCache,
    RankingScoreCache,
    cache_key,
    clear_all_caches,
    get_all_cache_stats,
    get_embedding_cache,
    get_file_cache,
    get_ranking_cache,
)


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_entry_creation(self):
        """Test cache entry is created with default values."""
        entry = CacheEntry(value="test")
        assert entry.value == "test"
        assert entry.access_count == 0
        assert entry.size_bytes == 0
        assert entry.created_at <= time.time()

    def test_entry_not_expired_without_ttl(self):
        """Test entry never expires when TTL is 0."""
        entry = CacheEntry(value="test")
        assert not entry.is_expired(0)
        assert not entry.is_expired(-1)

    def test_entry_expiration(self):
        """Test entry expiration with TTL."""
        entry = CacheEntry(value="test")
        entry.created_at = time.time() - 10  # Created 10 seconds ago
        assert entry.is_expired(5)  # 5 second TTL
        assert not entry.is_expired(15)  # 15 second TTL

    def test_entry_touch(self):
        """Test touch updates access time and count."""
        entry = CacheEntry(value="test")
        original_access = entry.last_accessed
        entry.touch()
        assert entry.access_count == 1
        assert entry.last_accessed >= original_access


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_basic_get_set(self):
        """Test basic get and set operations."""
        cache: LRUCache[str] = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        """Test cache miss returns None."""
        cache: LRUCache[str] = LRUCache(max_size=10)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache: LRUCache[str] = LRUCache(max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new key, should evict key2 (LRU)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still present
        assert cache.get("key4") == "value4"  # Newly added

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache: LRUCache[str] = LRUCache(max_size=10, ttl_seconds=0.1)
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"  # Not expired yet
        time.sleep(0.2)
        assert cache.get("key1") is None  # Now expired

    def test_delete(self):
        """Test delete operation."""
        cache: LRUCache[str] = LRUCache(max_size=10)
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False

    def test_clear(self):
        """Test clear operation."""
        cache: LRUCache[str] = LRUCache(max_size=10)
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        count = cache.clear()
        assert count == 2
        assert cache.size == 0

    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache: LRUCache[str] = LRUCache(max_size=10)
        cache.set("key1", "value1")

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("miss")  # Miss

        assert cache.hits == 2
        assert cache.misses == 1
        assert abs(cache.hit_rate - 0.666) < 0.01

    def test_stats(self):
        """Test statistics reporting."""
        cache: LRUCache[str] = LRUCache(max_size=10, name="test_cache")
        cache.set("key1", "value1", size_bytes=100)

        stats = cache.stats()
        assert stats["name"] == "test_cache"
        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["total_bytes"] == 100


class TestCacheKey:
    """Tests for cache_key function."""

    def test_deterministic(self):
        """Test cache_key is deterministic."""
        key1 = cache_key("prompt", ["kw1", "kw2"])
        key2 = cache_key("prompt", ["kw1", "kw2"])
        assert key1 == key2

    def test_different_args_different_keys(self):
        """Test different arguments produce different keys."""
        key1 = cache_key("prompt1", ["kw1"])
        key2 = cache_key("prompt2", ["kw1"])
        assert key1 != key2

    def test_handles_paths(self):
        """Test cache_key handles Path objects."""
        key1 = cache_key(Path("/some/path"), "value")
        key2 = cache_key(Path("/some/path"), "value")
        assert key1 == key2

    def test_handles_kwargs(self):
        """Test cache_key handles keyword arguments."""
        key1 = cache_key(a=1, b=2)
        key2 = cache_key(b=2, a=1)  # Order shouldn't matter
        assert key1 == key2


class TestFileContentCache:
    """Tests for FileContentCache class."""

    def test_set_and_get(self):
        """Test basic file content caching."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("content")
            f.flush()

            cache = FileContentCache()
            path = Path(f.name)

            cache.set(path, "cached content")
            assert cache.get(path) == "cached content"

    def test_invalidation_on_modification(self):
        """Test cache invalidation when file is modified."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("content")
            f.flush()

            cache = FileContentCache()
            path = Path(f.name)

            cache.set(path, "cached content")
            assert cache.get(path) == "cached content"

            # Modify the file
            time.sleep(0.1)  # Ensure mtime changes
            path.write_text("modified content")

            # Cache should be invalidated
            assert cache.get(path) is None

    def test_max_file_size(self):
        """Test files over max size are not cached."""
        cache = FileContentCache(max_file_size=10)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            path = Path(f.name)
            large_content = "x" * 100

            result = cache.set(path, large_content)
            assert result is False  # Too large


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    def test_set_and_get(self):
        """Test basic embedding caching."""
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3]

        cache.set("test text", embedding, model="test-model")
        result = cache.get("test text", model="test-model")

        assert result == embedding

    def test_different_models_different_keys(self):
        """Test different models produce different cache keys."""
        cache = EmbeddingCache()

        cache.set("text", [1.0], model="model-a")
        cache.set("text", [2.0], model="model-b")

        assert cache.get("text", model="model-a") == [1.0]
        assert cache.get("text", model="model-b") == [2.0]

    def test_ttl_expiration(self):
        """Test embedding cache TTL."""
        cache = EmbeddingCache(ttl_seconds=0.1)
        cache.set("text", [1.0, 2.0])

        assert cache.get("text") is not None
        time.sleep(0.2)
        assert cache.get("text") is None


class TestRankingScoreCache:
    """Tests for RankingScoreCache class."""

    def test_set_and_get(self):
        """Test basic ranking score caching."""
        cache = RankingScoreCache()
        path = Path("/test/file.py")
        factors = {"keyword_match": 0.5, "tfidf_similarity": 0.3}

        cache.set(
            file_path=path,
            prompt_hash="abc123",
            file_mtime=1000.0,
            score=0.75,
            factors=factors,
            algorithm="balanced",
        )

        result = cache.get(
            file_path=path,
            prompt_hash="abc123",
            file_mtime=1000.0,
            algorithm="balanced",
        )

        assert result is not None
        assert result["score"] == 0.75
        assert result["factors"] == factors

    def test_invalidation_on_file_change(self):
        """Test cache invalidation when file mtime changes."""
        cache = RankingScoreCache()
        path = Path("/test/file.py")

        cache.set(
            file_path=path,
            prompt_hash="abc123",
            file_mtime=1000.0,
            score=0.75,
            factors={},
            algorithm="balanced",
        )

        # Get with newer mtime should return None
        result = cache.get(
            file_path=path,
            prompt_hash="abc123",
            file_mtime=2000.0,  # File was modified
            algorithm="balanced",
        )

        assert result is None

    def test_different_prompts_different_entries(self):
        """Test different prompts produce different cache entries."""
        cache = RankingScoreCache()
        path = Path("/test/file.py")

        cache.set(
            file_path=path,
            prompt_hash="prompt1",
            file_mtime=1000.0,
            score=0.5,
            factors={},
            algorithm="balanced",
        )
        cache.set(
            file_path=path,
            prompt_hash="prompt2",
            file_mtime=1000.0,
            score=0.8,
            factors={},
            algorithm="balanced",
        )

        result1 = cache.get(path, "prompt1", 1000.0)
        result2 = cache.get(path, "prompt2", 1000.0)

        assert result1["score"] == 0.5
        assert result2["score"] == 0.8


class TestGlobalCaches:
    """Tests for global cache functions."""

    def test_get_caches_returns_singletons(self):
        """Test global cache getters return singletons."""
        cache1 = get_file_cache()
        cache2 = get_file_cache()
        assert cache1 is cache2

        emb1 = get_embedding_cache()
        emb2 = get_embedding_cache()
        assert emb1 is emb2

        rank1 = get_ranking_cache()
        rank2 = get_ranking_cache()
        assert rank1 is rank2

    def test_clear_all_caches(self):
        """Test clearing all global caches."""
        # Add some entries
        get_file_cache()._cache.set("test", (0, "content"))
        get_embedding_cache()._cache.set("test", [1.0])
        get_ranking_cache()._cache.set("test", {"score": 0.5})

        results = clear_all_caches()

        assert "file_content" in results
        assert "embeddings" in results
        assert "ranking_scores" in results

    def test_get_all_cache_stats(self):
        """Test getting statistics for all caches."""
        stats = get_all_cache_stats()

        assert "file_content" in stats
        assert "embeddings" in stats
        assert "ranking_scores" in stats

        for cache_stats in stats.values():
            assert "size" in cache_stats
            assert "hits" in cache_stats
            assert "misses" in cache_stats
