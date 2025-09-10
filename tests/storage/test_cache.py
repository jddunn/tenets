"""Tests for the caching system."""

import json
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from tenets.config import TenetsConfig
from tenets.models.analysis import ComplexityMetrics, FileAnalysis
from tenets.storage.cache import AnalysisCache, CacheManager, DiskCache, MemoryCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_cache_dir):
    """Create test configuration with temporary cache."""
    config = TenetsConfig()
    config.cache_dir = str(temp_cache_dir)
    config.cache.enabled = True
    config.cache.directory = str(temp_cache_dir)
    config.cache_ttl_days = 7
    config.max_cache_size_mb = 100
    return config


class TestMemoryCache:
    """Test suite for MemoryCache."""

    def test_initialization(self):
        """Test memory cache initialization."""
        cache = MemoryCache(max_size=10)

        assert cache.max_size == 10
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0

    def test_put_and_get(self):
        """Test putting and getting items."""
        cache = MemoryCache(max_size=5)

        # Put items
        cache.put("key1", "value1")
        cache.put("key2", {"data": "value2"})
        cache.put("key3", [1, 2, 3])

        # Get items
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == {"data": "value2"}
        assert cache.get("key3") == [1, 2, 3]
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = MemoryCache(max_size=3)

        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new item - should evict key2 (least recently used)
        cache.put("key4", "value4")

        assert cache.get("key1") == "value1"  # Still there
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"  # Still there
        assert cache.get("key4") == "value4"  # New item

    def test_update_existing_key(self):
        """Test updating an existing key."""
        cache = MemoryCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Update existing key
        cache.put("key1", "new_value1")

        assert cache.get("key1") == "new_value1"
        assert len(cache._cache) == 2  # No new entry added

    def test_access_order_update(self):
        """Test that access order is updated on get."""
        cache = MemoryCache(max_size=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Access in specific order
        cache.get("key2")
        cache.get("key1")

        # key3 should be least recently used
        assert cache._access_order[0] == "key3"
        assert cache._access_order[-1] == "key1"

    def test_clear(self):
        """Test clearing the cache."""
        cache = MemoryCache(max_size=5)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0
        assert cache.get("key1") is None


class TestDiskCache:
    """Test suite for DiskCache."""

    def test_initialization(self, temp_cache_dir):
        """Test disk cache initialization."""
        cache = DiskCache(temp_cache_dir, name="test")

        assert cache.cache_dir == temp_cache_dir
        assert cache.db_path == temp_cache_dir / "test.db"
        assert cache.db_path.exists()

        # Check database structure
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "cache" in tables

    def test_put_and_get(self, temp_cache_dir):
        """Test putting and getting items from disk cache."""
        cache = DiskCache(temp_cache_dir, name="test")

        # Put various types
        cache.put("string_key", "string_value")
        cache.put("dict_key", {"data": "value", "count": 42})
        cache.put("list_key", [1, 2, 3, "four"])

        # Get items
        assert cache.get("string_key") == "string_value"
        assert cache.get("dict_key") == {"data": "value", "count": 42}
        assert cache.get("list_key") == [1, 2, 3, "four"]
        assert cache.get("nonexistent") is None

    @pytest.mark.skipif(
        "freezegun" in sys.modules or any("freeze" in m for m in sys.modules),
        reason="TTL tests incompatible with freezegun",
    )
    def test_ttl_expiration(self, temp_cache_dir):
        """Test TTL expiration."""
        cache = DiskCache(temp_cache_dir, name="test")

        # Put item with 1 second TTL
        cache.put("expiring_key", "value", ttl=1)

        # Should be available immediately
        assert cache.get("expiring_key") == "value"

        # Wait for expiration
        time.sleep(2)

        # Should be expired
        assert cache.get("expiring_key") is None

    def test_metadata_storage(self, temp_cache_dir):
        """Test storing metadata with cache entries."""
        cache = DiskCache(temp_cache_dir, name="test")

        metadata = {"source": "test", "version": "1.0"}
        cache.put("key_with_meta", "value", metadata=metadata)

        # Verify metadata is stored
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute("SELECT metadata FROM cache WHERE key = ?", ("key_with_meta",))
            row = cursor.fetchone()
            stored_meta = json.loads(row[0])
            assert stored_meta == metadata

    def test_delete(self, temp_cache_dir):
        """Test deleting cache entries."""
        cache = DiskCache(temp_cache_dir, name="test")

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Delete one key
        assert cache.delete("key1") == True
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

        # Try deleting non-existent key
        assert cache.delete("nonexistent") == False

    def test_clear(self, temp_cache_dir):
        """Test clearing all cache entries."""
        cache = DiskCache(temp_cache_dir, name="test")

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    @pytest.mark.skipif(
        "freezegun" in sys.modules or any("freeze" in m for m in sys.modules),
        reason="Time-based cleanup tests incompatible with freezegun",
    )
    def test_cleanup_by_age(self, temp_cache_dir):
        """Test cleanup by age."""
        cache = DiskCache(temp_cache_dir, name="test")

        # Put items with different access times
        cache.put("old_key", "old_value")

        # Manually update access time to be old
        with sqlite3.connect(cache.db_path) as conn:
            old_date = datetime.now() - timedelta(days=10)
            conn.execute("UPDATE cache SET accessed_at = ? WHERE key = ?", (old_date, "old_key"))

        cache.put("new_key", "new_value")

        # Cleanup entries older than 7 days
        deleted = cache.cleanup(max_age_days=7)

        assert deleted > 0
        assert cache.get("old_key") is None
        assert cache.get("new_key") == "new_value"

    @pytest.mark.skipif(
        "freezegun" in sys.modules or any("freeze" in m for m in sys.modules),
        reason="Access time tests incompatible with freezegun",
    )
    def test_access_time_update(self, temp_cache_dir):
        """Test that access time is updated on get."""
        cache = DiskCache(temp_cache_dir, name="test")

        cache.put("key", "value")

        # Get initial access time
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute("SELECT accessed_at FROM cache WHERE key = ?", ("key",))
            initial_time = cursor.fetchone()[0]

        # Wait and access again
        time.sleep(0.1)
        cache.get("key")

        # Check access time was updated
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute("SELECT accessed_at FROM cache WHERE key = ?", ("key",))
            new_time = cursor.fetchone()[0]

        assert new_time > initial_time


class TestAnalysisCache:
    """Test suite for AnalysisCache."""

    def test_initialization(self, temp_cache_dir):
        """Test analysis cache initialization."""
        cache = AnalysisCache(temp_cache_dir)

        assert cache.cache_dir == temp_cache_dir
        assert cache.memory is not None
        assert cache.disk is not None

    def test_file_analysis_caching(self, temp_cache_dir):
        """Test caching file analysis results."""
        cache = AnalysisCache(temp_cache_dir)

        # Create a test file
        test_file = temp_cache_dir / "test.py"
        test_file.write_text("print('hello')")

        # Create analysis
        analysis = FileAnalysis(
            path=str(test_file),
            language="python",
            lines=1,
            size=14,
            complexity=ComplexityMetrics(cyclomatic=1),
        )

        # Cache it
        cache.put_file_analysis(test_file, analysis)

        # Retrieve from cache
        cached = cache.get_file_analysis(test_file)

        assert cached is not None
        assert cached.path == str(test_file)
        assert cached.language == "python"
        assert cached.lines == 1

    def test_cache_invalidation_on_file_change(self, temp_cache_dir):
        """Test cache invalidation when file is modified."""
        cache = AnalysisCache(temp_cache_dir)

        # Create and analyze file
        test_file = temp_cache_dir / "test.py"
        test_file.write_text("print('hello')")

        analysis = FileAnalysis(path=str(test_file), language="python")

        cache.put_file_analysis(test_file, analysis)

        # Verify cached
        assert cache.get_file_analysis(test_file) is not None

        # Modify file (changes mtime)
        time.sleep(0.01)  # Ensure different mtime
        test_file.write_text("print('modified')")

        # Cache should be invalidated
        assert cache.get_file_analysis(test_file) is None

    def test_memory_promotion(self, temp_cache_dir):
        """Test promotion from disk to memory cache."""
        cache = AnalysisCache(temp_cache_dir)

        test_file = temp_cache_dir / "test.py"
        test_file.write_text("code")

        analysis = FileAnalysis(path=str(test_file))

        # Put in cache (goes to both memory and disk)
        cache.put_file_analysis(test_file, analysis)

        # Clear memory cache
        cache.memory.clear()

        # Get should promote from disk to memory
        cached = cache.get_file_analysis(test_file)
        assert cached is not None

        # Should now be in memory
        key = cache._make_file_key(test_file)
        assert cache.memory.get(key) is not None

    def test_cache_key_generation(self, temp_cache_dir):
        """Test cache key generation for files."""
        cache = AnalysisCache(temp_cache_dir)

        file1 = Path("/path/to/file1.py")
        file2 = Path("/path/to/file2.py")
        file3 = Path("/other/path/file1.py")

        key1 = cache._make_file_key(file1)
        key2 = cache._make_file_key(file2)
        key3 = cache._make_file_key(file3)

        # Different files should have different keys
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

        # Same file should have same key
        assert key1 == cache._make_file_key(file1)


class TestCacheManager:
    """Test suite for CacheManager."""

    def test_initialization(self, config):
        """Test cache manager initialization."""
        manager = CacheManager(config)

        assert manager.config == config
        assert manager.cache_dir == Path(config.cache_dir)
        assert manager.analysis is not None
        assert manager.general is not None
        assert manager.memory is not None

    def test_get_or_compute_cached(self, config):
        """Test get_or_compute when value is cached."""
        manager = CacheManager(config)

        # Pre-populate cache
        manager.memory.put("test_key", "cached_value")

        compute_fn = Mock(return_value="computed_value")

        result = manager.get_or_compute("test_key", compute_fn)

        assert result == "cached_value"
        compute_fn.assert_not_called()  # Should not compute

    def test_get_or_compute_not_cached(self, config):
        """Test get_or_compute when value needs computing."""
        manager = CacheManager(config)

        def compute_fn():
            return {"data": "computed_value"}

        result = manager.get_or_compute("test_key", compute_fn, ttl=60)

        assert result == {"data": "computed_value"}

        # Should be cached now
        assert manager.memory.get("test_key") == {"data": "computed_value"}
        assert manager.general.get("test_key") == {"data": "computed_value"}

    def test_get_or_compute_disk_only(self, config):
        """Test get_or_compute with disk cache only."""
        manager = CacheManager(config)

        # Put in disk cache only
        manager.general.put("disk_key", "disk_value")

        compute_fn = Mock(return_value="computed_value")

        result = manager.get_or_compute("disk_key", compute_fn, use_memory=True)

        assert result == "disk_value"
        compute_fn.assert_not_called()

        # Should be promoted to memory
        assert manager.memory.get("disk_key") == "disk_value"

    def test_invalidate(self, config):
        """Test cache invalidation."""
        manager = CacheManager(config)

        # Populate caches
        manager.memory.put("test_key", "value")
        manager.general.put("test_key", "value")

        # Invalidate
        manager.invalidate("test_key")

        # Should be removed from both caches
        assert manager.memory.get("test_key") is None
        assert manager.general.get("test_key") is None

    def test_clear_all(self, config):
        """Test clearing all caches."""
        manager = CacheManager(config)

        # Populate various caches
        manager.memory.put("key1", "value1")
        manager.general.put("key2", "value2")
        manager.analysis.memory.put("key3", "value3")

        # Clear all
        manager.clear_all()

        # All should be empty
        assert manager.memory.get("key1") is None
        assert manager.general.get("key2") is None
        assert manager.analysis.memory.get("key3") is None

    def test_cleanup(self, config):
        """Test cache cleanup."""
        manager = CacheManager(config)

        # Add some data
        manager.general.put("key1", "value1")
        manager.general.put("key2", "value2")

        # Run cleanup
        stats = manager.cleanup()

        assert "analysis_deleted" in stats
        assert "general_deleted" in stats
        assert isinstance(stats["analysis_deleted"], int)
        assert isinstance(stats["general_deleted"], int)

    def test_exception_handling_in_compute(self, config):
        """Test exception handling when compute function fails."""
        manager = CacheManager(config)

        def failing_compute():
            raise ValueError("Computation failed")

        with pytest.raises(ValueError):
            manager.get_or_compute("fail_key", failing_compute)

        # Should not cache failed computation
        assert manager.memory.get("fail_key") is None
        assert manager.general.get("fail_key") is None
