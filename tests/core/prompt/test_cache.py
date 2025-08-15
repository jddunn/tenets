"""Unit tests for prompt caching system.

Tests intelligent caching with TTL management, invalidation strategies,
and cache statistics tracking.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from freezegun import freeze_time

from tenets.core.prompt.cache import (
    CacheEntry,
    PromptCache,
)


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    @freeze_time("2024-01-15 10:00:00")
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            ttl_seconds=3600,
            hit_count=0,
            metadata={"source": "test"},
        )

        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.created_at == datetime(2024, 1, 15, 10, 0, 0)
        assert entry.ttl_seconds == 3600
        assert entry.hit_count == 0
        assert entry.metadata["source"] == "test"

    @freeze_time("2024-01-15 10:00:00")
    def test_is_expired_not_expired(self):
        """Test checking if entry is not expired."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=datetime(2024, 1, 15, 9, 0, 0),  # 1 hour ago
            accessed_at=datetime.now(),
            ttl_seconds=7200,  # 2 hour TTL
            hit_count=0,
        )

        assert entry.is_expired() is False

    @freeze_time("2024-01-15 10:00:00")
    def test_is_expired_expired(self):
        """Test checking if entry is expired."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=datetime(2024, 1, 15, 8, 0, 0),  # 2 hours ago
            accessed_at=datetime(2024, 1, 15, 8, 30, 0),
            ttl_seconds=3600,  # 1 hour TTL
            hit_count=0,
        )

        assert entry.is_expired() is True

    def test_is_expired_no_ttl(self):
        """Test entry with no expiration (TTL <= 0)."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=datetime(2020, 1, 1),  # Very old
            accessed_at=datetime(2020, 1, 1),
            ttl_seconds=0,  # No expiration
            hit_count=0,
        )

        assert entry.is_expired() is False

        entry.ttl_seconds = -1
        assert entry.is_expired() is False

    @freeze_time("2024-01-15 10:00:00")
    def test_touch(self):
        """Test touching a cache entry."""
        entry = CacheEntry(
            key="test",
            value="data",
            created_at=datetime(2024, 1, 15, 9, 0, 0),
            accessed_at=datetime(2024, 1, 15, 9, 0, 0),
            ttl_seconds=3600,
            hit_count=5,
        )

        entry.touch()

        assert entry.accessed_at == datetime(2024, 1, 15, 10, 0, 0)
        assert entry.hit_count == 6


class TestPromptCache:
    """Test PromptCache class."""

    @pytest.fixture
    def cache(self):
        """Create cache instance without external manager."""
        return PromptCache(
            cache_manager=None,
            enable_memory_cache=True,
            enable_disk_cache=False,
            memory_cache_size=10,
        )

    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        manager = MagicMock()
        manager.general.get.return_value = None
        manager.general.put.return_value = None
        manager.general.clear.return_value = None
        return manager

    def test_initialization_no_manager(self, cache):
        """Test initialization without cache manager."""
        assert cache.cache_manager is None
        assert cache.enable_memory is True
        assert cache.enable_disk is False
        assert cache.memory_cache_size == 10
        assert len(cache.memory_cache) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_initialization_with_manager(self, mock_cache_manager):
        """Test initialization with cache manager."""
        cache = PromptCache(
            cache_manager=mock_cache_manager, enable_memory_cache=True, enable_disk_cache=True
        )

        assert cache.cache_manager is not None
        assert cache.enable_disk is True

    def test_generate_key_string(self, cache):
        """Test generating cache key from string."""
        key = cache._generate_key("prefix", "test content")

        assert key.startswith("prefix:")
        assert len(key) > len("prefix:")

        # Same content should generate same key
        key2 = cache._generate_key("prefix", "test content")
        assert key == key2

        # Different content should generate different key
        key3 = cache._generate_key("prefix", "different content")
        assert key != key3

    def test_generate_key_dict(self, cache):
        """Test generating cache key from dictionary."""
        data = {"key": "value", "number": 123}
        key = cache._generate_key("prefix", data)

        assert key.startswith("prefix:")

        # Same dict should generate same key (order shouldn't matter)
        data2 = {"number": 123, "key": "value"}
        key2 = cache._generate_key("prefix", data2)
        assert key == key2

    def test_generate_key_list(self, cache):
        """Test generating cache key from list."""
        data = ["item1", "item2", "item3"]
        key = cache._generate_key("prefix", data)

        assert key.startswith("prefix:")

        # Different list should generate different key
        data2 = ["item1", "item2"]
        key2 = cache._generate_key("prefix", data2)
        assert key != key2

    def test_calculate_ttl_base(self, cache):
        """Test basic TTL calculation."""
        ttl = cache._calculate_ttl(3600, "parsed_prompt", None)
        assert ttl == 3600

        ttl = cache._calculate_ttl(1800, "entity_recognition", {})
        assert ttl == 1800

    def test_calculate_ttl_github_modifiers(self, cache):
        """Test TTL calculation with GitHub metadata."""
        # Open GitHub issue - shorter TTL
        metadata = {"source": "github", "state": "open"}
        ttl = cache._calculate_ttl(3600, "external_content", metadata)
        assert ttl == int(3600 * 0.25)  # 25% of normal

        # Closed GitHub issue - longer TTL
        metadata = {"source": "github", "state": "closed"}
        ttl = cache._calculate_ttl(3600, "external_content", metadata)
        assert ttl == int(3600 * 4.0)  # 400% of normal

    def test_calculate_ttl_jira_modifiers(self, cache):
        """Test TTL calculation with JIRA metadata."""
        # Active JIRA ticket
        metadata = {"source": "jira", "status": "In Progress"}
        ttl = cache._calculate_ttl(3600, "external_content", metadata)
        assert ttl == int(3600 * 0.5)  # 50% of normal

        # Closed JIRA ticket
        metadata = {"source": "jira", "status": "Done"}
        ttl = cache._calculate_ttl(3600, "external_content", metadata)
        assert ttl == 3600  # Normal TTL

    def test_calculate_ttl_confidence_modifiers(self, cache):
        """Test TTL calculation with confidence scores."""
        # High confidence - longer TTL
        metadata = {"confidence": 0.9}
        ttl = cache._calculate_ttl(3600, "intent_detection", metadata)
        assert ttl == int(3600 * 1.5)  # 150% of normal

        # Low confidence - shorter TTL
        metadata = {"confidence": 0.3}
        ttl = cache._calculate_ttl(3600, "intent_detection", metadata)
        assert ttl == int(3600 * 0.5)  # 50% of normal

        # Medium confidence - normal TTL
        metadata = {"confidence": 0.6}
        ttl = cache._calculate_ttl(3600, "intent_detection", metadata)
        assert ttl == 3600

    def test_calculate_ttl_bounds(self, cache):
        """Test TTL bounds enforcement."""
        # Very short TTL should be bounded to minimum
        metadata = {"source": "github", "state": "open", "confidence": 0.1}
        ttl = cache._calculate_ttl(100, "external_content", metadata)
        assert ttl >= 60  # Minimum 1 minute

        # Very long TTL should be bounded to maximum
        metadata = {"source": "github", "state": "closed", "confidence": 0.95}
        ttl = cache._calculate_ttl(100000, "external_content", metadata)
        assert ttl <= 86400  # Maximum 24 hours

    @freeze_time("2024-01-15 10:00:00")
    def test_get_from_memory_valid(self, cache):
        """Test getting valid entry from memory cache."""
        # Add entry to memory cache
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            ttl_seconds=3600,
            hit_count=0,
        )
        cache.memory_cache["test_key"] = entry

        result = cache.get("test_key", check_disk=False)

        assert result == "test_value"
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 0
        assert entry.hit_count == 1

    @freeze_time("2024-01-15 10:00:00")
    def test_get_from_memory_expired(self, cache):
        """Test getting expired entry from memory cache."""
        # Add expired entry
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=datetime(2024, 1, 15, 8, 0, 0),  # 2 hours ago
            accessed_at=datetime(2024, 1, 15, 8, 0, 0),
            ttl_seconds=3600,  # 1 hour TTL
            hit_count=0,
        )
        cache.memory_cache["test_key"] = entry

        result = cache.get("test_key", check_disk=False)

        assert result is None
        assert "test_key" not in cache.memory_cache
        assert cache.stats["expirations"] == 1
        assert cache.stats["misses"] == 1

    def test_get_from_disk(self, mock_cache_manager):
        """Test getting entry from disk cache."""
        cache = PromptCache(
            cache_manager=mock_cache_manager, enable_memory_cache=True, enable_disk_cache=True
        )

        mock_cache_manager.general.get.return_value = "disk_value"

        result = cache.get("test_key")

        assert result == "disk_value"
        assert cache.stats["hits"] == 1
        mock_cache_manager.general.get.assert_called_once_with("test_key")

        # Should be promoted to memory cache
        assert "test_key" in cache.memory_cache

    def test_get_miss(self, cache):
        """Test cache miss."""
        result = cache.get("nonexistent_key")

        assert result is None
        assert cache.stats["misses"] == 1
        assert cache.stats["hits"] == 0

    @freeze_time("2024-01-15 10:00:00")
    def test_put_to_memory(self, cache):
        """Test putting entry in memory cache."""
        cache.put("test_key", "test_value", ttl_seconds=3600)

        assert "test_key" in cache.memory_cache
        entry = cache.memory_cache["test_key"]
        assert entry.value == "test_value"
        assert entry.ttl_seconds == 3600
        assert entry.created_at == datetime.now()

    def test_put_to_disk(self, mock_cache_manager):
        """Test putting entry in disk cache."""
        cache = PromptCache(
            cache_manager=mock_cache_manager, enable_memory_cache=False, enable_disk_cache=True
        )

        cache.put("test_key", "test_value", ttl_seconds=3600)

        mock_cache_manager.general.put.assert_called_once()
        call_args = mock_cache_manager.general.put.call_args
        assert call_args[0][0] == "test_key"
        assert call_args[0][1] == "test_value"
        assert call_args[1]["ttl"] == 3600

    def test_memory_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        cache.memory_cache_size = 3

        # Fill cache
        with freeze_time("2024-01-15 10:00:00"):
            cache.put("key1", "value1")

        with freeze_time("2024-01-15 10:01:00"):
            cache.put("key2", "value2")

        with freeze_time("2024-01-15 10:02:00"):
            cache.put("key3", "value3")

        # Access key1 and key3 to make key2 LRU
        with freeze_time("2024-01-15 10:03:00"):
            cache.get("key1")
            cache.get("key3")

        # Add new key - should evict key2
        with freeze_time("2024-01-15 10:04:00"):
            cache.put("key4", "value4")

        assert "key1" in cache.memory_cache
        assert "key2" not in cache.memory_cache  # Evicted
        assert "key3" in cache.memory_cache
        assert "key4" in cache.memory_cache
        assert cache.stats["evictions"] == 1

    def test_cache_parsed_prompt(self, cache):
        """Test caching parsed prompt results."""
        prompt = "implement authentication system"
        result = {"intent": "implement", "keywords": ["authentication"]}
        metadata = {"confidence": 0.85}

        cache.cache_parsed_prompt(prompt, result, metadata)

        # Should be able to retrieve
        cached = cache.get_parsed_prompt(prompt)
        assert cached == result

    def test_get_parsed_prompt(self, cache):
        """Test getting cached parsed prompt."""
        prompt = "fix the bug"
        result = {"intent": "debug"}

        # Cache it first
        cache.cache_parsed_prompt(prompt, result)

        # Retrieve
        cached = cache.get_parsed_prompt(prompt)
        assert cached == result

    def test_cache_external_content(self, cache):
        """Test caching external content."""
        url = "https://github.com/org/repo/issues/123"
        content = {"title": "Issue", "body": "Description"}
        metadata = {"source": "github", "state": "open"}

        cache.cache_external_content(url, content, metadata)

        # Should be able to retrieve
        cached = cache.get_external_content(url)
        assert cached == content

    def test_cache_entities(self, cache):
        """Test caching entity recognition results."""
        text = "The UserController class"
        entities = [{"name": "UserController", "type": "class"}]
        confidence = 0.9

        cache.cache_entities(text, entities, confidence)

        cached = cache.get_entities(text)
        assert cached == entities

    def test_cache_intent(self, cache):
        """Test caching intent detection result."""
        text = "implement new feature"
        intent = {"type": "implement", "confidence": 0.85}

        cache.cache_intent(text, intent, confidence=0.85)

        cached = cache.get_intent(text)
        assert cached == intent

    def test_invalidate_pattern(self, cache):
        """Test invalidating cache entries by pattern."""
        # Add multiple entries
        cache.put("prompt:abc123", "value1")
        cache.put("prompt:def456", "value2")
        cache.put("external:url123", "value3")
        cache.put("entities:text123", "value4")

        # Invalidate all prompt entries
        count = cache.invalidate("prompt:")

        assert count == 2
        assert "prompt:abc123" not in cache.memory_cache
        assert "prompt:def456" not in cache.memory_cache
        assert "external:url123" in cache.memory_cache
        assert "entities:text123" in cache.memory_cache

    def test_clear_all(self, cache):
        """Test clearing all cache entries."""
        # Add some entries
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.stats["hits"] = 10
        cache.stats["misses"] = 5

        cache.clear_all()

        assert len(cache.memory_cache) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0

    def test_clear_all_with_disk(self, mock_cache_manager):
        """Test clearing all cache including disk."""
        cache = PromptCache(cache_manager=mock_cache_manager, enable_disk_cache=True)

        cache.clear_all()

        mock_cache_manager.general.clear.assert_called_once()

    @freeze_time("2024-01-15 10:00:00")
    def test_cleanup_expired(self, cache):
        """Test cleaning up expired entries."""
        # Add mix of expired and valid entries
        cache.memory_cache["expired1"] = CacheEntry(
            key="expired1",
            value="value1",
            created_at=datetime(2024, 1, 15, 8, 0, 0),
            accessed_at=datetime(2024, 1, 15, 8, 0, 0),
            ttl_seconds=3600,  # Expired
            hit_count=0,
        )

        cache.memory_cache["valid1"] = CacheEntry(
            key="valid1",
            value="value2",
            created_at=datetime(2024, 1, 15, 9, 30, 0),
            accessed_at=datetime(2024, 1, 15, 9, 30, 0),
            ttl_seconds=3600,  # Still valid
            hit_count=0,
        )

        cache.memory_cache["expired2"] = CacheEntry(
            key="expired2",
            value="value3",
            created_at=datetime(2024, 1, 15, 7, 0, 0),
            accessed_at=datetime(2024, 1, 15, 7, 0, 0),
            ttl_seconds=3600,  # Expired
            hit_count=0,
        )

        count = cache.cleanup_expired()

        assert count == 2
        assert "expired1" not in cache.memory_cache
        assert "expired2" not in cache.memory_cache
        assert "valid1" in cache.memory_cache

    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        # Generate some activity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        cache.get("key1")  # Hit again

        stats = cache.get_stats()

        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3
        assert stats["evictions"] == 0
        assert stats["expirations"] == 0
        assert stats["memory_entries"] == 2
        assert stats["memory_size"] > 0

    def test_get_stats_empty_cache(self, cache):
        """Test getting stats for empty cache."""
        stats = cache.get_stats()

        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0
        assert stats["memory_entries"] == 0
        assert stats["memory_size"] == 0

    def test_warm_cache(self, cache):
        """Test cache warming (placeholder functionality)."""
        common_prompts = ["implement authentication", "fix bug", "understand architecture"]

        # Should not raise any errors
        cache.warm_cache(common_prompts)

        # Currently a no-op, but should work
        assert True
