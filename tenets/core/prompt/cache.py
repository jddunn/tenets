"""Caching system for prompt parsing results.

Provides intelligent caching for parsed prompts, external content fetches,
and entity recognition results with proper invalidation strategies.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from tenets.utils.logger import get_logger

# Optional import for storage
try:
    from tenets.storage.cache import CacheManager
except ImportError:
    CacheManager = None


@dataclass
class CacheEntry:
    """A cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    ttl_seconds: int
    hit_count: int = 0
    metadata: Dict[str, Any] = None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # No expiration

        age = datetime.now() - self.created_at
        return age.total_seconds() > self.ttl_seconds

    def touch(self):
        """Update access time and increment hit count."""
        self.accessed_at = datetime.now()
        self.hit_count += 1


class PromptCache:
    """Intelligent caching for prompt parsing operations."""

    # Default TTLs for different content types (in seconds)
    DEFAULT_TTLS = {
        "parsed_prompt": 3600,  # 1 hour for parsed prompts
        "external_content": 21600,  # 6 hours for external content
        "entity_recognition": 1800,  # 30 minutes for entity recognition
        "intent_detection": 1800,  # 30 minutes for intent detection
        "temporal_parsing": 3600,  # 1 hour for temporal parsing
    }

    # TTL modifiers based on content characteristics
    TTL_MODIFIERS = {
        "github_open": 0.25,  # 25% of normal TTL for open issues/PRs
        "github_closed": 4.0,  # 400% of normal TTL for closed issues/PRs
        "jira_active": 0.5,  # 50% of normal TTL for active tickets
        "notion_page": 2.0,  # 200% of normal TTL for Notion pages
        "high_confidence": 1.5,  # 150% of normal TTL for high confidence results
        "low_confidence": 0.5,  # 50% of normal TTL for low confidence results
    }

    def __init__(
        self,
        cache_manager: Optional[Any] = None,
        enable_memory_cache: bool = True,
        enable_disk_cache: bool = True,
        memory_cache_size: int = 100,
    ):
        """Initialize prompt cache.

        Args:
            cache_manager: External cache manager to use
            enable_memory_cache: Whether to use in-memory caching
            enable_disk_cache: Whether to use disk caching
            memory_cache_size: Maximum items in memory cache
        """
        self.logger = get_logger(__name__)
        self.cache_manager = cache_manager if cache_manager and CacheManager else None
        self.enable_memory = enable_memory_cache
        self.enable_disk = enable_disk_cache and self.cache_manager is not None

        # In-memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_cache_size = memory_cache_size

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def _generate_key(self, prefix: str, content: Union[str, Dict, List]) -> str:
        """Generate a cache key from content.

        Args:
            prefix: Key prefix for namespacing
            content: Content to hash

        Returns:
            Cache key string
        """
        # Convert content to string for hashing
        if isinstance(content, (dict, list)):
            content_str = json.dumps(content, sort_keys=True)
        else:
            content_str = str(content)

        # Generate hash
        hash_obj = hashlib.sha256(content_str.encode())
        hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars

        return f"{prefix}:{hash_hex}"

    def _calculate_ttl(
        self, base_ttl: int, content_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Calculate dynamic TTL based on content characteristics.

        Args:
            base_ttl: Base TTL in seconds
            content_type: Type of content being cached
            metadata: Additional metadata for TTL calculation

        Returns:
            Adjusted TTL in seconds
        """
        ttl = base_ttl

        if metadata:
            # Apply modifiers based on metadata
            if metadata.get("source") == "github":
                if metadata.get("state") == "open":
                    ttl = int(ttl * self.TTL_MODIFIERS["github_open"])
                elif metadata.get("state") == "closed":
                    ttl = int(ttl * self.TTL_MODIFIERS["github_closed"])

            elif metadata.get("source") == "jira":
                if metadata.get("status") in ["In Progress", "Open", "To Do"]:
                    ttl = int(ttl * self.TTL_MODIFIERS["jira_active"])

            elif metadata.get("source") == "notion":
                ttl = int(ttl * self.TTL_MODIFIERS["notion_page"])

            # Confidence-based modifiers (only if explicitly provided)
            if "confidence" in metadata:
                confidence = metadata.get("confidence")
                if confidence is not None:
                    try:
                        conf_val = float(confidence)
                    except (TypeError, ValueError):
                        conf_val = None
                    if conf_val is not None:
                        if conf_val >= 0.8:
                            ttl = int(ttl * self.TTL_MODIFIERS["high_confidence"])
                        elif conf_val < 0.5:
                            ttl = int(ttl * self.TTL_MODIFIERS["low_confidence"])

        # Ensure reasonable bounds
        min_ttl = 60  # 1 minute minimum
        max_ttl = 86400  # 24 hours maximum

        return max(min_ttl, min(ttl, max_ttl))

    def get(self, key: str, check_disk: bool = True) -> Optional[Any]:
        """Get a value from cache.

        Args:
            key: Cache key
            check_disk: Whether to check disk cache if not in memory

        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        if self.enable_memory and key in self.memory_cache:
            entry = self.memory_cache[key]

            if entry.is_expired():
                # Remove expired entry
                del self.memory_cache[key]
                self.stats["expirations"] += 1
                self.logger.debug(f"Cache expired for key: {key}")
            else:
                # Update access time
                entry.touch()
                self.stats["hits"] += 1
                self.logger.debug(f"Cache hit for key: {key} (memory)")
                return entry.value

        # Check disk cache if enabled
        if check_disk and self.enable_disk and self.cache_manager:
            disk_value = self.cache_manager.general.get(key)
            if disk_value is not None:
                self.stats["hits"] += 1
                self.logger.debug(f"Cache hit for key: {key} (disk)")

                # Promote to memory cache
                if self.enable_memory:
                    self._add_to_memory(
                        key, disk_value, self.DEFAULT_TTLS.get("parsed_prompt", 3600)
                    )

                return disk_value

        self.stats["misses"] += 1
        self.logger.debug(f"Cache miss for key: {key}")
        return None

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        write_disk: bool = True,
    ) -> None:
        """Put a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds (uses default if not specified)
            metadata: Additional metadata for TTL calculation
            write_disk: Whether to write to disk cache
        """
        # Use default TTL if not specified
        if ttl_seconds is None:
            ttl_seconds = self.DEFAULT_TTLS.get("parsed_prompt", 3600)

        # Add to memory cache
        if self.enable_memory:
            self._add_to_memory(key, value, ttl_seconds, metadata)

        # Add to disk cache
        if write_disk and self.enable_disk and self.cache_manager:
            self.cache_manager.general.put(key, value, ttl=ttl_seconds, metadata=metadata)
            self.logger.debug(f"Cached to disk: {key} (TTL: {ttl_seconds}s)")

    def _add_to_memory(
        self, key: str, value: Any, ttl_seconds: int, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add entry to memory cache with LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds
            metadata: Additional metadata
        """
        # Check if we need to evict
        if len(self.memory_cache) >= self.memory_cache_size:
            # Find least recently used entry
            lru_key = min(self.memory_cache.keys(), key=lambda k: self.memory_cache[k].accessed_at)
            del self.memory_cache[lru_key]
            self.stats["evictions"] += 1
            self.logger.debug(f"Evicted LRU entry: {lru_key}")

        # Add new entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            ttl_seconds=ttl_seconds,
            hit_count=0,
            metadata=metadata,
        )
        self.memory_cache[key] = entry
        self.logger.debug(f"Cached to memory: {key} (TTL: {ttl_seconds}s)")

    def cache_parsed_prompt(
        self, prompt: str, result: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cache a parsed prompt result.

        Args:
            prompt: Original prompt text
            result: Parsing result
            metadata: Additional metadata
        """
        key = self._generate_key("prompt", prompt)
        ttl = self._calculate_ttl(self.DEFAULT_TTLS["parsed_prompt"], "parsed_prompt", metadata)
        self.put(key, result, ttl, metadata)

    def get_parsed_prompt(self, prompt: str) -> Optional[Any]:
        """Get cached parsed prompt result.

        Args:
            prompt: Original prompt text

        Returns:
            Cached result or None
        """
        key = self._generate_key("prompt", prompt)
        return self.get(key)

    def cache_external_content(
        self, url: str, content: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cache external content fetch result.

        Args:
            url: URL that was fetched
            content: Fetched content
            metadata: Additional metadata (source, state, etc.)
        """
        key = self._generate_key("external", url)

        # Add URL to metadata
        if metadata is None:
            metadata = {}
        metadata["url"] = url

        ttl = self._calculate_ttl(
            self.DEFAULT_TTLS["external_content"], "external_content", metadata
        )
        self.put(key, content, ttl, metadata)

    def get_external_content(self, url: str) -> Optional[Any]:
        """Get cached external content.

        Args:
            url: URL to check

        Returns:
            Cached content or None
        """
        key = self._generate_key("external", url)
        return self.get(key)

    def cache_entities(self, text: str, entities: List[Any], confidence: float = 0.0) -> None:
        """Cache entity recognition results.

        Args:
            text: Text that was analyzed
            entities: Recognized entities
            confidence: Average confidence score
        """
        key = self._generate_key("entities", text)
        metadata = {"confidence": confidence, "count": len(entities)}
        ttl = self._calculate_ttl(
            self.DEFAULT_TTLS["entity_recognition"], "entity_recognition", metadata
        )
        self.put(key, entities, ttl, metadata)

    def get_entities(self, text: str) -> Optional[List[Any]]:
        """Get cached entity recognition results.

        Args:
            text: Text to check

        Returns:
            Cached entities or None
        """
        key = self._generate_key("entities", text)
        return self.get(key)

    def cache_intent(self, text: str, intent: Any, confidence: float = 0.0) -> None:
        """Cache intent detection result.

        Args:
            text: Text that was analyzed
            intent: Detected intent
            confidence: Confidence score
        """
        key = self._generate_key("intent", text)
        metadata = {"confidence": confidence}
        ttl = self._calculate_ttl(
            self.DEFAULT_TTLS["intent_detection"], "intent_detection", metadata
        )
        self.put(key, intent, ttl, metadata)

    def get_intent(self, text: str) -> Optional[Any]:
        """Get cached intent detection result.

        Args:
            text: Text to check

        Returns:
            Cached intent or None
        """
        key = self._generate_key("intent", text)
        return self.get(key)

    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.

        Args:
            pattern: Key pattern to match (prefix)

        Returns:
            Number of entries invalidated
        """
        count = 0

        # Invalidate memory cache
        if self.enable_memory:
            keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(pattern)]
            for key in keys_to_remove:
                del self.memory_cache[key]
                count += 1

            if count > 0:
                self.logger.info(f"Invalidated {count} memory cache entries matching: {pattern}")

        # Invalidate disk cache
        if self.enable_disk and self.cache_manager:
            # Note: This assumes the cache manager supports pattern-based deletion
            # If not, we'd need to iterate through all keys
            pass

        return count

    def clear_all(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        if self.enable_memory:
            self.memory_cache.clear()
            self.logger.info("Cleared memory cache")

        # Clear disk cache
        if self.enable_disk and self.cache_manager:
            self.cache_manager.general.clear()
            self.logger.info("Cleared disk cache")

        # Reset statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        count = 0

        if self.enable_memory:
            expired_keys = [k for k, v in self.memory_cache.items() if v.is_expired()]
            for key in expired_keys:
                del self.memory_cache[key]
                count += 1

            if count > 0:
                self.logger.info(f"Cleaned up {count} expired cache entries")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "expirations": self.stats["expirations"],
            "memory_entries": len(self.memory_cache) if self.enable_memory else 0,
            "memory_size": (
                sum(len(str(e.value)) for e in self.memory_cache.values())
                if self.enable_memory
                else 0
            ),
        }

    def warm_cache(self, common_prompts: List[str]) -> None:
        """Pre-warm cache with common prompts.

        Args:
            common_prompts: List of common prompts to pre-cache
        """
        # This would be called during initialization to pre-populate
        # the cache with commonly used prompts
        pass
