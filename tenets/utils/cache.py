"""Caching utilities for tenets.

Provides LRU caching with TTL support, file-based caching for expensive
computations like embeddings and ranking scores.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with value and metadata."""

    value: T
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry has expired based on TTL."""
        if ttl_seconds <= 0:
            return False  # No TTL, never expires
        return time.time() - self.created_at > ttl_seconds

    def touch(self) -> None:
        """Update last access time and increment counter."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache(Generic[T]):
    """Thread-safe LRU cache with optional TTL.

    Provides efficient caching with automatic eviction of least recently
    used entries when capacity is reached.

    Attributes:
        max_size: Maximum number of entries
        ttl_seconds: Time-to-live in seconds (0 = no expiration)
        hits: Number of cache hits
        misses: Number of cache misses
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 0,
        name: str = "cache",
    ):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries to store
            ttl_seconds: Time-to-live for entries (0 = no expiration)
            name: Name for logging/debugging
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.name = name

        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self.misses += 1
                return None

            if entry.is_expired(self.ttl_seconds):
                del self._cache[key]
                self.misses += 1
                return None

            entry.touch()
            self.hits += 1
            return entry.value

    def set(self, key: str, value: T, size_bytes: int = 0) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Optional size estimate for memory tracking
        """
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = CacheEntry(
                value=value,
                size_bytes=size_bytes,
            )

    def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            return count

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed,
        )
        del self._cache[lru_key]

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_size = sum(e.size_bytes for e in self._cache.values())
            return {
                "name": self.name,
                "size": self.size,
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hit_rate,
                "total_bytes": total_size,
                "ttl_seconds": self.ttl_seconds,
            }


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a cache key from arguments.

    Creates a deterministic hash from the provided arguments that can be
    used as a cache key.

    Args:
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key

    Returns:
        A hexadecimal hash string
    """
    # Serialize arguments to JSON for consistent hashing
    key_data = {
        "args": [_serialize_arg(a) for a in args],
        "kwargs": {k: _serialize_arg(v) for k, v in sorted(kwargs.items())},
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _serialize_arg(arg: Any) -> Any:
    """Serialize an argument for cache key generation."""
    if isinstance(arg, Path):
        return str(arg)
    if isinstance(arg, (list, tuple)):
        return [_serialize_arg(a) for a in arg]
    if isinstance(arg, dict):
        return {str(k): _serialize_arg(v) for k, v in arg.items()}
    if hasattr(arg, "__dict__"):
        return str(type(arg).__name__)
    return arg


class FileContentCache:
    """Cache for file contents with modification tracking.

    Automatically invalidates cache entries when files are modified.

    Attributes:
        max_size: Maximum number of files to cache
        max_file_size: Maximum file size to cache (bytes)
    """

    def __init__(
        self,
        max_size: int = 500,
        max_file_size: int = 1024 * 1024,  # 1MB default
    ):
        """Initialize file content cache.

        Args:
            max_size: Maximum number of files to cache
            max_file_size: Maximum file size to cache in bytes
        """
        self._cache: LRUCache[tuple[float, str]] = LRUCache(
            max_size=max_size,
            name="file_content",
        )
        self.max_file_size = max_file_size

    def get(self, path: Path) -> Optional[str]:
        """Get cached file content if still valid.

        Args:
            path: Path to file

        Returns:
            File content or None if not cached/stale
        """
        key = str(path.resolve())
        entry = self._cache.get(key)

        if entry is None:
            return None

        mtime, content = entry

        # Check if file was modified
        try:
            current_mtime = path.stat().st_mtime
            if current_mtime > mtime:
                self._cache.delete(key)
                return None
        except OSError:
            self._cache.delete(key)
            return None

        return content

    def set(self, path: Path, content: str) -> bool:
        """Cache file content.

        Args:
            path: Path to file
            content: File content

        Returns:
            True if cached, False if file too large
        """
        if len(content) > self.max_file_size:
            return False

        key = str(path.resolve())
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = time.time()

        self._cache.set(key, (mtime, content), size_bytes=len(content))
        return True

    def invalidate(self, path: Path) -> bool:
        """Invalidate cached content for a file.

        Args:
            path: Path to file

        Returns:
            True if entry was removed, False if not cached
        """
        return self._cache.delete(str(path.resolve()))

    def clear(self) -> int:
        """Clear all cached content.

        Returns:
            Number of entries cleared
        """
        return self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()


class EmbeddingCache:
    """Cache for text embeddings.

    Stores computed embeddings to avoid re-computation for repeated
    queries or unchanged files.

    Attributes:
        max_size: Maximum number of embeddings to cache
        ttl_seconds: Time-to-live for cached embeddings
    """

    def __init__(
        self,
        max_size: int = 2000,
        ttl_seconds: float = 3600,  # 1 hour default
    ):
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Time-to-live for cached embeddings
        """
        self._cache: LRUCache[list[float]] = LRUCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            name="embeddings",
        )

    def get(self, text: str, model: str = "default") -> Optional[list[float]]:
        """Get cached embedding.

        Args:
            text: Text that was embedded
            model: Model used for embedding

        Returns:
            Cached embedding vector or None
        """
        key = cache_key(text[:500], model)  # Truncate for key
        return self._cache.get(key)

    def set(
        self,
        text: str,
        embedding: list[float],
        model: str = "default",
    ) -> None:
        """Cache an embedding.

        Args:
            text: Text that was embedded
            embedding: Embedding vector
            model: Model used for embedding
        """
        key = cache_key(text[:500], model)
        self._cache.set(key, embedding, size_bytes=len(embedding) * 8)

    def clear(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of entries cleared
        """
        return self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()


class RankingScoreCache:
    """Cache for file ranking scores.

    Caches ranking scores for file-prompt pairs to speed up repeated
    queries on unchanged files.

    Attributes:
        max_size: Maximum number of scores to cache
        ttl_seconds: Time-to-live for cached scores
    """

    def __init__(
        self,
        max_size: int = 5000,
        ttl_seconds: float = 300,  # 5 minutes default
    ):
        """Initialize ranking score cache.

        Args:
            max_size: Maximum number of scores to cache
            ttl_seconds: Time-to-live for cached scores
        """
        self._cache: LRUCache[Dict[str, Any]] = LRUCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            name="ranking_scores",
        )

    def get(
        self,
        file_path: Path,
        prompt_hash: str,
        file_mtime: float,
        algorithm: str = "balanced",
    ) -> Optional[Dict[str, Any]]:
        """Get cached ranking score.

        Args:
            file_path: Path to file
            prompt_hash: Hash of the prompt
            file_mtime: File modification time
            algorithm: Ranking algorithm used

        Returns:
            Cached score data or None
        """
        key = cache_key(str(file_path), prompt_hash, algorithm)
        entry = self._cache.get(key)

        if entry is None:
            return None

        # Check if file was modified since caching
        if entry.get("mtime", 0) < file_mtime:
            self._cache.delete(key)
            return None

        return entry

    def set(
        self,
        file_path: Path,
        prompt_hash: str,
        file_mtime: float,
        score: float,
        factors: Dict[str, float],
        algorithm: str = "balanced",
    ) -> None:
        """Cache a ranking score.

        Args:
            file_path: Path to file
            prompt_hash: Hash of the prompt
            file_mtime: File modification time
            score: Computed relevance score
            factors: Individual ranking factors
            algorithm: Ranking algorithm used
        """
        key = cache_key(str(file_path), prompt_hash, algorithm)
        self._cache.set(
            key,
            {
                "score": score,
                "factors": factors,
                "mtime": file_mtime,
                "algorithm": algorithm,
            },
        )

    def clear(self) -> int:
        """Clear all cached scores.

        Returns:
            Number of entries cleared
        """
        return self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()


# Global cache instances for shared use
_file_cache: Optional[FileContentCache] = None
_embedding_cache: Optional[EmbeddingCache] = None
_ranking_cache: Optional[RankingScoreCache] = None
_cache_lock = threading.Lock()


def get_file_cache() -> FileContentCache:
    """Get or create the global file content cache."""
    global _file_cache
    with _cache_lock:
        if _file_cache is None:
            _file_cache = FileContentCache()
        return _file_cache


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache."""
    global _embedding_cache
    with _cache_lock:
        if _embedding_cache is None:
            _embedding_cache = EmbeddingCache()
        return _embedding_cache


def get_ranking_cache() -> RankingScoreCache:
    """Get or create the global ranking score cache."""
    global _ranking_cache
    with _cache_lock:
        if _ranking_cache is None:
            _ranking_cache = RankingScoreCache()
        return _ranking_cache


def clear_all_caches() -> Dict[str, int]:
    """Clear all global caches.

    Returns:
        Dictionary with count of cleared entries per cache
    """
    global _file_cache, _embedding_cache, _ranking_cache

    with _cache_lock:
        results = {}

        if _file_cache is not None:
            results["file_content"] = _file_cache.clear()

        if _embedding_cache is not None:
            results["embeddings"] = _embedding_cache.clear()

        if _ranking_cache is not None:
            results["ranking_scores"] = _ranking_cache.clear()

        return results


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches.

    Returns:
        Dictionary with stats for each cache type
    """
    stats = {}

    file_cache = get_file_cache()
    stats["file_content"] = file_cache.stats()

    embedding_cache = get_embedding_cache()
    stats["embeddings"] = embedding_cache.stats()

    ranking_cache = get_ranking_cache()
    stats["ranking_scores"] = ranking_cache.stats()

    return stats
