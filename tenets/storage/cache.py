"""Caching system for file analysis and other expensive operations.

This module provides a multi-level caching system with memory and disk caches
to speed up repeated operations.
"""

import hashlib
import json
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Dict
from contextlib import suppress

from tenets.config import TenetsConfig
from tenets.models.analysis import FileAnalysis
from tenets.utils.logger import get_logger

T = TypeVar("T")


class MemoryCache:
    """In-memory LRU cache for hot data."""

    def __init__(self, max_size: int = 1000):
        """Initialize memory cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self._cache = {}
        self._access_order = []
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        if key in self._cache:
            if key in self._access_order:
                self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict least recently used (single compound condition)
            if self._access_order and (lru_key := self._access_order.pop(0)) in self._cache:
                del self._cache[lru_key]
        self._cache[key] = value
        self._access_order.append(key)

    def delete(self, key: str) -> None:
        """Delete a key from the memory cache if present."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


class DiskCache:
    """SQLite-based disk cache for persistent storage."""

    def __init__(self, cache_dir: Path, name: str = "cache"):
        """Initialize disk cache.

        Args:
            cache_dir: Directory for cache storage
            name: Cache database name
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / f"{name}.db"
        self.logger = get_logger(__name__)

        self._init_db()

    def _init_db(self):
        """Initialize the cache database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    accessed_at TIMESTAMP NOT NULL,
                    ttl INTEGER,
                    metadata JSON
                )
            """
            )

            conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON cache(accessed_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON cache(ttl)")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT value, ttl, created_at FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()

            if row:
                value_blob, ttl, created_at = row

                # Check TTL
                if ttl:
                    created = datetime.fromisoformat(created_at)
                    if datetime.now() > created + timedelta(seconds=ttl):
                        # Expired
                        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                        return None

                # Update access time
                conn.execute(
                    "UPDATE cache SET accessed_at = ? WHERE key = ?", (datetime.now(), key)
                )

                # Deserialize value
                try:
                    # nosec B301 - Pickle limited to trusted internal cache storage
                    return pickle.loads(value_blob)  # nosec
                except Exception as e:
                    self.logger.warning(f"Failed to deserialize cache value: {e}")
                    return None

        return None

    def put(
        self, key: str, value: Any, ttl: Optional[int] = None, metadata: Optional[dict] = None
    ) -> None:
        """Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            metadata: Optional metadata
        """
        try:
            value_blob = pickle.dumps(value)
        except Exception as e:
            self.logger.warning(f"Failed to serialize value for caching: {e}")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, created_at, accessed_at, ttl, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    key,
                    value_blob,
                    datetime.now(),
                    datetime.now(),
                    ttl,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")

    def cleanup(self, max_age_days: int = 7, max_size_mb: int = 1000) -> int:
        """Clean up old or expired entries.

        Args:
            max_age_days: Delete entries older than this
            max_size_mb: Target maximum cache size in MB

        Returns:
            Number of entries deleted
        """
        deleted = 0

        with sqlite3.connect(self.db_path) as conn:
            # Delete expired entries
            cursor = conn.execute(
                """
                DELETE FROM cache
                WHERE (ttl IS NOT NULL AND datetime('now') > datetime(created_at, '+' || ttl || ' seconds'))
                   OR accessed_at < datetime('now', '-' || ? || ' days')
            """,
                (max_age_days,),
            )
            deleted += cursor.rowcount

            # Check size and remove LRU if needed
            cursor = conn.execute(
                "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
            )
            size_bytes = cursor.fetchone()[0]
            size_mb = size_bytes / (1024 * 1024)

            if size_mb > max_size_mb:
                # Delete least recently used until under limit
                cursor = conn.execute(
                    """
                    DELETE FROM cache
                    WHERE key IN (
                        SELECT key FROM cache
                        ORDER BY accessed_at ASC
                        LIMIT (SELECT COUNT(*) / 4 FROM cache)
                    )
                """
                )
                deleted += cursor.rowcount

                # VACUUM to reclaim space
                conn.execute("VACUUM")

        return deleted

    def close(self) -> None:
        """Close any open resources (no-op; uses per-call connections)."""
        return


class AnalysisCache:
    """Specialized cache for file analysis results."""

    def __init__(self, cache_dir: Path):
        """Initialize analysis cache.

        Args:
            cache_dir: Directory for cache storage
        """
        # Allow str inputs by converting to Path
        if not isinstance(cache_dir, Path):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.memory = MemoryCache(max_size=500)
        self.disk = DiskCache(cache_dir, name="analysis")
        self.logger = get_logger(__name__)

    def get_file_analysis(self, file_path: Path) -> Optional[FileAnalysis]:
        """Get cached analysis for a file.

        Args:
            file_path: Path to the file

        Returns:
            Cached FileAnalysis or None
        """
        # Generate cache key
        key = self._make_file_key(file_path)

        # Check memory cache first
        analysis = self.memory.get(key)
        if analysis:
            # Validate memory cache against file mtime too
            try:
                current_mtime = file_path.stat().st_mtime
                cached = self.disk.get(key)
                if cached and self._is_cache_valid(file_path, cached.get("mtime")):
                    return analysis
            except Exception:
                return analysis

        # Check disk cache
        cached = self.disk.get(key)
        if cached:
            # Validate cache
            if self._is_cache_valid(file_path, cached.get("mtime")):
                analysis = FileAnalysis.from_dict(cached["analysis"])
                # Promote to memory cache
                self.memory.put(key, analysis)
                return analysis
            else:
                # Invalidate stale cache
                self.disk.delete(key)
                self.memory.delete(key)

        return None

    def put_file_analysis(self, file_path: Path, analysis: FileAnalysis) -> None:
        """Cache file analysis.

        Args:
            file_path: Path to the file
            analysis: Analysis to cache
        """
        key = self._make_file_key(file_path)

        # Store in memory
        self.memory.put(key, analysis)

        # Store on disk with metadata
        try:
            mtime = file_path.stat().st_mtime
            cached_data = {
                "analysis": analysis.to_dict(),
                "mtime": mtime,
                "analyzer_version": "1.0",  # Track analyzer version
            }
            self.disk.put(key, cached_data, ttl=7 * 24 * 3600)  # 7 days TTL
        except Exception as e:
            self.logger.warning(f"Failed to cache analysis for {file_path}: {e}")

    def _make_file_key(self, file_path: Path) -> str:
        """Generate cache key for a file."""
        # Include absolute path to handle same filename in different directories
        abs_path = file_path.absolute()
        return hashlib.sha256(str(abs_path).encode()).hexdigest()

    def _is_cache_valid(self, file_path: Path, cached_mtime: Optional[float]) -> bool:
        """Check if cached data is still valid."""
        if cached_mtime is None:
            return False

        try:
            current_mtime = file_path.stat().st_mtime
            return float(current_mtime) == float(cached_mtime)
        except Exception:
            return False

    def close(self) -> None:
        """Close underlying caches."""
        with suppress(Exception):
            self.disk.close()


class CacheManager:
    """Manages all caching operations."""

    def __init__(self, config: TenetsConfig):
        """Initialize cache manager.

        Args:
            config: Tenets configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.logger = get_logger(__name__)

        # Initialize caches
        self.analysis = AnalysisCache(self.cache_dir / "analysis")
        self.general = DiskCache(self.cache_dir / "general")

        # Memory cache for hot data
        self.memory = MemoryCache(max_size=1000)

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        ttl: Optional[int] = None,
        use_memory: bool = True,
    ) -> T:
        """Get from cache or compute if missing.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Time to live in seconds
            use_memory: Whether to use memory cache

        Returns:
            Cached or computed value
        """
        # Check memory cache
        if use_memory:
            value = self.memory.get(key)
            if value is not None:
                return value

        # Check disk cache
        value = self.general.get(key)
        if value is not None:
            if use_memory:
                self.memory.put(key, value)
            return value

        # Compute value
        self.logger.debug(f"Cache miss for {key}, computing...")
        value = compute_fn()

        # Cache it
        if use_memory:
            self.memory.put(key, value)
        self.general.put(key, value, ttl=ttl)

        return value

    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        # Remove from memory cache
        self.memory.delete(key)
        # Remove from disk cache
        self.general.delete(key)

    def clear_all(self) -> None:
        """Clear all caches."""
        self.memory.clear()
        self.analysis.memory.clear()
        self.analysis.disk.clear()
        self.general.clear()

    def cleanup(self) -> dict[str, int]:
        """Clean up old cache entries.

        Returns:
            Statistics about cleanup
        """
        stats = {
            "analysis_deleted": self.analysis.disk.cleanup(
                max_age_days=self.config.cache_ttl_days,
                max_size_mb=self.config.max_cache_size_mb // 2,
            ),
            "general_deleted": self.general.cleanup(
                max_age_days=self.config.cache_ttl_days,
                max_size_mb=self.config.max_cache_size_mb // 2,
            ),
        }

        self.logger.info(f"Cache cleanup: {stats}")
        return stats
