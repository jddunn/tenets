"""Embedding cache management.

This module provides caching for embeddings to avoid recomputation
of expensive embedding operations.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from tenets.storage.cache import DiskCache
from tenets.utils.logger import get_logger


class EmbeddingCache:
    """Cache for embedding vectors.
    
    Uses a two-level cache:
    1. Memory cache for hot embeddings
    2. Disk cache for persistence
    """
    
    def __init__(
        self,
        cache_dir: Path,
        max_memory_items: int = 1000,
        ttl_days: int = 30
    ):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory for disk cache
            max_memory_items: Maximum items in memory cache
            ttl_days: Time to live for cached embeddings
        """
        self.logger = get_logger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._access_order: list[str] = []
        self.max_memory_items = max_memory_items
        
        # Disk cache
        self.disk_cache = DiskCache(
            self.cache_dir,
            name="embeddings"
        )
        self.ttl_seconds = ttl_days * 24 * 3600
        
    def get(
        self,
        text: str,
        model_name: str = "default"
    ) -> Optional[np.ndarray]:
        """Get cached embedding.
        
        Args:
            text: Text that was embedded
            model_name: Model used for embedding
            
        Returns:
            Cached embedding or None
        """
        key = self._make_key(text, model_name)
        
        # Check memory cache
        if key in self._memory_cache:
            # Move to end (LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return self._memory_cache[key]
            
        # Check disk cache
        cached = self.disk_cache.get(key)
        if cached is not None:
            # Validate it's an embedding
            if isinstance(cached, np.ndarray):
                # Promote to memory cache
                self._add_to_memory(key, cached)
                return cached
            else:
                self.logger.warning(f"Invalid cached embedding for {key}")
                self.disk_cache.delete(key)
                
        return None
        
    def put(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: str = "default"
    ):
        """Cache an embedding.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
            model_name: Model used
        """
        key = self._make_key(text, model_name)
        
        # Add to memory cache
        self._add_to_memory(key, embedding)
        
        # Add to disk cache
        self.disk_cache.put(
            key,
            embedding,
            ttl=self.ttl_seconds,
            metadata={
                'model': model_name,
                'dim': embedding.shape[0],
                'text_preview': text[:100]
            }
        )
        
    def get_batch(
        self,
        texts: list[str],
        model_name: str = "default"
    ) -> Dict[str, Optional[np.ndarray]]:
        """Get multiple cached embeddings.
        
        Args:
            texts: List of texts
            model_name: Model used
            
        Returns:
            Dict mapping text to embedding (or None if not cached)
        """
        results = {}
        
        for text in texts:
            results[text] = self.get(text, model_name)
            
        return results
        
    def put_batch(
        self,
        embeddings: Dict[str, np.ndarray],
        model_name: str = "default"
    ):
        """Cache multiple embeddings.
        
        Args:
            embeddings: Dict mapping text to embedding
            model_name: Model used
        """
        for text, embedding in embeddings.items():
            self.put(text, embedding, model_name)
            
    def _make_key(self, text: str, model_name: str) -> str:
        """Generate cache key.
        
        Args:
            text: Input text
            model_name: Model name
            
        Returns:
            Cache key
        """
        # Use first 50 chars + hash for readable keys
        text_preview = text[:50].replace('/', '_').replace(' ', '_')
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        
        return f"{model_name}:{text_preview}:{text_hash}"
        
    def _add_to_memory(self, key: str, embedding: np.ndarray):
        """Add to memory cache with LRU eviction.
        
        Args:
            key: Cache key
            embedding: Embedding to cache
        """
        # Check if we need to evict
        if len(self._memory_cache) >= self.max_memory_items:
            # Evict least recently used
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._memory_cache[lru_key]
                
        # Add new item
        self._memory_cache[key] = embedding
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
    def clear_memory(self):
        """Clear memory cache."""
        self._memory_cache.clear()
        self._access_order.clear()
        
    def clear_all(self):
        """Clear all caches."""
        self.clear_memory()
        self.disk_cache.clear()
        
    def cleanup(self) -> int:
        """Clean up old cache entries.
        
        Returns:
            Number of entries deleted
        """
        return self.disk_cache.cleanup(
            max_age_days=self.ttl_seconds // (24 * 3600)
        )
        
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return {
            'memory_items': len(self._memory_cache),
            'memory_size_mb': sum(
                e.nbytes for e in self._memory_cache.values()
            ) / (1024 * 1024),
            'access_order_length': len(self._access_order)
        }