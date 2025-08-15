"""Similarity computation utilities.

This module provides various similarity metrics including
cosine similarity and semantic similarity using embeddings.
"""

import math
from typing import List, Optional, Union, Tuple
import numpy as np

from tenets.utils.logger import get_logger


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    # Handle different input types
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()

    # Check dimensions
    if vec1.shape != vec2.shape:
        raise ValueError(f"Vectors must have same shape: {vec1.shape} != {vec2.shape}")

    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)

    # Clamp to [-1, 1] to handle floating point errors
    return float(np.clip(similarity, -1.0, 1.0))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance (>= 0)
    """
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()

    if vec1.shape != vec2.shape:
        raise ValueError(f"Vectors must have same shape: {vec1.shape} != {vec2.shape}")

    return float(np.linalg.norm(vec1 - vec2))


def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Manhattan (L1) distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Manhattan distance (>= 0)
    """
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()

    if vec1.shape != vec2.shape:
        raise ValueError(f"Vectors must have same shape: {vec1.shape} != {vec2.shape}")

    return float(np.sum(np.abs(vec1 - vec2)))


class SemanticSimilarity:
    """Compute semantic similarity using embeddings."""

    def __init__(self, model: Optional["EmbeddingModel"] = None, cache_embeddings: bool = True):
        """Initialize semantic similarity.

        Args:
            model: Embedding model to use (creates default if None)
            cache_embeddings: Cache computed embeddings
        """
        self.logger = get_logger(__name__)

        if model is None:
            from .embeddings import create_embedding_model

            self.model = create_embedding_model()
        else:
            self.model = model

        self.cache_embeddings = cache_embeddings
        self._cache = {} if cache_embeddings else None

    def compute(self, text1: str, text2: str, metric: str = "cosine") -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')

        Returns:
            Similarity score
        """
        # Get embeddings
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        # Compute similarity
        if metric == "cosine":
            return cosine_similarity(emb1, emb2)
        elif metric == "euclidean":
            # Convert distance to similarity
            dist = euclidean_distance(emb1, emb2)
            return 1.0 / (1.0 + dist)
        elif metric == "manhattan":
            # Convert distance to similarity
            dist = manhattan_distance(emb1, emb2)
            return 1.0 / (1.0 + dist)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def compute_batch(
        self, query: str, documents: List[str], metric: str = "cosine", top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Compute similarity between query and multiple documents.

        Args:
            query: Query text
            documents: List of documents
            metric: Similarity metric
            top_k: Return only top K results

        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        if not documents:
            return []

        # Get query embedding
        query_emb = self._get_embedding(query)

        # Get document embeddings (batch encode for efficiency)
        doc_embeddings = self.model.encode(documents)

        # Compute similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            if metric == "cosine":
                sim = cosine_similarity(query_emb, doc_emb)
            elif metric == "euclidean":
                dist = euclidean_distance(query_emb, doc_emb)
                sim = 1.0 / (1.0 + dist)
            elif metric == "manhattan":
                dist = manhattan_distance(query_emb, doc_emb)
                sim = 1.0 / (1.0 + dist)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            return similarities[:top_k]

        return similarities

    def find_similar(
        self, query: str, documents: List[str], threshold: float = 0.7, metric: str = "cosine"
    ) -> List[Tuple[int, float]]:
        """Find documents similar to query above threshold.

        Args:
            query: Query text
            documents: List of documents
            threshold: Similarity threshold
            metric: Similarity metric

        Returns:
            List of (index, similarity) for documents above threshold
        """
        similarities = self.compute_batch(query, documents, metric)
        return [(i, sim) for i, sim in similarities if sim >= threshold]

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self.cache_embeddings and text in self._cache:
            return self._cache[text]

        embedding = self.model.encode(text)

        if self.cache_embeddings:
            self._cache[text] = embedding

        return embedding

    def clear_cache(self):
        """Clear embedding cache."""
        if self._cache:
            self._cache.clear()
