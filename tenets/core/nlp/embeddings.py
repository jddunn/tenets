"""Embedding generation and management.

This module provides local embedding generation using sentence transformers.
No external API calls are made - everything runs locally.
"""

import hashlib
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np

from tenets.utils.logger import get_logger

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class EmbeddingModel:
    """Base class for embedding models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model.

        Args:
            model_name: Name of the model to use
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM

    def encode(
        self, texts: Union[str, List[str]], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        raise NotImplementedError

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


class LocalEmbeddings(EmbeddingModel):
    """Local embedding generation using sentence transformers.

    This runs completely locally with no external API calls.
    Models are downloaded and cached by sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize local embeddings.

        Args:
            model_name: Sentence transformer model name
            device: Device to use ('cpu', 'cuda', or None for auto)
            cache_dir: Directory to cache models
        """
        super().__init__(model_name)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Sentence transformers not available. "
                "Install with: pip install sentence-transformers"
            )

        try:
            # Determine device
            if device:
                self.device = device
            else:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model
            self.model = SentenceTransformer(
                model_name, device=self.device, cache_folder=str(cache_dir) if cache_dir else None
            )

            # Get actual embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            self.logger.info(
                f"Loaded {model_name} on {self.device}, " f"embedding dim: {self.embedding_dim}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: Text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: L2 normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        # Handle single text
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )

        if single_text:
            return embeddings[0]

        return embeddings

    def encode_file(
        self, file_path: Path, chunk_size: int = 1000, overlap: int = 100
    ) -> np.ndarray:
        """Encode a file with chunking for long files.

        Args:
            file_path: Path to file
            chunk_size: Characters per chunk
            overlap: Overlap between chunks

        Returns:
            Mean pooled embedding for the file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.logger.warning(f"Failed to read {file_path}: {e}")
            return np.zeros(self.embedding_dim)

        if not content:
            return np.zeros(self.embedding_dim)

        # Chunk the content
        chunks = []
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)

        if not chunks:
            return np.zeros(self.embedding_dim)

        # Encode chunks
        chunk_embeddings = self.encode(chunks, show_progress=False)

        # Mean pooling
        return np.mean(chunk_embeddings, axis=0)


class FallbackEmbeddings(EmbeddingModel):
    """Fallback embeddings using TF-IDF when ML not available."""

    def __init__(self, embedding_dim: int = 384):
        """Initialize fallback embeddings.

        Args:
            embedding_dim: Dimension for embeddings
        """
        super().__init__(model_name="tfidf-fallback")
        self.embedding_dim = embedding_dim

        from .keyword_extractor import TFIDFExtractor

        self.tfidf = TFIDFExtractor()

    def encode(
        self, texts: Union[str, List[str]], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """Generate pseudo-embeddings using TF-IDF.

        Args:
            texts: Text or list of texts
            batch_size: Ignored
            show_progress: Ignored

        Returns:
            Numpy array of pseudo-embeddings
        """
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]

        # Fit TF-IDF on texts
        self.tfidf.fit(texts)
        vectors = self.tfidf.transform(texts)

        # Pad or truncate to embedding_dim
        embeddings = []
        for vec in vectors:
            if len(vec) < self.embedding_dim:
                # Pad with zeros
                padded = vec + [0.0] * (self.embedding_dim - len(vec))
                embeddings.append(padded)
            else:
                # Truncate
                embeddings.append(vec[: self.embedding_dim])

        embeddings = np.array(embeddings)

        if single_text:
            return embeddings[0]

        return embeddings


def create_embedding_model(
    prefer_local: bool = True, model_name: Optional[str] = None, **kwargs
) -> EmbeddingModel:
    """Create best available embedding model.

    Args:
        prefer_local: Prefer local models over API-based
        model_name: Specific model to use
        **kwargs: Additional arguments for model

    Returns:
        EmbeddingModel instance
    """
    logger = get_logger(__name__)

    # Try local embeddings first
    if prefer_local and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            return LocalEmbeddings(model_name or "all-MiniLM-L6-v2", **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create local embeddings: {e}")

    # Fallback to TF-IDF
    logger.info("Using TF-IDF fallback for embeddings")
    return FallbackEmbeddings()
