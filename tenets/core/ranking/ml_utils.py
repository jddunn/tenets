"""Machine learning utilities for ranking.

This module provides ML-based ranking capabilities including semantic similarity
using embeddings, neural reranking, and other advanced techniques. All ML
dependencies are lazily loaded to keep the core package lightweight.

The module gracefully handles missing dependencies and provides fallback behavior
when ML features are not available.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tenets.utils.logger import get_logger

# Lazy imports for ML dependencies
_TORCH_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False
_SKLEARN_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore

try:
    import transformers
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    transformers = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    _SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None  # type: ignore
    sklearn_cosine_similarity = None  # type: ignore


class EmbeddingModel:
    """Wrapper for embedding models with caching.
    
    Provides a unified interface for different embedding models
    (SentenceTransformers, HuggingFace, OpenAI, etc.) with built-in
    caching and batch processing capabilities.
    
    Attributes:
        model_name: Name of the embedding model
        model: The loaded model instance
        cache_dir: Directory for caching embeddings
        embeddings_cache: In-memory cache for embeddings
        device: Device to run model on (cpu/cuda)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        """Initialize embedding model.
        
        Args:
            model_name: Name of the model to load
            cache_dir: Directory for caching embeddings
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.model = None
        self.cache_dir = cache_dir
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Determine device
        if device:
            self.device = device
        elif _TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the embedding model."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning(
                "SentenceTransformers not available. "
                "Install with: pip install sentence-transformers"
            )
            return
            
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.logger.info(f"Loaded embedding model: {self.model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        use_cache: bool = True
    ) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            use_cache: Use cached embeddings if available
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model:
            # Fallback to TF-IDF if available
            if _SKLEARN_AVAILABLE:
                return self._tfidf_fallback(texts)
            else:
                # Return random embeddings as last resort
                return self._random_fallback(texts)
                
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
            
        # Check cache
        embeddings = []
        texts_to_encode = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            if use_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self.embeddings_cache:
                    embeddings.append(self.embeddings_cache[cache_key])
                else:
                    texts_to_encode.append(text)
                    cache_indices.append(i)
            else:
                texts_to_encode.append(text)
                cache_indices.append(i)
                
        # Encode new texts
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Cache new embeddings
            if use_cache:
                for text, embedding in zip(texts_to_encode, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self.embeddings_cache[cache_key] = embedding
                    
            # Merge with cached embeddings
            if embeddings:
                # Reconstruct full list in original order
                result = np.zeros((len(texts), new_embeddings.shape[1]))
                cached_idx = 0
                new_idx = 0
                
                for i in range(len(texts)):
                    if i in cache_indices:
                        result[i] = new_embeddings[new_idx]
                        new_idx += 1
                    else:
                        result[i] = embeddings[cached_idx]
                        cached_idx += 1
                        
                embeddings = result
            else:
                embeddings = new_embeddings
        else:
            embeddings = np.array(embeddings)
            
        if single_text:
            return embeddings[0]
        return embeddings
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key string
        """
        # Use first 100 chars + hash for key
        text_preview = text[:100]
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{text_preview}_{text_hash}"
        
    def _tfidf_fallback(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Fallback to TF-IDF embeddings.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            TF-IDF vectors
        """
        if isinstance(texts, str):
            texts = [texts]
            
        vectorizer = TfidfVectorizer(max_features=384)  # Similar dimension to small models
        try:
            embeddings = vectorizer.fit_transform(texts).toarray()
            return embeddings
        except Exception:
            return self._random_fallback(texts)
            
    def _random_fallback(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate random embeddings as last resort.
        
        Args:
            texts: Text or list of texts
            
        Returns:
            Random vectors
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Generate deterministic "random" embeddings based on text hash
        embeddings = []
        for text in texts:
            # Use text hash as seed for reproducibility
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            embedding = np.random.randn(384)  # Standard embedding size
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
            
        return np.array(embeddings)
        
    def save_cache(self, path: Path):
        """Save embeddings cache to disk.
        
        Args:
            path: Path to save cache
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            self.logger.info(f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
            
    def load_cache(self, path: Path):
        """Load embeddings cache from disk.
        
        Args:
            path: Path to load cache from
        """
        try:
            if path.exists():
                with open(path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embeddings_cache)} embeddings from cache")
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")


def load_embedding_model(
    model_name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None
) -> Optional[EmbeddingModel]:
    """Load an embedding model.
    
    Args:
        model_name: Model name (default: all-MiniLM-L6-v2)
        cache_dir: Directory for caching
        device: Device to run on
        
    Returns:
        EmbeddingModel instance or None if unavailable
    """
    logger = get_logger(__name__)
    
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning(
            "ML features not available. Install with: pip install tenets[ml]"
        )
        return None
        
    try:
        model_name = model_name or "all-MiniLM-L6-v2"
        return EmbeddingModel(model_name, cache_dir, device)
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


def compute_similarity(
    model: EmbeddingModel,
    text1: str,
    text2: str,
    cache: Optional[Dict[str, Any]] = None
) -> float:
    """Compute semantic similarity between two texts.
    
    Args:
        model: Embedding model
        text1: First text
        text2: Second text
        cache: Optional cache dictionary
        
    Returns:
        Similarity score (0-1)
    """
    if not model:
        return 0.0
        
    try:
        # Get embeddings
        embeddings = model.encode([text1, text2])
        
        # Compute cosine similarity
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Similarity computation failed: {e}")
        return 0.0


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    # Handle different input types
    if _TORCH_AVAILABLE and torch is not None:
        if isinstance(vec1, torch.Tensor):
            vec1 = vec1.cpu().numpy()
        if isinstance(vec2, torch.Tensor):
            vec2 = vec2.cpu().numpy()
            
    # Ensure numpy arrays
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def batch_similarity(
    model: EmbeddingModel,
    query: str,
    documents: List[str],
    batch_size: int = 32
) -> List[float]:
    """Compute similarity between query and multiple documents.
    
    Args:
        model: Embedding model
        query: Query text
        documents: List of documents
        batch_size: Batch size for encoding
        
    Returns:
        List of similarity scores
    """
    if not model or not documents:
        return [0.0] * len(documents)
        
    try:
        # Encode query
        query_embedding = model.encode(query)
        
        # Encode documents in batches
        doc_embeddings = model.encode(documents, batch_size=batch_size)
        
        # Compute similarities
        similarities = []
        for doc_embedding in doc_embeddings:
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append(max(0.0, min(1.0, similarity)))
            
        return similarities
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Batch similarity computation failed: {e}")
        return [0.0] * len(documents)


class NeuralReranker:
    """Neural reranking model for improved ranking.
    
    Uses cross-encoder models to rerank initial results for better accuracy.
    This is more accurate than bi-encoders but slower, so it's typically
    used on top-K results.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the reranking model."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("CrossEncoder not available")
            return
            
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self.logger.info(f"Loaded reranking model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load reranking model: {e}")
            
    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, float]],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Rerank documents using cross-encoder.
        
        Args:
            query: Query text
            documents: List of (document_text, initial_score) tuples
            top_k: Number of top results to rerank
            
        Returns:
            Reranked list of (document_text, score) tuples
        """
        if not self.model or not documents:
            return documents
            
        try:
            # Take top-K for reranking
            docs_to_rerank = documents[:top_k]
            remaining_docs = documents[top_k:]
            
            # Prepare pairs for cross-encoder
            pairs = [(query, doc[0]) for doc in docs_to_rerank]
            
            # Get reranking scores
            scores = self.model.predict(pairs)
            
            # Combine with original scores (weighted average)
            reranked = []
            for i, (doc_text, orig_score) in enumerate(docs_to_rerank):
                # Combine original and reranking scores
                combined_score = 0.3 * orig_score + 0.7 * scores[i]
                reranked.append((doc_text, combined_score))
                
            # Sort by new scores
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            # Append remaining documents
            reranked.extend(remaining_docs)
            
            return reranked
            
        except Exception as e:
            self.logger.warning(f"Reranking failed: {e}")
            return documents


def check_ml_dependencies() -> Dict[str, bool]:
    """Check which ML dependencies are available.
    
    Returns:
        Dictionary of dependency availability
    """
    return {
        'torch': _TORCH_AVAILABLE,
        'transformers': _TRANSFORMERS_AVAILABLE,
        'sentence_transformers': _SENTENCE_TRANSFORMERS_AVAILABLE,
        'sklearn': _SKLEARN_AVAILABLE
    }


def get_available_models() -> List[str]:
    """Get list of available embedding models.
    
    Returns:
        List of model names
    """
    models = []
    
    if _SENTENCE_TRANSFORMERS_AVAILABLE:
        # Common small models
        models.extend([
            'all-MiniLM-L6-v2',
            'all-MiniLM-L12-v2',
            'all-mpnet-base-v2',
            'multi-qa-MiniLM-L6-cos-v1',
            'paraphrase-MiniLM-L6-v2'
        ])
        
    if _SKLEARN_AVAILABLE:
        models.append('tfidf')  # TF-IDF fallback
        
    return models


def estimate_embedding_memory(
    num_files: int,
    embedding_dim: int = 384
) -> Dict[str, float]:
    """Estimate memory requirements for embeddings.
    
    Args:
        num_files: Number of files to embed
        embedding_dim: Dimension of embeddings
        
    Returns:
        Dictionary with memory estimates
    """
    # Assume float32 (4 bytes per value)
    bytes_per_embedding = embedding_dim * 4
    total_bytes = num_files * bytes_per_embedding
    
    return {
        'per_file_mb': bytes_per_embedding / (1024 * 1024),
        'total_mb': total_bytes / (1024 * 1024),
        'total_gb': total_bytes / (1024 * 1024 * 1024)
    }


# Export key functions and classes
__all__ = [
    'EmbeddingModel',
    'NeuralReranker',
    'load_embedding_model',
    'compute_similarity',
    'cosine_similarity',
    'batch_similarity',
    'check_ml_dependencies',
    'get_available_models',
    'estimate_embedding_memory'
]