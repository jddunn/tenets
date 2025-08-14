"""Machine learning utilities for ranking.

This module provides ML-based ranking capabilities using NLP components.
All embedding and similarity logic is handled by the NLP package to avoid duplication.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tenets.utils.logger import get_logger

# Check ML availability through NLP package
try:
    from tenets.core.nlp.embeddings import LocalEmbeddings, SENTENCE_TRANSFORMERS_AVAILABLE
    from tenets.core.nlp.similarity import SemanticSimilarity, cosine_similarity
    ML_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE
except ImportError:
    ML_AVAILABLE = False
    LocalEmbeddings = None
    SemanticSimilarity = None
    cosine_similarity = None


class EmbeddingModel:
    """Wrapper for embedding models using NLP components.
    
    Provides a unified interface for different embedding models
    with built-in caching and batch processing capabilities.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """Initialize embedding model.

        Args:
            model_name: Name of the model to load
            cache_dir: Directory for caching embeddings
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.model = None

        if not ML_AVAILABLE:
            self.logger.warning(
                "ML features not available. Install with: pip install sentence-transformers"
            )
            return

        # Load model using NLP package
        try:
            self.model = LocalEmbeddings(
                model_name=model_name,
                device=device,
                cache_dir=cache_dir
            )
            self.logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        use_cache: bool = True,
    ) -> Union[list, Any]:  # Returns list or numpy array
        """Encode texts to embeddings.

        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            use_cache: Use cached embeddings if available

        Returns:
            Numpy array of embeddings or fallback list
        """
        if not self.model:
            # Fallback to TF-IDF
            return self._tfidf_fallback(texts)

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def _tfidf_fallback(self, texts: Union[str, List[str]]) -> list:
        """Fallback to TF-IDF when embeddings not available.

        Args:
            texts: Text or list of texts

        Returns:
            TF-IDF vectors as lists
        """
        from tenets.core.nlp.embeddings import FallbackEmbeddings
        
        fallback = FallbackEmbeddings()
        return fallback.encode(texts).tolist()


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

    if not ML_AVAILABLE:
        logger.warning("ML features not available. Install with: pip install sentence-transformers")
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
        cache: Optional cache dictionary (unused, for API compatibility)

    Returns:
        Similarity score (0-1)
    """
    if not model or not model.model:
        return 0.0

    try:
        # Use NLP similarity computation
        similarity_calc = SemanticSimilarity(model.model)
        return similarity_calc.compute(text1, text2)

    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Similarity computation failed: {e}")
        return 0.0


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
    if not model or not model.model or not documents:
        return [0.0] * len(documents)

    try:
        # Use NLP batch similarity
        similarity_calc = SemanticSimilarity(model.model)
        results = similarity_calc.compute_batch(query, documents)
        
        # Convert to list of scores in original order
        score_dict = dict(results)
        return [score_dict.get(i, 0.0) for i in range(len(documents))]

    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Batch similarity computation failed: {e}")
        return [0.0] * len(documents)


class NeuralReranker:
    """Neural reranking model for improved ranking.

    Uses cross-encoder models to rerank initial results for better accuracy.
    This is more accurate than bi-encoders but slower.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker.

        Args:
            model_name: Cross-encoder model name
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.model = None
        
        if not ML_AVAILABLE:
            self.logger.warning("Cross-encoder reranking not available without ML dependencies")
            return
            
        self._load_model()

    def _load_model(self):
        """Load the reranking model."""
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
    deps = {
        "sentence_transformers": ML_AVAILABLE,
        "torch": False,
        "transformers": False,
        "sklearn": False,
    }
    
    try:
        import torch
        deps["torch"] = True
    except ImportError:
        pass
        
    try:
        import transformers
        deps["transformers"] = True
    except ImportError:
        pass
        
    try:
        import sklearn
        deps["sklearn"] = True
    except ImportError:
        pass
    
    return deps


def get_available_models() -> List[str]:
    """Get list of available embedding models.

    Returns:
        List of model names
    """
    models = []

    if ML_AVAILABLE:
        # Common small models
        models.extend([
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2", 
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
            "paraphrase-MiniLM-L6-v2",
        ])

    # Always available fallback
    models.append("tfidf")

    return models


def estimate_embedding_memory(num_files: int, embedding_dim: int = 384) -> Dict[str, float]:
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
        "per_file_mb": bytes_per_embedding / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
    }


# Export key functions and classes
__all__ = [
    "EmbeddingModel",
    "NeuralReranker", 
    "load_embedding_model",
    "compute_similarity",
    "cosine_similarity",
    "batch_similarity",
    "check_ml_dependencies",
    "get_available_models",
    "estimate_embedding_memory",
]