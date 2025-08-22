"""Natural Language Processing and Machine Learning utilities.

This package provides all NLP/ML functionality for Tenets including:
- Tokenization and text processing
- Keyword extraction (YAKE, TF-IDF)
- Stopword management
- Embedding generation and caching
- Semantic similarity calculation

All ML features are optional and gracefully degrade when not available.
"""

from typing import List, Optional, Dict, Any

# Core NLP components (always available)
from .tokenizer import CodeTokenizer, TextTokenizer
from .stopwords import StopwordManager, StopwordSet
from .keyword_extractor import KeywordExtractor, TFIDFExtractor

# ML components (optional)
try:
    from .embeddings import EmbeddingModel, LocalEmbeddings
    from .similarity import SemanticSimilarity, cosine_similarity
    from .cache import EmbeddingCache

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

    # Provide stub classes
    class EmbeddingModel:
        """Stub for when ML features not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("ML features not available. Install with: pip install tenets[ml]")

    LocalEmbeddings = EmbeddingModel

    class SemanticSimilarity:
        """Stub for when ML features not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("ML features not available. Install with: pip install tenets[ml]")

    def cosine_similarity(a, b):
        """Stub for when ML features not available."""
        return 0.0

    class EmbeddingCache:
        """Stub for when ML features not available."""

        def __init__(self, *args, **kwargs):
            pass


__all__ = [
    # Core NLP
    "CodeTokenizer",
    "TextTokenizer",
    "StopwordManager",
    "StopwordSet",
    "KeywordExtractor",
    "TFIDFExtractor",
    # ML features
    "ML_AVAILABLE",
    "EmbeddingModel",
    "LocalEmbeddings",
    "SemanticSimilarity",
    "cosine_similarity",
    "EmbeddingCache",
    # Convenience functions
    "extract_keywords",
    "tokenize_code",
    "compute_similarity",
]


def extract_keywords(
    text: str, max_keywords: int = 20, use_yake: bool = True, language: str = "en"
) -> List[str]:
    """Extract keywords from text using best available method.

    Args:
        text: Input text
        max_keywords: Maximum keywords to extract
        use_yake: Try YAKE first if available
        language: Language for YAKE

    Returns:
        List of extracted keywords
    """
    extractor = KeywordExtractor(use_yake=use_yake, language=language)
    return extractor.extract(text, max_keywords=max_keywords)


def tokenize_code(
    code: str, language: Optional[str] = None, use_stopwords: bool = False
) -> List[str]:
    """Tokenize code with language-aware processing.

    Args:
        code: Source code to tokenize
        language: Programming language (auto-detect if None)
        use_stopwords: Filter stopwords

    Returns:
        List of tokens
    """
    tokenizer = CodeTokenizer(use_stopwords=use_stopwords)
    return tokenizer.tokenize(code, language=language)


def compute_similarity(text1: str, text2: str, method: str = "auto") -> float:
    """Compute similarity between two texts.

    Args:
        text1: First text
        text2: Second text
        method: 'semantic'|'tfidf'|'auto'

    Returns:
        Similarity score (0-1)
    """
    if method == "semantic" or (method == "auto" and ML_AVAILABLE):
        sim = SemanticSimilarity()
        return sim.compute(text1, text2)
    else:
        # Fallback to TF-IDF
        extractor = TFIDFExtractor()
        extractor.fit([text1, text2])
        vec1 = extractor.transform([text1])[0]
        vec2 = extractor.transform([text2])[0]

        # Compute cosine similarity manually
        import math

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 * norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
