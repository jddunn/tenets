"""Relevance ranking system for Tenets.

This package provides sophisticated file ranking capabilities using multiple
strategies from simple keyword matching to advanced ML-based semantic analysis.
The ranking system is designed to efficiently identify the most relevant files
for a given prompt or query.

Main components:
- RelevanceRanker: Main orchestrator for ranking operations
- RankingFactors: Comprehensive factors used for scoring
- RankedFile: File with ranking information
- Ranking strategies: Fast, Balanced, Thorough, ML
- TF-IDF and BM25 calculators for text similarity

Example usage:
    >>> from tenets.core.ranking import RelevanceRanker, create_ranker
    >>> from tenets.models.context import PromptContext
    >>> 
    >>> # Create ranker with config
    >>> ranker = create_ranker(algorithm="balanced")
    >>> 
    >>> # Parse prompt
    >>> prompt_context = PromptContext(text="implement OAuth authentication")
    >>> 
    >>> # Rank files
    >>> ranked_files = ranker.rank_files(files, prompt_context)
    >>> 
    >>> # Get top relevant files
    >>> for file in ranked_files[:10]:
    ...     print(f"{file.path}: {file.relevance_score:.3f}")
"""

from typing import List, Optional

from tenets.config import TenetsConfig

# Import main classes
from .factors import (
    FactorWeight,
    RankedFile,
    RankingExplainer,
    RankingFactors,
)
from .ranker import (
    RankingAlgorithm,
    RankingStats,
    RelevanceRanker,
    create_ranker,
)

# Import strategies
from .strategies import (
    BalancedRankingStrategy,
    FastRankingStrategy,
    MLRankingStrategy,
    RankingStrategy,
    ThoroughRankingStrategy,
)

# Import TF-IDF/BM25
from .tfidf import BM25Calculator, TFIDFCalculator

# Import ML utilities (optional)
try:
    from .ml_utils import (
        EmbeddingModel,
        NeuralReranker,
        batch_similarity,
        check_ml_dependencies,
        compute_similarity,
        cosine_similarity,
        estimate_embedding_memory,
        get_available_models,
        load_embedding_model,
    )
    
    ML_AVAILABLE = True
except ImportError:
    # ML features not available
    ML_AVAILABLE = False
    
    # Provide stub functions
    def check_ml_dependencies():
        """Check ML dependencies (stub)."""
        return {
            'torch': False,
            'transformers': False,
            'sentence_transformers': False,
            'sklearn': False
        }
    
    def get_available_models():
        """Get available models (stub)."""
        return []


# Version info
__version__ = "0.1.0"


# Public API exports
__all__ = [
    # Main ranker
    "RelevanceRanker",
    "create_ranker",
    "RankingAlgorithm",
    "RankingStats",
    
    # Factors and scoring
    "RankingFactors",
    "RankedFile",
    "RankingExplainer",
    "FactorWeight",
    
    # Strategies
    "RankingStrategy",
    "FastRankingStrategy",
    "BalancedRankingStrategy",
    "ThoroughRankingStrategy",
    "MLRankingStrategy",
    
    # Text similarity
    "TFIDFCalculator",
    "BM25Calculator",
    
    # ML utilities (if available)
    "ML_AVAILABLE",
    "check_ml_dependencies",
    "get_available_models",
]

# Add ML exports if available
if ML_AVAILABLE:
    __all__.extend([
        "EmbeddingModel",
        "NeuralReranker",
        "load_embedding_model",
        "compute_similarity",
        "cosine_similarity",
        "batch_similarity",
        "estimate_embedding_memory",
    ])


def get_default_ranker(config: Optional[TenetsConfig] = None) -> RelevanceRanker:
    """Get a default configured ranker.
    
    Convenience function to quickly get a working ranker with
    sensible defaults.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Configured RelevanceRanker instance
    """
    if config is None:
        config = TenetsConfig()
        
    # Use config's algorithm or default to balanced
    algorithm = config.ranking.algorithm or "balanced"
    
    return create_ranker(
        config=config,
        algorithm=algorithm,
        use_stopwords=config.ranking.use_stopwords
    )


def rank_files_simple(
    files: List,
    prompt: str,
    algorithm: str = "balanced",
    threshold: float = 0.1
) -> List:
    """Simple interface for ranking files.
    
    Provides a simplified API for quick ranking without needing
    to manage ranker instances or configurations.
    
    Args:
        files: List of FileAnalysis objects
        prompt: Search prompt or query
        algorithm: Ranking algorithm to use
        threshold: Minimum relevance score
        
    Returns:
        List of files sorted by relevance above threshold
        
    Example:
        >>> from tenets.core.ranking import rank_files_simple
        >>> relevant_files = rank_files_simple(
        ...     files, 
        ...     "authentication logic",
        ...     algorithm="thorough"
        ... )
    """
    from tenets.models.context import PromptContext
    
    # Create temporary config with threshold
    config = TenetsConfig()
    config.ranking.algorithm = algorithm
    config.ranking.threshold = threshold
    
    # Create ranker
    ranker = create_ranker(config=config, algorithm=algorithm)
    
    # Parse prompt
    prompt_context = PromptContext(text=prompt)
    
    # Rank files
    try:
        ranked = ranker.rank_files(files, prompt_context)
        return ranked
    finally:
        # Clean up
        ranker.shutdown()


def explain_ranking(
    files: List,
    prompt: str,
    algorithm: str = "balanced",
    top_n: int = 10
) -> str:
    """Get explanation of why files ranked the way they did.
    
    Useful for debugging and understanding ranking behavior.
    
    Args:
        files: List of FileAnalysis objects
        prompt: Search prompt
        algorithm: Algorithm used
        top_n: Number of top files to explain
        
    Returns:
        Formatted explanation string
        
    Example:
        >>> from tenets.core.ranking import explain_ranking
        >>> explanation = explain_ranking(files, "database models")
        >>> print(explanation)
    """
    from tenets.models.context import PromptContext
    
    # Create ranker
    config = TenetsConfig()
    config.ranking.algorithm = algorithm
    ranker = create_ranker(config=config, algorithm=algorithm)
    
    # Parse prompt
    prompt_context = PromptContext(text=prompt)
    
    # Get strategy for weights
    strategy = ranker.strategies.get(RankingAlgorithm(algorithm))
    if not strategy:
        return "No strategy available for explanation"
        
    # Rank files with explanation
    try:
        ranker.rank_files(files, prompt_context, explain=True)
        
        # Get ranked files with factors
        ranked_files = []
        for file in files[:top_n]:
            if hasattr(file, 'relevance_score'):
                # Create RankedFile for explanation
                rf = RankedFile(
                    analysis=file,
                    score=file.relevance_score,
                    factors=RankingFactors(),  # Would need actual factors
                    rank=file.relevance_rank
                )
                ranked_files.append(rf)
                
        # Generate explanation
        explainer = RankingExplainer()
        return explainer.explain_ranking(
            ranked_files,
            strategy.get_weights(),
            top_n=top_n
        )
    finally:
        ranker.shutdown()


# Initialize default components on import
_default_tfidf = None
_default_bm25 = None


def get_default_tfidf(use_stopwords: bool = False) -> TFIDFCalculator:
    """Get default TF-IDF calculator instance.
    
    Args:
        use_stopwords: Whether to filter stopwords
        
    Returns:
        TFIDFCalculator instance
    """
    global _default_tfidf
    if _default_tfidf is None or _default_tfidf.use_stopwords != use_stopwords:
        _default_tfidf = TFIDFCalculator(use_stopwords=use_stopwords)
    return _default_tfidf


def get_default_bm25(use_stopwords: bool = False) -> BM25Calculator:
    """Get default BM25 calculator instance.
    
    Args:
        use_stopwords: Whether to filter stopwords
        
    Returns:
        BM25Calculator instance
    """
    global _default_bm25
    if _default_bm25 is None or _default_bm25.use_stopwords != use_stopwords:
        _default_bm25 = BM25Calculator(use_stopwords=use_stopwords)
    return _default_bm25