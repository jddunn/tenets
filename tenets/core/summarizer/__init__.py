"""Content summarization system for Tenets.

This package provides intelligent text and code summarization capabilities
using multiple strategies from simple extraction to advanced ML approaches.
The summarization system helps compress large codebases to fit within token
limits while preserving the most important information.

Main components:
- Summarizer: Main orchestrator for summarization operations
- Strategies: Different summarization approaches (extractive, compressive, etc.)
- LLMSummarizer: Integration with Large Language Models (costs $)

Example usage:
    >>> from tenets.core.summarizer import Summarizer, create_summarizer
    >>>
    >>> # Create summarizer
    >>> summarizer = create_summarizer(mode="extractive")
    >>>
    >>> # Summarize text
    >>> result = summarizer.summarize(
    ...     long_text,
    ...     target_ratio=0.3  # Compress to 30% of original
    ... )
    >>>
    >>> print(f"Reduced by {result.reduction_percent:.1f}%")
"""

from typing import Optional, Union

from tenets.config import TenetsConfig

# Import main classes
from .summarizer import (
    BatchSummarizationResult,
    SummarizationMode,
    SummarizationResult,
    Summarizer,
)

# Import strategies
from .strategies import (
    CompressiveStrategy,
    ExtractiveStrategy,
    SummarizationStrategy,
    TextRankStrategy,
    TransformerStrategy,
)

# Import LLM support
from .llm import (
    LLMConfig,
    LLMProvider,
    LLMSummarizer,
    LLMSummaryStrategy,
    create_llm_summarizer,
)

# Version info
__version__ = "0.1.0"

# Check for ML support
ML_AVAILABLE = False
try:
    import transformers

    ML_AVAILABLE = True
except ImportError:
    pass


# Public API exports
__all__ = [
    # Main summarizer
    "Summarizer",
    "SummarizationMode",
    "SummarizationResult",
    "BatchSummarizationResult",
    # Strategies
    "SummarizationStrategy",
    "ExtractiveStrategy",
    "CompressiveStrategy",
    "TextRankStrategy",
    "TransformerStrategy",
    # LLM support
    "LLMProvider",
    "LLMConfig",
    "LLMSummarizer",
    "LLMSummaryStrategy",
    "create_llm_summarizer",
    # Utilities
    "create_summarizer",
    "estimate_compression",
    "ML_AVAILABLE",
]


def create_summarizer(
    config: Optional[TenetsConfig] = None, mode: str = "auto", enable_cache: bool = True
) -> Summarizer:
    """Create a configured summarizer.

    Convenience function to quickly create a summarizer with
    sensible defaults.

    Args:
        config: Optional configuration
        mode: Default summarization mode
        enable_cache: Whether to enable caching

    Returns:
        Configured Summarizer instance

    Example:
        >>> summarizer = create_summarizer(mode="extractive")
        >>> result = summarizer.summarize(text, target_ratio=0.25)
    """
    if config is None:
        config = TenetsConfig()

    return Summarizer(config=config, default_mode=mode, enable_cache=enable_cache)


def estimate_compression(text: str, target_ratio: float = 0.3, mode: str = "extractive") -> dict:
    """Estimate compression results without actually summarizing.

    Useful for planning and understanding how much compression
    is possible for given text.

    Args:
        text: Text to analyze
        target_ratio: Target compression ratio
        mode: Summarization mode

    Returns:
        Dictionary with estimates

    Example:
        >>> estimate = estimate_compression(long_text, 0.25)
        >>> print(f"Expected output: ~{estimate['expected_length']} chars")
    """
    original_length = len(text)
    expected_length = int(original_length * target_ratio)

    # Estimate based on mode
    if mode == "extractive":
        # Extractive typically achieves 80-90% of target
        achievable_ratio = target_ratio * 1.1
    elif mode == "compressive":
        # Compressive can achieve closer to target
        achievable_ratio = target_ratio * 1.05
    elif mode == "textrank":
        # TextRank similar to extractive
        achievable_ratio = target_ratio * 1.1
    elif mode in ["transformer", "llm"]:
        # ML models can hit target precisely
        achievable_ratio = target_ratio
    else:
        achievable_ratio = target_ratio * 1.1

    achievable_length = int(original_length * achievable_ratio)

    # Estimate quality based on compression level
    if target_ratio >= 0.5:
        quality = "high"
        info_preserved = "90-95%"
    elif target_ratio >= 0.3:
        quality = "good"
        info_preserved = "75-85%"
    elif target_ratio >= 0.2:
        quality = "moderate"
        info_preserved = "60-75%"
    else:
        quality = "low"
        info_preserved = "40-60%"

    return {
        "original_length": original_length,
        "target_ratio": target_ratio,
        "expected_length": expected_length,
        "achievable_length": achievable_length,
        "achievable_ratio": achievable_ratio,
        "quality": quality,
        "info_preserved": info_preserved,
        "recommended_mode": _recommend_mode(text, target_ratio),
    }


def _recommend_mode(text: str, target_ratio: float) -> str:
    """Recommend best summarization mode.

    Args:
        text: Text to summarize
        target_ratio: Target ratio

    Returns:
        Recommended mode name
    """
    text_length = len(text)

    # Check if text looks like code
    code_indicators = ["def ", "class ", "function ", "import ", "{", "}"]
    is_code = sum(1 for ind in code_indicators if ind in text) >= 2

    if is_code:
        return "extractive"  # Best for preserving code structure

    if text_length < 500:
        return "extractive"  # Short text
    elif text_length < 2000:
        return "compressive"  # Medium text, remove redundancy
    elif text_length < 10000:
        return "textrank"  # Longer text, use graph-based
    else:
        if ML_AVAILABLE and target_ratio < 0.3:
            return "transformer"  # Very long text, aggressive compression
        else:
            return "extractive"  # Fallback for very long text


# Batch processing utilities
def summarize_files(
    files: list,
    target_ratio: float = 0.3,
    mode: str = "auto",
    config: Optional[TenetsConfig] = None,
) -> BatchSummarizationResult:
    """Summarize multiple files in batch.

    Convenience function for batch processing.

    Args:
        files: List of FileAnalysis objects or text strings
        target_ratio: Target compression ratio
        mode: Summarization mode
        config: Optional configuration

    Returns:
        BatchSummarizationResult

    Example:
        >>> from tenets.core.summarizer import summarize_files
        >>> results = summarize_files(file_list, target_ratio=0.25)
        >>> print(f"Compressed {results.files_processed} files")
    """
    summarizer = create_summarizer(config=config, mode=mode)
    return summarizer.batch_summarize(files, target_ratio=target_ratio)


def quick_summary(text: str, max_length: int = 500) -> str:
    """Quick summary with simple length constraint.

    Convenience function for quick summarization without
    needing to manage summarizer instances.

    Args:
        text: Text to summarize
        max_length: Maximum length in characters

    Returns:
        Summarized text

    Example:
        >>> from tenets.core.summarizer import quick_summary
        >>> summary = quick_summary(long_text, max_length=200)
    """
    if len(text) <= max_length:
        return text

    # Calculate ratio needed
    target_ratio = max_length / len(text)

    summarizer = create_summarizer(mode="extractive")
    result = summarizer.summarize(text, target_ratio=target_ratio, max_length=max_length)

    return result.summary


# Code-specific summarization
def summarize_code(
    code: str, language: str = "python", preserve_structure: bool = True, target_ratio: float = 0.3
) -> str:
    """Summarize code while preserving structure.

    Specialized function for code summarization that maintains
    imports, signatures, and key structural elements.

    Args:
        code: Source code
        language: Programming language
        preserve_structure: Keep imports and signatures
        target_ratio: Target compression ratio

    Returns:
        Summarized code

    Example:
        >>> from tenets.core.summarizer import summarize_code
        >>> summary = summarize_code(
        ...     long_module,
        ...     language="python",
        ...     target_ratio=0.25
        ... )
    """
    from tenets.models.analysis import FileAnalysis

    # Create a temporary FileAnalysis
    file = FileAnalysis(
        path=f"temp.{language}",
        language=language,
        content=code,
        size=len(code),
        lines=code.count("\n") + 1,
    )

    summarizer = create_summarizer()
    result = summarizer.summarize_file(
        file, target_ratio=target_ratio, preserve_structure=preserve_structure
    )

    return result.summary


# LLM integration helpers
def estimate_llm_cost(
    text: str, provider: str = "openai", model: str = "gpt-3.5-turbo", target_ratio: float = 0.3
) -> dict:
    """Estimate cost of LLM summarization.

    Calculate expected API costs before summarizing.

    Args:
        text: Text to summarize
        provider: LLM provider
        model: Model name
        target_ratio: Target compression ratio

    Returns:
        Cost estimate dictionary

    Example:
        >>> from tenets.core.summarizer import estimate_llm_cost
        >>> cost = estimate_llm_cost(text, "openai", "gpt-4")
        >>> print(f"Estimated cost: ${cost['total_cost']:.4f}")
    """
    try:
        llm = create_llm_summarizer(provider, model)
        return llm.estimate_cost(text)
    except Exception as e:
        return {"error": str(e), "total_cost": 0.0, "currency": "USD"}


# Strategy selection helper
def select_best_strategy(text: str, target_ratio: float, constraints: Optional[dict] = None) -> str:
    """Select best summarization strategy for given text.

    Analyzes text characteristics and constraints to recommend
    the optimal summarization approach.

    Args:
        text: Text to analyze
        target_ratio: Target compression ratio
        constraints: Optional constraints (time, quality, cost)

    Returns:
        Recommended strategy name

    Example:
        >>> from tenets.core.summarizer import select_best_strategy
        >>> strategy = select_best_strategy(
        ...     text,
        ...     0.25,
        ...     {'max_time': 1.0, 'quality': 'high'}
        ... )
        >>> print(f"Recommended: {strategy}")
    """
    constraints = constraints or {}

    # Check constraints
    max_time = constraints.get("max_time", float("inf"))
    quality = constraints.get("quality", "medium")
    allow_llm = constraints.get("allow_llm", False)
    require_ml = constraints.get("require_ml", False)

    text_length = len(text)

    # Time-constrained selection
    if max_time < 0.5:
        return "extractive"  # Fastest

    # Quality-based selection
    if quality == "high":
        if allow_llm:
            return "llm"
        elif ML_AVAILABLE:
            return "transformer"
        else:
            return "textrank"
    elif quality == "low":
        return "compressive"  # Fast but lower quality

    # ML requirement
    if require_ml:
        if ML_AVAILABLE:
            return "transformer"
        else:
            raise ValueError("ML required but not available")

    # Default recommendation
    return _recommend_mode(text, target_ratio)
