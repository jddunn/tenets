"""Ranker package exports."""

# Expose logger getter for tests that patch tenets.core.ranking.ranker.get_logger
from tenets.utils.logger import get_logger  # noqa: F401

# Expose SentenceTransformer at package level for tests that patch it
try:
    from sentence_transformers import SentenceTransformer  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

# Expose cosine_similarity for patching in tests
try:  # pragma: no cover - optional dependency
    from torch.nn.functional import cosine_similarity  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover

    def cosine_similarity(*args, **kwargs):  # type: ignore
        raise RuntimeError("cosine_similarity not available")


# Import and re-export ranker classes/strategies
from .ranker import (  # noqa: F401,E402
    BalancedRankingStrategy,
    FastRankingStrategy,
    RankedFile,
    RankingAlgorithm,
    RankingFactors,
    RelevanceRanker,
    TFIDFCalculator,
    ThoroughRankingStrategy,
)
