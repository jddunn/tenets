"""Ranking API (clean minimal surface).

Public imports:
        RelevanceRanker, RankingAlgorithm
Internal helpers (still importable if needed):
        RankingFactors, RankedFile, TFIDFCalculator, strategies
"""

from .ranker import (
    BalancedRankingStrategy,
    FastRankingStrategy,
    RankedFile,
    RankingAlgorithm,
    RankingFactors,
    RelevanceRanker,
    TFIDFCalculator,
    ThoroughRankingStrategy,
)

__all__ = ["RelevanceRanker", "RankingAlgorithm"]
