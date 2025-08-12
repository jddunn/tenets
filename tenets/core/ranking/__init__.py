"""Ranking API (clean minimal surface).

Public imports:
        RelevanceRanker, RankingAlgorithm
Internal helpers (still importable if needed):
        RankingFactors, RankedFile, TFIDFCalculator, strategies
"""

from .ranker import (
    RelevanceRanker,
    RankingAlgorithm,
    RankingFactors,
    RankedFile,
    TFIDFCalculator,
    FastRankingStrategy,
    BalancedRankingStrategy,
    ThoroughRankingStrategy,
)

__all__ = ["RelevanceRanker", "RankingAlgorithm"]
