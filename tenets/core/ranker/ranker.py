"""Compatibility shim for ranker module.

Re-exports from `tenets.core.ranking.ranker`.
"""

from __future__ import annotations
import warnings

warnings.warn(
    "tenets.core.ranker.ranker is deprecated; use tenets.core.ranking",
    DeprecationWarning,
    stacklevel=2,
)
from tenets.core.ranking.ranker import *  # noqa: F401,F403
from tenets.core.ranking.ranker.ranker import (
    RankingAlgorithm,  # noqa: F401
    RankingFactors,  # noqa: F401
    RankedFile,  # noqa: F401
    RelevanceRanker,  # noqa: F401
)
