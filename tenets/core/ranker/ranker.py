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
