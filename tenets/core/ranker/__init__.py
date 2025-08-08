"""Deprecated ranker import path shim.

Use `tenets.core.ranking` moving forward.
"""
from __future__ import annotations
import warnings
warnings.warn(
    "tenets.core.ranker is deprecated; use tenets.core.ranking",
    DeprecationWarning,
    stacklevel=2,
)
from tenets.core.ranking import RelevanceRanker  # noqa: F401
