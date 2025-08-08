"""Distiller module - Extract and aggregate relevant context from codebases.

The distiller is responsible for the main 'distill' command functionality:
1. Understanding what the user wants (prompt parsing)
2. Finding relevant files (discovery)
3. Ranking by importance (intelligence)
4. Packing within token limits (optimization)
5. Formatting for output (presentation)
"""

from tenets.core.distiller.distiller import Distiller
from tenets.core.distiller.aggregator import ContextAggregator
from tenets.core.distiller.optimizer import TokenOptimizer
from tenets.core.distiller.formatter import ContextFormatter

__all__ = [
    "Distiller",
    "ContextAggregator",
    "TokenOptimizer",
    "ContextFormatter",
]
