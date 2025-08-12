"""Instiller module for managing and injecting tenets.

The instiller system handles the lifecycle of tenets (guiding principles)
and their strategic injection into generated context to maintain consistency
across AI interactions.
"""

from tenets.core.instiller.injector import InjectionPosition, TenetInjector
from tenets.core.instiller.instiller import InstillationResult, Instiller
from tenets.core.instiller.manager import TenetManager

__all__ = [
    "TenetManager",
    "TenetInjector",
    "InjectionPosition",
    "Instiller",
    "InstillationResult",
]
