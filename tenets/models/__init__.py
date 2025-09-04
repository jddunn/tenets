"""
Models module for Tenets.

This module contains data models and structures used throughout the Tenets package.
"""

# Just re-export what the main package expects
from .context import ContextResult

__all__ = [
    "ContextResult",
]
