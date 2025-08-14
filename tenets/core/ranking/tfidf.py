"""Compatibility shim for TF-IDF/BM25 calculators.

This module preserves legacy imports like:

    from tenets.core.ranking.tfidf import TFIDFCalculator, BM25Calculator

by re-exporting the implementations from the centralized NLP module.
"""

from ..nlp.tfidf import BM25Calculator, TFIDFCalculator  # noqa: F401

__all__ = [
    "TFIDFCalculator",
    "BM25Calculator",
]
