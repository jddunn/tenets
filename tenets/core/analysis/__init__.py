"""Analysis package.

Re-exports the main CodeAnalyzer after directory reorganization.

This module intentionally re-exports ``CodeAnalyzer`` so callers can import
``tenets.core.analysis.CodeAnalyzer``. The implementation lives in
``analyzer.py`` and does not import this package-level module, so exposing
the symbol here will not create a circular import.
"""

from .analyzer import CodeAnalyzer  # noqa: F401
