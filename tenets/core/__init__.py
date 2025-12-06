"""Core subsystem of Tenets.

This package aggregates core functionality such as analysis, distillation,
ranking, sessions, and related utilities.

It exposes a stable import path for documentation and users:
- tenets.core.analysis
- tenets.core.ranking
- tenets.core.session
- tenets.core.instiller
- tenets.core.git
- tenets.core.summarizer
"""

# Eager imports to avoid Python 3.14 import recursion with lazy __getattr__
# (see recursion in importlib resolution). These modules are light enough to
# import at package load, and this prevents circular lazy loading.
from . import analysis, git, instiller, ranking, session, summarizer

__all__ = ["analysis", "git", "instiller", "ranking", "session", "summarizer"]
