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

# Re-export common subpackages for convenience
from . import analysis, ranking, session, instiller, git, summarizer  # noqa: F401
