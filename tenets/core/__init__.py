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

# Use lazy loading to avoid importing heavy ML dependencies at startup
# Submodules are imported on first access via __getattr__

__all__ = ["analysis", "git", "instiller", "ranking", "session", "summarizer"]


def __getattr__(name):
    """Lazy load submodules to improve import performance."""
    if name == "analysis":
        from . import analysis

        return analysis
    elif name == "ranking":
        from . import ranking

        return ranking
    elif name == "session":
        from . import session

        return session
    elif name == "instiller":
        from . import instiller

        return instiller
    elif name == "git":
        from . import git

        return git
    elif name == "summarizer":
        from . import summarizer

        return summarizer
    raise AttributeError(f"module 'tenets.core' has no attribute '{name}'")
