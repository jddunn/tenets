"""Prompt core package."""
try:
    from .parser import *  # noqa: F401,F403
except Exception:
    # Parser may not be ready yet in early setups
    pass
