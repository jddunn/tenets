"""Shared helpers for CLI command modules."""

from pathlib import Path
from typing import Union


def normalize_path(p: Union[str, Path]) -> str:
    """Return a normalized absolute path string for stable testing/logging.

    Ensures platform-appropriate formatting and avoids returning Path objects
    so that tests can compare against stringified call arguments consistently.
    """
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)
