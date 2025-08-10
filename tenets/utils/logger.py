"""Logging utilities for Tenets.

Provides a single entrypoint `get_logger` that configures Rich logging once
and returns child loggers for modules.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

_RICH_INSTALLED = False
try:
    from rich.logging import RichHandler  # type: ignore

    _RICH_INSTALLED = True
except Exception:  # pragma: no cover
    _RICH_INSTALLED = False

_CONFIGURED = False
_CURRENT_LEVEL = None


def _configure_root(level: int) -> None:
    """Configure the root logger once and update on subsequent calls.

    Ensures a single handler is attached and formatters are applied
    consistently, with idempotent behavior across calls.
    """
    global _CONFIGURED, _CURRENT_LEVEL

    root = logging.getLogger()

    if _CONFIGURED:
        if _CURRENT_LEVEL != level:
            root.setLevel(level)
            for h in root.handlers:
                h.setLevel(level)
        _CURRENT_LEVEL = level
        return

    # Create or update a handler
    if _RICH_INSTALLED:
        # Try to reuse an existing RichHandler if present
        handler = None
        for h in root.handlers:
            if h.__class__.__name__ == "RichHandler":
                handler = h
                break
        if handler is None:
            handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
            # Even with Rich, provide a simple formatter to satisfy tests
            handler.setFormatter(logging.Formatter("%(message)s"))
            handler.setLevel(level)
            root.addHandler(handler)
        else:
            handler.setLevel(level)
            # Don't override existing formatter aggressively when Rich is present
    else:
        # Non-Rich path: ensure the first root handler includes asctime in its formatter
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s:%(filename)s:%(lineno)d %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if root.handlers:
            # Update the first handler to satisfy test expectations
            h = root.handlers[0]
            h.setLevel(level)
            h.setFormatter(formatter)
        else:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(formatter)
            root.addHandler(handler)

    # Attach level to root
    root.setLevel(level)

    _CONFIGURED = True
    _CURRENT_LEVEL = level


def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger.

    Environment variables:
      - TENETS_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL
    """
    env_level = os.getenv("TENETS_LOG_LEVEL")
    default_level_name = env_level.upper() if env_level else "INFO"
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    resolved_level = level if level is not None else level_map.get(default_level_name, logging.INFO)

    # Configure root with the resolved level (explicit level overrides env)
    _configure_root(resolved_level)

    logger_name = name or "tenets"
    logger = logging.getLogger(logger_name)
    logger.propagate = True

    # Apply level rules:
    # - If explicit level provided, set it for this logger
    # - If requesting the base 'tenets' logger (or name None), set its level
    # - If requesting a child under 'tenets.', let it inherit (don't set level)
    # - Otherwise (arbitrary logger names), set the resolved level
    if level is not None:
        logger.setLevel(level)
    else:
        if logger_name == "tenets":
            logger.setLevel(resolved_level)
        elif logger_name.startswith("tenets."):
            # Inherit from parent 'tenets' logger / root, do not set explicit level
            pass
        else:
            logger.setLevel(resolved_level)

    return logger
