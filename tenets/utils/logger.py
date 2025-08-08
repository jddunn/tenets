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
    global _CONFIGURED, _CURRENT_LEVEL
    # If already configured, just update levels if different
    if _CONFIGURED:
        if _CURRENT_LEVEL != level:
            root = logging.getLogger()
            root.setLevel(level)
            for h in root.handlers:
                h.setLevel(level)
            _CURRENT_LEVEL = level
        return

    handlers = []
    if _RICH_INSTALLED:
        handlers.append(RichHandler(rich_tracebacks=True, show_time=True, show_path=False))
        fmt = "%(message)s"
    else:
        handlers.append(logging.StreamHandler())
        fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    _CONFIGURED = True
    _CURRENT_LEVEL = level


def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger.

    Environment variables:
      - TENETS_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL
    """
    env_level = os.getenv("TENETS_LOG_LEVEL")
    # Default to ERROR unless explicitly overridden
    default_level_name = env_level.upper() if env_level else "ERROR"
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    resolved_level = level if level is not None else level_map.get(default_level_name, logging.ERROR)

    _configure_root(resolved_level)

    logger = logging.getLogger(name or "tenets")
    logger.propagate = True
    logger.setLevel(resolved_level)
    return logger
