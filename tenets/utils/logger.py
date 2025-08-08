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


def _configure_root(level: int) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers = []
    if _RICH_INSTALLED:
        handlers.append(
            RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        )
        fmt = "%(message)s"
    else:
        handlers.append(logging.StreamHandler())
        fmt = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    _CONFIGURED = True


def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger.

    Environment variables:
      - TENETS_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR|CRITICAL
    """
    env_level = os.getenv("TENETS_LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    resolved_level = level if level is not None else level_map.get(env_level, logging.INFO)

    _configure_root(resolved_level)

    logger = logging.getLogger(name or "tenets")
    # Avoid double propagation noise if user configures logging separately
    logger.propagate = True
    logger.setLevel(resolved_level)
    return logger
