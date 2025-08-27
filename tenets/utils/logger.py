"""Logging utilities for Tenets.

Provides a single entrypoint `get_logger` that configures Rich logging once
and returns child loggers for modules.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# Lazy load Rich to improve import performance
RichHandler = None
Console = None

def _check_rich_available():
    """Check if Rich is available."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("rich")
        return spec is not None
    except (ImportError, AttributeError):
        return False

_RICH_INSTALLED = _check_rich_available()

def _ensure_rich_imported():
    """Import Rich when actually needed."""
    global RichHandler, Console
    if RichHandler is None and _RICH_INSTALLED:
        try:
            from rich.logging import RichHandler as _RichHandler
            from rich.console import Console as _Console
            RichHandler = _RichHandler
            Console = _Console
        except Exception:
            pass

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
        if level != _CURRENT_LEVEL:
            root.setLevel(level)
            for h in root.handlers:
                h.setLevel(level)
        _CURRENT_LEVEL = level
        return

    # Create or update a handler
    if _RICH_INSTALLED:
        _ensure_rich_imported()  # Import Rich when needed
        # Try to reuse an existing RichHandler if present
        handler = None
        for h in root.handlers:
            if h.__class__.__name__ == "RichHandler":
                handler = h
                break
        if handler is None and RichHandler is not None:
            # Force a reasonable width to prevent character wrapping
            import shutil

            # Get terminal width or use a reasonable default
            terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
            # Use at least 120 columns to prevent wrapping
            width = max(120, terminal_width)
            console = Console(width=width, force_terminal=True, legacy_windows=False) if Console else None
            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                console=console,
                markup=True,
                log_time_format="[%X]",
            )
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
    elif logger_name == "tenets":
        logger.setLevel(resolved_level)
    elif logger_name.startswith("tenets."):
        # Inherit from parent 'tenets' logger / root, do not set explicit level
        pass
    else:
        logger.setLevel(resolved_level)

    return logger
