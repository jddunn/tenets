"""Logging utilities for Tenets.

Provides a single entrypoint `get_logger` that configures Rich logging once
and returns child loggers for modules.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# Lazy load Rich to improve import performance
RichHandler = None
Console = None


def _check_rich_available():
    """Check if Rich is available and colors are not disabled."""
    import os

    # Respect NO_COLOR standard (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False

    # Also check FORCE_COLOR=0
    if os.environ.get("FORCE_COLOR") == "0":
        return False

    try:
        # Try a simple import instead of using importlib.util which seems to hang
        import rich

        return True
    except ImportError:
        return False


# Defer the check to avoid import-time hangs
_RICH_INSTALLED = None


def _ensure_rich_imported():
    """Import Rich when actually needed."""
    global RichHandler, Console, _RICH_INSTALLED

    # Check Rich availability on first use if not already checked
    if _RICH_INSTALLED is None:
        _RICH_INSTALLED = _check_rich_available()

    if RichHandler is None and _RICH_INSTALLED:
        try:
            from rich.console import Console as _Console
            from rich.logging import RichHandler as _RichHandler

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

    # Check Rich availability if not already checked
    global _RICH_INSTALLED
    if _RICH_INSTALLED is None:
        _RICH_INSTALLED = _check_rich_available()

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
            console = (
                Console(width=width, force_terminal=True, legacy_windows=False) if Console else None
            )
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


class _MinLevelFilter(logging.Filter):
    """Drop records below a minimum level, independent of handler.setLevel().

    The root logger's handler levels get re-synced whenever get_logger() runs at
    a new level; a filter survives that, so the MCP stderr stream stays clean even
    as modules log at INFO during request handling.
    """

    def __init__(self, min_level: int):
        super().__init__()
        self.min_level = min_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= self.min_level


def configure_logging(
    stderr_level: int = logging.WARNING,
    file_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Route detailed logs to a file while keeping stderr quiet.

    Used by the MCP server: stderr (which the MCP client surfaces as ``[ERROR]``)
    receives only ``stderr_level`` and above, enforced by a filter that survives
    later level changes; ``log_file`` receives ``file_level`` and above for
    debugging. Replaces the previous basicConfig that leaked INFO onto stderr.
    """
    global _CONFIGURED, _CURRENT_LEVEL

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(stderr_level)
    stderr_handler.addFilter(_MinLevelFilter(stderr_level))
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(stderr_handler)

    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)
            file_handler.addFilter(_MinLevelFilter(file_level))
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(file_handler)
        except OSError:
            pass

    effective = min(stderr_level, file_level)
    root.setLevel(effective)
    _CONFIGURED = True
    _CURRENT_LEVEL = effective
