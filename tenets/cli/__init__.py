"""Tenets CLI package.

This package contains the Typer application and command groupings used by the
``tenets`` command-line interface.

Modules
-------
- :mod:`tenets.cli.app` exposes the top-level Typer ``app`` and ``run()``.
- :mod:`tenets.cli.commands` contains individual subcommands and groups.

Typical usage
-------------
>>> from tenets.cli.app import app  # noqa: F401
>>> # or programmatically invoke
>>> # from tenets.cli.app import run; run()
"""

from . import commands as commands  # re-export commands package for convenience

# Re-export the Typer app and run helper for documentation and imports
from .app import app, run  # noqa: F401

__all__ = [
    "app",
    "run",
    "commands",
]
