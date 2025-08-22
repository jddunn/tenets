"""Entry point for running tenets CLI as a module.

This allows running tenets with:
    python -m tenets.cli

instead of:
    python -m tenets.cli.app
"""

from .app import run

if __name__ == "__main__":
    run()
