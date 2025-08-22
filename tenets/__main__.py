"""Entry point for running tenets as a module.

This allows running tenets with:
    python -m tenets

instead of:
    python -m tenets.cli.app
"""

from tenets.cli.app import run

if __name__ == "__main__":
    run()
