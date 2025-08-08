"""CLI command modules."""

# Import command modules to make them available
from tenets.cli.commands import (
    distill as distill_command,
    instill as instill_command,
    examine as examine_command,
    chronicle as chronicle_command,
    momentum as momentum_command,
)

__all__ = [
    "distill_command",
    "instill_command",
    "examine_command",
    "chronicle_command",
    "momentum_command",
]
