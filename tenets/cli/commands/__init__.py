"""CLI command modules."""

# Import command modules to make them available
from tenets.cli.commands import chronicle as chronicle_command
from tenets.cli.commands import distill as distill_command
from tenets.cli.commands import examine as examine_command
from tenets.cli.commands import instill as instill_command
from tenets.cli.commands import momentum as momentum_command
from tenets.cli.commands import system_instruction as system_instruction_command

__all__ = [
    "distill_command",
    "instill_command",
    "examine_command",
    "chronicle_command",
    "momentum_command",
    "system_instruction_command",
]
