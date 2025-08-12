"""Tenets CLI application."""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich import print
from rich.console import Console

from tenets import __version__
from tenets.cli.commands import (
    chronicle_command,
    distill_command,
    examine_command,
    instill_command,
    momentum_command,
)
from tenets.cli.commands.config import config_app
from tenets.cli.commands.session import session_app
from tenets.cli.commands.tenet import tenet_app
from tenets.cli.commands.viz import viz_app
from tenets.utils.logger import get_logger

# Create main app
app = typer.Typer(
    name="tenets",
    help="Context that feeds your prompts - intelligent code aggregation and analysis.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

console = Console()

# Add subcommand groups
app.add_typer(tenet_app, name="tenet", help="Manage guiding principles (tenets)")
app.add_typer(session_app, name="session", help="Manage development sessions")
app.add_typer(viz_app, name="viz", help="Visualize codebase insights")
app.add_typer(config_app, name="config", help="Configuration management")

# Register main commands
app.command()(distill_command.distill)
app.command()(instill_command.instill)
app.command()(examine_command.examine)
app.command()(chronicle_command.chronicle)
app.command()(momentum_command.momentum)


@app.command()
def version(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed version info")
):
    """Show version information."""
    if verbose:
        console.print(f"[bold]Tenets[/bold] v{__version__}")
        console.print("Context that feeds your prompts")
        console.print("\n[dim]Features:[/dim]")
        console.print("  • Intelligent context distillation")
        console.print("  • Guiding principles (tenets) system")
        console.print("  • Git-aware code analysis")
        console.print("  • Multi-factor relevance ranking")
        console.print("  • Token-optimized aggregation")
        console.print("\n[dim]Built by manic.agency[/dim]")
    else:
        print(f"tenets v{__version__}")


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-essential output"),
    silent: bool = typer.Option(False, "--silent", help="Only show errors"),
):
    """
    Tenets - Context that feeds your prompts.

    Distill relevant context from your codebase and instill guiding principles
    to maintain consistency across AI interactions.
    """
    # Store options in context for commands to access
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet or silent
    ctx.obj["silent"] = silent

    # Configure logging level
    import logging

    if verbose:
        get_logger(level=logging.DEBUG)
    elif quiet or silent:
        get_logger(level=logging.ERROR)
    else:
        # Default to WARNING to avoid noisy INFO
        get_logger(level=logging.WARNING)


def run():
    """Run the CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    run()
