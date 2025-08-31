"""Tenets CLI application."""

import sys

import typer
from rich import print
from rich.console import Console

# Import lightweight commands immediately
from tenets.cli.commands.config import config_app
from tenets.cli.commands.session import session_app
from tenets.cli.commands.system_instruction import app as system_instruction_app
from tenets.cli.commands.tenet import tenet_app

# Create main app
app = typer.Typer(
    name="tenets",
    help="Context that feeds your prompts - intelligent code aggregation and analysis.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

console = Console()


def _check_git_availability(ctx: typer.Context) -> bool:
    """Check if git is available and warn if not.

    Returns:
        bool: True if git is available, False otherwise
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "--version"], check=False, capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Only show warning if not in silent mode and if invoked command might use git
    if not ctx.obj.get("silent", False):
        # Check if user is using git-related commands
        invoked_command = ctx.invoked_subcommand
        git_related_commands = ["chronicle", "momentum", "examine", "distill"]

        if invoked_command in git_related_commands:
            # Rich console doesn't have stderr parameter, use file parameter instead
            import sys

            from rich.console import Console

            err_console = Console(stderr=True, file=sys.stderr)
            err_console.print(
                "[yellow]⚠ Git is not available or not in PATH.[/yellow]\n"
                "[dim]Git-related features (history analysis, authorship tracking) will be disabled.[/dim]\n"
                "[dim]All other features will work normally. To enable git features:[/dim]\n"
                "[dim]  • Install git: https://git-scm.com/downloads[/dim]\n"
                "[dim]  • Ensure git is in your system PATH[/dim]\n"
            )

    return False


# Register subcommand groups
# Lightweight commands are already imported above
app.add_typer(tenet_app, name="tenet", help="Manage guiding principles (tenets)")
app.add_typer(session_app, name="session", help="Manage development sessions")
app.add_typer(config_app, name="config", help="Configuration management")
app.add_typer(
    system_instruction_app,
    name="system-instruction",
    help="Manage system instruction (system prompt)",
)

# Delay import of heavy commands until they're actually called
# These imports are the slow ones that load ML libraries

# Import momentum (relatively lightweight)
from tenets.cli.commands.momentum import momentum as momentum_app

app.add_typer(momentum_app, name="momentum", help="Track team momentum and velocity")

# Import chronicle (medium weight, git operations)
from tenets.cli.commands.chronicle import chronicle as chronicle_app

app.add_typer(chronicle_app, name="chronicle", help="Analyze git history over time")

# Import examine (medium weight)
from tenets.cli.commands.examine import examine as examine_app

app.add_typer(examine_app, name="examine", help="Comprehensive code examination")

# Import viz (medium weight)
from tenets.cli.commands.viz import viz_app

app.add_typer(viz_app, name="viz", help="Visualize codebase insights")

# Register the heavy main commands - these are what's slowing us down
# We'll import them conditionally only when they're actually needed
import sys as _sys

if len(_sys.argv) > 1 and _sys.argv[1] in ["distill", "instill"]:
    # Only import these heavy commands if they're being called
    from tenets.cli.commands.distill import distill
    from tenets.cli.commands.instill import instill

    app.command()(distill)
    app.command()(instill)
else:
    # Create placeholder commands for help text
    @app.command(name="distill")
    def distill_placeholder(
        ctx: typer.Context,
        prompt: str = typer.Argument(..., help="Query or task to build context for"),
    ):
        """Distill relevant context from codebase for AI prompts."""
        # Import and run the real command
        from tenets.cli.commands.distill import distill

        # Remove the placeholder and register the real command
        app.registered_commands = [c for c in app.registered_commands if c.name != "distill"]
        app.command()(distill)
        # Re-invoke with the real command
        ctx.obj = ctx.obj or {}
        return ctx.invoke(distill, prompt=prompt)

    @app.command(name="instill")
    def instill_placeholder(ctx: typer.Context):
        """Apply tenets (guiding principles) to context."""
        # Import and run the real command
        from tenets.cli.commands.instill import instill

        # Remove the placeholder and register the real command
        app.registered_commands = [c for c in app.registered_commands if c.name != "instill"]
        app.command()(instill)
        # Re-invoke with the real command
        return ctx.invoke(instill)


@app.command()
def version(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed version info"),
):
    """Show version information."""
    if verbose:
        from tenets import __version__

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
        from tenets import __version__

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

    # Check git availability and warn if needed
    _check_git_availability(ctx)

    # Configure logging level
    import logging

    from tenets.utils.logger import get_logger

    # Configure both root logger and tenets logger
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("tenets").setLevel(logging.DEBUG)
        # Show debug output immediately
        logger = get_logger(__name__)
        logger.debug("Verbose mode enabled")
    elif quiet or silent:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger("tenets").setLevel(logging.ERROR)
    else:
        # Default to INFO for tenets, WARNING for others
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("tenets").setLevel(logging.INFO)


def run():
    """Run the CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e!s}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    run()
