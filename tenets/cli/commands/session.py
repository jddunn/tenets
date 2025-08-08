"""Session management commands."""

from __future__ import annotations

import typer
from rich.console import Console

session_app = typer.Typer(help="Manage development sessions")
console = Console()


@session_app.command()
def create(name: str = typer.Argument(..., help="Session name")):
    """Create a new session."""
    console.print(f"[green]Session created:[/green] {name}")


@session_app.command("list")
def list_cmd():
    """List sessions (placeholder)."""
    console.print("No sessions implemented yet.")


@session_app.command()
def show(name: str):
    """Show session details (placeholder)."""
    console.print(f"Session: {name}")


@session_app.command()
def delete(name: str):
    """Delete a session (placeholder)."""
    console.print(f"[red]Deleted session:[/red] {name}")
