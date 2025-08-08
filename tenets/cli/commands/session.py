"""Session management commands."""

from __future__ import annotations

import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tenets.config import TenetsConfig
from tenets.storage.session_db import SessionDB

session_app = typer.Typer(help="Manage development sessions")
console = Console()


def _get_db() -> SessionDB:
    return SessionDB(TenetsConfig())


@session_app.command()
def create(name: str = typer.Argument(..., help="Session name")):
    """Create a new session (no-op if it already exists)."""
    db = _get_db()
    existing = db.get_session(name)
    if existing:
        console.print(f"[yellow]Session already exists:[/yellow] {name}")
        raise typer.Exit(code=0)
    db.create_session(name)
    console.print(f"[green]✓ Created session:[/green] {name}")


@session_app.command("list")
def list_cmd():
    """List sessions."""
    db = _get_db()
    sessions = db.list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return
    table = Table(title="Sessions")
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Metadata", style="magenta")
    for s in sessions:
        table.add_row(
            s.name,
            s.created_at.isoformat(timespec="seconds"),
            json.dumps(s.metadata),
        )
    console.print(table)


@session_app.command()
def show(name: str = typer.Argument(..., help="Session name")):
    """Show session details."""
    db = _get_db()
    sess = db.get_session(name)
    if not sess:
        console.print(f"[red]Session not found:[/red] {name}")
        raise typer.Exit(1)
    console.print(
        Panel(
            f"Name: {sess.name}\nCreated: {sess.created_at.isoformat(timespec='seconds')}\nMetadata: {json.dumps(sess.metadata, indent=2)}",
            title=f"Session: {sess.name}",
        )
    )


@session_app.command()
def delete(
    name: str = typer.Argument(..., help="Session name"),
    keep_context: bool = typer.Option(
        False, "--keep-context", help="Do not delete stored context artifacts"
    ),
):
    """Delete a session (and its stored context unless --keep-context)."""
    db = _get_db()
    deleted = db.delete_session(name, purge_context=not keep_context)
    if deleted:
        console.print(f"[red]Deleted session:[/red] {name}")
    else:
        console.print(f"[yellow]No such session:[/yellow] {name}")


@session_app.command("add")
def add_context(
    name: str = typer.Argument(..., help="Session name"),
    kind: str = typer.Argument(..., help="Content kind tag (e.g. note, context_result)"),
    file: typer.FileText = typer.Argument(..., help="File whose content to attach"),
):
    """Attach arbitrary content file to a session (stored as text)."""
    db = _get_db()
    content = file.read()
    db.add_context(name, kind=kind, content=content)
    console.print(f"[green]✓ Added {kind} to session:[/green] {name}")


@session_app.command("reset")
def reset_session(name: str = typer.Argument(..., help="Session name")):
    """Reset (delete and recreate) a session and purge its context."""
    db = _get_db()
    db.delete_session(name, purge_context=True)
    db.create_session(name)
    console.print(f"[green]✓ Reset session:[/green] {name}")
