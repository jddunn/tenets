"""Session management commands."""

from __future__ import annotations

import json
from typing import Optional

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
    """Create a new session or activate it if it already exists."""
    db = _get_db()
    existing = db.get_session(name)
    if existing:
        # If it exists, just mark it active and exit successfully
        db.set_active(name, True)
        console.print(f"[green]✓ Activated session:[/green] {name}")
        return
    db.create_session(name)
    db.set_active(name, True)
    console.print(f"[green]✓ Created session:[/green] {name}")


@session_app.command("start")
def start(name: str = typer.Argument(..., help="Session name")):
    """Start (create or activate) a session (alias of create)."""
    return create(name)


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
    table.add_column("Active", style="green")
    table.add_column("Created", style="green")
    table.add_column("Metadata", style="magenta")
    for s in sessions:
        # Coerce potential MagicMocks to plain serializable types for display
        meta = s.metadata if isinstance(s.metadata, dict) else {}
        is_active = "yes" if meta.get("active") else ""
        table.add_row(
            str(s.name),
            str(is_active),
            str(
                getattr(s.created_at, "isoformat", lambda **_: str(s.created_at))(
                    timespec="seconds"
                )
            ),
            json.dumps(meta),
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
            f"Name: {sess.name}\nActive: {bool(sess.metadata.get('active'))}\nCreated: {sess.created_at.isoformat(timespec='seconds')}\nMetadata: {json.dumps(sess.metadata, indent=2)}",
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


@session_app.command("clear")
def clear_all(keep_context: bool = typer.Option(False, "--keep-context", help="Keep artifacts")):
    """Delete ALL sessions (optionally keep artifacts)."""
    db = _get_db()
    count = db.delete_all_sessions(purge_context=not keep_context)
    if count:
        console.print(f"[red]Deleted {count} session(s).[/red]")
    else:
        console.print("[dim]No sessions to delete.[/dim]")


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
    db.set_active(name, True)
    console.print(f"[green]✓ Reset session:[/green] {name}")


@session_app.command("resume")
def resume(name: Optional[str] = typer.Argument(None, help="Session name (optional)")):
    """Mark a session as active (load/resume existing session).

    If NAME is omitted, resumes the most recently active session.
    """
    db = _get_db()
    target = name
    if not target:
        active = db.get_active_session()
        if not active:
            console.print("[red]No active session. Specify a NAME to resume.[/red]")
            raise typer.Exit(1)
        target = active.name
    sess = db.get_session(target)
    if not sess:
        console.print(f"[red]Session not found:[/red] {target}")
        raise typer.Exit(1)
    db.set_active(target, True)
    console.print(f"[green]✓ Resumed session:[/green] {target}")


@session_app.command("exit")
def exit_session(name: Optional[str] = typer.Argument(None, help="Session name (optional)")):
    """Mark a session as inactive (exit/end session).

    If NAME is omitted, exits the current active session.
    """
    db = _get_db()
    target = name
    if not target:
        active = db.get_active_session()
        if not active:
            console.print("[red]No active session to exit.[/red]")
            raise typer.Exit(1)
        target = active.name
    sess = db.get_session(target)
    if not sess:
        console.print(f"[red]Session not found:[/red] {target}")
        raise typer.Exit(1)
    db.set_active(target, False)
    console.print(f"[yellow]Exited session:[/yellow] {target}")
