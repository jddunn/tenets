"""Manage the persistent BM25/TF-IDF corpus index (tenets >= 0.11.0).

The index builds automatically on the first rank/distill and is reused after; these
commands let you inspect, warm, or clear it.
"""

import contextlib
import shutil
import sqlite3
from pathlib import Path

import typer
from rich.console import Console

index_app = typer.Typer(help="Manage the persistent corpus index")
console = Console()


def _index_dir() -> Path:
    from tenets.config import TenetsConfig

    return Path(TenetsConfig().cache_dir) / "index"


def _cached_roots(db_path: Path) -> list:
    """Scan roots that have a persisted index (keys are ``corpus_index::<root>``)."""
    try:
        with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
            rows = conn.execute("SELECT key FROM cache").fetchall()
    except sqlite3.Error:
        return []
    return sorted(
        r[0].split("corpus_index::", 1)[-1] for r in rows if "corpus_index::" in (r[0] or "")
    )


@index_app.command("status")
def status() -> None:
    """Show the persistent index location, size, and cached scan roots."""
    idx_dir = _index_dir()
    db = idx_dir / "corpus_index.db"
    console.print(f"[bold]Index dir:[/bold] {idx_dir}")
    if not db.exists():
        console.print("[yellow]No index yet[/yellow] — it builds on the first rank/distill.")
        return
    size_mb = db.stat().st_size / (1024 * 1024)
    console.print(f"[bold]Index db:[/bold]  {db.name}  ([cyan]{size_mb:.1f} MB[/cyan])")
    roots = _cached_roots(db)
    if roots:
        console.print(f"[bold]Cached scan roots ({len(roots)}):[/bold]")
        for r in roots:
            console.print(f"  • {r}")
    else:
        console.print("[dim](no cached roots recorded)[/dim]")


@index_app.command("clear")
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete the persistent corpus index (it rebuilds on the next query)."""
    idx_dir = _index_dir()
    if not idx_dir.exists():
        console.print("No index to clear.")
        return
    if not yes and not typer.confirm(f"Delete the corpus index at {idx_dir}?"):
        raise typer.Abort()
    shutil.rmtree(idx_dir, ignore_errors=True)
    console.print(f"[green]Cleared[/green] {idx_dir}")


@index_app.command("build")
def build(
    path: Path = typer.Argument(Path("."), help="Directory to index / warm"),
) -> None:
    """Warm the persistent index for a path so the first real query is fast."""
    from tenets import Tenets

    console.print(f"Building corpus index for [cyan]{path}[/cyan] …")
    result = Tenets().rank_files(prompt="warm the corpus index", paths=str(path))
    n = len(getattr(result, "files", []) or [])
    console.print(f"[green]Indexed[/green] {n} files for {path}")
