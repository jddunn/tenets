"""Instill command - apply guiding principles to context."""

from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from tenets import Tenets

console = Console()


def instill(
    session: Optional[str] = typer.Option(
        None,
        "--session", "-s",
        help="Target session for instillation"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Re-instill even if already applied"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be instilled without applying"
    ),
    list_pending: bool = typer.Option(
        False,
        "--list-pending",
        help="List pending tenets and exit"
    ),
    ctx: typer.Context = typer.Context,
):
    """
    Instill guiding principles (tenets) into your context.
    
    This command applies all pending tenets to ensure consistent coding principles
    are maintained across AI interactions. Tenets are strategically injected into
    generated context to combat context drift.
    
    Examples:
    
        # Instill all pending tenets
        tenets instill
        
        # Instill into specific session
        tenets instill --session oauth-work
        
        # Re-instill all tenets (force)
        tenets instill --force
        
        # Preview what would be instilled
        tenets instill --dry-run
        
        # List pending tenets
        tenets instill --list-pending
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    
    try:
        tenets = Tenets()
        
        # Check if tenet system is available
        if not tenets.tenet_manager:
            console.print("[red]Error:[/red] Tenet system is not available.")
            console.print("This may be due to missing dependencies.")
            raise typer.Exit(1)
        
        # List pending if requested
        if list_pending:
            pending = tenets.get_pending_tenets(session=session)
            
            if not pending:
                console.print("[yellow]No pending tenets found.[/yellow]")
                if not session:
                    console.print("Add tenets with: [bold]tenets tenet add \"Your principle\"[/bold]")
                return
            
            # Create table
            table = Table(title=f"Pending Tenets{f' (Session: {session})' if session else ''}")
            table.add_column("ID", style="cyan", width=12)
            table.add_column("Content", style="white")
            table.add_column("Priority", style="yellow")
            table.add_column("Category", style="blue")
            
            for tenet in pending:
                table.add_row(
                    str(tenet.id)[:8] + "...",
                    tenet.content[:60] + "..." if len(tenet.content) > 60 else tenet.content,
                    tenet.priority.value,
                    tenet.category.value if tenet.category else "-"
                )
            
            console.print(table)
            return
        
        # Get pending tenets
        pending_tenets = tenets.get_pending_tenets(session=session)
        
        if not pending_tenets and not force:
            console.print("[yellow]No pending tenets to instill.[/yellow]")
            console.print("\nAdd tenets with: [bold]tenets tenet add \"Your principle\"[/bold]")
            console.print("Or use [bold]--force[/bold] to re-instill existing tenets.")
            return
        
        # Dry run mode
        if dry_run:
            console.print("[bold]Would instill the following tenets:[/bold]\n")
            
            for i, tenet in enumerate(pending_tenets, 1):
                priority_color = {
                    "critical": "red",
                    "high": "yellow",
                    "medium": "blue",
                    "low": "dim"
                }.get(tenet.priority.value, "white")
                
                console.print(
                    f"{i}. [[{priority_color}]{tenet.priority.value.upper()}[/{priority_color}]] "
                    f"{tenet.content}"
                )
                
                if tenet.category:
                    console.print(f"   Category: {tenet.category.value}")
                console.print(f"   Added: {tenet.created_at.strftime('%Y-%m-%d %H:%M')}")
                console.print()
            
            console.print(f"\n[dim]Total: {len(pending_tenets)} tenet(s)[/dim]")
            return
        
        # Instill tenets
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Instilling tenets...", total=None)
            
            result = tenets.instill_tenets(
                session=session,
                force=force
            )
        
        # Show results
        if not quiet:
            console.print(Panel(
                f"[green]✓[/green] Successfully instilled {result['count']} tenet(s)\n\n"
                f"Session: {session or 'global'}\n"
                f"Strategy: {result['strategy']}",
                title="🌟 Tenets Instilled",
                border_style="green"
            ))
            
            if result.get('tenets') and verbose:
                console.print("\n[bold]Instilled tenets:[/bold]")
                for tenet in result['tenets']:
                    console.print(f"  • {tenet}")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)