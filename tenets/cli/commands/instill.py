"""Instill command - Smart injection of guiding principles into context.

This command provides comprehensive control over tenet injection including:
- Multiple injection frequency modes (always, periodic, adaptive, manual)
- Session-aware injection tracking
- Complexity analysis for smart injection
- History and statistics viewing
- Export capabilities for analysis
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from tenets import Tenets
from tenets.config import TenetsConfig

console = Console()


def instill(
    # Session management
    session: Optional[str] = typer.Option(
        None, "--session", "-s", help="Target session for instillation"
    ),
    
    # Injection control
    force: bool = typer.Option(
        False, "--force", "-f", help="Force injection regardless of frequency settings"
    ),
    frequency: Optional[str] = typer.Option(
        None, "--frequency", help="Override injection frequency (always/periodic/adaptive/manual)"
    ),
    interval: Optional[int] = typer.Option(
        None, "--interval", help="Override injection interval for periodic mode"
    ),
    
    # Analysis and preview
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be instilled without applying"
    ),
    analyze: bool = typer.Option(
        False, "--analyze", help="Analyze injection patterns and effectiveness"
    ),
    stats: bool = typer.Option(
        False, "--stats", help="Show injection statistics"
    ),
    
    # Listing options
    list_pending: bool = typer.Option(
        False, "--list-pending", help="List pending tenets and exit"
    ),
    list_history: bool = typer.Option(
        False, "--list-history", help="Show injection history for session"
    ),
    list_sessions: bool = typer.Option(
        False, "--list-sessions", help="List all tracked sessions"
    ),
    
    # File pinning
    add_file: Optional[list[str]] = typer.Option(
        None,
        "--add-file",
        "-F",
        help="Pin a file for future distill operations (can be passed multiple times)",
    ),
    add_folder: Optional[list[str]] = typer.Option(
        None,
        "--add-folder",
        "-D",
        help="Pin all files in a folder (respects .gitignore)",
    ),
    remove_file: Optional[list[str]] = typer.Option(
        None,
        "--remove-file",
        help="Unpin a file from the session",
    ),
    list_pinned: bool = typer.Option(
        False, "--list-pinned", help="List pinned files for the session and exit"
    ),
    
    # Session management
    reset_session: bool = typer.Option(
        False, "--reset-session", help="Reset injection history for the session"
    ),
    clear_all_sessions: bool = typer.Option(
        False, "--clear-all-sessions", help="Clear all session histories (requires confirmation)"
    ),
    
    # Export options
    export_history: Optional[Path] = typer.Option(
        None, "--export-history", help="Export injection history to file (JSON or CSV)"
    ),
    export_format: str = typer.Option(
        "json", "--export-format", help="Format for export (json/csv)"
    ),
    
    # Configuration
    set_frequency: Optional[str] = typer.Option(
        None, "--set-frequency", help="Set default injection frequency and save to config"
    ),
    set_interval: Optional[int] = typer.Option(
        None, "--set-interval", help="Set default injection interval and save to config"
    ),
    show_config: bool = typer.Option(
        False, "--show-config", help="Show current injection configuration"
    ),
    
    # Context
    ctx: typer.Context = typer.Context,
):
    """
    Smart injection of guiding principles (tenets) into your context.

    This command manages the injection of tenets with intelligent frequency control,
    session tracking, and complexity-aware adaptation. Tenets are strategically
    placed to maintain consistent coding principles across AI interactions.

    INJECTION MODES:
        always   - Inject into every distilled context
        periodic - Inject every Nth distillation
        adaptive - Smart injection based on complexity
        manual   - Only inject when forced

    Examples:

        # Standard injection (uses configured frequency)
        tenets instill

        # Force injection regardless of frequency
        tenets instill --force

        # Session-specific injection
        tenets instill --session oauth-work

        # Set injection to every 5th distill
        tenets instill --set-frequency periodic --set-interval 5

        # View injection statistics
        tenets instill --stats --session oauth-work

        # Analyze effectiveness
        tenets instill --analyze

        # Pin files for guaranteed inclusion
        tenets instill --add-file src/core.py --session main

        # Export history for analysis
        tenets instill --export-history analysis.json

        # Reset session tracking
        tenets instill --reset-session --session oauth-work
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        # Load configuration
        config = TenetsConfig()
        tenets_instance = Tenets(config)

        # Check if tenet system is available
        if not hasattr(tenets_instance, 'instiller') or not tenets_instance.instiller:
            console.print("[red]Error:[/red] Tenet system is not available.")
            console.print("This may be due to missing dependencies or configuration issues.")
            raise typer.Exit(1)

        instiller = tenets_instance.instiller

        # ============= Configuration Commands =============

        if show_config:
            _show_injection_config(config)
            return

        if set_frequency:
            _set_injection_frequency(config, set_frequency, set_interval)
            return

        # ============= Session Management =============

        if list_sessions:
            _list_sessions(instiller)
            return

        if clear_all_sessions:
            if typer.confirm("Clear all session histories? This cannot be undone."):
                _clear_all_sessions(instiller)
            return

        if reset_session:
            if not session:
                console.print("[red]Error:[/red] --reset-session requires --session")
                raise typer.Exit(1)
            _reset_session(instiller, session)
            return

        # ============= Listing Commands =============

        if list_history:
            _show_injection_history(instiller, session)
            return

        if stats:
            _show_statistics(instiller, session)
            return

        if analyze:
            _analyze_effectiveness(instiller, session)
            return

        if list_pending:
            _list_pending_tenets(tenets_instance, session)
            return

        # ============= File Pinning =============

        if list_pinned:
            _list_pinned_files(tenets_instance, session)
            return

        if add_file or add_folder or remove_file:
            _manage_pinned_files(
                tenets_instance,
                session,
                add_file,
                add_folder,
                remove_file
            )
            if not (force or dry_run):  # Only manage files, don't instill
                return

        # ============= Export =============

        if export_history:
            _export_history(instiller, export_history, export_format, session)
            return

        # ============= Main Instillation Logic =============

        # Get injection frequency configuration
        if frequency:
            injection_frequency = frequency
        else:
            injection_frequency = config.tenet.injection_frequency

        if interval:
            injection_interval = interval
        else:
            injection_interval = config.tenet.injection_interval

        # Get or create session
        session_name = session or "default"

        # Get session history if exists
        if session_name in instiller.session_histories:
            history = instiller.session_histories[session_name]
            session_info = history.get_stats()
        else:
            session_info = None

        # Show session status
        if not quiet and session_info:
            console.print(
                Panel(
                    f"Session: [cyan]{session_name}[/cyan]\n"
                    f"Distills: {session_info['total_distills']}\n"
                    f"Injections: {session_info['total_injections']}\n"
                    f"Rate: {session_info['injection_rate']:.1%}\n"
                    f"Avg Complexity: {session_info['average_complexity']:.2f}",
                    title="Session Status",
                    border_style="blue",
                )
            )

        # Check if injection should occur
        if not force and injection_frequency != "manual":
            # Simulate a distill to check frequency
            test_context = "# Test Context\n\nChecking injection frequency..."
            
            # This won't actually inject, just checks frequency
            result = instiller.instill(
                test_context,
                session=session_name,
                force=False,
                check_frequency=True,
            )
            
            # Check if injection was skipped
            last_record = instiller.metrics_tracker.instillations[-1] if instiller.metrics_tracker.instillations else None
            if last_record and last_record.get("skip_reason"):
                skip_reason = last_record["skip_reason"]
                
                if not quiet:
                    console.print(
                        f"[yellow]Injection skipped:[/yellow] {skip_reason}\n"
                        f"Use --force to override or wait for next trigger."
                    )
                    
                    # Show when next injection will occur
                    if "periodic" in injection_frequency:
                        next_at = ((session_info['total_distills'] // injection_interval) + 1) * injection_interval
                        console.print(f"Next injection at distill #{next_at}")
                        
                return

        # Dry run mode
        if dry_run:
            _dry_run_instillation(tenets_instance, session_name, injection_frequency)
            return

        # Perform instillation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Instilling tenets...", total=None)

            # Create sample context for demonstration
            sample_context = (
                "# Sample Context\n\n"
                "This is a demonstration of tenet injection.\n"
                "The instiller will analyze this context and inject "
                "appropriate tenets based on the configured strategy."
            )

            result = instiller.instill(
                sample_context,
                session=session_name,
                force=force,
                strategy=None,  # Use configured strategy
                check_frequency=not force,
            )

        # Get the last instillation result
        if instiller._cache:
            last_key = list(instiller._cache.keys())[-1]
            last_result = instiller._cache[last_key]
            
            if not quiet:
                _show_instillation_result(last_result, verbose)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


# ============= Helper Functions =============

def _show_injection_config(config: TenetsConfig) -> None:
    """Show current injection configuration."""
    table = Table(title="Tenet Injection Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")

    settings = [
        ("Frequency Mode", config.tenet.injection_frequency, "When to inject tenets"),
        ("Injection Interval", str(config.tenet.injection_interval), "For periodic mode"),
        ("Complexity Threshold", f"{config.tenet.session_complexity_threshold:.2f}", "Triggers adaptive injection"),
        ("Min Session Length", str(config.tenet.min_session_length), "Before first injection"),
        ("Max Per Context", str(config.tenet.max_per_context), "Maximum tenets to inject"),
        ("Strategy", config.tenet.injection_strategy, "Placement strategy"),
        ("Decay Rate", f"{config.tenet.decay_rate:.2f}", "How quickly tenets decay"),
        ("Reinforcement Interval", str(config.tenet.reinforcement_interval), "Reinforce critical tenets"),
        ("Track History", str(config.tenet.track_injection_history), "Track per-session patterns"),
        ("Session Aware", str(config.tenet.session_aware), "Use session patterns"),
    ]

    for setting, value, desc in settings:
        table.add_row(setting, value, desc)

    console.print(table)


def _set_injection_frequency(
    config: TenetsConfig,
    frequency: str,
    interval: Optional[int]
) -> None:
    """Set and save injection frequency configuration."""
    valid_frequencies = ["always", "periodic", "adaptive", "manual"]
    
    if frequency not in valid_frequencies:
        console.print(f"[red]Error:[/red] Invalid frequency. Must be one of: {', '.join(valid_frequencies)}")
        raise typer.Exit(1)

    # Update configuration
    config.tenet.injection_frequency = frequency
    
    if interval:
        config.tenet.injection_interval = interval

    # Save to config file
    config_file = config.config_file or Path(".tenets.yml")
    config.save(config_file)

    console.print(
        Panel(
            f"[green]✓[/green] Injection frequency set to: [cyan]{frequency}[/cyan]\n"
            f"Interval: {interval or config.tenet.injection_interval}\n"
            f"Configuration saved to: {config_file}",
            title="Configuration Updated",
            border_style="green",
        )
    )


def _list_sessions(instiller) -> None:
    """List all tracked sessions."""
    if not instiller.session_histories:
        console.print("[yellow]No sessions tracked yet.[/yellow]")
        return

    table = Table(title="Tracked Sessions")
    table.add_column("Session", style="cyan")
    table.add_column("Distills", justify="right")
    table.add_column("Injections", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Last Injection", style="dim")
    table.add_column("Complexity", justify="right")

    for session_id, history in instiller.session_histories.items():
        stats = history.get_stats()
        
        last_injection = "Never"
        if history.last_injection:
            time_ago = datetime.now() - history.last_injection
            if time_ago.days > 0:
                last_injection = f"{time_ago.days}d ago"
            elif time_ago.seconds > 3600:
                last_injection = f"{time_ago.seconds // 3600}h ago"
            else:
                last_injection = f"{time_ago.seconds // 60}m ago"

        table.add_row(
            session_id,
            str(stats["total_distills"]),
            str(stats["total_injections"]),
            f"{stats['injection_rate']:.1%}",
            last_injection,
            f"{stats['average_complexity']:.2f}"
        )

    console.print(table)


def _clear_all_sessions(instiller) -> None:
    """Clear all session histories."""
    count = len(instiller.session_histories)
    instiller.session_histories.clear()
    instiller._save_session_histories()
    
    console.print(f"[green]✓[/green] Cleared {count} session histories")


def _reset_session(instiller, session: str) -> None:
    """Reset a specific session's history."""
    if instiller.reset_session_history(session):
        console.print(f"[green]✓[/green] Reset injection history for session: {session}")
    else:
        console.print(f"[yellow]No history found for session: {session}[/yellow]")


def _show_injection_history(instiller, session: Optional[str]) -> None:
    """Show detailed injection history."""
    records = instiller.metrics_tracker.instillations
    
    if session:
        records = [r for r in records if r.get("session") == session]
        title = f"Injection History - Session: {session}"
    else:
        title = "Injection History - All Sessions"

    if not records:
        console.print("[yellow]No injection history found.[/yellow]")
        return

    # Show recent injections
    table = Table(title=title)
    table.add_column("Time", style="dim")
    table.add_column("Session", style="cyan")
    table.add_column("Tenets", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Strategy", style="green")
    table.add_column("Complexity", justify="right")
    table.add_column("Status", style="yellow")

    # Show last 20 records
    for record in records[-20:]:
        timestamp = datetime.fromisoformat(record["timestamp"])
        time_str = timestamp.strftime("%Y-%m-%d %H:%M")
        
        status = "Skipped" if record.get("skip_reason") else "Injected"
        if record.get("skip_reason"):
            status = f"Skip: {record['skip_reason'][:20]}"

        table.add_row(
            time_str,
            record.get("session", "default"),
            str(record["tenet_count"]),
            str(record["token_increase"]),
            record["strategy"],
            f"{record.get('complexity', 0):.2f}",
            status
        )

    console.print(table)


def _show_statistics(instiller, session: Optional[str]) -> None:
    """Show injection statistics."""
    metrics = instiller.metrics_tracker.get_metrics(session)
    
    if "message" in metrics:
        console.print(f"[yellow]{metrics['message']}[/yellow]")
        return

    # Create statistics display
    title = f"Injection Statistics - {session or 'All Sessions'}"
    
    stats_text = f"""
[bold]Overall:[/bold]
  Total Instillations: {metrics['total_instillations']}
  Total Tenets: {metrics['total_tenets_instilled']}
  Total Tokens Added: {metrics['total_token_increase']:,}
  Avg Tenets/Context: {metrics['avg_tenets_per_context']:.1f}
  Avg Tokens/Context: {metrics['avg_token_increase']:.0f}
  Avg Complexity: {metrics['avg_complexity']:.2f}

[bold]Strategy Distribution:[/bold]"""

    for strategy, count in metrics.get('strategy_distribution', {}).items():
        stats_text += f"\n  {strategy}: {count}"

    if metrics.get('skip_distribution'):
        stats_text += "\n\n[bold]Skip Reasons:[/bold]"
        for reason, count in metrics['skip_distribution'].items():
            stats_text += f"\n  {reason}: {count}"

    if metrics.get('top_tenets'):
        stats_text += "\n\n[bold]Most Used Tenets:[/bold]"
        for tenet_id, count in metrics['top_tenets'][:5]:
            stats_text += f"\n  {tenet_id[:8]}...: {count} times"

    console.print(Panel(stats_text, title=title, border_style="blue"))


def _analyze_effectiveness(instiller, session: Optional[str]) -> None:
    """Analyze injection effectiveness."""
    analysis = instiller.analyze_effectiveness(session)
    
    # Create tree display
    tree = Tree("🎯 Tenet Injection Analysis")
    
    # Configuration
    config_branch = tree.add("⚙️ Configuration")
    for key, value in analysis['configuration'].items():
        config_branch.add(f"{key}: {value}")
    
    # Metrics
    metrics_branch = tree.add("📊 Metrics")
    metrics = analysis['instillation_metrics']
    if metrics and "message" not in metrics:
        metrics_branch.add(f"Total Injections: {metrics.get('total_instillations', 0)}")
        metrics_branch.add(f"Average Complexity: {metrics.get('avg_complexity', 0):.2f}")
        metrics_branch.add(f"Total Tokens: {metrics.get('total_token_increase', 0):,}")
    
    # Tenet effectiveness
    tenet_branch = tree.add("📈 Tenet Effectiveness")
    tenet_data = analysis['tenet_effectiveness']
    if tenet_data:
        tenet_branch.add(f"Total Tenets: {tenet_data.get('total_tenets', 0)}")
        
        if tenet_data.get('by_priority'):
            priority_branch = tenet_branch.add("By Priority")
            for priority, data in tenet_data['by_priority'].items():
                priority_branch.add(f"{priority}: {data}")
    
    # Recommendations
    if analysis['recommendations']:
        rec_branch = tree.add("💡 Recommendations")
        for rec in analysis['recommendations']:
            rec_branch.add(rec)
    
    console.print(tree)


def _list_pending_tenets(tenets_instance, session: Optional[str]) -> None:
    """List pending tenets."""
    pending = tenets_instance.get_pending_tenets(session=session)

    if not pending:
        console.print("[yellow]No pending tenets found.[/yellow]")
        if not session:
            console.print('Add tenets with: [bold]tenets tenet add "Your principle"[/bold]')
        return

    # Create table
    table = Table(title=f"Pending Tenets{f' (Session: {session})' if session else ''}")
    table.add_column("ID", style="cyan", width=12)
    table.add_column("Content", style="white")
    table.add_column("Priority", style="yellow")
    table.add_column("Category", style="blue")
    table.add_column("Injections", justify="right")

    for tenet in pending:
        content_preview = (
            tenet.content[:60] + "..." 
            if len(tenet.content) > 60 
            else tenet.content
        )
        
        table.add_row(
            str(tenet.id)[:8] + "...",
            content_preview,
            tenet.priority.value,
            tenet.category.value if tenet.category else "-",
            str(tenet.metrics.injection_count)
        )

    console.print(table)


def _list_pinned_files(tenets_instance, session: Optional[str]) -> None:
    """List pinned files for a session."""
    sess_name = session or "default"
    pinned_map = tenets_instance.config.custom.get("pinned_files", {})
    files = sorted(pinned_map.get(sess_name, [])) if pinned_map else []
    
    if not files:
        console.print(f"[yellow]No pinned files for session: {sess_name}[/yellow]")
    else:
        console.print(
            Panel(
                "\n".join(files),
                title=f"Pinned Files ({sess_name})",
                border_style="green"
            )
        )


def _manage_pinned_files(
    tenets_instance,
    session: Optional[str],
    add_files: Optional[list[str]],
    add_folders: Optional[list[str]],
    remove_files: Optional[list[str]]
) -> None:
    """Manage pinned files for a session."""
    sess_name = session or "default"
    added = 0
    removed = 0
    
    # Add individual files
    if add_files:
        for f in add_files:
            if tenets_instance.add_file_to_session(f, session=sess_name):
                added += 1
                if not quiet:
                    console.print(f"[green]✓[/green] Pinned: {f}")
    
    # Add folders
    if add_folders:
        for d in add_folders:
            count = tenets_instance.add_folder_to_session(d, session=sess_name)
            added += count
            if not quiet and count > 0:
                console.print(f"[green]✓[/green] Pinned {count} files from: {d}")
    
    # Remove files
    if remove_files:
        pinned_map = tenets_instance.config.custom.get("pinned_files", {})
        if sess_name in pinned_map:
            for f in remove_files:
                resolved = str(Path(f).resolve())
                if resolved in pinned_map[sess_name]:
                    pinned_map[sess_name].remove(resolved)
                    removed += 1
                    if not quiet:
                        console.print(f"[yellow]✗[/yellow] Unpinned: {f}")
    
    if added > 0 or removed > 0:
        summary = []
        if added > 0:
            summary.append(f"[green]{added} pinned[/green]")
        if removed > 0:
            summary.append(f"[yellow]{removed} unpinned[/yellow]")
        console.print(f"Files updated: {', '.join(summary)}")


def _dry_run_instillation(
    tenets_instance,
    session: str,
    frequency: str
) -> None:
    """Show what would be instilled without actually doing it."""
    pending = tenets_instance.get_pending_tenets(session=session)
    
    if not pending:
        console.print("[yellow]No tenets would be instilled (none pending).[/yellow]")
        return
    
    console.print("[bold]Would instill the following tenets:[/bold]\n")
    
    for i, tenet in enumerate(pending[:tenets_instance.config.tenet.max_per_context], 1):
        priority_color = {
            "critical": "red",
            "high": "yellow",
            "medium": "blue",
            "low": "dim",
        }.get(tenet.priority.value, "white")
        
        console.print(
            f"{i}. [[{priority_color}]{tenet.priority.value.upper()}[/{priority_color}]] "
            f"{tenet.content}"
        )
        
        if tenet.category:
            console.print(f"   Category: {tenet.category.value}")
        console.print(f"   Added: {tenet.created_at.strftime('%Y-%m-%d %H:%M')}")
        console.print(f"   Previous injections: {tenet.metrics.injection_count}")
        console.print()
    
    console.print(f"\n[dim]Frequency: {frequency}")
    console.print(f"Total: {len(pending[:tenets_instance.config.tenet.max_per_context])} tenet(s)[/dim]")


def _export_history(
    instiller,
    output_path: Path,
    format: str,
    session: Optional[str]
) -> None:
    """Export injection history."""
    try:
        instiller.export_instillation_history(output_path, format=format, session=session)
        console.print(f"[green]✓[/green] Exported history to: {output_path}")
    except Exception as e:
        console.print(f"[red]Error exporting history:[/red] {str(e)}")
        raise typer.Exit(1)


def _show_instillation_result(result, verbose: bool) -> None:
    """Show the result of an instillation."""
    if result.skip_reason:
        console.print(
            Panel(
                f"[yellow]Injection skipped[/yellow]\n"
                f"Reason: {result.skip_reason}",
                title="⏭️ Skipped",
                border_style="yellow",
            )
        )
    else:
        info = [
            f"[green]✓[/green] Successfully instilled {len(result.tenets_instilled)} tenet(s)",
            f"Session: {result.session or 'global'}",
            f"Strategy: {result.strategy_used}",
            f"Token increase: {result.token_increase}",
            f"Complexity: {result.complexity_score:.2f}",
        ]
        
        if verbose and result.tenets_instilled:
            info.append("\nTenets instilled:")
            for tenet in result.tenets_instilled:
                info.append(f"  • {tenet.content[:50]}...")
        
        console.print(
            Panel(
                "\n".join(info),
                title="🌟 Tenets Instilled",
                border_style="green",
            )
        )