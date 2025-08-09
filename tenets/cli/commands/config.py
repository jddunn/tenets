"""Configuration management commands."""

from pathlib import Path
import json
import yaml
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

from tenets.config import TenetsConfig
from tenets.models.llm import SUPPORTED_MODELS
from tenets.storage.cache import CacheManager

console = Console()

# Create config subcommand app
config_app = typer.Typer(help="Configuration management", no_args_is_help=True)


@config_app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
):
    """Create a starter .tenets.yml configuration file.

    Examples:
        tenets config init
        tenets config init --force
    """
    config_file = Path(".tenets.yml")

    if config_file.exists() and not force:
        console.print(f"[yellow]Config file {config_file} already exists.[/yellow]")
        console.print("Use --force to overwrite.")
        raise typer.Exit(1)

    # Starter config template (aligned with TenetsConfig schema)
    starter_config = """# .tenets.yml - Tenets configuration
# https://github.com/jddunn/tenets

max_tokens: 100000

ranking:
  algorithm: balanced        # fast, balanced, thorough, ml, custom
  threshold: 0.10            # 0.0–1.0 (lower includes more files)

scanner:
  respect_gitignore: true
  follow_symlinks: false
  max_file_size: 5000000
  additional_ignore_patterns:
    - "*.generated.*"
    - vendor/

output:
  default_format: markdown   # markdown, xml, json

cache:
  enabled: true
  ttl_days: 7
  max_size_mb: 500
  # directory: ~/.tenets/cache

git:
  enabled: true

# Tenet system
tenet:
  auto_instill: true
  max_per_context: 5
  reinforcement: true
"""

    config_file.write_text(starter_config)
    console.print(f"[green]✓[/green] Created {config_file}")

    console.print("\nNext steps:")
    console.print("1. Edit .tenets.yml to customize for your project")
    console.print("2. Run 'tenets config show' to verify settings")
    console.print("3. Lower ranking.threshold to include more files if needed")


@config_app.command("show")
def config_show(
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Specific key to show"),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format: yaml, json"),
):
    """Show current configuration.

    Examples:
        tenets config show
        tenets config show --key context.max_tokens
        tenets config show --format json
    """
    try:
        config = TenetsConfig()

        if key == "models":
            # Special case: show model information
            _show_model_info()
            return

        config_dict = config.to_dict()

        if key:
            # Navigate to specific key
            parts = key.split(".")
            value = config_dict
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    console.print(f"[red]Key not found: {key}[/red]")
                    raise typer.Exit(1)

            # Display the value
            if isinstance(value, (dict, list)):
                if format == "json":
                    console.print_json(data=value)
                else:
                    console.print(yaml.dump({key: value}, default_flow_style=False))
            else:
                console.print(f"{key}: {value}")
        else:
            # Show full config
            if format == "json":
                console.print_json(data=config_dict)
            else:
                yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
                syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=False)
                console.print(syntax)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key (e.g., context.max_tokens)"),
    value: str = typer.Argument(..., help="Value to set"),
):
    """Set a configuration value.

    Examples:
        tenets config set context.max_tokens 150000
        tenets config set context.ranking thorough
        tenets config set tenets.auto_instill false
    """
    console.print(f"[yellow]Config set command - coming soon![/yellow]")
    console.print(f"Would set: {key} = {value}")


@config_app.command("clear-cache")
def config_clear_cache(confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")):
    """Wipe all Tenets caches (analysis + general)."""
    if not confirm:
        typer.confirm("This will delete all cached analysis. Continue?", abort=True)
    cfg = TenetsConfig()
    mgr = CacheManager(cfg)
    mgr.clear_all()
    console.print("[red]Cache cleared.[/red]")


@config_app.command("cleanup-cache")
def config_cleanup_cache():
    """Cleanup old / oversized cache entries respecting TTL and size policies."""
    cfg = TenetsConfig()
    mgr = CacheManager(cfg)
    stats = mgr.analysis.disk.cleanup(
        max_age_days=cfg.cache.ttl_days, max_size_mb=cfg.cache.max_size_mb // 2
    )
    stats_general = mgr.general.cleanup(
        max_age_days=cfg.cache.ttl_days, max_size_mb=cfg.cache.max_size_mb // 2
    )
    console.print(
        Panel(
            f"Analysis deletions: {stats}\nGeneral deletions: {stats_general}",
            title="Cache Cleanup",
            border_style="yellow",
        )
    )


@config_app.command("cache-stats")
def config_cache_stats():
    """Show basic cache directory info."""
    cfg = TenetsConfig()
    cache_dir = Path(cfg.cache.directory or (Path.home() / ".tenets" / "cache"))
    if not cache_dir.exists():
        console.print("[dim]Cache directory does not exist.[/dim]")
        return
    total_size = 0
    file_count = 0
    for p in cache_dir.rglob("*"):
        if p.is_file():
            file_count += 1
            try:
                total_size += p.stat().st_size
            except Exception:
                pass
    mb = total_size / (1024 * 1024)
    console.print(
        Panel(
            f"Path: {cache_dir}\nFiles: {file_count}\nSize: {mb:.2f} MB\nTTL days: {cfg.cache.ttl_days}\nMax size MB: {cfg.cache.max_size_mb}",
            title="Cache Stats",
            border_style="cyan",
        )
    )


def _show_model_info():
    """Display information about supported models."""
    from rich.table import Table

    table = Table(title="Supported LLM Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="blue")
    table.add_column("Context", style="green", justify="right")
    table.add_column("Input $/1K", style="yellow", justify="right")
    table.add_column("Output $/1K", style="red", justify="right")

    for model in SUPPORTED_MODELS:
        context_k = model["context_tokens"] // 1000
        context_str = f"{context_k}K" if context_k < 1000 else f"{context_k // 1000}M"

        table.add_row(
            model["name"],
            model["provider"],
            context_str,
            f"${model['input_price']:.5f}",
            f"${model['output_price']:.5f}",
        )

    console.print(table)

    console.print("\n[dim]Use --model flag with distill command to target specific models.[/dim]")
