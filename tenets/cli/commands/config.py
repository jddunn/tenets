"""Configuration management commands."""

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

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

# File ranking configuration
ranking:
  algorithm: balanced        # fast, balanced, thorough, ml, custom
  threshold: 0.10            # 0.0–1.0 (lower includes more files)
  use_stopwords: false      # Filter programming stopwords
  use_embeddings: false     # Use ML embeddings (requires tenets[ml])

# Content summarization configuration
summarizer:
  default_mode: auto        # extractive, compressive, textrank, transformer, llm, auto
  target_ratio: 0.3         # Compress to 30% of original
  enable_cache: true        # Cache summaries
  preserve_code_structure: true  # Keep imports/signatures
  
  # LLM configuration (optional, costs $)
  # llm_provider: openai    # openai, anthropic, openrouter
  # llm_model: gpt-3.5-turbo
  # llm_temperature: 0.3
  
  # ML configuration
  enable_ml_strategies: true  # Enable transformer models
  quality_threshold: medium   # low, medium, high

# File scanning configuration
scanner:
  respect_gitignore: true
  follow_symlinks: false
  max_file_size: 5000000
  additional_ignore_patterns:
    - "*.generated.*"
    - vendor/

# Output formatting
output:
  default_format: markdown   # markdown, xml, json
  compression_threshold: 10000  # Summarize files larger than this
  summary_ratio: 0.25          # Target compression for large files

# Caching configuration
cache:
  enabled: true
  ttl_days: 7
  max_size_mb: 500
  # directory: ~/.tenets/cache

# Git integration
git:
  enabled: true
  include_history: true
  history_limit: 100

# Tenet system (guiding principles)
tenet:
  auto_instill: true        # Auto-apply tenets to context
  max_per_context: 5        # Max tenets per context
  reinforcement: true       # Reinforce critical tenets
"""

    config_file.write_text(starter_config)
    console.print(f"[green]✓[/green] Created {config_file}")

    console.print("\nNext steps:")
    console.print("1. Edit .tenets.yml to customize for your project")
    console.print("2. Run 'tenets config show' to verify settings")
    console.print("3. Lower ranking.threshold to include more files if needed")
    console.print("4. Configure summarization for large codebases")


@config_app.command("show")
def config_show(
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Specific key to show"),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format: yaml, json"),
):
    """Show current configuration.

    Examples:
        tenets config show
        tenets config show --key summarizer
        tenets config show --key ranking.algorithm
        tenets config show --format json
    """
    try:
        config = TenetsConfig()

        if key == "models":
            # Special case: show model information
            _show_model_info()
            return
        elif key == "summarizers":
            # Show summarization strategies
            _show_summarizer_info()
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
    key: str = typer.Argument(..., help="Configuration key (e.g., summarizer.target_ratio)"),
    value: str = typer.Argument(..., help="Value to set"),
    save: bool = typer.Option(False, "--save", "-s", help="Save to config file"),
):
    """Set a configuration value.

    Examples:
        tenets config set max_tokens 150000
        tenets config set ranking.algorithm thorough
        tenets config set summarizer.default_mode extractive --save
        tenets config set summarizer.llm_model gpt-4 --save
    """
    try:
        # Load current config
        config = TenetsConfig()

        # Parse the key path
        parts = key.split(".")

        # Navigate to the parent object
        obj = config
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                console.print(f"[red]Invalid configuration key: {key}[/red]")
                raise typer.Exit(1)

        # Set the value
        attr_name = parts[-1]
        if hasattr(obj, attr_name):
            # Parse value type based on existing type
            current_value = getattr(obj, attr_name)
            if isinstance(current_value, bool):
                parsed_value = value.lower() in ["true", "yes", "1"]
            elif isinstance(current_value, int):
                parsed_value = int(value)
            elif isinstance(current_value, float):
                parsed_value = float(value)
            elif isinstance(current_value, list):
                parsed_value = value.split(",")
            else:
                parsed_value = value

            setattr(obj, attr_name, parsed_value)
            console.print(f"[green]✓[/green] Set {key} = {parsed_value}")

            # Save if requested
            if save:
                config_file = config.config_file or Path(".tenets.yml")
                config.save(config_file)
                console.print(f"[green]✓[/green] Saved to {config_file}")
        else:
            console.print(f"[red]Invalid configuration attribute: {attr_name}[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error setting configuration:[/red] {str(e)}")
        raise typer.Exit(1)


@config_app.command("validate")
def config_validate(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Config file to validate"),
):
    """Validate configuration file.

    Examples:
        tenets config validate
        tenets config validate --file custom-config.yml
    """
    try:
        if file:
            config = TenetsConfig(config_file=file)
            console.print(f"[green]✓[/green] Configuration file {file} is valid")
        else:
            config = TenetsConfig()
            if config.config_file:
                console.print(f"[green]✓[/green] Configuration file {config.config_file} is valid")
            else:
                console.print("[green]✓[/green] Using default configuration (no config file)")

        # Show key settings
        table = Table(title="Key Configuration Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Max Tokens", str(config.max_tokens))
        table.add_row("Ranking Algorithm", config.ranking.algorithm)
        table.add_row("Ranking Threshold", f"{config.ranking.threshold:.2f}")
        table.add_row("Summarizer Mode", config.summarizer.default_mode)
        table.add_row("Summarizer Ratio", f"{config.summarizer.target_ratio:.2f}")
        table.add_row("Cache Enabled", str(config.cache.enabled))
        table.add_row("Git Enabled", str(config.git.enabled))
        table.add_row("Auto-instill Tenets", str(config.tenet.auto_instill))

        console.print(table)

    except Exception as e:
        console.print(f"[red]✗[/red] Configuration validation failed: {str(e)}")
        raise typer.Exit(1)


@config_app.command("clear-cache")
def config_clear_cache(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """Wipe all Tenets caches (analysis + general + summaries)."""
    if not confirm:
        typer.confirm("This will delete all cached analysis and summaries. Continue?", abort=True)
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
    """Show detailed cache statistics."""
    cfg = TenetsConfig()
    cache_dir = Path(cfg.cache.directory or (Path.home() / ".tenets" / "cache"))
    if not cache_dir.exists():
        console.print("[dim]Cache directory does not exist.[/dim]")
        return

    # Gather statistics
    total_size = 0
    file_count = 0
    cache_types = {"analysis": 0, "summary": 0, "other": 0}

    for p in cache_dir.rglob("*"):
        if p.is_file():
            file_count += 1
            try:
                size = p.stat().st_size
                total_size += size

                # Categorize cache files
                if "analysis" in str(p):
                    cache_types["analysis"] += size
                elif "summary" in str(p) or "summarize" in str(p):
                    cache_types["summary"] += size
                else:
                    cache_types["other"] += size
            except Exception:
                pass

    mb = total_size / (1024 * 1024)

    # Create statistics table
    table = Table(title="Cache Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Cache Path", str(cache_dir))
    table.add_row("Total Files", str(file_count))
    table.add_row("Total Size", f"{mb:.2f} MB")
    table.add_row("Analysis Cache", f"{cache_types['analysis'] / (1024 * 1024):.2f} MB")
    table.add_row("Summary Cache", f"{cache_types['summary'] / (1024 * 1024):.2f} MB")
    table.add_row("Other Cache", f"{cache_types['other'] / (1024 * 1024):.2f} MB")
    table.add_row("TTL Days", str(cfg.cache.ttl_days))
    table.add_row("Max Size MB", str(cfg.cache.max_size_mb))

    console.print(table)


def _show_model_info():
    """Display information about supported LLM models."""
    from rich.table import Table

    table = Table(title="Supported LLM Models for Distillation")
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


def _show_summarizer_info():
    """Display information about summarization strategies."""
    from rich.table import Table

    table = Table(title="Summarization Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Speed", style="green")
    table.add_column("Quality", style="yellow")
    table.add_column("Requirements", style="red")

    strategies = [
        {
            "name": "extractive",
            "description": "Selects important sentences",
            "speed": "Fast",
            "quality": "Good",
            "requirements": "None",
        },
        {
            "name": "compressive",
            "description": "Removes redundant content",
            "speed": "Fast",
            "quality": "Good",
            "requirements": "None",
        },
        {
            "name": "textrank",
            "description": "Graph-based ranking",
            "speed": "Medium",
            "quality": "Very Good",
            "requirements": "None",
        },
        {
            "name": "transformer",
            "description": "Neural summarization",
            "speed": "Slow",
            "quality": "Excellent",
            "requirements": "pip install tenets[ml]",
        },
        {
            "name": "llm",
            "description": "LLM-based (GPT, Claude)",
            "speed": "Slow",
            "quality": "Best",
            "requirements": "API key + costs $",
        },
        {
            "name": "auto",
            "description": "Auto-selects best strategy",
            "speed": "Varies",
            "quality": "Adaptive",
            "requirements": "None",
        },
    ]

    for strategy in strategies:
        table.add_row(
            strategy["name"],
            strategy["description"],
            strategy["speed"],
            strategy["quality"],
            strategy["requirements"],
        )

    console.print(table)

    console.print("\n[bold]Configuration:[/bold]")
    console.print("Set in .tenets.yml under 'summarizer' section:")
    console.print("  default_mode: auto")
    console.print("  target_ratio: 0.3")
    console.print("  llm_provider: openai")
    console.print("  llm_model: gpt-3.5-turbo")

    console.print("\n[bold]Environment Variables:[/bold]")
    console.print("  TENETS_SUMMARIZER_DEFAULT_MODE=extractive")
    console.print("  TENETS_SUMMARIZER_TARGET_RATIO=0.25")
    console.print("  OPENAI_API_KEY=sk-...")
    console.print("  ANTHROPIC_API_KEY=sk-ant-...")


@config_app.command("export")
def config_export(
    output: Path = typer.Argument(..., help="Output file path"),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format: yaml, json"),
):
    """Export current configuration to file.

    Examples:
        tenets config export my-config.yml
        tenets config export config.json --format json
    """
    try:
        config = TenetsConfig()

        # Ensure correct extension
        if format == "json" and not output.suffix == ".json":
            output = output.with_suffix(".json")
        elif format == "yaml" and output.suffix not in [".yml", ".yaml"]:
            output = output.with_suffix(".yml")

        config.save(output)
        console.print(f"[green]✓[/green] Configuration exported to {output}")

    except Exception as e:
        console.print(f"[red]Error exporting configuration:[/red] {str(e)}")
        raise typer.Exit(1)


@config_app.command("diff")
def config_diff(
    file1: Optional[Path] = typer.Option(None, "--file1", help="First config file"),
    file2: Optional[Path] = typer.Option(None, "--file2", help="Second config file"),
):
    """Show differences between configurations.

    Examples:
        tenets config diff  # Compare current vs defaults
        tenets config diff --file1 old.yml --file2 new.yml
    """
    try:
        # Load configurations
        if file1:
            config1 = TenetsConfig(config_file=file1)
            label1 = str(file1)
        else:
            config1 = TenetsConfig()
            label1 = "Current"

        if file2:
            config2 = TenetsConfig(config_file=file2)
            label2 = str(file2)
        else:
            # Create default config for comparison
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
                # Empty file gives defaults
                f.write("")
                temp_path = Path(f.name)
            config2 = TenetsConfig(config_file=temp_path)
            temp_path.unlink()
            label2 = "Defaults"

        # Get dictionaries
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()

        # Find differences
        differences = _find_differences(dict1, dict2)

        if not differences:
            console.print(f"[green]No differences between {label1} and {label2}[/green]")
        else:
            table = Table(title=f"Configuration Differences: {label1} vs {label2}")
            table.add_column("Key", style="cyan")
            table.add_column(label1, style="yellow")
            table.add_column(label2, style="green")

            for key, val1, val2 in differences:
                table.add_row(key, str(val1), str(val2))

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error comparing configurations:[/red] {str(e)}")
        raise typer.Exit(1)


def _find_differences(dict1: dict, dict2: dict, prefix: str = "") -> list:
    """Find differences between two dictionaries recursively."""
    differences = []

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        val1 = dict1.get(key)
        val2 = dict2.get(key)

        if isinstance(val1, dict) and isinstance(val2, dict):
            # Recurse into nested dicts
            differences.extend(_find_differences(val1, val2, full_key))
        elif val1 != val2:
            differences.append((full_key, val1, val2))

    return differences
