"""Configuration management commands."""

from pathlib import Path
import json
import yaml
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.syntax import Syntax

from tenets.config import TenetsConfig
from tenets.models.llm import SUPPORTED_MODELS

console = Console()

# Create config subcommand app
config_app = typer.Typer(
    help="Configuration management",
    no_args_is_help=True
)


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
    
    # Starter config template
    starter_config = """# .tenets.yml - Tenets configuration
# https://github.com/jddunn/tenets

# Context generation settings
context:
  max_tokens: 100000        # Maximum tokens for generated context
  ranking: balanced         # Ranking algorithm: fast, balanced, thorough
  include_git: true         # Include git context by default
  summarize_long_files: true

# Tenet system settings
tenets:
  auto_instill: true        # Automatically apply tenets to context
  max_per_context: 5        # Maximum tenets to inject per context
  reinforcement: true       # Reinforce critical principles

# File scanning settings
scanner:
  respect_gitignore: true   # Respect .gitignore files
  follow_symlinks: false    # Follow symbolic links
  max_file_size: 5000000    # Max file size in bytes (5MB)

# Patterns to ignore
ignore:
  # Version control
  - .git/
  - .svn/
  
  # Dependencies
  - node_modules/
  - vendor/
  - venv/
  - .venv/
  
  # Build outputs
  - build/
  - dist/
  - "*.egg-info"
  
  # IDE
  - .idea/
  - .vscode/
  
  # Generated files
  - "*.generated.*"
  - "*.min.js"
  - "*.min.css"

# Include patterns (if specified, only these are included)
# include:
#   - "*.py"
#   - "*.js"
#   - "*.ts"

# Output preferences
output:
  format: markdown          # Default format: markdown, xml, json
  
# Cache settings
cache:
  enabled: true
  ttl_days: 7              # Cache time-to-live in days
  max_size_mb: 500         # Maximum cache size
"""
    
    config_file.write_text(starter_config)
    console.print(f"[green]✓[/green] Created {config_file}")
    
    console.print("\nNext steps:")
    console.print("1. Edit .tenets.yml to customize for your project")
    console.print("2. Run 'tenets examine' to test your configuration")
    console.print("3. Add project-specific ignore patterns as needed")


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
            f"${model['output_price']:.5f}"
        )
    
    console.print(table)
    
    console.print("\n[dim]Use --model flag with distill command to target specific models.[/dim]")