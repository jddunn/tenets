"""Distill command - extract relevant context from codebase."""

import json
from pathlib import Path
from typing import Optional, List

import typer
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from tenets import Tenets
from tenets.models.llm import get_model_pricing

console = Console()


def distill(
    prompt: str = typer.Argument(
        ..., 
        help="Your query or task (can be text or URL to GitHub issue, etc.)"
    ),
    path: Path = typer.Argument(
        Path("."), 
        help="Path to analyze (directory or files)"
    ),
    # Output options
    format: str = typer.Option(
        "markdown", 
        "--format", "-f",
        help="Output format: markdown, xml, json"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save output to file instead of stdout"
    ),
    # Analysis options
    mode: str = typer.Option(
        "balanced",
        "--mode", "-m",
        help="Analysis mode: fast (keywords only), balanced (default), thorough (deep analysis)"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Target LLM model for token counting (e.g., gpt-4o, claude-3-opus)"
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens for context (overrides model default)"
    ),
    # Filtering
    include: Optional[str] = typer.Option(
        None,
        "--include", "-i",
        help="Include file patterns (e.g., '*.py,*.js')"
    ),
    exclude: Optional[str] = typer.Option(
        None,
        "--exclude", "-e", 
        help="Exclude file patterns (e.g., 'test_*,*.backup')"
    ),
    # Features
    no_git: bool = typer.Option(
        False,
        "--no-git",
        help="Disable git context inclusion"
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session", "-s",
        help="Use session for stateful context building"
    ),
    # Info options
    estimate_cost: bool = typer.Option(
        False,
        "--estimate-cost",
        help="Show token usage and cost estimate"
    ),
    show_stats: bool = typer.Option(
        False,
        "--stats",
        help="Show statistics about context generation"
    ),
    # Context options
    ctx: typer.Context = typer.Context,
):
    """
    Distill relevant context from your codebase for any prompt.
    
    This command extracts and aggregates the most relevant files, documentation,
    and git history based on your query, optimizing for LLM token limits.
    
    Examples:
    
        # Basic usage
        tenets distill "implement OAuth2 authentication"
        
        # From a GitHub issue
        tenets distill https://github.com/org/repo/issues/123
        
        # Specific path with options
        tenets distill "add caching layer" ./src --mode thorough --max-tokens 50000
        
        # Filter by file types
        tenets distill "review API" --include "*.py,*.yaml" --exclude "test_*"
        
        # Save to file with cost estimate
        tenets distill "debug login" -o context.md --model gpt-4o --estimate-cost
    """
    # Get verbosity from context
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)
    
    try:
        # Initialize tenets
        tenets = Tenets()
        
        # Parse include/exclude patterns
        include_patterns = include.split(",") if include else None
        exclude_patterns = exclude.split(",") if exclude else None
        
        # Show progress unless quiet
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(f"Distilling context for: {prompt[:50]}...", total=None)
                
                # Distill context
                result = tenets.distill(
                    prompt=prompt,
                    files=path,
                    format=format,
                    model=model,
                    max_tokens=max_tokens,
                    mode=mode,
                    include_git=not no_git,
                    session_name=session,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                )
        else:
            # No progress bar in quiet mode
            result = tenets.distill(
                prompt=prompt,
                files=path,
                format=format,
                model=model,
                max_tokens=max_tokens,
                mode=mode,
                include_git=not no_git,
                session_name=session,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
        
        # Format output
        if format == "json":
            output_text = json.dumps(result.to_dict(), indent=2)
        else:
            output_text = result.context
        
        # Show cost estimation if requested
        if estimate_cost and model:
            cost_info = tenets.estimate_cost(result, model)
            
            if not quiet:
                console.print(Panel(
                    f"[bold]Token Usage[/bold]\n"
                    f"Context tokens: {cost_info['input_tokens']:,}\n"
                    f"Est. response: {cost_info['output_tokens']:,}\n"
                    f"Total tokens: {cost_info['input_tokens'] + cost_info['output_tokens']:,}\n\n"
                    f"[bold]Cost Estimate[/bold]\n"
                    f"Context cost: ${cost_info['input_cost']:.4f}\n"
                    f"Response cost: ${cost_info['output_cost']:.4f}\n"
                    f"Total cost: ${cost_info['total_cost']:.4f}",
                    title=f"💰 Cost Estimate for {model}",
                    border_style="yellow"
                ))
        
        # Show statistics if requested
        if show_stats and not quiet:
            metadata = result.metadata
            console.print(Panel(
                f"[bold]Distillation Statistics[/bold]\n"
                f"Mode: {metadata.get('mode', 'unknown')}\n"
                f"Files found: {metadata.get('files_analyzed', 0)}\n"
                f"Files included: {metadata.get('files_included', 0)}\n"
                f"Token usage: {result.token_count:,} / {max_tokens or 'model default'}\n"
                f"Analysis time: {metadata.get('analysis_time', '?')}s",
                title="📊 Statistics",
                border_style="blue"
            ))
        
        # Output result
        if output:
            output.write_text(output_text, encoding='utf-8')
            if not quiet:
                console.print(f"[green]✓[/green] Context saved to {output}")
        else:
            if format == "json":
                console.print_json(output_text)
            else:
                print(output_text)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)