"""Distill command - extract relevant context from codebase."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from tenets import Tenets

console = Console()

# Expose a module-level pyperclip symbol so tests can patch it even if it's not installed
try:  # pragma: no cover - optional dependency presence varies by env
    import pyperclip as _pyperclip  # type: ignore

    pyperclip = _pyperclip
except Exception:  # pragma: no cover
    pyperclip = None  # type: ignore


def distill(
    prompt: str = typer.Argument(
        ..., help="Your query or task (can be text or URL to GitHub issue, etc.)"
    ),
    path: Path = typer.Argument(Path(), help="Path to analyze (directory or files)"),
    # Output options
    format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown, xml, json"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save output to file instead of stdout"
    ),
    # Analysis options
    mode: str = typer.Option(
        "balanced",
        "--mode",
        "-m",
        help="Analysis mode: fast (keywords only), balanced (default), thorough (deep analysis)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Target LLM model for token counting (e.g., gpt-4o, claude-3-opus)"
    ),
    max_tokens: Optional[int] = typer.Option(
        None, "--max-tokens", help="Maximum tokens for context (overrides model default)"
    ),
    # Filtering
    include: Optional[str] = typer.Option(
        None, "--include", "-i", help="Include file patterns (e.g., '*.py,*.js')"
    ),
    exclude: Optional[str] = typer.Option(
        None, "--exclude", "-e", help="Exclude file patterns (e.g., 'test_*,*.backup')"
    ),
    # Features
    no_git: bool = typer.Option(False, "--no-git", help="Disable git context inclusion"),
    use_stopwords: bool = typer.Option(
        False, "--use-stopwords", help="Enable stopword filtering for keyword analysis"
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Include full content for all ranked files within token budget (no summarization)",
    ),
    condense: bool = typer.Option(
        False,
        "--condense",
        help="Condense whitespace (collapse large blank runs, trim trailing spaces) before counting tokens",
    ),
    remove_comments: bool = typer.Option(
        False,
        "--remove-comments",
        help="Strip comments (heuristic, language-aware) before counting tokens",
    ),
    session: Optional[str] = typer.Option(
        None, "--session", "-s", help="Use session for stateful context building"
    ),
    # Info options
    estimate_cost: bool = typer.Option(
        False, "--estimate-cost", help="Show token usage and cost estimate"
    ),
    show_stats: bool = typer.Option(
        False, "--stats", help="Show statistics about context generation"
    ),
    copy: bool = typer.Option(
        False,
        "--copy",
        help="Copy distilled context to clipboard (also enabled automatically if config.output.copy_on_distill)",
    ),
    # Context options
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
    ctx_obj_local = {}
    try:
        _ctx = click.get_current_context(silent=True)
        if _ctx and _ctx.obj:
            ctx_obj_local = _ctx.obj
    except Exception:
        ctx_obj_local = {}
    state = ctx_obj_local or {}
    verbose = state.get("verbose", False)
    quiet = state.get("quiet", False)

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
                    use_stopwords=use_stopwords,
                    session_name=session,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                    full=full,
                    condense=condense,
                    remove_comments=remove_comments,
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
                use_stopwords=use_stopwords,
                session_name=session,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                full=full,
                condense=condense,
                remove_comments=remove_comments,
            )

        # Prepare metadata and interactivity flags
        raw_meta = getattr(result, "metadata", {})
        metadata = raw_meta if isinstance(raw_meta, dict) else {}
        files_included = metadata.get("files_included", 0)
        files_analyzed = metadata.get("files_analyzed", 0)
        token_count = getattr(result, "token_count", 0)
        try:
            token_count = int(token_count)
        except Exception:
            token_count = 0
        interactive = (output is None) and (not quiet) and sys.stdout.isatty()

        # Format output
        if format == "json":
            output_text = json.dumps(result.to_dict(), indent=2)
        else:
            output_text = result.context

        # Build summary details
        include_display = ",".join(include_patterns) if include_patterns else "(none)"
        exclude_display = ",".join(exclude_patterns) if exclude_patterns else "(none)"
        git_display = "disabled" if no_git else "enabled (ranking only)"
        session_display = session or "(none)"
        max_tokens_display = str(max_tokens) if max_tokens else "model default"

        # Show a concise summary before content in interactive mode
        if interactive:
            console.print(
                Panel(
                    f"[bold]Prompt[/bold]: {str(prompt)[:80]}\n"
                    f"Path: {path!s}\n"
                    f"Mode: {metadata.get('mode', 'unknown')}  •  Format: {format}\n"
                    f"Full: {metadata.get('full_mode', full)}  •  Condense: {metadata.get('condense', condense)}  •  Remove Comments: {metadata.get('remove_comments', remove_comments)}\n"
                    f"Files: {files_included}/{files_analyzed}  •  Tokens: {token_count:,} / {max_tokens_display}\n"
                    f"Include: {include_display}\n"
                    f"Exclude: {exclude_display}\n"
                    f"Git: {git_display}  •  Session: {session_display}",
                    title="Tenets Context",
                    border_style="green",
                )
            )

        # Output result
        if output:
            output.write_text(output_text, encoding="utf-8")
            if not quiet:
                console.print(f"[green]✓[/green] Context saved to {output}")
        elif format == "json":
            # Emit pure JSON without Rich formatting to keep stdout clean for parsers/tests
            print(output_text)
        else:
            # Draw clear context boundaries in interactive TTY only
            if interactive:
                console.rule("Context")
            print(output_text)
            if interactive:
                console.rule("End")

        # Clipboard copy (after output so piping still works)
        do_copy = copy
        try:
            # Check config flag (best-effort; Tenets() instance may expose config)
            cfg = getattr(tenets, "config", None)
            if cfg and getattr(getattr(cfg, "output", None), "copy_on_distill", False):
                do_copy = True or copy
        except Exception:
            pass
        if do_copy:
            copied = False
            text_to_copy = (
                output_text if format != "json" else json.dumps(result.to_dict(), indent=2)
            )
            # Try pyperclip first
            try:  # pragma: no cover - environment dependent
                if pyperclip is not None:
                    pyperclip.copy(text_to_copy)  # type: ignore[attr-defined]
                    copied = True
                else:
                    raise RuntimeError("no pyperclip")
            except Exception:
                # Fallbacks by platform
                try:
                    import platform
                    import shutil
                    import subprocess

                    plat = platform.system().lower()
                    if "windows" in plat:
                        # Use clip
                        p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, close_fds=True)
                        p.communicate(input=text_to_copy.encode("utf-8"))
                        copied = p.returncode == 0
                    elif "darwin" in plat and shutil.which("pbcopy"):
                        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                        p.communicate(input=text_to_copy.encode("utf-8"))
                        copied = p.returncode == 0
                    elif shutil.which("xclip"):
                        p = subprocess.Popen(
                            ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE
                        )
                        p.communicate(input=text_to_copy.encode("utf-8"))
                        copied = p.returncode == 0
                    elif shutil.which("wl-copy"):
                        p = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
                        p.communicate(input=text_to_copy.encode("utf-8"))
                        copied = p.returncode == 0
                except Exception:
                    copied = False
            if copied and not quiet:
                console.print("[cyan]📋 Context copied to clipboard[/cyan]")
            elif not copied and do_copy and not quiet:
                console.print(
                    "[yellow]Warning:[/yellow] Unable to copy to clipboard (missing pyperclip/xclip/pbcopy)."
                )

        # Show cost estimation if requested
        if estimate_cost and model:
            cost_info = tenets.estimate_cost(result, model)

            if not quiet:
                console.print(
                    Panel(
                        f"[bold]Token Usage[/bold]\n"
                        f"Context tokens: {cost_info['input_tokens']:,}\n"
                        f"Est. response: {cost_info['output_tokens']:,}\n"
                        f"Total tokens: {cost_info['input_tokens'] + cost_info['output_tokens']:,}\n\n"
                        f"[bold]Cost Estimate[/bold]\n"
                        f"Context cost: ${cost_info['input_cost']:.4f}\n"
                        f"Response cost: ${cost_info['output_cost']:.4f}\n"
                        f"Total cost: ${cost_info['total_cost']:.4f}",
                        title=f"💰 Cost Estimate for {model}",
                        border_style="yellow",
                    )
                )

        # If no files included, provide actionable suggestions. Avoid contaminating JSON stdout.
        if files_included == 0 and format != "json" and output is None:
            if interactive:
                console.print(
                    Panel(
                        "No files were included in the context.\n\n"
                        "Try: \n"
                        "• Increase --max-tokens\n"
                        "• Relax filters: remove or adjust --include/--exclude\n"
                        "• Use --mode thorough for deeper analysis\n"
                        "• Run with --verbose to see why files were skipped\n"
                        "• Add --stats to view generation metrics",
                        title="Suggestions",
                        border_style="red",
                    )
                )
            else:
                # Plain output for non-interactive (piped) environments
                print("No files were included in the context.")
                print("Suggestions")
                print("- Increase --max-tokens")
                print("- Relax filters: remove or adjust --include/--exclude")
                print("- Use --mode thorough for deeper analysis")
                print("- Run with --verbose to see why files were skipped")
                print("- Add --stats to view generation metrics")

        # Show statistics if requested
        if show_stats and not quiet:
            console.print(
                Panel(
                    f"[bold]Distillation Statistics[/bold]\n"
                    f"Mode: {metadata.get('mode', 'unknown')}\n"
                    f"Files found: {files_analyzed}\n"
                    f"Files included: {files_included}\n"
                    f"Token usage: {token_count:,} / {max_tokens or 'model default'}\n"
                    f"Analysis time: {metadata.get('analysis_time', '?')}s",
                    title="📊 Statistics",
                    border_style="blue",
                )
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e!s}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
