"""Distill command - extract relevant context from codebase."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
import typer
from rich import print
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from tenets import Tenets
from tenets.utils.timing import CommandTimer

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
        "markdown", "--format", "-f", help="Output format: markdown, xml, json, html"
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
    include_tests: bool = typer.Option(
        False, "--include-tests", help="Include test files (overrides default exclusion)"
    ),
    exclude_tests: bool = typer.Option(
        False,
        "--exclude-tests",
        help="Explicitly exclude test files (even for test-related prompts)",
    ),
    include_minified: bool = typer.Option(
        False,
        "--include-minified",
        help="Include minified/built files (*.min.js, dist/, etc.) normally excluded",
    ),
    # Features
    no_git: bool = typer.Option(False, "--no-git", help="Disable git context inclusion"),
    ml: bool = typer.Option(
        False, "--ml", help="Enable ML features (embeddings, transformers) for better ranking"
    ),
    reranker: bool = typer.Option(
        False,
        "--reranker",
        help="Enable neural cross-encoder reranking for highest accuracy (requires --ml)",
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
    timeout: Optional[int] = typer.Option(
        None,
        "--timeout",
        "-t",
        help="Timeout in seconds for distill (default: config distill_timeout, <=0 disables)",
    ),
    docstring_weight: Optional[float] = typer.Option(
        None,
        "--docstring-weight",
        min=0.0,
        max=1.0,
        help="Weight for including docstrings in summaries (0=never, 0.5=balanced, 1.0=always)",
    ),
    no_summarize_imports: bool = typer.Option(
        False,
        "--no-summarize-imports",
        help="Disable import summarization (show all imports verbatim)",
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
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed debug information including keyword matching"
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

        # Basic usage (BM25 text similarity)
        tenets distill "implement OAuth2 authentication"

        # With ML embeddings for better semantic matching
        tenets distill "fix authentication bug" --ml

        # With neural reranking for highest accuracy
        tenets distill "optimize database performance" --ml --reranker

        # From a GitHub issue
        tenets distill https://github.com/org/repo/issues/123

        # Specific path with options
        tenets distill "add caching layer" ./src --mode thorough --max-tokens 50000

        # Filter by file types
        tenets distill "review API" --include "*.py,*.yaml" --exclude "test_*"

        # Save to file with cost estimate
        tenets distill "debug login" -o context.md --model gpt-4o --estimate-cost
    """
    # Get verbosity from context (but parameter takes precedence)
    ctx_obj_local = {}
    try:
        _ctx = click.get_current_context(silent=True)
        if _ctx and _ctx.obj:
            ctx_obj_local = _ctx.obj
    except Exception:
        ctx_obj_local = {}
    state = ctx_obj_local or {}
    # Use the verbose parameter directly (it overrides context)
    quiet = state.get("quiet", False)

    # Initialize timer - suppress output in JSON/HTML modes when not outputting to file
    is_json_quiet = format.lower() == "json" and not output
    is_html_quiet = format.lower() == "html" and not output
    timer = CommandTimer(console, quiet or is_json_quiet or is_html_quiet)

    try:
        # Start timing
        timer.start("Initializing tenets...")

        # Initialize tenets
        tenets = Tenets()

        # Override ML settings if specified
        if ml or reranker:
            # Enable ML features in config
            if hasattr(tenets, "config") and hasattr(tenets.config, "ranking"):
                tenets.config.ranking.use_ml = True
                tenets.config.ranking.use_embeddings = True
                if reranker:
                    tenets.config.ranking.use_reranker = True

        # Parse include/exclude patterns
        include_patterns = include.split(",") if include else None
        exclude_patterns = exclude.split(",") if exclude else None

        # Determine timeout precedence: CLI flag > config > disabled
        effective_timeout = timeout
        if effective_timeout is None:
            try:
                effective_timeout = getattr(getattr(tenets, "config", None), "distill_timeout", None)
            except Exception:
                effective_timeout = None
        try:
            if effective_timeout is not None:
                effective_timeout = float(effective_timeout)
        except (TypeError, ValueError):
            effective_timeout = None
        if effective_timeout is not None and effective_timeout <= 0:
            effective_timeout = None

        # Determine test inclusion based on CLI flags
        # Priority: exclude_tests flag > include_tests flag > automatic detection
        test_inclusion = None
        if exclude_tests:
            test_inclusion = False  # Explicitly exclude tests
        elif include_tests:
            test_inclusion = True  # Explicitly include tests
        # If neither flag is set, let the prompt analysis decide (test_inclusion = None)

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
                    full=full,
                    condense=condense,
                    remove_comments=remove_comments,
                    include_tests=test_inclusion,
                    docstring_weight=docstring_weight,
                    summarize_imports=not no_summarize_imports,
                    timeout=effective_timeout,
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
                full=full,
                condense=condense,
                remove_comments=remove_comments,
                include_tests=test_inclusion,
                docstring_weight=docstring_weight,
                summarize_imports=not no_summarize_imports,
                timeout=effective_timeout,
            )

        # Prepare metadata and interactivity flags
        raw_meta = getattr(result, "metadata", {})
        metadata = raw_meta if isinstance(raw_meta, dict) else {}

        # Emit timeout warning on stderr only
        if metadata.get("timed_out"):
            err_console = Console(stderr=True, file=sys.stderr)
            limit_display = metadata.get("timeout_seconds")
            limit_txt = f" (limit: {limit_display}s)" if limit_display else ""
            err_console.print(
                f"[yellow]Timeout:[/yellow] distill exceeded time budget{limit_txt}; returning partial context."
            )

        # Show verbose debug information if requested
        if verbose and not quiet:
            console.print("\n[yellow]═══ Verbose Debug Information ═══[/yellow]")

            # Show parsing details
            if "prompt_context" in metadata:
                pc = metadata["prompt_context"]
                console.print("\n[cyan]Prompt Parsing:[/cyan]")
                console.print(f"  Task Type: {pc.get('task_type', 'unknown')}")
                console.print(f"  Intent: {pc.get('intent', 'unknown')}")
                console.print(f"  Keywords: {pc.get('keywords', [])}")
                console.print(f"  Synonyms: {pc.get('synonyms', [])}")
                console.print(f"  Entities: {pc.get('entities', [])}")

            # Show NLP normalization details
            if "nlp_normalization" in metadata:
                nn = metadata["nlp_normalization"]
                console.print("\n[cyan]NLP Normalization:[/cyan]")
                kw = nn.get("keywords", {})
                console.print(
                    f"  Keywords normalized: {kw.get('original_total', 0)} -> {kw.get('total', 0)}"
                )
                # Print up to 5 examples of normalization steps
                norm_map = kw.get("normalized", {})
                shown = 0
                for k, info in norm_map.items():
                    console.print(
                        f"    - {k}: steps={info.get('steps', [])}, variants={info.get('variants', [])}"
                    )
                    shown += 1
                    if shown >= 5:
                        break
                ent = nn.get("entities", {})
                console.print(
                    f"  Entities recognized: {ent.get('total', 0)} (variation counts: top {min(5, len(ent.get('variation_counts', {})))} shown)"
                )
                vc = ent.get("variation_counts", {})
                shown = 0
                for name, cnt in vc.items():
                    console.print(f"    - {name}: {cnt} variants")
                    shown += 1
                    if shown >= 5:
                        break

            # Show ranking details
            if "ranking_details" in metadata:
                rd = metadata["ranking_details"]
                console.print("\n[cyan]Ranking Details:[/cyan]")
                console.print(f"  Algorithm: {rd.get('algorithm', 'unknown')}")
                console.print(f"  Threshold: {rd.get('threshold', 0.1)}")
                console.print(f"  Files Ranked: {rd.get('files_ranked', 0)}")
                console.print(f"  Files Above Threshold: {rd.get('files_above_threshold', 0)}")

                # Show top ranked files
                if "top_files" in rd:
                    console.print("\n[cyan]Top Ranked Files:[/cyan]")
                    for i, file_info in enumerate(rd["top_files"][:10], 1):
                        console.print(
                            f"  {i}. {file_info['path']} (score: {file_info['score']:.3f})"
                        )
                        if "match_details" in file_info:
                            md = file_info["match_details"]
                            console.print(
                                f"      Keywords matched: {md.get('keywords_matched', [])}"
                            )
                            console.print(
                                f"      Semantic score: {md.get('semantic_score', 0):.3f}"
                            )

            # Show aggregation details
            if "aggregation_details" in metadata:
                ad = metadata["aggregation_details"]
                console.print("\n[cyan]Aggregation Details:[/cyan]")
                console.print(f"  Strategy: {ad.get('strategy', 'unknown')}")
                console.print(f"  Min Relevance: {ad.get('min_relevance', 0)}")
                console.print(f"  Files Considered: {ad.get('files_considered', 0)}")
                console.print(f"  Files Rejected: {ad.get('files_rejected', 0)}")
                if "rejection_reasons" in ad:
                    console.print("\n  [yellow]Rejection Reasons:[/yellow]")
                    for reason, count in ad["rejection_reasons"].items():
                        console.print(f"    {reason}: {count} files")

            console.print("\n[yellow]═════════════════════════════[/yellow]\n")
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

        # Stop timing
        timing_result = timer.stop("Context distillation complete")

        # Build summary details
        include_display_raw = ",".join(include_patterns) if include_patterns else "(none)"
        exclude_display_raw = ",".join(exclude_patterns) if exclude_patterns else "(none)"
        git_display = "disabled" if no_git else "enabled (ranking only)"
        session_display_raw = session or "(none)"
        max_tokens_display = str(max_tokens) if max_tokens else "model default"

        # Escape dynamic strings for Rich markup safety
        prompt_text = escape(str(prompt)[:80])
        path_text = escape(str(path))
        include_display = escape(include_display_raw)
        exclude_display = escape(exclude_display_raw)
        session_display = escape(session_display_raw)

        # Show a concise summary before content in interactive mode
        if interactive:
            console.print(
                Panel(
                    f"[bold]Prompt[/bold]: {prompt_text}\n"
                    f"Path: {path_text}\n"
                    f"Mode: {metadata.get('mode', 'unknown')}  •  Format: {format}\n"
                    f"Full: {metadata.get('full_mode', full)}  •  Condense: {metadata.get('condense', condense)}  •  Remove Comments: {metadata.get('remove_comments', remove_comments)}\n"
                    f"Files: {files_included}/{files_analyzed}  •  Tokens: {token_count:,} / {max_tokens_display}\n"
                    f"Include: {include_display}\n"
                    f"Exclude: {exclude_display}\n"
                    f"Git: {git_display}  •  Session: {session_display}\n"
                    f"[dim]Time: {timing_result.formatted_duration}[/dim]",
                    title="Tenets Context",
                    border_style="green",
                )
            )

        # Output result
        if output:
            output.write_text(output_text, encoding="utf-8")
            if not quiet:
                console.print(
                    f"[green]✓[/green] Context saved to {escape(str(output))} [dim]({timing_result.formatted_duration})[/dim]"
                )

                # If HTML format and interactive, offer to open in browser
                if format == "html" and interactive:
                    import click

                    if click.confirm(
                        "\nWould you like to open it in your browser now?", default=False
                    ):
                        import webbrowser

                        # Ensure absolute path for file URI
                        file_path = output.resolve()
                        webbrowser.open(file_path.as_uri())
                        console.print("[green]✓[/green] Opened in browser")
        elif format in ["json", "xml", "html"]:
            # For HTML/XML/JSON, save to a default file if no output specified
            if interactive:
                # Auto-generate filename with timestamp and prompt info
                import re
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Create safe prompt snippet for filename
                prompt_str = prompt[:50] if prompt else "context"
                safe_prompt = re.sub(r"[^\w\-_\s]", "", prompt_str)
                safe_prompt = safe_prompt.replace(" ", "_")[:30]
                safe_prompt = re.sub(r"_+", "_", safe_prompt)

                # Determine file extension
                ext = format.lower()
                default_file = Path(f"distill_{safe_prompt}_{timestamp}.{ext}")
                default_file.write_text(output_text, encoding="utf-8")
                console.print(
                    f"[green]✓[/green] {format.upper()} context saved to {escape(str(default_file))} [dim]({timing_result.formatted_duration})[/dim]"
                )

                # Offer to open in browser for HTML, or folder for XML/JSON
                import click

                if format == "html":
                    if click.confirm(
                        "\nWould you like to open it in your browser now?", default=False
                    ):
                        import webbrowser

                        # Ensure absolute path for file URI
                        file_path = default_file.resolve()
                        webbrowser.open(file_path.as_uri())
                        console.print("[green]✓[/green] Opened in browser")
                    else:
                        console.print(
                            "[cyan]💡 Tip:[/cyan] Open the file in a browser or use --output to specify a different path"
                        )
                # For XML/JSON, offer to open the folder
                elif click.confirm(
                    f"\nWould you like to open the folder containing the {format.upper()} file?",
                    default=False,
                ):
                    import platform

                    folder = default_file.parent.resolve()
                    if platform.system() == "Windows":
                        import os

                        os.startfile(folder)
                    elif platform.system() == "Darwin":  # macOS
                        import subprocess

                        subprocess.run(["open", folder], check=False)
                    else:  # Linux
                        import subprocess

                        subprocess.run(["xdg-open", folder], check=False)
                    console.print(f"[green]✓[/green] Opened folder: {folder}")
            else:
                # Non-interactive mode: print raw output for piping
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
                console.print(
                    f"[cyan]📋 Context copied to clipboard[/cyan] [dim]({timing_result.formatted_duration} total)[/dim]"
                )
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
                    f"Analysis time: {metadata.get('analysis_time', '?')}s\n"
                    f"Total time: [green]{timing_result.formatted_duration}[/green]",
                    title="📊 Statistics",
                    border_style="blue",
                )
            )

    except Exception as e:
        # Stop timer on error
        if timer.start_time and not timer.end_time:
            timing_result = timer.stop("Operation failed")
            if not quiet:
                console.print(f"[dim]Failed after {timing_result.formatted_duration}[/dim]")

        # Escape dynamic error text to avoid Rich markup parsing issues (e.g., stray [ or ]).
        console.print(f"[red]Error:[/red] {escape(str(e))}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)
