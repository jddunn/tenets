"""System instruction command - Manage the system instruction/prompt."""

from pathlib import Path
from typing import Optional

import typer
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from tenets.config import TenetsConfig

console = Console()

# Create system-instruction subcommand app
system_app = typer.Typer(help="Manage system instruction", no_args_is_help=True)


@system_app.command("set")
def set_instruction(
    instruction: Optional[str] = typer.Argument(None, help="System instruction text"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Read from file"),
    enable: bool = typer.Option(True, "--enable/--disable", help="Enable auto-injection"),
    position: Optional[str] = typer.Option(None, "--position", help="Injection position"),
    format: Optional[str] = typer.Option(None, "--format", help="Format type"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save to config"),
):
    """Set the system instruction that will be injected at session start.

    Examples:
        # Set directly
        tenets system-instruction set "You are a helpful coding assistant"

        # Set from file
        tenets system-instruction set --file system_prompt.md

        # Set with options
        tenets system-instruction set "Context here" --position after_header --format xml

        # Disable auto-injection
        tenets system-instruction set --disable
    """
    try:
        config = TenetsConfig()

        # Get instruction text
        if file:
            if not file.exists():
                console.print(f"[red]Error:[/red] File not found: {file}")
                raise typer.Exit(1)
            instruction_text = file.read_text()
        elif instruction:
            instruction_text = instruction
        else:
            # No instruction provided, just updating settings
            instruction_text = config.tenet.system_instruction

        # Update configuration
        if instruction_text:
            config.tenet.system_instruction = instruction_text

        config.tenet.system_instruction_enabled = enable

        if position:
            config.tenet.system_instruction_position = position

        if format:
            config.tenet.system_instruction_format = format

        # Save if requested
        if save:
            config_file = config.config_file or Path(".tenets.yml")
            config.save(config_file)
            console.print(f"[green]✓[/green] Configuration saved to {config_file}")

        # Show confirmation
        console.print(
            Panel(
                f"System instruction {'enabled' if enable else 'disabled'}\n"
                f"Position: {config.tenet.system_instruction_position}\n"
                f"Format: {config.tenet.system_instruction_format}\n"
                f"Length: {len(instruction_text or '')} chars",
                title="System Instruction Updated",
                border_style="green",
            )
        )

        if instruction_text and len(instruction_text) < 500:
            console.print("\n[bold]Instruction:[/bold]")
            console.print(Panel(instruction_text, border_style="blue"))

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@system_app.command("show")
def show_instruction(
    raw: bool = typer.Option(False, "--raw", help="Show raw text without formatting"),
):
    """Show the current system instruction.

    Examples:
        tenets system-instruction show
        tenets system-instruction show --raw
    """
    try:
        config = TenetsConfig()

        if not config.tenet.system_instruction:
            console.print("[yellow]No system instruction configured.[/yellow]")
            console.print(
                '\nSet one with: [bold]tenets system-instruction set "Your instruction"[/bold]'
            )
            return

        instruction = config.tenet.system_instruction

        if raw:
            print(instruction)
        else:
            # Show formatted
            console.print(
                Panel(
                    f"Status: {'[green]Enabled[/green]' if config.tenet.system_instruction_enabled else '[red]Disabled[/red]'}\n"
                    f"Position: {config.tenet.system_instruction_position}\n"
                    f"Format: {config.tenet.system_instruction_format}\n"
                    f"Once per session: {config.tenet.system_instruction_once_per_session}\n"
                    f"Length: {len(instruction)} characters",
                    title="System Instruction Configuration",
                    border_style="blue",
                )
            )

            console.print("\n[bold]Instruction Content:[/bold]")

            # Use syntax highlighting if it looks like code
            if any(
                keyword in instruction.lower()
                for keyword in ["def ", "class ", "function", "import"]
            ):
                syntax = Syntax(instruction, "python", theme="monokai")
                console.print(syntax)
            else:
                console.print(Panel(instruction, border_style="dim"))

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@system_app.command("clear")
def clear_instruction(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear the system instruction.

    Examples:
        tenets system-instruction clear
        tenets system-instruction clear --yes
    """
    if not confirm:
        confirm = typer.confirm("Clear the system instruction?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            return

    try:
        config = TenetsConfig()
        config.tenet.system_instruction = None
        config.tenet.system_instruction_enabled = False

        config_file = config.config_file or Path(".tenets.yml")
        config.save(config_file)

        console.print("[green]✓[/green] System instruction cleared")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@system_app.command("test")
def test_instruction(
    session: Optional[str] = typer.Option(None, "--session", help="Test with session"),
):
    """Test system instruction injection on sample content.

    Examples:
        tenets system-instruction test
        tenets system-instruction test --session my-session
    """
    try:
        from tenets import Tenets

        config = TenetsConfig()

        if not config.tenet.system_instruction:
            console.print("[yellow]No system instruction configured.[/yellow]")
            return

        # Create sample content
        sample_content = """# Sample Project Documentation

## Overview
This is sample content to test system instruction injection.

## Code Example
```python
def hello():
    return "world"
```

The rest of the content continues here..."""

        # Test injection
        tenets = Tenets(config)
        instiller = tenets.instiller

        # Inject system instruction
        modified, metadata = instiller.inject_system_instruction(
            sample_content, format="markdown", session=session
        )

        if metadata.get("system_instruction_injected"):
            console.print(
                Panel(
                    f"[green]✓[/green] System instruction injected\n"
                    f"Position: {metadata.get('system_instruction_position')}\n"
                    f"Token increase: {metadata.get('token_increase', 0)}",
                    title="Injection Test Result",
                    border_style="green",
                )
            )

            console.print("\n[bold]Modified Content:[/bold]")
            console.print(Panel(modified[:1000] + "..." if len(modified) > 1000 else modified))
        else:
            reason = metadata.get("reason", "unknown")
            console.print(f"[yellow]System instruction not injected: {reason}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@system_app.command("export")
def export_instruction(
    output: Path = typer.Argument(..., help="Output file path"),
):
    """Export system instruction to file.

    Examples:
        tenets system-instruction export system_prompt.txt
        tenets system-instruction export prompts/main.md
    """
    try:
        config = TenetsConfig()

        if not config.tenet.system_instruction:
            console.print("[yellow]No system instruction to export.[/yellow]")
            return

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(config.tenet.system_instruction)

        console.print(f"[green]✓[/green] Exported to {output}")
        console.print(f"Size: {len(config.tenet.system_instruction)} characters")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@system_app.command("validate")
def validate_instruction(
    check_tokens: bool = typer.Option(False, "--tokens", help="Check token count"),
    max_tokens: int = typer.Option(1000, "--max-tokens", help="Maximum allowed tokens"),
):
    """Validate the current system instruction.

    Examples:
        tenets system-instruction validate
        tenets system-instruction validate --tokens --max-tokens 500
    """
    try:
        config = TenetsConfig()

        if not config.tenet.system_instruction:
            console.print("[yellow]No system instruction to validate.[/yellow]")
            return

        instruction = config.tenet.system_instruction
        issues = []
        warnings = []

        # Check length
        if len(instruction) > 5000:
            warnings.append(f"Instruction is quite long ({len(instruction)} chars)")
        elif len(instruction) < 10:
            issues.append(f"Instruction seems too short ({len(instruction)} chars)")

        # Check for common issues
        if not instruction.strip():
            issues.append("Instruction is empty or only whitespace")

        if instruction.count("\n") > 50:
            warnings.append(f"Instruction has many lines ({instruction.count(chr(10))} lines)")

        # Token count check (optional)
        if check_tokens:
            # Simple token estimation (actual tokenization would need tiktoken)
            estimated_tokens = len(instruction.split()) * 1.3
            if estimated_tokens > max_tokens:
                issues.append(
                    f"Estimated tokens ({int(estimated_tokens)}) exceeds max ({max_tokens})"
                )

            console.print(f"\n[bold]Token Estimate:[/bold] ~{int(estimated_tokens)} tokens")

        # Check format compatibility
        format_type = config.tenet.system_instruction_format
        if format_type == "xml" and not ("<" in instruction and ">" in instruction):
            warnings.append("Format is 'xml' but instruction doesn't contain XML tags")
        elif format_type == "markdown" and not any(md in instruction for md in ["#", "*", "`"]):
            warnings.append("Format is 'markdown' but no markdown formatting detected")

        # Display results
        if not issues and not warnings:
            console.print(
                Panel(
                    "[green]✓[/green] System instruction is valid\n"
                    f"Length: {len(instruction)} characters\n"
                    f"Lines: {instruction.count(chr(10)) + 1}\n"
                    f"Format: {format_type}",
                    title="Validation Passed",
                    border_style="green",
                )
            )
        else:
            if issues:
                console.print("\n[red]Issues found:[/red]")
                for issue in issues:
                    console.print(f"  • {issue}")

            if warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in warnings:
                    console.print(f"  • {warning}")

            if issues:
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@system_app.command("edit")
def edit_instruction(
    editor: Optional[str] = typer.Option(None, "--editor", "-e", help="Editor to use"),
):
    """Open system instruction in editor for editing.

    Examples:
        tenets system-instruction edit
        tenets system-instruction edit --editor vim
        tenets system-instruction edit -e nano
    """
    try:
        import tempfile
        import subprocess
        import os

        config = TenetsConfig()

        # Get current instruction or empty string
        current_instruction = config.tenet.system_instruction or ""

        # Create temp file with current instruction
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            tmp.write(current_instruction)
            tmp_path = tmp.name

        try:
            # Determine editor
            if not editor:
                editor = os.environ.get("EDITOR", "nano")

            # Open editor
            subprocess.call([editor, tmp_path])

            # Read edited content
            with open(tmp_path, "r") as f:
                new_instruction = f.read()

            # Check if changed
            if new_instruction != current_instruction:
                # Update configuration
                config.tenet.system_instruction = new_instruction
                config.tenet.system_instruction_enabled = True

                # Save
                config_file = config.config_file or Path(".tenets.yml")
                config.save(config_file)

                console.print(
                    Panel(
                        f"[green]✓[/green] System instruction updated\n"
                        f"Length: {len(new_instruction)} characters\n"
                        f"Saved to: {config_file}",
                        title="Instruction Edited",
                        border_style="green",
                    )
                )
            else:
                console.print("[yellow]No changes made.[/yellow]")

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


# Re-export the Typer app for mounting by main CLI
app = system_app
