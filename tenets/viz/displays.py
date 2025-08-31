"""Terminal display utilities for CLI visualization.

This module provides rich terminal display capabilities including
tables, progress bars, charts, and formatted output for CLI commands.
"""

import math
import shutil
from typing import Any, Dict, List, Optional, Union

from .base import DisplayConfig


class TerminalDisplay:
    """Terminal display utilities for rich CLI output.

    Provides methods for displaying data in the terminal with colors,
    formatting, and various visualization styles.
    """

    def __init__(self, config: Optional[DisplayConfig] = None):
        """Initialize terminal display.

        Args:
            config: Display configuration
        """
        self.config = config or DisplayConfig()
        self.terminal_width = shutil.get_terminal_size().columns

        # ANSI color codes
        self.colors = {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "reset": "\033[0m",
            "bold": "\033[1m",
            "dim": "\033[2m",
            "italic": "\033[3m",
            "underline": "\033[4m",
        }

    def display_header(
        self, title: str, subtitle: Optional[str] = None, style: str = "single"
    ) -> None:
        """Display a formatted header.

        Args:
            title: Header title
            subtitle: Optional subtitle
            style: Border style (single, double, heavy)
        """
        # Border characters - check if terminal supports Unicode
        import sys
        import locale
        encoding = sys.stdout.encoding or locale.getpreferredencoding()
        supports_unicode = encoding and 'utf' in encoding.lower()
        
        if supports_unicode:
            borders = {
                "single": {"h": "─", "v": "│", "tl": "┌", "tr": "┐", "bl": "└", "br": "┘"},
                "double": {"h": "═", "v": "║", "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝"},
                "heavy": {"h": "━", "v": "┃", "tl": "┏", "tr": "┓", "bl": "┗", "br": "┛"},
            }
        else:
            # Fallback ASCII characters for cp1252 and similar
            borders = {
                "single": {"h": "-", "v": "|", "tl": "+", "tr": "+", "bl": "+", "br": "+"},
                "double": {"h": "=", "v": "|", "tl": "+", "tr": "+", "bl": "+", "br": "+"},
                "heavy": {"h": "=", "v": "|", "tl": "+", "tr": "+", "bl": "+", "br": "+"},
            }

        border = borders.get(style, borders["single"])
        width = min(self.config.max_width, self.terminal_width)

        # Top border
        print(f"{border['tl']}{border['h'] * (width - 2)}{border['tr']}")

        # Title
        title_line = f"{border['v']} {self.colorize(title, 'bold')} "
        padding = width - self._visible_length(title_line) - 1
        print(f"{title_line}{' ' * padding}{border['v']}")

        # Subtitle if provided
        if subtitle:
            subtitle_line = f"{border['v']} {self.colorize(subtitle, 'dim')} "
            padding = width - self._visible_length(subtitle_line) - 1
            print(f"{subtitle_line}{' ' * padding}{border['v']}")

        # Bottom border
        print(f"{border['bl']}{border['h'] * (width - 2)}{border['br']}")
        print()

    def display_table(
        self,
        headers: List[str],
        rows: List[List[Any]],
        title: Optional[str] = None,
        align: Optional[List[str]] = None,
    ) -> None:
        """Display a formatted table.

        Args:
            headers: Table headers
            rows: Table rows
            title: Optional table title
            align: Column alignment (left, right, center)
        """
        if not headers or not rows:
            return

        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows[: self.config.max_rows]:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Ensure table fits terminal
        total_width = sum(col_widths) + len(col_widths) * 3 + 1
        if total_width > self.terminal_width:
            scale = self.terminal_width / total_width
            col_widths = [max(4, int(w * scale)) for w in col_widths]

        # Display title
        if title:
            print(f"\n{self.colorize(title, 'bold')}")
            print("─" * min(total_width, self.terminal_width))

        # Display headers
        header_line = "│"
        for i, header in enumerate(headers):
            if i < len(col_widths):
                header_str = str(header)[: col_widths[i]]
                header_line += f" {self.colorize(header_str.ljust(col_widths[i]), 'bold')} │"
        print(header_line)

        # Separator
        sep_line = "├"
        for i, width in enumerate(col_widths):
            sep_line += "─" * (width + 2)
            sep_line += "┼" if i < len(col_widths) - 1 else "┤"
        print(sep_line)

        # Display rows
        for row_idx, row in enumerate(rows):
            if row_idx >= self.config.max_rows:
                print(f"... and {len(rows) - row_idx} more rows")
                break

            row_line = "│"
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    cell_str = str(cell)[: col_widths[i]]

                    # Apply alignment
                    if align and i < len(align):
                        if align[i] == "right":
                            cell_str = cell_str.rjust(col_widths[i])
                        elif align[i] == "center":
                            cell_str = cell_str.center(col_widths[i])
                        else:
                            cell_str = cell_str.ljust(col_widths[i])
                    else:
                        cell_str = cell_str.ljust(col_widths[i])

                    row_line += f" {cell_str} │"
            print(row_line)

        # Bottom border
        bottom_line = "└"
        for i, width in enumerate(col_widths):
            bottom_line += "─" * (width + 2)
            bottom_line += "┴" if i < len(col_widths) - 1 else "┘"
        print(bottom_line)
        print()

    def display_metrics(
        self, metrics: Dict[str, Any], title: Optional[str] = None, columns: int = 2
    ) -> None:
        """Display metrics in a grid layout.

        Args:
            metrics: Dictionary of metric name to value
            title: Optional title
            columns: Number of columns
        """
        if title:
            print(f"\n{self.colorize(title, 'bold')}")
            print("─" * 40)

        items = list(metrics.items())
        rows = math.ceil(len(items) / columns)

        for row in range(rows):
            line = ""
            for col in range(columns):
                idx = row * columns + col
                if idx < len(items):
                    name, value = items[idx]
                    # Format metric
                    metric_str = f"{name}: {self.colorize(str(value), 'cyan')}"
                    line += metric_str.ljust(self.terminal_width // columns)
            print(line)
        print()

    def display_distribution(
        self,
        distribution: Union[Dict[str, int], List[int]],
        title: Optional[str] = None,
        labels: Optional[List[str]] = None,
        char: str = "█",
    ) -> None:
        """Display distribution as horizontal bar chart.

        Args:
            distribution: Distribution data
            title: Optional title
            labels: Labels for values if distribution is a list
            char: Character to use for bars
        """
        if title:
            print(f"\n{self.colorize(title, 'bold')}")

        # Convert to dict if list
        if isinstance(distribution, list):
            if labels and len(labels) == len(distribution):
                distribution = dict(zip(labels, distribution))
            else:
                distribution = {f"Cat {i + 1}": v for i, v in enumerate(distribution)}

        if not distribution:
            return

        max_value = max(distribution.values()) if distribution else 1
        max_label_len = max(len(str(k)) for k in distribution.keys())

        # Calculate bar width
        bar_width = min(40, self.terminal_width - max_label_len - 10)

        for label, value in distribution.items():
            bar_len = int((value / max_value) * bar_width) if max_value > 0 else 0
            bar = char * bar_len

            # Color based on value
            if value / max_value > 0.75:
                bar = self.colorize(bar, "red")
            elif value / max_value > 0.5:
                bar = self.colorize(bar, "yellow")
            else:
                bar = self.colorize(bar, "green")

            print(f"{str(label).rjust(max_label_len)} │ {bar} {value}")
        print()

    def display_list(
        self, items: List[str], title: Optional[str] = None, style: str = "bullet"
    ) -> None:
        """Display a formatted list.

        Args:
            items: List items
            title: Optional title
            style: List style (bullet, numbered, checkbox)
        """
        if title:
            print(f"\n{self.colorize(title, 'bold')}")

        for i, item in enumerate(items):
            if style == "numbered":
                prefix = f"{i + 1}."
            elif style == "checkbox":
                prefix = "☐"
            else:  # bullet
                prefix = "•"

            # Handle multi-line items
            lines = str(item).split("\n")
            print(f"  {prefix} {lines[0]}")
            for line in lines[1:]:
                print(f"      {line}")
        print()

    def create_progress_bar(
        self, current: float, total: float, width: int = 30, show_percentage: bool = True
    ) -> str:
        """Create a progress bar string.

        Args:
            current: Current value
            total: Total value
            width: Bar width
            show_percentage: Whether to show percentage

        Returns:
            str: Progress bar string
        """
        if total == 0:
            percentage = 0
        else:
            percentage = (current / total) * 100

        filled = int((current / max(1, total)) * width)
        bar = "█" * filled + "░" * (width - filled)

        # Color based on percentage
        if percentage >= 80:
            bar = self.colorize(bar, "green")
        elif percentage >= 50:
            bar = self.colorize(bar, "yellow")
        else:
            bar = self.colorize(bar, "red")

        if show_percentage:
            return f"[{bar}] {percentage:.1f}%"
        else:
            return f"[{bar}] {current}/{total}"

    def display_warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message: Warning message
        """
        print(f"\n{self.colorize('⚠️  WARNING:', 'yellow')} {message}")

    def display_error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: Error message
        """
        print(f"\n{self.colorize('❌ ERROR:', 'red')} {message}")

    def display_success(self, message: str) -> None:
        """Display a success message.

        Args:
            message: Success message
        """
        print(f"\n{self.colorize('✅ SUCCESS:', 'green')} {message}")

    def colorize(self, text: str, color: str) -> str:
        """Add color to text.

        Args:
            text: Text to colorize
            color: Color name or style

        Returns:
            str: Colored text
        """
        if not self.config.use_colors:
            return text

        color_code = self.colors.get(color, "")
        reset = self.colors["reset"]

        return f"{color_code}{text}{reset}"

    def _visible_length(self, text: str) -> int:
        """Calculate visible length of text (excluding ANSI codes).

        Args:
            text: Text with potential ANSI codes

        Returns:
            int: Visible length
        """
        import re

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_text = ansi_escape.sub("", text)
        return len(clean_text)


class ProgressDisplay:
    """Progress indicator for long-running operations.

    Provides spinner and progress bar functionality for CLI operations.
    """

    def __init__(self):
        """Initialize progress display."""
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_spinner = 0

    def spinner(self, message: str = "Processing") -> str:
        """Get next spinner frame.

        Args:
            message: Message to display

        Returns:
            str: Spinner frame with message
        """
        char = self.spinner_chars[self.current_spinner]
        self.current_spinner = (self.current_spinner + 1) % len(self.spinner_chars)
        return f"\r{char} {message}..."

    def update_progress(self, current: int, total: int, message: str = "Progress") -> str:
        """Update progress display.

        Args:
            current: Current item
            total: Total items
            message: Progress message

        Returns:
            str: Progress string
        """
        percentage = (current / max(1, total)) * 100
        bar_width = 30
        filled = int((current / max(1, total)) * bar_width)
        bar = "=" * filled + "-" * (bar_width - filled)

        return f"\r{message}: [{bar}] {current}/{total} ({percentage:.1f}%)"
