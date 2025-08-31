"""Timing utilities for CLI commands with intelligent duration formatting.

This module provides comprehensive timing utilities including:
- Intelligent duration formatting (μs, ms, s, m, h)
- Decorator for timing functions and methods
- Context manager for timing code blocks
- Benchmarking utilities
- CLI command timers with console output

Examples:
    Using the timing decorator::

        from tenets.utils.timing import timed

        @timed()
        def process_files(files):
            # Function automatically timed
            return analyze(files)

        # With custom options
        @timed(name="Analysis", log_output=True, include_args=True)
        def analyze_codebase(path, mode="fast"):
            return results

    Using context manager::

        from tenets.utils.timing import timed_operation

        with timed_operation("Building context") as timer:
            result = build_context()
            # Access timing: timer.duration

    Direct timing::

        from tenets.utils.timing import CommandTimer

        timer = CommandTimer()
        timer.start("Processing...")
        # ... do work ...
        result = timer.stop("Complete")
        print(f"Took {result.formatted_duration}")
"""

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

if TYPE_CHECKING:
    from rich.console import Console

import logging

# Lazy load Rich to improve import performance
Console = None
Text = None


def _ensure_rich_imported():
    """Import Rich when actually needed."""
    global Console, Text
    if Console is None:
        try:
            from rich.console import Console as _Console
            from rich.text import Text as _Text

            Console = _Console
            Text = _Text
        except ImportError:
            pass


__all__ = [
    "CommandTimer",
    "TimedMixin",
    "TimingResult",
    "benchmark_operation",
    "format_duration",
    "format_progress_time",
    "format_time_range",
    "timed",
    "timed_operation",
]


@dataclass
class TimingResult:
    """Container for timing information."""

    start_time: float
    end_time: float
    duration: float
    formatted_duration: str
    start_datetime: datetime
    end_datetime: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with timing data including ISO format timestamps
        """
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "duration_seconds": self.duration,
            "duration_ms": self.duration * 1000,
            "formatted_duration": self.formatted_duration,
            "start_datetime": self.start_datetime.isoformat(),
            "end_datetime": self.end_datetime.isoformat(),
        }


def format_duration(seconds: float) -> str:
    """
    Format duration intelligently based on length.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string with appropriate units

    Examples:
        0.123 -> "123ms"
        1.5 -> "1.50s"
        65 -> "1m 5s"
        3665 -> "1h 1m 5s"
    """
    if seconds < 0.001:
        # Microseconds
        microseconds = seconds * 1_000_000
        # Round up to at least 1 for very small values
        if 0 < microseconds < 1:
            return "1μs"
        return f"{microseconds:.0f}μs"
    elif seconds < 1:
        # Milliseconds
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        # Seconds with 2 decimal places
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        # Minutes and seconds
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        if secs == 0:
            return f"{minutes}m"
        return f"{minutes}m {secs}s"
    else:
        # Hours, minutes, and seconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or (hours == 0 and minutes == 0):
            parts.append(f"{secs}s")

        return " ".join(parts)


def format_time_range(start: datetime, end: datetime) -> str:
    """
    Format a time range for display.

    Args:
        start: Start datetime
        end: End datetime

    Returns:
        Formatted time range string

    Examples:
        Same day: "10:30:45 - 10:31:23"
        Different days: "2024-01-15 10:30:45 - 2024-01-16 08:15:23"
    """
    if start.date() == end.date():
        # Same day - just show times
        return f"{start.strftime('%H:%M:%S')} - {end.strftime('%H:%M:%S')}"
    else:
        # Different days - show full datetime
        return f"{start.strftime('%Y-%m-%d %H:%M:%S')} - {end.strftime('%Y-%m-%d %H:%M:%S')}"


class CommandTimer:
    """Timer for CLI commands with formatted output."""

    def __init__(self, console: Optional[Any] = None, quiet: bool = False):
        """
        Initialize command timer.

        Args:
            console: Rich console for output
            quiet: If True, suppress timing output
        """
        _ensure_rich_imported()
        self.console = console or (Console() if Console else None)
        self.quiet = quiet
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.start_datetime: Optional[datetime] = None
        self.end_datetime: Optional[datetime] = None
        # Check if we can use emojis
        self._use_emojis = self._check_emoji_support()

    def start(self, message: Optional[str] = None) -> None:
        """
        Start the timer.

        Args:
            message: Optional message to display when starting
        """
        self.start_time = time.perf_counter()
        self.start_datetime = datetime.now()

        if not self.quiet and message and self.console:
            timestamp = self.start_datetime.strftime("%H:%M:%S")
            if self._use_emojis:
                self.console.print(f"[dim]🕐 {timestamp}[/dim] {message}")
            else:
                self.console.print(f"[dim][{timestamp}][/dim] {message}")

    def stop(self, message: Optional[str] = None) -> TimingResult:
        """
        Stop the timer and return timing information.

        Args:
            message: Optional message to display when stopping

        Returns:
            TimingResult with all timing information
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started")

        self.end_time = time.perf_counter()
        self.end_datetime = datetime.now()

        duration = self.end_time - self.start_time
        formatted = format_duration(duration)

        result = TimingResult(
            start_time=self.start_time,
            end_time=self.end_time,
            duration=duration,
            formatted_duration=formatted,
            start_datetime=self.start_datetime,
            end_datetime=self.end_datetime,
        )

        if not self.quiet and self.console:
            timestamp = self.end_datetime.strftime("%H:%M:%S")
            if message:
                if self._use_emojis:
                    self.console.print(
                        f"[dim]✅ {timestamp}[/dim] {message} [green]({formatted})[/green]"
                    )
                else:
                    self.console.print(
                        f"[dim][OK {timestamp}][/dim] {message} [green]({formatted})[/green]"
                    )
            elif self._use_emojis:
                self.console.print(
                    f"[dim]✅ {timestamp}[/dim] Completed in [green]{formatted}[/green]"
                )
            else:
                self.console.print(
                    f"[dim][OK {timestamp}][/dim] Completed in [green]{formatted}[/green]"
                )

        return result

    def _check_emoji_support(self) -> bool:
        """Check if the terminal supports emoji characters.

        Returns:
            True if emojis are supported, False otherwise
        """
        import locale
        import sys

        # Check encoding
        encoding = sys.stdout.encoding or locale.getpreferredencoding()
        if encoding:
            encoding = encoding.lower()
            # Common encodings that don't support emojis
            if any(enc in encoding for enc in ["cp1252", "cp437", "ascii", "latin"]):
                return False

        # UTF-8 and similar support emojis
        return True

    def display_summary(self, result: Optional[TimingResult] = None) -> None:
        """
        Display a timing summary.

        Args:
            result: TimingResult to display (uses last result if None)
        """
        if result is None and self.end_time is not None:
            result = TimingResult(
                start_time=self.start_time,
                end_time=self.end_time,
                duration=self.end_time - self.start_time,
                formatted_duration=format_duration(self.end_time - self.start_time),
                start_datetime=self.start_datetime,
                end_datetime=self.end_datetime,
            )

        if result and not self.quiet and self.console:
            self.console.print("\n[cyan]═══ Timing Summary ═══[/cyan]")
            self.console.print(f"  Started:  {result.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            self.console.print(f"  Finished: {result.end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            self.console.print(f"  Duration: [green]{result.formatted_duration}[/green]")


@contextmanager
def timed_operation(
    name: str, console: Optional[Any] = None, quiet: bool = False, show_summary: bool = False
):
    """
    Context manager for timing operations.

    Args:
        name: Name of the operation
        console: Rich console for output
        quiet: If True, suppress output
        show_summary: If True, show timing summary at end

    Yields:
        CommandTimer instance

    Example:
        with timed_operation("Building context", console) as timer:
            # ... do work ...
            pass
    """
    timer = CommandTimer(console, quiet)
    timer.start(f"Starting: {name}")

    try:
        yield timer
    finally:
        result = timer.stop(f"Finished: {name}")
        if show_summary:
            timer.display_summary(result)


def format_progress_time(elapsed: float, total: Optional[float] = None) -> str:
    """
    Format elapsed time with optional ETA.

    Args:
        elapsed: Elapsed time in seconds
        total: Total expected time (for ETA calculation)

    Returns:
        Formatted string with elapsed and optional ETA
    """
    elapsed_str = format_duration(elapsed)

    if total and total > elapsed:
        remaining = total - elapsed
        eta_str = format_duration(remaining)
        return f"{elapsed_str} / ~{format_duration(total)} (ETA: {eta_str})"

    return elapsed_str


def timed(
    name: Optional[str] = None,
    log_output: bool = False,
    console: Optional[Any] = None,
    quiet: bool = False,
    include_args: bool = False,
    include_result: bool = False,
    threshold_ms: Optional[float] = None,
) -> Callable:
    """Decorator to time function execution.

    Args:
        name: Custom name for the operation (defaults to function name)
        log_output: If True, log timing to logger
        console: Rich console for output (creates one if needed and not quiet)
        quiet: If True, suppress all output
        include_args: If True, include function arguments in output
        include_result: If True, include return value info in output
        threshold_ms: Only log if duration exceeds this threshold (milliseconds)

    Returns:
        Decorated function that tracks timing

    Examples:
        Basic usage::

            @timed()
            def process_data(data):
                return transform(data)

        With logging::

            @timed(log_output=True, threshold_ms=100)
            def slow_operation():
                # Only logs if takes > 100ms
                time.sleep(0.2)

        With arguments::

            @timed(include_args=True, include_result=True)
            def api_call(endpoint, method="GET"):
                # Logs: "api_call(endpoint='/users', method='GET') -> 200 (45ms)"
                return response

        Class methods::

            class Processor:
                @timed(name="Processing")
                def process(self, items):
                    return [self.transform(i) for i in items]

    Note:
        The timing information is also attached to the function as
        `func._last_timing` for programmatic access.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Determine operation name
            op_name = name or func.__name__

            # Start timing
            timer = CommandTimer(console, quiet)

            # Build context string if needed
            context = ""
            if include_args and not quiet:
                # Format arguments
                arg_strs = [repr(a) for a in args[:3]]  # Limit to first 3 args
                if len(args) > 3:
                    arg_strs.append("...")
                kwarg_strs = [f"{k}={v!r}" for k, v in list(kwargs.items())[:3]]
                if len(kwargs) > 3:
                    kwarg_strs.append("...")
                all_args = arg_strs + kwarg_strs
                if all_args:
                    context = f"({', '.join(all_args)})"

            # Start with context
            timer.start(f"{op_name}{context}" if not quiet else None)

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Stop timing
                timing_result = timer.stop()

                # Check threshold
                if threshold_ms and timing_result.duration * 1000 < threshold_ms:
                    # Below threshold, don't output
                    pass
                else:
                    # Format output message
                    msg_parts = [op_name]
                    if context:
                        msg_parts.append(context)
                    if include_result and not quiet:
                        # Add result info
                        if hasattr(result, "__len__"):
                            msg_parts.append(f" -> {len(result)} items")
                        elif isinstance(result, (int, float, str, bool)):
                            msg_parts.append(f" -> {result}")
                        else:
                            msg_parts.append(f" -> {type(result).__name__}")
                    msg_parts.append(f" ({timing_result.formatted_duration})")

                    # Output based on settings
                    if log_output:
                        logger = logging.getLogger(func.__module__)
                        logger.info("".join(msg_parts))
                    elif not quiet and console:
                        console.print(f"[dim]⏱ {''.join(msg_parts)}[/dim]")

                # Attach timing to function for programmatic access
                wrapper._last_timing = timing_result  # type: ignore

                return result

            except Exception as e:
                # Stop timer on error
                timing_result = timer.stop()

                # Attach timing to function even on error for programmatic access
                wrapper._last_timing = timing_result  # type: ignore

                # Log error with timing
                if log_output:
                    logger = logging.getLogger(func.__module__)
                    logger.error(f"{op_name} failed after {timing_result.formatted_duration}: {e}")
                elif not quiet and console:
                    console.print(
                        f"[red]✗ {op_name} failed after {timing_result.formatted_duration}[/red]"
                    )

                # Re-raise the exception
                raise

        return wrapper

    return decorator


def benchmark_operation(func, *args, iterations: int = 1, **kwargs) -> Tuple[Any, TimingResult]:
    """
    Benchmark a function execution.

    Args:
        func: Function to benchmark
        *args: Arguments to pass to function
        iterations: Number of iterations for averaging
        **kwargs: Keyword arguments to pass to function

    Returns:
        Tuple of (function result, timing result)
    """
    start_time = time.perf_counter()
    start_datetime = datetime.now()

    # Run the function
    for i in range(iterations - 1):
        func(*args, **kwargs)

    # Last iteration captures the result
    result = func(*args, **kwargs)

    end_time = time.perf_counter()
    end_datetime = datetime.now()

    total_duration = end_time - start_time
    avg_duration = total_duration / iterations

    timing_result = TimingResult(
        start_time=start_time,
        end_time=end_time,
        duration=avg_duration if iterations > 1 else total_duration,
        formatted_duration=format_duration(avg_duration if iterations > 1 else total_duration),
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    return result, timing_result


class TimedMixin:
    """Mixin class to add timing capabilities to any class.

    Example:
        class MyAnalyzer(TimedMixin):
            def analyze(self, data):
                with self.timed_method("analysis"):
                    return process(data)

            def get_timing_summary(self):
                return self.format_timing_summary()
    """

    def __init__(self, *args, **kwargs):
        """Initialize timing tracking."""
        super().__init__(*args, **kwargs)
        self._timing_history: list[TimingResult] = []
        self._current_timers: dict[str, CommandTimer] = {}

    @contextmanager
    def timed_method(self, name: str, quiet: bool = True):
        """Context manager for timing a method.

        Args:
            name: Name of the operation
            quiet: If True, suppress output

        Yields:
            CommandTimer instance
        """
        timer = CommandTimer(quiet=quiet)
        self._current_timers[name] = timer
        timer.start()

        try:
            yield timer
        finally:
            result = timer.stop()
            self._timing_history.append(result)
            del self._current_timers[name]

    def get_total_time(self) -> float:
        """Get total time spent in timed operations.

        Returns:
            Total duration in seconds
        """
        return sum(t.duration for t in self._timing_history)

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get summary of all timing data.

        Returns:
            Dictionary with timing statistics
        """
        if not self._timing_history:
            return {"total_operations": 0, "total_time": 0}

        durations = [t.duration for t in self._timing_history]
        return {
            "total_operations": len(self._timing_history),
            "total_time": sum(durations),
            "total_time_formatted": format_duration(sum(durations)),
            "average_time": sum(durations) / len(durations),
            "average_time_formatted": format_duration(sum(durations) / len(durations)),
            "min_time": min(durations),
            "min_time_formatted": format_duration(min(durations)),
            "max_time": max(durations),
            "max_time_formatted": format_duration(max(durations)),
        }

    def format_timing_summary(self) -> str:
        """Format timing summary as a string.

        Returns:
            Formatted timing summary
        """
        summary = self.get_timing_summary()
        if summary["total_operations"] == 0:
            return "No timed operations"

        return (
            f"Operations: {summary['total_operations']}, "
            f"Total: {summary['total_time_formatted']}, "
            f"Avg: {summary['average_time_formatted']}, "
            f"Min: {summary['min_time_formatted']}, "
            f"Max: {summary['max_time_formatted']}"
        )
