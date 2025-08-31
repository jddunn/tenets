"""Tests for timing utilities."""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from tenets.utils.timing import (
    CommandTimer,
    TimingResult,
    TimedMixin,
    benchmark_operation,
    format_duration,
    format_progress_time,
    format_time_range,
    timed,
    timed_operation,
)


class TestFormatDuration:
    """Test duration formatting."""

    def test_microseconds(self):
        """Test formatting for microsecond durations."""
        assert format_duration(0.0000005) == "1μs"
        assert format_duration(0.0000999) == "100μs"

    def test_milliseconds(self):
        """Test formatting for millisecond durations."""
        assert format_duration(0.001) == "1ms"
        assert format_duration(0.123) == "123ms"
        assert format_duration(0.999) == "999ms"

    def test_seconds(self):
        """Test formatting for second durations."""
        assert format_duration(1.0) == "1.00s"
        assert format_duration(1.5) == "1.50s"
        assert format_duration(59.99) == "59.99s"

    def test_minutes(self):
        """Test formatting for minute durations."""
        assert format_duration(60) == "1m"
        assert format_duration(65) == "1m 5s"
        assert format_duration(120) == "2m"
        assert format_duration(3599) == "59m 59s"

    def test_hours(self):
        """Test formatting for hour durations."""
        assert format_duration(3600) == "1h"
        assert format_duration(3665) == "1h 1m 5s"
        assert format_duration(7200) == "2h"
        assert format_duration(7260) == "2h 1m"
        assert format_duration(86400) == "24h"


class TestFormatTimeRange:
    """Test time range formatting."""

    def test_same_day(self):
        """Test formatting when start and end are on the same day."""
        start = datetime(2024, 1, 15, 10, 30, 45)
        end = datetime(2024, 1, 15, 10, 31, 23)
        result = format_time_range(start, end)
        assert result == "10:30:45 - 10:31:23"

    def test_different_days(self):
        """Test formatting when start and end are on different days."""
        start = datetime(2024, 1, 15, 10, 30, 45)
        end = datetime(2024, 1, 16, 8, 15, 23)
        result = format_time_range(start, end)
        assert result == "2024-01-15 10:30:45 - 2024-01-16 08:15:23"


class TestCommandTimer:
    """Test CommandTimer class."""

    def test_basic_timing(self):
        """Test basic start/stop timing."""
        timer = CommandTimer(quiet=True)
        timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        result = timer.stop()

        assert result.duration >= 0.01
        assert result.duration < 0.1  # Should be much less than 100ms
        assert isinstance(result.formatted_duration, str)
        assert isinstance(result.start_datetime, datetime)
        assert isinstance(result.end_datetime, datetime)

    def test_with_messages(self):
        """Test timing with console messages."""
        mock_console = MagicMock()
        timer = CommandTimer(console=mock_console, quiet=False)

        timer.start("Starting operation")
        mock_console.print.assert_called_once()
        assert "Starting operation" in str(mock_console.print.call_args)

        result = timer.stop("Operation complete")
        assert mock_console.print.call_count == 2
        assert "Operation complete" in str(mock_console.print.call_args)
        assert result.formatted_duration in str(mock_console.print.call_args)

    def test_quiet_mode(self):
        """Test that quiet mode suppresses output."""
        mock_console = MagicMock()
        timer = CommandTimer(console=mock_console, quiet=True)

        timer.start("Starting")
        timer.stop("Stopping")

        mock_console.print.assert_not_called()

    def test_display_summary(self):
        """Test displaying timing summary."""
        mock_console = MagicMock()
        timer = CommandTimer(console=mock_console, quiet=False)

        timer.start()
        result = timer.stop()
        timer.display_summary(result)

        # Should have printed summary
        calls = str(mock_console.print.call_args_list)
        assert "Timing Summary" in calls
        assert "Started:" in calls
        assert "Finished:" in calls
        assert "Duration:" in calls

    def test_error_without_start(self):
        """Test that stopping without starting raises an error."""
        timer = CommandTimer(quiet=True)
        with pytest.raises(RuntimeError, match="Timer was not started"):
            timer.stop()


class TestTimedOperation:
    """Test timed_operation context manager."""

    def test_basic_context_manager(self):
        """Test basic usage of timed_operation."""
        mock_console = MagicMock()
        
        with timed_operation("test operation", console=mock_console, quiet=False) as timer:
            time.sleep(0.01)
            assert timer.start_time is not None

        # Should have printed start and stop messages
        assert mock_console.print.call_count >= 2

    def test_with_exception(self):
        """Test that timer stops even with exception."""
        mock_console = MagicMock()
        
        with pytest.raises(ValueError):
            with timed_operation("test", console=mock_console, quiet=True) as timer:
                assert timer.start_time is not None
                raise ValueError("Test error")

        # Timer should have been stopped
        assert timer.end_time is not None

    def test_with_summary(self):
        """Test context manager with summary display."""
        mock_console = MagicMock()
        
        with timed_operation("test", console=mock_console, quiet=False, show_summary=True):
            time.sleep(0.01)

        # Should have printed summary
        calls = str(mock_console.print.call_args_list)
        assert "Timing Summary" in calls


class TestFormatProgressTime:
    """Test progress time formatting."""

    def test_elapsed_only(self):
        """Test formatting with only elapsed time."""
        result = format_progress_time(65.5)
        assert result == "1m 5s"

    def test_with_total_and_eta(self):
        """Test formatting with total time and ETA."""
        result = format_progress_time(30, total=90)
        assert "30.00s" in result
        assert "1m 30s" in result  # Total
        assert "ETA: 1m" in result  # Remaining

    def test_when_exceeded(self):
        """Test formatting when elapsed exceeds total."""
        result = format_progress_time(100, total=90)
        assert result == "1m 40s"  # Just shows elapsed


class TestBenchmarkOperation:
    """Test benchmark_operation function."""

    def test_single_iteration(self):
        """Test benchmarking with single iteration."""
        def test_func(x):
            return x * 2

        result, timing = benchmark_operation(test_func, 5, iterations=1)
        
        assert result == 10
        assert timing.duration >= 0
        assert isinstance(timing.formatted_duration, str)

    def test_multiple_iterations(self):
        """Test benchmarking with multiple iterations."""
        call_count = 0
        
        def test_func():
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)
            return "result"

        result, timing = benchmark_operation(test_func, iterations=3)
        
        assert result == "result"
        assert call_count == 3
        assert timing.duration >= 0.001  # Average should be at least 1ms

    def test_with_kwargs(self):
        """Test benchmarking with keyword arguments."""
        def test_func(a, b=2):
            return a + b

        result, timing = benchmark_operation(test_func, 3, b=5, iterations=1)
        
        assert result == 8
        assert timing.duration >= 0


class TestTimingResultDataclass:
    """Test TimingResult dataclass."""

    def test_creation(self):
        """Test creating a TimingResult."""
        start = datetime.now()
        end = start + timedelta(seconds=1.5)
        
        result = TimingResult(
            start_time=1000.0,
            end_time=1001.5,
            duration=1.5,
            formatted_duration="1.50s",
            start_datetime=start,
            end_datetime=end,
        )
        
        assert result.start_time == 1000.0
        assert result.end_time == 1001.5
        assert result.duration == 1.5
        assert result.formatted_duration == "1.50s"
        assert result.start_datetime == start
        assert result.end_datetime == end

    def test_to_dict(self):
        """Test TimingResult to_dict conversion."""
        start = datetime(2024, 1, 15, 10, 30, 45)
        end = start + timedelta(seconds=1.5)
        
        result = TimingResult(
            start_time=1000.0,
            end_time=1001.5,
            duration=1.5,
            formatted_duration="1.50s",
            start_datetime=start,
            end_datetime=end,
        )
        
        data = result.to_dict()
        
        assert data["duration"] == 1.5
        assert data["duration_seconds"] == 1.5
        assert data["duration_ms"] == 1500
        assert data["formatted_duration"] == "1.50s"
        assert data["start_datetime"] == start.isoformat()
        assert data["end_datetime"] == end.isoformat()


class TestTimedDecorator:
    """Test the @timed decorator."""
    
    def test_basic_decorator(self):
        """Test basic decorator functionality."""
        @timed(quiet=True)
        def test_func(x):
            time.sleep(0.01)
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        assert hasattr(test_func, '_last_timing')
        assert test_func._last_timing.duration >= 0.01
    
    def test_decorator_with_name(self):
        """Test decorator with custom name."""
        mock_console = MagicMock()
        
        @timed(name="Custom Operation", console=mock_console, quiet=False)
        def test_func():
            time.sleep(0.01)
            return "result"
        
        result = test_func()
        
        assert result == "result"
        # Should have called console.print for start and stop
        assert mock_console.print.call_count >= 2
    
    def test_decorator_with_args(self):
        """Test decorator with include_args option."""
        mock_console = MagicMock()
        
        @timed(include_args=True, console=mock_console, quiet=False)
        def test_func(a, b=2):
            return a + b
        
        result = test_func(3, b=5)
        
        assert result == 8
        # Check that args were included in output
        calls = str(mock_console.print.call_args_list)
        assert "test_func" in calls
    
    def test_decorator_with_threshold(self):
        """Test decorator with timing threshold."""
        mock_console = MagicMock()
        
        @timed(threshold_ms=100, console=mock_console, quiet=False)
        def fast_func():
            time.sleep(0.01)  # 10ms, below threshold
            return "fast"
        
        @timed(threshold_ms=10, console=mock_console, quiet=False)
        def slow_func():
            time.sleep(0.02)  # 20ms, above threshold
            return "slow"
        
        fast_func()
        slow_func()
        
        # Only slow_func should have printed (above threshold)
        # Fast func should not print due to threshold
        assert mock_console.print.call_count >= 2  # At least from slow_func
    
    def test_decorator_with_exception(self):
        """Test decorator behavior with exceptions."""
        @timed(quiet=True)
        def failing_func():
            time.sleep(0.01)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_func()
        
        # Should still have timing even with exception
        assert hasattr(failing_func, '_last_timing')
        assert failing_func._last_timing.duration >= 0.01
    
    def test_decorator_logging(self):
        """Test decorator with log output."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            @timed(log_output=True, quiet=True)
            def test_func():
                time.sleep(0.01)
                return "result"
            
            result = test_func()
            
            assert result == "result"
            # Should have logged timing info
            mock_logger.info.assert_called()


class TestTimedMixin:
    """Test the TimedMixin class."""
    
    def test_basic_mixin(self):
        """Test basic mixin functionality."""
        class TestClass(TimedMixin):
            def process(self, data):
                with self.timed_method("processing"):
                    time.sleep(0.01)
                    return data.upper()
        
        obj = TestClass()
        result = obj.process("hello")
        
        assert result == "HELLO"
        assert obj.get_total_time() >= 0.01
        
        summary = obj.get_timing_summary()
        assert summary["total_operations"] == 1
        assert summary["total_time"] >= 0.01
    
    def test_multiple_operations(self):
        """Test mixin with multiple timed operations."""
        class TestClass(TimedMixin):
            def operation_a(self):
                with self.timed_method("op_a"):
                    time.sleep(0.01)
                    return "a"
            
            def operation_b(self):
                with self.timed_method("op_b"):
                    time.sleep(0.01)
                    return "b"
        
        obj = TestClass()
        obj.operation_a()
        obj.operation_b()
        
        summary = obj.get_timing_summary()
        assert summary["total_operations"] == 2
        assert summary["total_time"] >= 0.02
        assert "min_time" in summary
        assert "max_time" in summary
        assert "average_time" in summary
    
    def test_mixin_summary_formatting(self):
        """Test mixin timing summary formatting."""
        class TestClass(TimedMixin):
            def work(self):
                with self.timed_method("work"):
                    time.sleep(0.01)
        
        obj = TestClass()
        
        # No operations yet
        assert obj.format_timing_summary() == "No timed operations"
        
        # After operation
        obj.work()
        summary_str = obj.format_timing_summary()
        
        assert "Operations: 1" in summary_str
        assert "Total:" in summary_str
        assert "Avg:" in summary_str
    
    def test_mixin_nested_operations(self):
        """Test mixin with nested timed operations."""
        class TestClass(TimedMixin):
            def outer_operation(self):
                with self.timed_method("outer"):
                    time.sleep(0.01)
                    self.inner_operation()
                    return "outer"
            
            def inner_operation(self):
                with self.timed_method("inner"):
                    time.sleep(0.01)
                    return "inner"
        
        obj = TestClass()
        result = obj.outer_operation()
        
        assert result == "outer"
        
        summary = obj.get_timing_summary()
        assert summary["total_operations"] == 2  # Both outer and inner
        assert summary["total_time"] >= 0.02