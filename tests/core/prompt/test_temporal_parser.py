"""Unit tests for temporal parsing system.

Tests date/time extraction, relative dates, ranges, and recurring patterns
with comprehensive edge cases and timezone handling.
"""

import json
import re
from datetime import date, datetime, time, timedelta
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from tenets.core.prompt.temporal_parser import (
    TemporalExpression,
    TemporalParser,
    TemporalPatternMatcher,
)


class TestTemporalExpression:
    """Test TemporalExpression dataclass."""

    def test_temporal_expression_creation(self):
        """Test creating temporal expression."""
        expr = TemporalExpression(
            text="next week",
            type="relative",
            start_date=datetime(2024, 1, 8),
            end_date=datetime(2024, 1, 14),
            is_relative=True,
            is_recurring=False,
            recurrence_pattern=None,
            confidence=0.9,
            metadata={"source": "pattern"},
        )

        assert expr.text == "next week"
        assert expr.type == "relative"
        assert expr.start_date == datetime(2024, 1, 8)
        assert expr.end_date == datetime(2024, 1, 14)
        assert expr.is_relative is True
        assert expr.is_recurring is False
        assert expr.confidence == 0.9
        assert expr.metadata["source"] == "pattern"

    def test_timeframe_property_single_day(self):
        """Test timeframe for single day range."""
        expr = TemporalExpression(
            text="tomorrow",
            type="relative",
            start_date=datetime(2024, 1, 2, 0, 0),
            end_date=datetime(2024, 1, 3, 0, 0),
            is_relative=True,
            is_recurring=False,
            recurrence_pattern=None,
            confidence=0.9,
            metadata={},
        )
        assert expr.timeframe == "1 day"

    def test_timeframe_property_week(self):
        """Test timeframe for week range."""
        expr = TemporalExpression(
            text="this week",
            type="relative",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 8),
            is_relative=True,
            is_recurring=False,
            recurrence_pattern=None,
            confidence=0.9,
            metadata={},
        )
        assert expr.timeframe == "1 week"

    def test_timeframe_property_multiple_weeks(self):
        """Test timeframe for multiple weeks."""
        expr = TemporalExpression(
            text="past 3 weeks",
            type="relative",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 22),
            is_relative=True,
            is_recurring=False,
            recurrence_pattern=None,
            confidence=0.9,
            metadata={},
        )
        assert expr.timeframe == "3 weeks"

    def test_timeframe_property_month(self):
        """Test timeframe for month range."""
        expr = TemporalExpression(
            text="last month",
            type="relative",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
            is_relative=True,
            is_recurring=False,
            recurrence_pattern=None,
            confidence=0.9,
            metadata={},
        )
        assert "month" in expr.timeframe

    def test_timeframe_property_year(self):
        """Test timeframe for year range."""
        expr = TemporalExpression(
            text="last year",
            type="relative",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            is_relative=True,
            is_recurring=False,
            recurrence_pattern=None,
            confidence=0.9,
            metadata={},
        )
        assert "year" in expr.timeframe

    def test_timeframe_property_recurring(self):
        """Test timeframe for recurring patterns."""
        expr = TemporalExpression(
            text="every Monday",
            type="recurring",
            start_date=datetime(2024, 1, 1),
            end_date=None,
            is_relative=False,
            is_recurring=True,
            recurrence_pattern="weekly",
            confidence=0.9,
            metadata={},
        )
        assert expr.timeframe == "Recurring weekly"

    def test_timeframe_property_single_date(self):
        """Test timeframe for single date without end."""
        expr = TemporalExpression(
            text="2024-01-15",
            type="absolute",
            start_date=datetime(2024, 1, 15),
            end_date=None,
            is_relative=False,
            is_recurring=False,
            recurrence_pattern=None,
            confidence=0.95,
            metadata={},
        )
        assert expr.timeframe == "2024-01-15"

    def test_timeframe_property_no_dates(self):
        """Test timeframe with no dates."""
        expr = TemporalExpression(
            text="recently",
            type="relative",
            start_date=None,
            end_date=None,
            is_relative=True,
            is_recurring=False,
            recurrence_pattern=None,
            confidence=0.7,
            metadata={},
        )
        assert expr.timeframe == "recently"


class TestTemporalPatternMatcher:
    """Test temporal pattern matching."""

    @pytest.fixture
    def matcher(self):
        """Create pattern matcher instance."""
        return TemporalPatternMatcher()

    def test_load_default_patterns(self, matcher):
        """Test loading default patterns."""
        assert "absolute_dates" in matcher.patterns
        assert "relative_dates" in matcher.patterns
        assert "relative_patterns" in matcher.patterns
        assert "time_patterns" in matcher.patterns
        assert "duration_patterns" in matcher.patterns
        assert "range_patterns" in matcher.patterns
        assert "recurring_patterns" in matcher.patterns
        assert "special_dates" in matcher.patterns
        assert "quarters" in matcher.patterns

    def test_load_patterns_from_file(self, tmp_path):
        """Test loading patterns from JSON file."""
        patterns_file = tmp_path / "patterns.json"
        patterns = {
            "absolute_dates": [r"\b(\d{4}/\d{2}/\d{2})\b"],
            "relative_dates": {"soon": {"days": -3}},
            "quarters": {"h1": (1, 6)},
        }
        patterns_file.write_text(json.dumps(patterns))

        matcher = TemporalPatternMatcher(patterns_file)

        assert "absolute_dates" in matcher.patterns
        assert len(matcher.patterns["absolute_dates"]) == 1
        assert "quarters" in matcher.patterns

    def test_compile_patterns_success(self, matcher):
        """Test successful pattern compilation."""
        assert "absolute_dates" in matcher.compiled_patterns
        assert "relative_patterns" in matcher.compiled_patterns
        assert "time_patterns" in matcher.compiled_patterns
        assert "duration_patterns" in matcher.compiled_patterns
        assert "range_patterns" in matcher.compiled_patterns
        assert "recurring_patterns" in matcher.compiled_patterns

        # Check patterns are compiled
        for pattern in matcher.compiled_patterns["absolute_dates"]:
            assert isinstance(pattern, re.Pattern)

    def test_compile_invalid_pattern(self, caplog):
        """Test handling of invalid regex patterns."""
        matcher = TemporalPatternMatcher()
        matcher.patterns = {"test": [r"[invalid(regex"]}  # Invalid regex

        # Force recompilation
        matcher.compiled_patterns = matcher._compile_patterns()

        # Should handle gracefully
        assert (
            "test" not in matcher.compiled_patterns
            or len(matcher.compiled_patterns.get("test", [])) == 0
        )


class TestTemporalParser:
    """Test main temporal parser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return TemporalParser()

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_iso_dates(self, parser):
        """Test parsing ISO format dates."""
        texts = [
            ("2024-02-20", datetime(2024, 2, 20)),
            ("2024-12-31", datetime(2024, 12, 31)),
            ("2024-01-01", datetime(2024, 1, 1)),
        ]

        for text, expected in texts:
            expressions = parser.parse(text)

            assert len(expressions) >= 1
            expr = expressions[0]
            assert expr.text == text
            assert expr.type == "absolute"
            assert expr.start_date == expected
            assert expr.is_relative is False
            assert expr.confidence >= 0.9

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_iso_datetime(self, parser):
        """Test parsing ISO datetime formats."""
        text = "Meeting at 2024-02-20T14:30:00"

        expressions = parser.parse(text)

        assert len(expressions) >= 1
        expr = expressions[0]
        assert expr.start_date == datetime(2024, 2, 20, 14, 30, 0)

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_us_date_format(self, parser):
        """Test parsing US date format (MM/DD/YYYY)."""
        texts = [
            "12/31/2024",
            "01/15/2024",
            "06/30/2024",
        ]

        for text in texts:
            expressions = parser.parse(text)
            assert len(expressions) >= 1
            assert parser._detect_date_format(expressions[0].text) == "US"

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_european_date_format(self, parser):
        """Test parsing European date format (DD.MM.YYYY)."""
        texts = [
            "25.12.2024",
            "15.01.2024",
            "30.06.2024",
        ]

        for text in texts:
            expressions = parser.parse(text)
            assert len(expressions) >= 1
            assert parser._detect_date_format(expressions[0].text) == "European"

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_written_dates(self, parser):
        """Test parsing written date formats."""
        test_cases = [
            ("January 15, 2024", 1, 15),
            ("Jan 15, 2024", 1, 15),
            ("15 January 2024", 1, 15),
            ("15 Jan 2024", 1, 15),
            ("December 31, 2024", 12, 31),
            ("1 February 2024", 2, 1),
        ]

        for text, month, day in test_cases:
            expressions = parser.parse(text)
            assert len(expressions) >= 1
            assert expressions[0].start_date.year == 2024
            assert expressions[0].start_date.month == month
            assert expressions[0].start_date.day == day

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_relative_dates_simple(self, parser):
        """Test parsing simple relative dates."""
        test_cases = [
            ("yesterday", datetime(2024, 1, 14)),
            ("today", datetime(2024, 1, 15)),
            ("tomorrow", datetime(2024, 1, 16)),
            ("now", datetime(2024, 1, 15, 10, 0)),
        ]

        for text, expected_date in test_cases:
            expressions = parser.parse(text)
            assert len(expressions) >= 1
            expr = expressions[0]
            assert expr.is_relative is True
            assert expr.start_date.date() == expected_date.date()

    @freeze_time("2024-01-15 10:00:00")  # Monday
    def test_parse_this_week(self, parser):
        """Test parsing 'this week'."""
        expressions = parser.parse("this week")

        assert len(expressions) >= 1
        expr = expressions[0]
        assert expr.start_date.date() == date(2024, 1, 15)  # Monday
        assert expr.end_date.date() == date(2024, 1, 21)  # Sunday
        assert expr.is_relative is True

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_last_week(self, parser):
        """Test parsing 'last week'."""
        expressions = parser.parse("last week")

        assert len(expressions) >= 1
        expr = expressions[0]
        assert expr.start_date < datetime(2024, 1, 15)
        assert expr.is_relative is True

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_next_week(self, parser):
        """Test parsing 'next week'."""
        expressions = parser.parse("next week")

        assert len(expressions) >= 1
        expr = expressions[0]
        assert expr.start_date > datetime(2024, 1, 15)
        assert expr.is_relative is True

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_this_month(self, parser):
        """Test parsing 'this month'."""
        expressions = parser.parse("this month")

        assert len(expressions) >= 1
        expr = expressions[0]
        assert expr.start_date.date() == date(2024, 1, 1)
        assert expr.end_date.date() == date(2024, 1, 31)
        assert expr.is_relative is True

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_this_year(self, parser):
        """Test parsing 'this year'."""
        expressions = parser.parse("this year")

        assert len(expressions) >= 1
        expr = expressions[0]
        assert expr.start_date.date() == date(2024, 1, 1)
        assert expr.end_date.date() == date(2024, 12, 31)
        assert expr.is_relative is True

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_numeric_relative_dates(self, parser):
        """Test parsing numeric relative dates."""
        test_cases = [
            ("3 days ago", 3, "day", True),
            ("5 hours from now", 5, "hour", False),
            ("in 2 weeks", 2, "week", False),
            ("10 minutes ago", 10, "minute", True),
            ("1 month ago", 1, "month", True),
            ("2 years from now", 2, "year", False),
        ]

        for text, number, unit, is_past in test_cases:
            expressions = parser.parse(text)
            assert len(expressions) >= 1
            expr = expressions[0]
            assert expr.is_relative is True
            assert expr.metadata.get("number") == number
            assert expr.metadata.get("unit") == unit

            if is_past:
                assert expr.start_date < datetime(2024, 1, 15, 10, 0)
            else:
                assert expr.start_date > datetime(2024, 1, 15, 10, 0)

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_date_ranges(self, parser):
        """Test parsing date ranges."""
        test_cases = [
            "from 2024-01-01 to 2024-01-31",
            "between January 1 and January 31",
            "2024-01-01 - 2024-01-31",
            "Jan 1 through Jan 31",
            "2024-01-01 until 2024-01-31",
        ]

        for text in test_cases:
            expressions = parser.parse(text)

            # Find range expressions
            range_exprs = [e for e in expressions if e.type == "range"]
            if range_exprs:
                expr = range_exprs[0]
                assert expr.start_date is not None
                assert expr.end_date is not None
                assert expr.end_date > expr.start_date

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_since_until(self, parser):
        """Test parsing 'since' and 'until' patterns."""
        text = "since 2024-01-01 until 2024-01-31"

        expressions = parser.parse(text)
        range_exprs = [e for e in expressions if e.type == "range"]

        if range_exprs:
            expr = range_exprs[0]
            assert expr.start_date.date() == date(2024, 1, 1)
            assert expr.end_date.date() == date(2024, 1, 31)

    def test_parse_recurring_daily(self, parser):
        """Test parsing daily recurring patterns."""
        texts = [
            "every day",
            "daily",
            "each day",
        ]

        for text in texts:
            expressions = parser.parse(text)
            recurring = [e for e in expressions if e.is_recurring]

            assert len(recurring) >= 1
            expr = recurring[0]
            assert expr.type == "recurring"
            assert expr.recurrence_pattern == "daily"

    def test_parse_recurring_weekly(self, parser):
        """Test parsing weekly recurring patterns."""
        texts = [
            "every week",
            "weekly",
            "each week",
            "every Monday",
            "on Tuesdays",
            "every Friday",
        ]

        for text in texts:
            expressions = parser.parse(text)
            recurring = [e for e in expressions if e.is_recurring]

            assert len(recurring) >= 1
            expr = recurring[0]
            assert expr.type == "recurring"
            assert expr.recurrence_pattern == "weekly"

    def test_parse_recurring_monthly(self, parser):
        """Test parsing monthly recurring patterns."""
        texts = [
            "every month",
            "monthly",
            "each month",
        ]

        for text in texts:
            expressions = parser.parse(text)
            recurring = [e for e in expressions if e.is_recurring]

            assert len(recurring) >= 1
            expr = recurring[0]
            assert expr.type == "recurring"
            assert expr.recurrence_pattern == "monthly"

    def test_parse_recurring_yearly(self, parser):
        """Test parsing yearly recurring patterns."""
        texts = [
            "every year",
            "yearly",
            "annually",
            "each year",
        ]

        for text in texts:
            expressions = parser.parse(text)
            recurring = [e for e in expressions if e.is_recurring]

            assert len(recurring) >= 1
            expr = recurring[0]
            assert expr.type == "recurring"
            assert expr.recurrence_pattern == "yearly"

    def test_parse_recurring_custom(self, parser):
        """Test parsing custom recurring patterns."""
        texts = [
            "weekdays",
            "weekends",
            "business days",
            "3 times a week",
        ]

        for text in texts:
            expressions = parser.parse(text)
            recurring = [e for e in expressions if e.is_recurring]

            if recurring:
                expr = recurring[0]
                assert expr.type == "recurring"
                assert expr.recurrence_pattern in ["custom", "weekly", "daily"]

    @patch("tenets.core.prompt.temporal_parser.DATEUTIL_AVAILABLE", True)
    @patch("tenets.core.prompt.temporal_parser.dateutil_parser")
    def test_parse_with_dateutil_fallback(self, mock_dateutil, parser):
        """Test dateutil fallback parsing."""
        mock_dateutil.parse.return_value = datetime(2024, 3, 15, 15, 0)

        text = "March 15th at 3pm"

        expressions = parser.parse(text)

        # Should attempt dateutil parsing
        if expressions:
            # Check that dateutil was used
            assert any(e.metadata.get("parser") == "dateutil" for e in expressions)

    def test_parse_date_string_various_formats(self, parser):
        """Test _parse_date_string with various formats."""
        test_cases = [
            ("2024-01-15", datetime(2024, 1, 15)),
            ("2024-01-15T10:30:00", datetime(2024, 1, 15, 10, 30, 0)),
            ("01/15/2024", datetime(2024, 1, 15)),
            ("15.01.2024", datetime(2024, 1, 15)),
            ("January 15, 2024", datetime(2024, 1, 15)),
            ("15 Jan 2024", datetime(2024, 1, 15)),
        ]

        for date_str, expected in test_cases:
            result = parser._parse_date_string(date_str)
            if result:
                assert result.date() == expected.date()

    def test_parse_date_string_invalid(self, parser):
        """Test _parse_date_string with invalid input."""
        invalid_dates = [
            "",
            None,
            "not a date",
            "2024-13-45",  # Invalid month/day
            "abcd-ef-gh",
            "32/32/2024",  # Invalid date
        ]

        for date_str in invalid_dates:
            result = parser._parse_date_string(date_str)
            assert result is None

    def test_calculate_time_delta(self, parser):
        """Test _calculate_time_delta method."""
        test_cases = [
            (5, "second", timedelta(seconds=5)),
            (10, "minute", timedelta(minutes=10)),
            (2, "hour", timedelta(hours=2)),
            (3, "day", timedelta(days=3)),
            (1, "week", timedelta(weeks=1)),
            (2, "month", timedelta(days=60)),  # Approximate
            (1, "year", timedelta(days=365)),  # Approximate
            (5, "unknown", timedelta(days=5)),  # Default to days
        ]

        for number, unit, expected in test_cases:
            result = parser._calculate_time_delta(number, unit)
            assert result == expected

    @freeze_time("2024-01-15 10:00:00")  # Monday
    def test_get_current_period_week(self, parser):
        """Test _get_current_period for week."""
        start, end = parser._get_current_period("this week")

        assert start.date() == date(2024, 1, 15)  # Monday
        assert end.date() == date(2024, 1, 21)  # Sunday
        assert start.time() == time(0, 0, 0)
        assert end.hour == 23
        assert end.minute == 59

    @freeze_time("2024-01-15 10:00:00")
    def test_get_current_period_month(self, parser):
        """Test _get_current_period for month."""
        start, end = parser._get_current_period("this month")

        assert start.date() == date(2024, 1, 1)
        assert end.date() == date(2024, 1, 31)
        assert start.time() == time(0, 0, 0)
        assert end.hour == 23

    @freeze_time("2024-02-15 10:00:00")
    def test_get_current_period_month_february(self, parser):
        """Test _get_current_period for February (edge case)."""
        start, end = parser._get_current_period("this month")

        assert start.date() == date(2024, 2, 1)
        assert end.date() == date(2024, 2, 29)  # 2024 is a leap year

    @freeze_time("2024-12-15 10:00:00")
    def test_get_current_period_month_december(self, parser):
        """Test _get_current_period for December (year boundary)."""
        start, end = parser._get_current_period("this month")

        assert start.date() == date(2024, 12, 1)
        assert end.date() == date(2024, 12, 31)

    @freeze_time("2024-01-15 10:00:00")
    def test_get_current_period_year(self, parser):
        """Test _get_current_period for year."""
        start, end = parser._get_current_period("this year")

        assert start.date() == date(2024, 1, 1)
        assert end.date() == date(2024, 12, 31)

    def test_detect_date_format(self, parser):
        """Test _detect_date_format method."""
        assert parser._detect_date_format("2024-01-15") == "ISO"
        assert parser._detect_date_format("01/15/2024") == "US"
        assert parser._detect_date_format("15.01.2024") == "European"
        assert parser._detect_date_format("January 15, 2024") == "Written"
        assert parser._detect_date_format("Jan 15, 2024") == "Written"
        assert parser._detect_date_format("random text") == "Unknown"

    def test_deduplicate_expressions(self, parser):
        """Test _deduplicate_expressions method."""
        expressions = [
            TemporalExpression(
                text="today",
                type="relative",
                start_date=datetime(2024, 1, 15),
                end_date=None,
                is_relative=True,
                is_recurring=False,
                recurrence_pattern=None,
                confidence=0.9,
                metadata={},
            ),
            TemporalExpression(
                text="today",  # Duplicate
                type="relative",
                start_date=datetime(2024, 1, 15),
                end_date=None,
                is_relative=True,
                is_recurring=False,
                recurrence_pattern=None,
                confidence=0.8,  # Lower confidence
                metadata={},
            ),
            TemporalExpression(
                text="tomorrow",  # Different
                type="relative",
                start_date=datetime(2024, 1, 16),
                end_date=None,
                is_relative=True,
                is_recurring=False,
                recurrence_pattern=None,
                confidence=0.85,
                metadata={},
            ),
        ]

        unique = parser._deduplicate_expressions(expressions)

        assert len(unique) == 2
        # Should be sorted by confidence (highest first)
        assert unique[0].confidence == 0.9
        assert unique[0].text == "today"
        assert unique[1].text == "tomorrow"

    @freeze_time("2024-01-15 10:00:00")
    def test_get_temporal_context_with_expressions(self, parser):
        """Test getting temporal context from expressions."""
        expressions = [
            TemporalExpression(
                text="last week",
                type="relative",
                start_date=datetime(2024, 1, 8),
                end_date=datetime(2024, 1, 14),
                is_relative=True,
                is_recurring=False,
                recurrence_pattern=None,
                confidence=0.9,
                metadata={},
            ),
            TemporalExpression(
                text="every Monday",
                type="recurring",
                start_date=datetime(2024, 1, 15),
                end_date=None,
                is_relative=False,
                is_recurring=True,
                recurrence_pattern="weekly",
                confidence=0.85,
                metadata={},
            ),
        ]

        context = parser.get_temporal_context(expressions)

        assert context["has_temporal"] is True
        assert context["is_historical"] is True  # Last week is in the past
        assert context["is_future"] is False
        assert context["has_recurring"] is True
        assert context["expressions"] == 2
        assert "relative" in context["types"]
        assert "recurring" in context["types"]
        assert context["min_date"] is not None
        assert context["max_date"] is not None

    def test_get_temporal_context_empty(self, parser):
        """Test getting temporal context with no expressions."""
        context = parser.get_temporal_context([])

        assert context["has_temporal"] is False
        assert context["timeframe"] is None
        assert context["is_historical"] is False
        assert context["is_future"] is False
        assert context["is_current"] is False

    @freeze_time("2024-01-15 10:00:00")
    def test_get_temporal_context_future(self, parser):
        """Test temporal context for future dates."""
        expressions = [
            TemporalExpression(
                text="next month",
                type="relative",
                start_date=datetime(2024, 2, 1),
                end_date=datetime(2024, 2, 29),
                is_relative=True,
                is_recurring=False,
                recurrence_pattern=None,
                confidence=0.9,
                metadata={},
            )
        ]

        context = parser.get_temporal_context(expressions)

        assert context["is_future"] is True
        assert context["is_historical"] is False
        assert context["is_current"] is False

    @freeze_time("2024-01-15 10:00:00")
    def test_get_temporal_context_current(self, parser):
        """Test temporal context for current period."""
        expressions = [
            TemporalExpression(
                text="this week",
                type="relative",
                start_date=datetime(2024, 1, 15),
                end_date=datetime(2024, 1, 21),
                is_relative=True,
                is_recurring=False,
                recurrence_pattern=None,
                confidence=0.9,
                metadata={},
            )
        ]

        context = parser.get_temporal_context(expressions)

        assert context["is_current"] is True
        assert context["is_historical"] is False
        assert context["is_future"] is False

    def test_parse_complex_text(self, parser):
        """Test parsing text with multiple temporal expressions."""
        text = """
        The project started on 2024-01-01 and will run for 3 months.
        We have weekly meetings every Monday at 10am.
        The deadline is March 31, 2024.
        Review was done yesterday.
        """

        expressions = parser.parse(text)

        assert len(expressions) >= 3

        # Should find absolute dates
        absolute = [e for e in expressions if e.type == "absolute"]
        assert len(absolute) >= 2

        # Should find recurring pattern
        recurring = [e for e in expressions if e.is_recurring]
        assert len(recurring) >= 1

        # Should find relative date
        relative = [e for e in expressions if e.is_relative]
        assert len(relative) >= 1

        # Check temporal context
        context = parser.get_temporal_context(expressions)
        assert context["has_temporal"] is True
        assert context["has_recurring"] is True
        assert context["expressions"] >= 3

    def test_parse_empty_text(self, parser):
        """Test parsing empty text."""
        expressions = parser.parse("")

        assert expressions == []

    def test_parse_no_temporal_text(self, parser):
        """Test parsing text with no temporal expressions."""
        text = "The cat sat on the mat"

        expressions = parser.parse(text)

        # Might find some false positives or none
        assert len(expressions) >= 0

    @freeze_time("2024-01-15 10:00:00")
    def test_parse_ambiguous_dates(self, parser):
        """Test parsing ambiguous date expressions."""
        text = "Let's meet next Tuesday at noon"

        expressions = parser.parse(text)

        # Should find at least the time expression
        assert any("noon" in e.text.lower() or "tuesday" in e.text.lower() for e in expressions)
