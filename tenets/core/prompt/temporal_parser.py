"""Enhanced temporal parsing for dates, times, and ranges.

Supports multiple date formats, natural language expressions, recurring
patterns, and date ranges with comprehensive parsing capabilities.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenets.utils.logger import get_logger

# Try to import dateutil for advanced parsing
try:
    from dateutil import parser as dateutil_parser
    from dateutil.relativedelta import relativedelta

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    dateutil_parser = None
    relativedelta = None


@dataclass
class TemporalExpression:
    """Parsed temporal expression with metadata."""

    text: str  # Original text
    type: str  # 'absolute', 'relative', 'range', 'recurring'
    start_date: Optional[datetime]  # Start date/time
    end_date: Optional[datetime]  # End date/time (for ranges)
    is_relative: bool  # Whether it's relative to current time
    is_recurring: bool  # Whether it's a recurring pattern
    recurrence_pattern: Optional[str]  # Recurrence pattern (daily, weekly, etc.)
    confidence: float  # Confidence in parsing
    metadata: Dict[str, Any]  # Additional metadata

    @property
    def timeframe(self) -> str:
        """Get human-readable timeframe description."""
        if self.is_recurring:
            return f"Recurring {self.recurrence_pattern}"
        elif self.start_date and self.end_date:
            duration = self.end_date - self.start_date
            if duration.days == 1:
                return "1 day"
            elif duration.days < 7:
                return f"{duration.days} days"
            elif duration.days < 30:
                weeks = duration.days // 7
                return f"{weeks} week{'s' if weeks > 1 else ''}"
            elif duration.days < 365:
                months = duration.days // 30
                return f"{months} month{'s' if months > 1 else ''}"
            else:
                years = duration.days // 365
                return f"{years} year{'s' if years > 1 else ''}"
        elif self.start_date:
            return self.start_date.strftime("%Y-%m-%d")
        else:
            return self.text


class TemporalPatternMatcher:
    """Pattern-based temporal expression matching."""

    def __init__(self, patterns_file: Optional[Path] = None):
        """Initialize with temporal patterns.

        Args:
            patterns_file: Path to temporal patterns JSON file
        """
        self.logger = get_logger(__name__)
        self.patterns = self._load_patterns(patterns_file)
        self.compiled_patterns = self._compile_patterns()

    def _load_patterns(self, patterns_file: Optional[Path]) -> Dict[str, Any]:
        """Load temporal patterns from JSON file."""
        if patterns_file is None:
            patterns_file = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "patterns"
                / "temporal_patterns.json"
            )

        if patterns_file.exists():
            try:
                with open(patterns_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load temporal patterns: {e}")

        # Return default patterns if file not found
        return self._get_default_patterns()

    def _get_default_patterns(self) -> Dict[str, Any]:
        """Get comprehensive default temporal patterns."""
        return {
            "absolute_dates": [
                # ISO formats
                r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)\b",  # Full ISO
                r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\b",  # ISO without timezone
                r"\b(\d{4}-\d{2}-\d{2})\b",  # YYYY-MM-DD
                # US formats
                r"\b(\d{1,2}/\d{1,2}/\d{4})\b",  # MM/DD/YYYY
                r"\b(\d{1,2}/\d{1,2}/\d{2})\b",  # MM/DD/YY
                r"\b(\d{1,2}-\d{1,2}-\d{4})\b",  # MM-DD-YYYY
                # European formats
                r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b",  # DD.MM.YYYY
                r"\b(\d{1,2}\.\d{1,2}\.\d{2})\b",  # DD.MM.YY
                # Written formats
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b",
                r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
                r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b",
            ],
            "relative_dates": {
                # Time-based relatives
                "now": timedelta(seconds=0),
                "today": timedelta(days=0),
                "yesterday": timedelta(days=1),
                "tomorrow": timedelta(days=-1),
                "tonight": timedelta(hours=0),  # Special handling needed
                # Day-based relatives
                "day before yesterday": timedelta(days=2),
                "day after tomorrow": timedelta(days=-2),
                # Week-based relatives
                "this week": timedelta(days=0),  # Current week
                "last week": timedelta(weeks=1),
                "next week": timedelta(weeks=-1),
                "past week": timedelta(weeks=1),
                "previous week": timedelta(weeks=1),
                "coming week": timedelta(weeks=-1),
                # Month-based relatives
                "this month": timedelta(days=0),  # Current month
                "last month": timedelta(days=30),
                "next month": timedelta(days=-30),
                "past month": timedelta(days=30),
                "previous month": timedelta(days=30),
                # Year-based relatives
                "this year": timedelta(days=0),  # Current year
                "last year": timedelta(days=365),
                "next year": timedelta(days=-365),
                # Recent/upcoming
                "recently": timedelta(days=7),
                "recent": timedelta(days=7),
                "soon": timedelta(days=-7),
                "lately": timedelta(days=14),
                "earlier": timedelta(hours=3),
                "later": timedelta(hours=-3),
            },
            "relative_patterns": [
                # Numeric relative patterns
                r"\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b",
                r"\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+from\s+now\b",
                r"\bin\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bwithin\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bafter\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bbefore\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bpast\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bnext\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\blast\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bprevious\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
            ],
            "time_patterns": [
                # 12-hour format
                r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm)\b",
                r"\b(\d{1,2})\s*(AM|PM|am|pm)\b",
                # 24-hour format
                r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b",
                # Written times
                r"\b(noon|midnight|midday|dawn|dusk|morning|afternoon|evening|night)\b",
            ],
            "duration_patterns": [
                # Duration expressions
                r"\bfor\s+(?:the\s+)?(?:past|last|next|coming)?\s*(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bduring\s+(?:the\s+)?(?:past|last|next|coming)?\s*(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bthroughout\s+(?:the\s+)?(?:past|last|next|coming)?\s*(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
                r"\bover\s+(?:the\s+)?(?:past|last|next|coming)?\s*(\d+)\s+(second|minute|hour|day|week|month|year)s?\b",
            ],
            "range_patterns": [
                # Date ranges
                r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\b|$)",
                r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\b|$)",
                r"\b(.+?)\s*-\s*(.+?)(?:\b|$)",  # Date1 - Date2
                r"\b(.+?)\s+through\s+(.+?)(?:\b|$)",
                r"\b(.+?)\s+until\s+(.+?)(?:\b|$)",
                r"\bsince\s+(.+?)(?:\s+until\s+(.+?))?(?:\b|$)",
            ],
            "recurring_patterns": [
                # Recurring expressions
                r"\bevery\s+(second|minute|hour|day|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                r"\b(daily|weekly|monthly|yearly|annually|hourly)\b",
                r"\b(weekdays|weekends|workdays|business\s+days)\b",
                r"\bon\s+(mondays|tuesdays|wednesdays|thursdays|fridays|saturdays|sundays)\b",
                r"\b(\d+)\s+times?\s+(?:a|per)\s+(second|minute|hour|day|week|month|year)\b",
            ],
            "special_dates": {
                # Holidays and special dates
                "new year": "01-01",
                "new years": "01-01",
                "christmas": "12-25",
                "halloween": "10-31",
                "valentine": "02-14",
                "independence day": "07-04",
                "thanksgiving": "11-fourth-thursday",  # Special handling needed
                "easter": "varies",  # Special calculation needed
                "black friday": "11-friday-after-thanksgiving",
                "cyber monday": "11-monday-after-thanksgiving",
            },
            "quarters": {
                "q1": (1, 3),
                "q2": (4, 6),
                "q3": (7, 9),
                "q4": (10, 12),
                "first quarter": (1, 3),
                "second quarter": (4, 6),
                "third quarter": (7, 9),
                "fourth quarter": (10, 12),
                "last quarter": (10, 12),
            },
        }

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for efficiency."""
        compiled = {}

        # Compile absolute date patterns
        compiled["absolute_dates"] = []
        for pattern in self.patterns.get("absolute_dates", []):
            try:
                compiled["absolute_dates"].append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                self.logger.warning(f"Invalid date pattern: {pattern} - {e}")

        # Compile relative patterns
        compiled["relative_patterns"] = []
        for pattern in self.patterns.get("relative_patterns", []):
            try:
                compiled["relative_patterns"].append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                self.logger.warning(f"Invalid relative pattern: {pattern} - {e}")

        # Compile time patterns
        compiled["time_patterns"] = []
        for pattern in self.patterns.get("time_patterns", []):
            try:
                compiled["time_patterns"].append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                self.logger.warning(f"Invalid time pattern: {pattern} - {e}")

        # Compile duration patterns
        compiled["duration_patterns"] = []
        for pattern in self.patterns.get("duration_patterns", []):
            try:
                compiled["duration_patterns"].append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                self.logger.warning(f"Invalid duration pattern: {pattern} - {e}")

        # Compile range patterns
        compiled["range_patterns"] = []
        for pattern in self.patterns.get("range_patterns", []):
            try:
                compiled["range_patterns"].append(re.compile(pattern, re.IGNORECASE | re.DOTALL))
            except re.error as e:
                self.logger.warning(f"Invalid range pattern: {pattern} - {e}")

        # Compile recurring patterns
        compiled["recurring_patterns"] = []
        for pattern in self.patterns.get("recurring_patterns", []):
            try:
                compiled["recurring_patterns"].append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                self.logger.warning(f"Invalid recurring pattern: {pattern} - {e}")

        return compiled


class TemporalParser:
    """Main temporal parser combining all approaches."""

    def __init__(self, patterns_file: Optional[Path] = None):
        """Initialize temporal parser.

        Args:
            patterns_file: Path to temporal patterns JSON file
        """
        self.logger = get_logger(__name__)
        self.pattern_matcher = TemporalPatternMatcher(patterns_file)
        self.now = datetime.now()

    def parse(self, text: str) -> List[TemporalExpression]:
        """Parse temporal expressions from text.

        Args:
            text: Text to parse

        Returns:
            List of temporal expressions
        """
        expressions = []

        # 1. Check for absolute dates
        absolute_exprs = self._parse_absolute_dates(text)
        expressions.extend(absolute_exprs)

        # 2. Check for relative dates
        relative_exprs = self._parse_relative_dates(text)
        expressions.extend(relative_exprs)

        # 3. Check for date ranges
        range_exprs = self._parse_date_ranges(text)
        expressions.extend(range_exprs)

        # 4. Check for recurring patterns
        recurring_exprs = self._parse_recurring_patterns(text)
        expressions.extend(recurring_exprs)

        # 5. Try dateutil parser if available
        if DATEUTIL_AVAILABLE and not expressions:
            dateutil_exprs = self._parse_with_dateutil(text)
            expressions.extend(dateutil_exprs)

        # Remove duplicates and sort by position
        unique_expressions = self._deduplicate_expressions(expressions)

        return unique_expressions

    def _parse_absolute_dates(self, text: str) -> List[TemporalExpression]:
        """Parse absolute date expressions."""
        expressions = []

        for pattern in self.pattern_matcher.compiled_patterns.get("absolute_dates", []):
            for match in pattern.finditer(text):
                date_str = match.group(0)
                parsed_date = self._parse_date_string(date_str)

                if parsed_date:
                    expr = TemporalExpression(
                        text=date_str,
                        type="absolute",
                        start_date=parsed_date,
                        end_date=None,
                        is_relative=False,
                        is_recurring=False,
                        recurrence_pattern=None,
                        confidence=0.95,
                        metadata={
                            "format": self._detect_date_format(date_str),
                            "position": match.span(),
                        },
                    )
                    expressions.append(expr)

        return expressions

    def _parse_relative_dates(self, text: str) -> List[TemporalExpression]:
        """Parse relative date expressions."""
        expressions = []
        text_lower = text.lower()

        # Check for simple relative dates
        for phrase, delta in self.pattern_matcher.patterns.get("relative_dates", {}).items():
            if phrase in text_lower:
                # Find position
                pos = text_lower.find(phrase)

                # Calculate date based on delta
                if phrase in ["this week", "this month", "this year"]:
                    # Special handling for "this" periods
                    start_date, end_date = self._get_current_period(phrase)
                else:
                    # Regular delta calculation
                    target_date = self.now - delta
                    start_date = target_date
                    end_date = None

                expr = TemporalExpression(
                    text=phrase,
                    type="relative",
                    start_date=start_date,
                    end_date=end_date,
                    is_relative=True,
                    is_recurring=False,
                    recurrence_pattern=None,
                    confidence=0.9,
                    metadata={
                        "position": (pos, pos + len(phrase)),
                        "delta_days": delta.days if isinstance(delta, timedelta) else 0,
                    },
                )
                expressions.append(expr)

        # Check for numeric relative patterns (e.g., "3 days ago")
        for pattern in self.pattern_matcher.compiled_patterns.get("relative_patterns", []):
            for match in pattern.finditer(text):
                full_text = match.group(0)

                # Extract number and unit
                groups = match.groups()
                if len(groups) >= 2:
                    number = int(groups[0])
                    unit = groups[1].lower()

                    # Calculate delta
                    delta = self._calculate_time_delta(number, unit)

                    # Determine if it's past or future
                    if any(
                        word in full_text.lower() for word in ["ago", "past", "last", "previous"]
                    ):
                        target_date = self.now - delta
                    else:
                        target_date = self.now + delta

                    expr = TemporalExpression(
                        text=full_text,
                        type="relative",
                        start_date=target_date,
                        end_date=None,
                        is_relative=True,
                        is_recurring=False,
                        recurrence_pattern=None,
                        confidence=0.85,
                        metadata={"position": match.span(), "number": number, "unit": unit},
                    )
                    expressions.append(expr)

        return expressions

    def _parse_date_ranges(self, text: str) -> List[TemporalExpression]:
        """Parse date range expressions."""
        expressions = []

        for pattern in self.pattern_matcher.compiled_patterns.get("range_patterns", []):
            for match in pattern.finditer(text):
                full_text = match.group(0)
                groups = match.groups()

                if len(groups) >= 2:
                    start_text = groups[0]
                    end_text = groups[1] if groups[1] else None

                    # Parse start and end dates
                    start_date = self._parse_date_string(start_text)
                    end_date = self._parse_date_string(end_text) if end_text else None

                    # If we couldn't parse as dates, try as relative expressions
                    if not start_date:
                        start_exprs = self._parse_relative_dates(start_text)
                        if start_exprs:
                            start_date = start_exprs[0].start_date

                    if end_text and not end_date:
                        end_exprs = self._parse_relative_dates(end_text)
                        if end_exprs:
                            end_date = end_exprs[0].start_date

                    if start_date:
                        expr = TemporalExpression(
                            text=full_text,
                            type="range",
                            start_date=start_date,
                            end_date=end_date,
                            is_relative=False,
                            is_recurring=False,
                            recurrence_pattern=None,
                            confidence=0.8,
                            metadata={
                                "position": match.span(),
                                "start_text": start_text,
                                "end_text": end_text,
                            },
                        )
                        expressions.append(expr)

        return expressions

    def _parse_recurring_patterns(self, text: str) -> List[TemporalExpression]:
        """Parse recurring pattern expressions."""
        expressions = []

        for pattern in self.pattern_matcher.compiled_patterns.get("recurring_patterns", []):
            for match in pattern.finditer(text):
                full_text = match.group(0)

                # Determine recurrence pattern
                pattern_text = full_text.lower()
                if "daily" in pattern_text or "every day" in pattern_text:
                    recurrence = "daily"
                elif "weekly" in pattern_text or "every week" in pattern_text:
                    recurrence = "weekly"
                elif "monthly" in pattern_text or "every month" in pattern_text:
                    recurrence = "monthly"
                elif "yearly" in pattern_text or "annually" in pattern_text:
                    recurrence = "yearly"
                elif "hourly" in pattern_text or "every hour" in pattern_text:
                    recurrence = "hourly"
                elif any(
                    day in pattern_text
                    for day in [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                        "saturday",
                        "sunday",
                    ]
                ):
                    recurrence = "weekly"
                else:
                    recurrence = "custom"

                expr = TemporalExpression(
                    text=full_text,
                    type="recurring",
                    start_date=self.now,  # Start from now
                    end_date=None,
                    is_relative=False,
                    is_recurring=True,
                    recurrence_pattern=recurrence,
                    confidence=0.85,
                    metadata={"position": match.span(), "pattern": pattern_text},
                )
                expressions.append(expr)

        return expressions

    def _parse_with_dateutil(self, text: str) -> List[TemporalExpression]:
        """Try to parse with dateutil as fallback."""
        expressions = []

        if not DATEUTIL_AVAILABLE:
            return expressions

        # Try to find date-like substrings
        words = text.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 5, len(words) + 1)):  # Max 5 words
                phrase = " ".join(words[i:j])

                try:
                    parsed = dateutil_parser.parse(phrase, fuzzy=False)

                    expr = TemporalExpression(
                        text=phrase,
                        type="absolute",
                        start_date=parsed,
                        end_date=None,
                        is_relative=False,
                        is_recurring=False,
                        recurrence_pattern=None,
                        confidence=0.7,  # Lower confidence for fuzzy parsing
                        metadata={"parser": "dateutil", "fuzzy": True},
                    )
                    expressions.append(expr)
                    break  # Found a date, skip overlapping phrases

                except (ValueError, TypeError):
                    continue

        return expressions

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse a date string into datetime."""
        if not date_str:
            return None

        # Try common formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%m/%d/%Y",
            "%m/%d/%y",
            "%d.%m.%Y",
            "%d.%m.%y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        # Try dateutil if available
        if DATEUTIL_AVAILABLE:
            try:
                return dateutil_parser.parse(date_str)
            except (ValueError, TypeError):
                pass

        return None

    def _calculate_time_delta(self, number: int, unit: str) -> timedelta:
        """Calculate timedelta from number and unit."""
        unit = unit.lower().rstrip("s")  # Remove plural

        if unit in ["second", "sec"]:
            return timedelta(seconds=number)
        elif unit in ["minute", "min"]:
            return timedelta(minutes=number)
        elif unit in ["hour", "hr"]:
            return timedelta(hours=number)
        elif unit in ["day"]:
            return timedelta(days=number)
        elif unit in ["week", "wk"]:
            return timedelta(weeks=number)
        elif unit in ["month", "mon"]:
            return timedelta(days=number * 30)  # Approximate
        elif unit in ["year", "yr"]:
            return timedelta(days=number * 365)  # Approximate
        else:
            return timedelta(days=number)  # Default to days

    def _get_current_period(self, period: str) -> Tuple[datetime, datetime]:
        """Get start and end dates for current period."""
        if "week" in period:
            # Get current week (Monday to Sunday)
            weekday = self.now.weekday()
            start = self.now - timedelta(days=weekday)
            end = start + timedelta(days=6)
        elif "month" in period:
            # Get current month
            start = self.now.replace(day=1)
            # Get last day of month
            if self.now.month == 12:
                end = self.now.replace(year=self.now.year + 1, month=1, day=1) - timedelta(days=1)
            else:
                end = self.now.replace(month=self.now.month + 1, day=1) - timedelta(days=1)
        elif "year" in period:
            # Get current year
            start = self.now.replace(month=1, day=1)
            end = self.now.replace(month=12, day=31)
        else:
            start = self.now
            end = self.now

        # Set times to start/end of day
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = end.replace(hour=23, minute=59, second=59, microsecond=999999)

        return start, end

    def _detect_date_format(self, date_str: str) -> str:
        """Detect the format of a date string."""
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            return "ISO"
        elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
            return "US"
        elif re.match(r"\d{1,2}\.\d{1,2}\.\d{4}", date_str):
            return "European"
        elif re.match(r"[A-Za-z]+ \d{1,2}, \d{4}", date_str):
            return "Written"
        else:
            return "Unknown"

    def _deduplicate_expressions(
        self, expressions: List[TemporalExpression]
    ) -> List[TemporalExpression]:
        """Remove duplicate temporal expressions."""
        seen = set()
        unique = []

        for expr in expressions:
            # Create a key for deduplication
            key = (expr.text.lower(), expr.type, expr.start_date, expr.end_date)

            if key not in seen:
                seen.add(key)
                unique.append(expr)

        # Sort by confidence (highest first)
        unique.sort(key=lambda x: x.confidence, reverse=True)

        return unique

    def get_temporal_context(self, expressions: List[TemporalExpression]) -> Dict[str, Any]:
        """Get overall temporal context from expressions.

        Args:
            expressions: List of temporal expressions

        Returns:
            Temporal context summary
        """
        if not expressions:
            return {
                "has_temporal": False,
                "timeframe": None,
                "is_historical": False,
                "is_future": False,
                "is_current": False,
            }

        # Find overall timeframe
        all_dates = []
        for expr in expressions:
            if expr.start_date:
                all_dates.append(expr.start_date)
            if expr.end_date:
                all_dates.append(expr.end_date)

        if all_dates:
            min_date = min(all_dates)
            max_date = max(all_dates)

            # Determine temporal orientation
            is_historical = max_date < self.now
            is_future = min_date > self.now
            is_current = min_date <= self.now <= max_date

            # Calculate timeframe
            if min_date == max_date:
                timeframe = min_date.strftime("%Y-%m-%d")
            else:
                duration = max_date - min_date
                if duration.days == 0:
                    timeframe = "today"
                elif duration.days == 1:
                    timeframe = "1 day"
                elif duration.days < 7:
                    timeframe = f"{duration.days} days"
                elif duration.days < 30:
                    timeframe = f"{duration.days // 7} weeks"
                elif duration.days < 365:
                    timeframe = f"{duration.days // 30} months"
                else:
                    timeframe = f"{duration.days // 365} years"
        else:
            is_historical = False
            is_future = False
            is_current = True
            timeframe = "unspecified"

        # Check for recurring patterns
        has_recurring = any(expr.is_recurring for expr in expressions)

        return {
            "has_temporal": True,
            "timeframe": timeframe,
            "is_historical": is_historical,
            "is_future": is_future,
            "is_current": is_current,
            "has_recurring": has_recurring,
            "expressions": len(expressions),
            "types": list(set(expr.type for expr in expressions)),
            "min_date": min_date.isoformat() if all_dates else None,
            "max_date": max_date.isoformat() if all_dates else None,
        }
