"""Centralized programming patterns loader for NLP.

This module loads programming patterns from the JSON file and provides
utilities for pattern matching. Consolidates duplicate logic from
parser.py and strategies.py.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tenets.utils.logger import get_logger


class ProgrammingPatterns:
    """Loads and manages programming patterns from JSON.

    This class provides centralized access to programming patterns,
    eliminating duplication between parser.py and strategies.py.

    Attributes:
        patterns: Dictionary of pattern categories loaded from JSON
        logger: Logger instance
        compiled_patterns: Cache of compiled regex patterns
    """

    def __init__(self, patterns_file: Optional[Path] = None):
        """Initialize programming patterns from JSON file.

        Args:
            patterns_file: Path to patterns JSON file (uses default if None)
        """
        self.logger = get_logger(__name__)

        # Default patterns file location
        if patterns_file is None:
            patterns_file = (
                Path(__file__).parent.parent.parent
                / "data"
                / "patterns"
                / "programming_patterns.json"
            )

        self.patterns = self._load_patterns(patterns_file)
        self.compiled_patterns = {}
        self._compile_all_patterns()

    def _load_patterns(self, patterns_file: Path) -> Dict:
        """Load patterns from JSON file.

        Args:
            patterns_file: Path to JSON file

        Returns:
            Dictionary of programming patterns
        """
        try:
            with open(patterns_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.logger.info(f"Loaded programming patterns from {patterns_file}")
                return data
        except FileNotFoundError:
            self.logger.warning(f"Patterns file not found: {patterns_file}, using defaults")
            return self._get_default_patterns()
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse patterns JSON: {e}, using defaults")
            return self._get_default_patterns()

    def _get_default_patterns(self) -> Dict:
        """Get default patterns if JSON file not available.

        Returns:
            Dictionary of default programming patterns
        """
        return {
            "auth": {
                "keywords": ["auth", "login", "oauth", "jwt", "token", "session"],
                "patterns": [r"\bauth\w*\b", r"\blogin\b", r"\btoken\b"],
                "importance": 0.9,
            },
            "api": {
                "keywords": ["api", "rest", "endpoint", "route", "http"],
                "patterns": [r"\bapi\b", r"\bendpoint\b", r"\broute\b"],
                "importance": 0.85,
            },
            "database": {
                "keywords": ["database", "db", "sql", "query", "model"],
                "patterns": [r"\bSELECT\b", r"\bINSERT\b", r"\.query\("],
                "importance": 0.8,
            },
        }

    def _compile_all_patterns(self):
        """Compile all regex patterns for efficiency."""
        for category, config in self.patterns.items():
            if "patterns" in config:
                self.compiled_patterns[category] = []
                for pattern in config["patterns"]:
                    try:
                        compiled = re.compile(pattern, re.IGNORECASE)
                        self.compiled_patterns[category].append(compiled)
                    except re.error as e:
                        self.logger.warning(f"Invalid regex pattern in {category}: {pattern} - {e}")

    def extract_programming_keywords(self, text: str) -> List[str]:
        """Extract programming-specific keywords from text.

        This replaces the duplicate methods in parser.py and strategies.py.

        Args:
            text: Input text to extract keywords from

        Returns:
            List of unique programming keywords found
        """
        keywords = set()
        text_lower = text.lower()

        # Check each category
        for category, config in self.patterns.items():
            # Check if any category keywords appear in text
            category_keywords = config.get("keywords", [])
            for keyword in category_keywords:
                if keyword.lower() in text_lower:
                    keywords.add(keyword)

            # Check regex patterns
            if category in self.compiled_patterns:
                for pattern in self.compiled_patterns[category]:
                    if pattern.search(text):
                        # Add the category name as a keyword
                        keywords.add(category)
                        # Also add any matched keywords from this category
                        for keyword in category_keywords[:3]:  # Top 3 keywords
                            keywords.add(keyword)
                        break

        return sorted(list(keywords))

    def analyze_code_patterns(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """Analyze code for pattern matches and scoring.

        This replaces _analyze_code_patterns in strategies.py.

        Args:
            content: File content to analyze
            keywords: Keywords from prompt for relevance checking

        Returns:
            Dictionary of pattern scores by category
        """
        scores = {}

        for category, config in self.patterns.items():
            # Check if category is relevant to keywords
            category_keywords = config.get("keywords", [])
            relevant = any(kw.lower() in [k.lower() for k in keywords] for kw in category_keywords)

            if relevant and category in self.compiled_patterns:
                category_score = 0.0
                patterns = self.compiled_patterns[category]

                # Count pattern matches
                for pattern in patterns:
                    matches = len(pattern.findall(content))
                    if matches > 0:
                        # Logarithmic scaling for match count
                        category_score += min(1.0, matches / 10)

                # Normalize by number of patterns and apply importance
                if patterns:
                    normalized_score = category_score / len(patterns)
                    importance = config.get("importance", 0.5)
                    scores[f"pattern_{category}"] = normalized_score * importance

        # Calculate overall pattern score
        if scores:
            scores["overall"] = sum(scores.values()) / len(scores)
        else:
            scores["overall"] = 0.0

        return scores

    def get_pattern_categories(self) -> List[str]:
        """Get list of all pattern categories.

        Returns:
            List of category names
        """
        return list(self.patterns.keys())

    def get_category_keywords(self, category: str) -> List[str]:
        """Get keywords for a specific category.

        Args:
            category: Category name

        Returns:
            List of keywords for the category
        """
        if category in self.patterns:
            return self.patterns[category].get("keywords", [])
        return []

    def get_category_importance(self, category: str) -> float:
        """Get importance score for a category.

        Args:
            category: Category name

        Returns:
            Importance score (0-1)
        """
        if category in self.patterns:
            return self.patterns[category].get("importance", 0.5)
        return 0.5

    def match_patterns(self, text: str, category: str) -> List[Tuple[str, int, int]]:
        """Find all pattern matches in text for a category.

        Args:
            text: Text to search
            category: Pattern category

        Returns:
            List of (matched_text, start_pos, end_pos) tuples
        """
        matches = []

        if category in self.compiled_patterns:
            for pattern in self.compiled_patterns[category]:
                for match in pattern.finditer(text):
                    matches.append((match.group(), match.start(), match.end()))

        return matches


# Singleton instance for global access
_patterns_instance = None


def get_programming_patterns() -> ProgrammingPatterns:
    """Get singleton instance of programming patterns.

    Returns:
        ProgrammingPatterns instance
    """
    global _patterns_instance
    if _patterns_instance is None:
        _patterns_instance = ProgrammingPatterns()
    return _patterns_instance


def extract_programming_keywords(text: str) -> List[str]:
    """Convenience function to extract programming keywords.

    Args:
        text: Input text

    Returns:
        List of programming keywords
    """
    patterns = get_programming_patterns()
    return patterns.extract_programming_keywords(text)


def analyze_code_patterns(content: str, keywords: List[str]) -> Dict[str, float]:
    """Convenience function to analyze code patterns.

    Args:
        content: File content
        keywords: Prompt keywords

    Returns:
        Dictionary of pattern scores
    """
    patterns = get_programming_patterns()
    return patterns.analyze_code_patterns(content, keywords)
