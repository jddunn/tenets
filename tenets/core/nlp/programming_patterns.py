"""Centralized programming patterns loader for NLP.

This module loads programming patterns from the JSON file and provides
utilities for pattern matching. Consolidates duplicate logic from
parser.py and strategies.py.
"""

import json
import math
import re
from pathlib import Path
from typing import Optional, Tuple

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
                / "pattterns"  # Note: folder is misspelled in filesystem
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
            with open(patterns_file, encoding="utf-8") as f:
                data = json.load(f)
                self.logger.info(f"Loaded programming patterns from {patterns_file}")

                # Always convert to standardized format
                if "concepts" in data:
                    return self._convert_concepts_format(data["concepts"])
                else:
                    # Assume it's already in the right format
                    return data
        except FileNotFoundError:
            self.logger.warning(f"Patterns file not found: {patterns_file}, using defaults")
            return self._get_default_patterns()
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse patterns JSON: {e}, using defaults")
            return self._get_default_patterns()

    def _convert_concepts_format(self, concepts: Dict) -> Dict:
        """Convert concepts format to standardized pattern format.

        Args:
            concepts: Dictionary of concept -> keywords

        Returns:
            Standardized pattern dictionary
        """
        converted = {}
        for category, keywords in concepts.items():
            if isinstance(keywords, list):
                converted[category] = {
                    "keywords": keywords[:10],  # Top 10 keywords
                    "patterns": self._generate_patterns_for_keywords(
                        keywords[:5]
                    ),  # Pattern from top 5
                    "importance": self._calculate_importance(category),
                }
        return converted

    def _calculate_importance(self, category: str) -> float:
        """Calculate importance score for a category.

        Args:
            category: Category name

        Returns:
            Importance score between 0 and 1
        """
        # Core categories get higher importance
        high_importance = {"auth", "authentication", "security", "api", "database"}
        medium_importance = {"frontend", "backend", "testing", "configuration"}

        if category in high_importance:
            return 0.9
        elif category in medium_importance:
            return 0.7
        else:
            return 0.5

    def _generate_patterns_for_keywords(self, keywords: List[str]) -> List[str]:
        """Generate regex patterns from keywords.

        Args:
            keywords: List of keywords

        Returns:
            List of regex patterns
        """
        patterns = []
        for kw in keywords[:5]:  # Limit to avoid too many patterns
            # Create word boundary pattern for each keyword
            patterns.append(rf"\b{re.escape(kw)}\w*\b")
        return patterns

    def _get_default_patterns(self) -> Dict:
        """Get default patterns if JSON file not available.

        Returns:
            Dictionary of default programming patterns
        """
        return {
            "auth": {
                "keywords": [
                    "auth",
                    "authentication",
                    "login",
                    "oauth",
                    "jwt",
                    "token",
                    "session",
                    "credential",
                    "password",
                    "signin",
                ],
                "patterns": [
                    r"\bauth\w*\b",
                    r"\blogin\b",
                    r"\btoken\b",
                    r"\boauth\w*\b",
                    r"\bjwt\b",
                ],
                "importance": 0.9,
            },
            "api": {
                "keywords": [
                    "api",
                    "rest",
                    "endpoint",
                    "route",
                    "http",
                    "graphql",
                    "grpc",
                    "webhook",
                    "request",
                    "response",
                ],
                "patterns": [
                    r"\bapi\b",
                    r"\bendpoint\b",
                    r"\broute\b",
                    r"\brest\w*\b",
                    r"\bhttp\w*\b",
                ],
                "importance": 0.9,
            },
            "database": {
                "keywords": [
                    "database",
                    "db",
                    "sql",
                    "query",
                    "model",
                    "schema",
                    "migration",
                    "orm",
                    "transaction",
                    "index",
                ],
                "patterns": [
                    r"\bSELECT\b",
                    r"\bINSERT\b",
                    r"\.query\(",
                    r"\bdatabase\b",
                    r"\bdb\b",
                ],
                "importance": 0.8,
            },
            "testing": {
                "keywords": [
                    "test",
                    "testing",
                    "unit",
                    "integration",
                    "mock",
                    "stub",
                    "assertion",
                    "coverage",
                    "spec",
                    "tdd",
                ],
                "patterns": [r"\btest\w*\b", r"\bspec\b", r"\bmock\b", r"\bassert\w*\b"],
                "importance": 0.7,
            },
            "frontend": {
                "keywords": [
                    "ui",
                    "ux",
                    "frontend",
                    "component",
                    "react",
                    "vue",
                    "angular",
                    "dom",
                    "css",
                    "html",
                ],
                "patterns": [r"\bcomponent\b", r"\brender\b", r"\bdom\b", r"\bui\b"],
                "importance": 0.7,
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
                # Check if keyword appears as a substring in text
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

        Args:
            content: File content to analyze
            keywords: Keywords from prompt for relevance checking

        Returns:
            Dictionary of pattern scores by category
        """
        scores = {}

        # Lower case keywords for comparison
        keywords_lower = [kw.lower() for kw in keywords]

        for category, config in self.patterns.items():
            # Check if category is relevant to keywords
            category_keywords = config.get("keywords", [])

            # More sophisticated relevance check
            relevance_score = self._calculate_relevance(category_keywords, keywords_lower)

            if relevance_score > 0 and category in self.compiled_patterns:
                category_score = 0.0
                patterns = self.compiled_patterns[category]

                # Count pattern matches with better scoring
                for pattern in patterns:
                    matches = pattern.findall(content)
                    if matches:
                        # Use logarithmic scaling with base 2 for smoother curve
                        match_score = math.log2(len(matches) + 1) / math.log2(
                            11
                        )  # Normalized to ~1.0 at 10 matches
                        category_score += min(1.0, match_score)

                # Normalize and apply importance and relevance
                if patterns:
                    normalized_score = category_score / len(patterns)
                    importance = config.get("importance", 0.5)
                    # Include relevance in final score
                    scores[category] = normalized_score * importance * (0.5 + 0.5 * relevance_score)

        # Calculate overall pattern score as weighted average
        if scores:
            total_weight = sum(self.patterns[cat].get("importance", 0.5) for cat in scores)
            scores["overall"] = sum(
                scores[cat] * self.patterns[cat].get("importance", 0.5) / total_weight
                for cat in scores
                if cat != "overall"
            )
        else:
            scores["overall"] = 0.0

        return scores

    def _calculate_relevance(
        self, category_keywords: List[str], prompt_keywords: List[str]
    ) -> float:
        """Calculate relevance score between category and prompt keywords.

        Args:
            category_keywords: Keywords for the category
            prompt_keywords: Keywords from the prompt (lowercase)

        Returns:
            Relevance score between 0 and 1
        """
        if not category_keywords or not prompt_keywords:
            return 0.0

        matches = 0
        for cat_kw in category_keywords:
            cat_kw_lower = cat_kw.lower()
            # Check exact match or substring match
            if cat_kw_lower in prompt_keywords:
                matches += 1.0
            else:
                # Check partial matches
                for prompt_kw in prompt_keywords:
                    if cat_kw_lower in prompt_kw or prompt_kw in cat_kw_lower:
                        matches += 0.5
                        break

        # Return normalized score
        return min(1.0, matches / max(3, len(category_keywords) * 0.3))

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
        # Handle common aliases
        category_map = {
            "auth": "authentication",
            "config": "configuration",
            "db": "database",
        }
        actual_category = category_map.get(category, category)

        if actual_category in self.patterns:
            return self.patterns[actual_category].get("keywords", [])
        return []

    def get_category_importance(self, category: str) -> float:
        """Get importance score for a category.

        Args:
            category: Category name

        Returns:
            Importance score (0-1)
        """
        # Handle common aliases
        category_map = {
            "auth": "authentication",
            "config": "configuration",
            "db": "database",
        }
        actual_category = category_map.get(category, category)

        if actual_category in self.patterns:
            return self.patterns[actual_category].get("importance", 0.5)
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

        # Handle common aliases
        category_map = {
            "auth": "authentication",
            "config": "configuration",
            "db": "database",
        }
        actual_category = category_map.get(category, category)

        if actual_category in self.compiled_patterns:
            for pattern in self.compiled_patterns[actual_category]:
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
