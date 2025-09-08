"""Simplified programming patterns loader for NLP.

This module provides a working version that avoids the import hang issue.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tenets.utils.logger import get_logger


class ProgrammingPatterns:
    """Loads and manages programming patterns from JSON."""

    def __init__(self, patterns_file: Optional[Path] = None):
        """Initialize programming patterns from JSON file."""
        self.logger = get_logger(__name__)
        self.patterns = {}
        self.compiled_patterns = {}
        
        # Load patterns immediately to ensure compiled_patterns is populated
        self._patterns_file = patterns_file
        self._load_patterns()

    def _load_patterns(self):
        """Load patterns from file or use defaults."""
        if self._patterns_file is None:
            self._patterns_file = (
                Path(__file__).parent.parent.parent
                / "data"
                / "patterns"
                / "programming_patterns.json"
            )
        
        try:
            if self._patterns_file.exists():
                with open(self._patterns_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if "concepts" in data:
                        # Simple conversion without complex logic
                        self.patterns = {k: {"keywords": v[:10] if isinstance(v, list) else [], 
                                            "patterns": [], "importance": 0.5} 
                                       for k, v in data.get("concepts", {}).items()}
                    else:
                        self.patterns = data
            else:
                self.patterns = self._get_default_patterns()
        except Exception as e:
            self.logger.debug(f"Could not load patterns: {e}")
            self.patterns = self._get_default_patterns()
        
        # Compile regex patterns for each category
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.compiled_patterns = {}
        for category, data in self.patterns.items():
            if isinstance(data, dict):
                compiled = []
                # Compile patterns if they exist
                if "patterns" in data and data["patterns"]:
                    for pattern in data["patterns"]:
                        try:
                            compiled.append(re.compile(pattern, re.IGNORECASE))
                        except re.error:
                            self.logger.debug(f"Invalid regex pattern: {pattern}")
                # If no patterns, create patterns from keywords
                elif "keywords" in data:
                    for keyword in data["keywords"]:
                        try:
                            # Create word boundary patterns for keywords
                            pattern = r'\b' + re.escape(keyword) + r'\b'
                            compiled.append(re.compile(pattern, re.IGNORECASE))
                        except re.error:
                            pass
                self.compiled_patterns[category] = compiled

    def _get_default_patterns(self) -> Dict:
        """Get minimal default patterns."""
        return {
            "auth": {
                "keywords": ["auth", "authenticate", "login", "oauth", "jwt", "token", "password"],
                "patterns": [r"\bauth\w*\b", r"\blogin\b", r"\boauth\d?\b", r"\bjwt\b"],
                "importance": 0.8
            },
            "api": {
                "keywords": ["api", "endpoint", "rest", "http", "route", "request", "response"],
                "patterns": [r"\bapi\b", r"\bendpoint\b", r"\brest\b", r"\bhttp\w*\b"],
                "importance": 0.7
            },
            "database": {
                "keywords": ["database", "sql", "query", "table", "schema", "migration"],
                "patterns": [r"\bdb\b", r"\bsql\b", r"\bquery\b", r"\btable\b"],
                "importance": 0.6
            },
            "testing": {
                "keywords": ["test", "mock", "assert", "spec", "unit", "integration"],
                "patterns": [r"\btest\w*\b", r"\bmock\w*\b", r"\bassert\w*\b"],
                "importance": 0.5
            },
        }

    def _ensure_loaded(self):
        """Ensure patterns are loaded (for backward compatibility)."""
        if not self.patterns:
            self._load_patterns()

    def extract_programming_keywords(self, text: str) -> List[str]:
        """Extract programming-specific keywords from text."""
        self._ensure_loaded()
        if not text:
            return []
        
        keywords = []
        text_lower = text.lower()
        
        for category_data in self.patterns.values():
            if isinstance(category_data, dict) and "keywords" in category_data:
                for keyword in category_data["keywords"]:
                    if keyword.lower() in text_lower:
                        keywords.append(keyword)
        
        return keywords[:30]  # Limit output

    def analyze_code_patterns(self, content: str, keywords: List[str]) -> Dict[str, float]:
        """Analyze code patterns in content."""
        self._ensure_loaded()
        scores = {}
        content_lower = content.lower()
        
        # Calculate scores for each category
        for category, data in self.patterns.items():
            if isinstance(data, dict):
                score = 0.0
                
                # Check keywords
                if "keywords" in data:
                    keyword_matches = sum(1 for kw in data["keywords"] if kw.lower() in content_lower)
                    score = keyword_matches / max(len(data["keywords"]), 1)
                
                # Check compiled patterns
                if category in self.compiled_patterns:
                    pattern_matches = sum(1 for pattern in self.compiled_patterns[category] 
                                        if pattern.search(content))
                    if self.compiled_patterns[category]:
                        pattern_score = pattern_matches / len(self.compiled_patterns[category])
                        score = max(score, pattern_score)
                
                scores[category] = min(score, 1.0)  # Cap at 1.0
        
        # Calculate overall score
        if scores:
            scores["overall"] = sum(scores.values()) / len(scores)
        else:
            scores["overall"] = 0.0
        
        return scores

    def get_pattern_categories(self) -> List[str]:
        """Get list of available pattern categories."""
        self._ensure_loaded()
        return list(self.patterns.keys())

    def get_category_keywords(self, category: str) -> List[str]:
        """Get keywords for a specific category."""
        self._ensure_loaded()
        if category in self.patterns:
            data = self.patterns[category]
            if isinstance(data, dict) and "keywords" in data:
                return data["keywords"]
        return []

    def get_category_importance(self, category: str) -> float:
        """Get importance score for a category."""
        self._ensure_loaded()
        if category in self.patterns:
            data = self.patterns[category]
            if isinstance(data, dict) and "importance" in data:
                return data["importance"]
        return 0.5  # Default importance

    def match_patterns(self, text: str, category: str) -> List[Tuple[str, int, int]]:
        """Match patterns for a specific category in text.
        
        Returns list of (matched_text, start_pos, end_pos) tuples.
        """
        self._ensure_loaded()
        matches = []
        
        if category in self.compiled_patterns:
            for pattern in self.compiled_patterns[category]:
                for match in pattern.finditer(text):
                    matches.append((match.group(), match.start(), match.end()))
        
        return matches


# Singleton instance - but create lazily to avoid import hang
_patterns_instance = None


def get_programming_patterns() -> ProgrammingPatterns:
    """Get singleton instance of programming patterns."""
    global _patterns_instance
    if _patterns_instance is None:
        _patterns_instance = ProgrammingPatterns()
    return _patterns_instance


def extract_programming_keywords(text: str) -> List[str]:
    """Convenience function to extract programming keywords."""
    patterns = get_programming_patterns()
    return patterns.extract_programming_keywords(text)


def analyze_code_patterns(content: str, keywords: List[str]) -> Dict[str, float]:
    """Convenience function to analyze code patterns."""
    patterns = get_programming_patterns()
    return patterns.analyze_code_patterns(content, keywords)