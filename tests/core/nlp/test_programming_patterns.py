"""Tests for programming patterns utilities."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from tenets.core.nlp.programming_patterns import (
    ProgrammingPatterns,
    get_programming_patterns,
    extract_programming_keywords,
    analyze_code_patterns,
)


class TestProgrammingPatterns:
    """Test suite for ProgrammingPatterns."""

    def test_initialization_default(self):
        """Test initialization with default patterns."""
        patterns = ProgrammingPatterns()

        assert patterns.patterns is not None
        assert isinstance(patterns.patterns, dict)
        assert len(patterns.compiled_patterns) > 0

    def test_initialization_custom_file(self, tmp_path):
        """Test initialization with custom patterns file."""
        patterns_file = tmp_path / "patterns.json"
        patterns_data = {
            "test_pattern": {
                "keywords": ["test", "unit", "mock"],
                "patterns": [r"\\btest_\\w+\\b"],
                "importance": 0.7,
            }
        }
        patterns_file.write_text(json.dumps(patterns_data))

        patterns = ProgrammingPatterns(patterns_file)

        assert "test_pattern" in patterns.patterns
        assert patterns.patterns["test_pattern"]["keywords"] == ["test", "unit", "mock"]

    def test_load_patterns_file_not_found(self):
        """Test loading with non-existent file."""
        patterns = ProgrammingPatterns(Path("/nonexistent/file.json"))

        # Should use defaults
        assert patterns.patterns is not None
        assert "auth" in patterns.patterns  # Default pattern

    def test_extract_programming_keywords(self):
        """Test extracting programming keywords."""
        patterns = ProgrammingPatterns()

        text = "Implement OAuth2 authentication with JWT tokens for the REST API endpoint"
        keywords = patterns.extract_programming_keywords(text)

        assert isinstance(keywords, list)
        # Should find auth-related keywords
        assert any("auth" in kw.lower() or "oauth" in kw.lower() for kw in keywords)

    def test_extract_keywords_empty(self):
        """Test extracting from empty text."""
        patterns = ProgrammingPatterns()

        keywords = patterns.extract_programming_keywords("")
        assert keywords == []

    def test_analyze_code_patterns(self):
        """Test analyzing code patterns."""
        patterns = ProgrammingPatterns()

        code = '''
        def authenticate_user(username, password):
            """Authenticate user with OAuth2."""
            token = generate_jwt_token(username)
            return token
            
        @app.route('/api/login')
        def login():
            return authenticate_user()
        '''

        keywords = ["auth", "login", "api"]
        scores = patterns.analyze_code_patterns(code, keywords)

        assert isinstance(scores, dict)
        assert "overall" in scores
        # Should detect auth and API patterns
        assert scores["overall"] > 0

    def test_get_pattern_categories(self):
        """Test getting pattern categories."""
        patterns = ProgrammingPatterns()

        categories = patterns.get_pattern_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0
        assert all(isinstance(cat, str) for cat in categories)

    def test_get_category_keywords(self):
        """Test getting keywords for category."""
        patterns = ProgrammingPatterns()

        # Assuming 'auth' is a default category
        keywords = patterns.get_category_keywords("auth")

        assert isinstance(keywords, list)
        assert len(keywords) > 0

        # Non-existent category
        keywords = patterns.get_category_keywords("nonexistent")
        assert keywords == []

    def test_get_category_importance(self):
        """Test getting category importance."""
        patterns = ProgrammingPatterns()

        importance = patterns.get_category_importance("auth")
        assert isinstance(importance, float)
        assert 0 <= importance <= 1

        # Non-existent category should return default
        importance = patterns.get_category_importance("nonexistent")
        assert importance == 0.5

    def test_match_patterns(self):
        """Test pattern matching."""
        patterns = ProgrammingPatterns()

        text = "User authentication is handled by the login() function"

        # Assuming 'auth' category has auth-related patterns
        matches = patterns.match_patterns(text, "auth")

        assert isinstance(matches, list)
        # Each match should be (matched_text, start, end)
        for match in matches:
            assert isinstance(match, tuple)
            assert len(match) == 3
            assert isinstance(match[0], str)
            assert isinstance(match[1], int)
            assert isinstance(match[2], int)

    def test_compiled_patterns(self):
        """Test pattern compilation."""
        patterns = ProgrammingPatterns()

        # Should have compiled patterns
        assert len(patterns.compiled_patterns) > 0

        # Each category should have compiled regex patterns
        for category, pattern_list in patterns.compiled_patterns.items():
            assert isinstance(pattern_list, list)
            # Each pattern should be compiled regex
            for pattern in pattern_list:
                assert hasattr(pattern, "match")  # Regex object

    def test_invalid_regex_handling(self):
        """Test handling of invalid regex patterns."""
        patterns_data = {
            "bad_pattern": {
                "keywords": ["test"],
                "patterns": ["[invalid(regex"],  # Invalid regex
                "importance": 0.5,
            }
        }

        with patch("builtins.open", mock_open(read_data=json.dumps(patterns_data))):
            patterns = ProgrammingPatterns(Path("dummy.json"))

            # Should handle invalid regex gracefully
            assert (
                "bad_pattern" not in patterns.compiled_patterns
                or len(patterns.compiled_patterns["bad_pattern"]) == 0
            )


class TestGlobalFunctions:
    """Test suite for global utility functions."""

    def test_get_programming_patterns_singleton(self):
        """Test singleton pattern instance."""
        patterns1 = get_programming_patterns()
        patterns2 = get_programming_patterns()

        # Should return same instance
        assert patterns1 is patterns2

    def test_extract_programming_keywords_global(self):
        """Test global keyword extraction."""
        text = "Build a REST API with authentication"
        keywords = extract_programming_keywords(text)

        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_analyze_code_patterns_global(self):
        """Test global pattern analysis."""
        code = "def api_endpoint():\n    return authenticate()"
        keywords = ["api", "auth"]

        scores = analyze_code_patterns(code, keywords)

        assert isinstance(scores, dict)
        assert "overall" in scores
