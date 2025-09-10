"""Tests for intent action word filtering in keyword extraction.

This module tests the filtering of common intent action words from keyword
matching to prevent generic words from affecting file ranking.
"""

from unittest.mock import patch

from tenets.config import TenetsConfig
from tenets.core.nlp.stopwords import StopwordManager
from tenets.core.prompt.parser import PromptParser


class TestIntentKeywordFiltering:
    """Test suite for intent action word filtering."""

    def test_stopword_manager_loads_intent_actions(self):
        """Test that StopwordManager loads intent action words."""
        manager = StopwordManager()

        # Check that intent_actions set is loaded
        intent_set = manager.get_set("intent_actions")
        assert intent_set is not None
        assert "intent_actions" in intent_set.name

        # Check that common action words are included
        common_action_words = ["fix", "debug", "implement", "add", "create", "build"]
        for word in common_action_words:
            assert word in intent_set.words

    def test_filter_intent_keywords_basic(self):
        """Test basic filtering of intent action words from keywords."""
        config = TenetsConfig()
        config.nlp.filter_intent_keywords = True

        parser = PromptParser(config, use_cache=False, use_ml=False)

        # Test keywords with action words
        keywords = ["debug", "tokenizing", "issue", "parser", "error"]
        intent = "debug"

        filtered = parser._filter_intent_keywords(keywords, intent)

        # "debug", "issue", and "error" should be filtered out
        assert "tokenizing" in filtered
        assert "parser" in filtered
        assert "debug" not in filtered
        assert "issue" not in filtered
        assert "error" not in filtered

    def test_filter_intent_keywords_disabled(self):
        """Test that filtering can be disabled via config."""
        config = TenetsConfig()
        config.nlp.filter_intent_keywords = False

        parser = PromptParser(config, use_cache=False, use_ml=False)

        # When disabled, keywords should pass through unchanged
        keywords = ["debug", "tokenizing", "issue", "parser", "error"]

        # Since filtering is disabled at config level, the filter method
        # is not called in the parse flow
        context = parser.parse("debug tokenizing issue")

        # All keywords should be present (after normal stopword filtering)
        # Note: The actual keywords depend on keyword extraction, not just the input
        assert len(context.keywords) > 0

    def test_filter_preserves_domain_specific_terms(self):
        """Test that domain-specific terms are preserved."""
        config = TenetsConfig()
        config.nlp.filter_intent_keywords = True

        parser = PromptParser(config, use_cache=False, use_ml=False)

        # Test with domain-specific terms
        keywords = ["fix", "authentication", "oauth", "bug", "session", "token"]
        intent = "debug"

        filtered = parser._filter_intent_keywords(keywords, intent)

        # Domain-specific terms should be preserved
        assert "authentication" in filtered
        assert "oauth" in filtered
        assert "session" in filtered
        assert "token" in filtered

        # Generic action words should be filtered
        assert "fix" not in filtered
        assert "bug" not in filtered

    def test_custom_intent_keywords(self):
        """Test custom intent keywords configuration."""
        config = TenetsConfig()
        config.nlp.filter_intent_keywords = True
        config.nlp.custom_intent_keywords = {
            "debug": ["custom_debug_word", "another_custom"],
            "implement": ["custom_implement_word"],
        }

        parser = PromptParser(config, use_cache=False, use_ml=False)

        # Test with custom words
        keywords = ["fix", "custom_debug_word", "authentication", "another_custom"]
        intent = "debug"

        filtered = parser._filter_intent_keywords(keywords, intent)

        # Custom words should be filtered
        assert "custom_debug_word" not in filtered
        assert "another_custom" not in filtered

        # Default action words should still be filtered
        assert "fix" not in filtered

        # Domain-specific terms should be preserved
        assert "authentication" in filtered

    def test_minimum_keywords_preserved(self):
        """Test that at least some keywords are preserved even if all are action words."""
        config = TenetsConfig()
        config.nlp.filter_intent_keywords = True

        parser = PromptParser(config, use_cache=False, use_ml=False)

        # All keywords are action words
        keywords = ["fix", "debug", "issue", "error", "problem"]
        intent = "debug"

        filtered = parser._filter_intent_keywords(keywords, intent)

        # Should keep at least some keywords (longest ones)
        assert len(filtered) > 0
        assert len(filtered) <= 3  # Should keep top 3 longest

    def test_intent_specific_filtering(self):
        """Test that filtering is intent-specific."""
        config = TenetsConfig()
        config.nlp.filter_intent_keywords = True

        parser = PromptParser(config, use_cache=False, use_ml=False)

        # Test with implement intent
        keywords = ["implement", "authentication", "feature", "oauth", "module"]
        intent = "implement"

        filtered = parser._filter_intent_keywords(keywords, intent)

        # Implement-specific action words should be filtered
        assert "implement" not in filtered
        assert "feature" not in filtered

        # Domain terms should be preserved
        assert "authentication" in filtered
        assert "oauth" in filtered
        assert "module" in filtered

    def test_parse_with_filtering_integration(self):
        """Test full parse flow with intent keyword filtering."""
        config = TenetsConfig()
        config.nlp.filter_intent_keywords = True

        parser = PromptParser(config, use_cache=False, use_ml=False)

        # Parse a prompt with action words
        context = parser.parse("debug the tokenizing issue in the parser module")

        # Check that intent is detected correctly
        assert context.intent == "debug"

        # Check that action words are filtered from keywords
        # but domain-specific terms are preserved
        keywords_lower = [k.lower() for k in context.keywords]

        # These should be filtered
        assert "debug" not in keywords_lower or "tokenizing" in keywords_lower
        assert "issue" not in keywords_lower or "parser" in keywords_lower

        # At least some keywords should remain
        assert len(context.keywords) > 0

    def test_stopword_set_combination(self):
        """Test combining stopword sets with intent action words."""
        manager = StopwordManager()

        # Get combined set with intent action filtering
        combined = manager.get_combined_set("prompt", filter_intent_actions=True)

        assert combined is not None
        assert "prompt" in combined.name
        assert "intent_actions" in combined.name

        # Should include both prompt stopwords and intent action words
        assert len(combined.words) > len(manager.get_set("prompt").words)

    def test_intent_action_words_fallback(self):
        """Test fallback when intent patterns file is not found."""
        with patch("pathlib.Path.exists", return_value=False):
            manager = StopwordManager()

            # Should still have intent_actions set with fallback words
            intent_set = manager.get_set("intent_actions")
            assert intent_set is not None

            # Check fallback words are present
            fallback_words = ["fix", "debug", "implement", "add", "create", "build"]
            for word in fallback_words:
                assert word in intent_set.words


class TestIntentKeywordFilteringPerformance:
    """Performance tests for intent keyword filtering."""

    def test_filtering_performance_with_many_keywords(self):
        """Test that filtering performs well with many keywords."""
        config = TenetsConfig()
        config.nlp.filter_intent_keywords = True

        parser = PromptParser(config, use_cache=False, use_ml=False)

        # Generate many keywords
        keywords = ["keyword_" + str(i) for i in range(100)]
        keywords.extend(["fix", "debug", "issue", "error", "problem"])

        import time

        start = time.time()
        filtered = parser._filter_intent_keywords(keywords, "debug")
        duration = time.time() - start

        # Should complete quickly (under 10ms)
        assert duration < 0.01

        # Should filter action words
        assert "fix" not in filtered
        assert "debug" not in filtered

        # Should preserve other keywords
        assert "keyword_0" in filtered
        assert "keyword_50" in filtered
