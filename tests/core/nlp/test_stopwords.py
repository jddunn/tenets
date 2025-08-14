"""Tests for stopword management."""

import pytest
from pathlib import Path

from tenets.core.nlp.stopwords import StopwordSet, StopwordManager


class TestStopwordSet:
    """Test suite for StopwordSet."""
    
    def test_initialization(self):
        """Test StopwordSet initialization."""
        words = {"the", "a", "an", "is", "are"}
        stopword_set = StopwordSet(
            name="test",
            words=words,
            description="Test stopwords"
        )
        
        assert stopword_set.name == "test"
        assert stopword_set.words == words
        assert stopword_set.description == "Test stopwords"
        
    def test_contains(self):
        """Test checking if word is in set."""
        stopword_set = StopwordSet(
            name="test",
            words={"the", "a", "an"},
            description="Test"
        )
        
        assert "the" in stopword_set
        assert "THE" in stopword_set  # Case insensitive
        assert "python" not in stopword_set
        
    def test_filter(self):
        """Test filtering words."""
        stopword_set = StopwordSet(
            name="test",
            words={"the", "a", "is"},
            description="Test"
        )
        
        words = ["the", "quick", "brown", "fox", "is", "fast"]
        filtered = stopword_set.filter(words)
        
        assert "the" not in filtered
        assert "is" not in filtered
        assert "quick" in filtered
        assert "brown" in filtered
        assert len(filtered) == 4


class TestStopwordManager:
    """Test suite for StopwordManager."""
    
    def test_initialization(self):
        """Test StopwordManager initialization."""
        manager = StopwordManager()
        
        # Should have default sets loaded
        code_set = manager.get_set('code')
        prompt_set = manager.get_set('prompt')
        
        assert code_set is not None
        assert prompt_set is not None
        
    def test_get_set(self):
        """Test getting stopword sets."""
        manager = StopwordManager()
        
        code_set = manager.get_set('code')
        assert isinstance(code_set, StopwordSet)
        assert code_set.name == 'code'
        
        # Non-existent set
        assert manager.get_set('nonexistent') is None
        
    def test_add_custom_set(self):
        """Test adding custom stopword set."""
        manager = StopwordManager()
        
        custom_words = {"foo", "bar", "baz"}
        custom_set = manager.add_custom_set(
            name="custom",
            words=custom_words,
            description="Custom stopwords"
        )
        
        assert custom_set.name == "custom"
        assert custom_set.words == {w.lower() for w in custom_words}
        
        # Should be retrievable
        retrieved = manager.get_set("custom")
        assert retrieved == custom_set
        
    def test_combine_sets(self):
        """Test combining stopword sets."""
        manager = StopwordManager()
        
        # Add custom sets
        manager.add_custom_set("set1", {"a", "b", "c"}, "Set 1")
        manager.add_custom_set("set2", {"c", "d", "e"}, "Set 2")
        
        # Combine them
        combined = manager.combine_sets(["set1", "set2"], name="combined")
        
        assert combined.name == "combined"
        assert combined.words == {"a", "b", "c", "d", "e"}
        assert "Combined from" in combined.description
        
    def test_code_stopwords(self):
        """Test code stopword set has expected words."""
        manager = StopwordManager()
        code_set = manager.get_set('code')
        
        # Should have minimal stopwords for code
        common_words = ["the", "a", "is", "are", "to", "of", "and", "or", "in"]
        for word in common_words:
            assert word in code_set.words
            
        # Should be minimal set
        assert len(code_set.words) < 50  # Minimal set
        
    def test_prompt_stopwords(self):
        """Test prompt stopword set."""
        manager = StopwordManager()
        prompt_set = manager.get_set('prompt')
        
        # Should have more aggressive stopwords
        assert len(prompt_set.words) > len(manager.get_set('code').words)
        
        # Should include action words
        action_words = ["make", "create", "implement", "add", "get", "set"]
        for word in action_words:
            if word in prompt_set.words:
                assert word in prompt_set.words
                
    def test_load_from_file(self, tmp_path):
        """Test loading stopwords from file."""
        # Create test file
        stopword_file = tmp_path / "test_stopwords.txt"
        stopword_file.write_text(
            "# Comment line\n"
            "the\n"
            "a\n"
            "an\n"
            "\n"  # Empty line
            "is\n"
            "are\n"
        )
        
        manager = StopwordManager(data_dir=tmp_path)
        loaded_set = manager._load_set_from_file(
            stopword_file,
            name="test",
            description="Test set"
        )
        
        assert loaded_set.name == "test"
        assert "the" in loaded_set.words
        assert "a" in loaded_set.words
        assert "is" in loaded_set.words
        assert "#" not in str(loaded_set.words)  # Comments excluded
        
    def test_case_insensitive(self):
        """Test case insensitive handling."""
        manager = StopwordManager()
        
        custom_set = manager.add_custom_set(
            "test",
            {"The", "A", "AN"},  # Mixed case
            "Test"
        )
        
        # Should be stored lowercase
        assert "the" in custom_set.words
        assert "a" in custom_set.words
        assert "an" in custom_set.words
        
        # Should match case-insensitive
        assert "THE" in custom_set
        assert "The" in custom_set
        assert "the" in custom_set