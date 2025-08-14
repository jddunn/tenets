"""Tests for the summarizer module with NLP integration."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from tenets.config import TenetsConfig
from tenets.core.summarizer.summarizer import (
    FileSummarizer,
    Summarizer,
    SummarizationMode,
    SummarizationResult,
    BatchSummarizationResult
)
from tenets.core.summarizer.strategies import (
    ExtractiveStrategy,
    CompressiveStrategy,
    TextRankStrategy,
    NLPEnhancedStrategy
)
from tenets.models.analysis import FileAnalysis, CodeStructure
from tenets.models.summary import FileSummary


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    config.summarizer.enable_ml_strategies = False  # Disable ML in tests
    return config


@pytest.fixture
def summarizer(config):
    """Create Summarizer instance."""
    return Summarizer(config=config, enable_cache=False)


@pytest.fixture
def file_summarizer():
    """Create FileSummarizer instance."""
    return FileSummarizer(model="gpt-4")


@pytest.fixture
def temp_files(tmp_path):
    """Create temporary test files."""
    # Python file with docstring
    python_file = tmp_path / "module.py"
    python_file.write_text(
        '''
"""This is a module docstring.

This module provides utility functions for
processing data and handling files.
"""

import os
import sys

def process_data(data):
    """Process the input data."""
    return data * 2

class DataProcessor:
    """A class for processing data."""
    
    def __init__(self):
        self.data = []
    
    def add(self, item):
        """Add an item to the processor."""
        self.data.append(item)
'''
    )

    # Python file with comments
    comment_file = tmp_path / "commented.py"
    comment_file.write_text(
        """
# This file contains utility functions
# for string manipulation and text processing
#
# Author: Test User
# Date: 2024-01-01

def reverse_string(s):
    return s[::-1]

def capitalize_words(text):
    return ' '.join(word.capitalize() for word in text.split())
"""
    )

    # Large text for summarization
    large_text_file = tmp_path / "large.txt"
    large_text = " ".join([f"Sentence {i}. This is content for sentence {i}." for i in range(100)])
    large_text_file.write_text(large_text)

    return {
        "python": python_file,
        "comment": comment_file,
        "large_text": large_text_file,
    }


class TestSummarizer:
    """Test suite for main Summarizer class."""
    
    def test_initialization(self, config):
        """Test summarizer initialization."""
        summarizer = Summarizer(config=config)
        
        assert summarizer.config == config
        assert summarizer.default_mode == SummarizationMode.AUTO
        assert len(summarizer.strategies) > 0
        assert SummarizationMode.EXTRACTIVE in summarizer.strategies
        assert SummarizationMode.COMPRESSIVE in summarizer.strategies
        
    def test_summarize_empty_text(self, summarizer):
        """Test summarizing empty text."""
        result = summarizer.summarize("")
        
        assert isinstance(result, SummarizationResult)
        assert result.summary == ""
        assert result.compression_ratio == 1.0
        assert result.strategy_used == "none"
        
    def test_summarize_extractive(self, summarizer):
        """Test extractive summarization."""
        text = "This is the first sentence. This is the second sentence. This is the third sentence. This is the fourth sentence."
        
        result = summarizer.summarize(
            text,
            mode=SummarizationMode.EXTRACTIVE,
            target_ratio=0.5
        )
        
        assert isinstance(result, SummarizationResult)
        assert len(result.summary) < len(text)
        assert result.strategy_used == "extractive"
        assert result.compression_ratio < 1.0
        
    def test_summarize_compressive(self, summarizer):
        """Test compressive summarization."""
        text = "The quick brown fox jumps over the lazy dog. The dog was sleeping under the tree."
        
        result = summarizer.summarize(
            text,
            mode=SummarizationMode.COMPRESSIVE,
            target_ratio=0.5
        )
        
        assert isinstance(result, SummarizationResult)
        assert len(result.summary) <= len(text)
        assert result.strategy_used == "compressive"
        
    @patch('tenets.core.summarizer.strategies.SKLEARN_AVAILABLE', True)
    def test_summarize_textrank(self, summarizer):
        """Test TextRank summarization."""
        text = " ".join([f"Sentence {i}." for i in range(10)])
        
        # Need to manually add TextRank strategy if sklearn available
        try:
            summarizer.strategies[SummarizationMode.TEXTRANK] = TextRankStrategy()
            
            result = summarizer.summarize(
                text,
                mode=SummarizationMode.TEXTRANK,
                target_ratio=0.3
            )
            
            assert isinstance(result, SummarizationResult)
            assert result.strategy_used == "textrank"
        except ImportError:
            # Skip if sklearn not available
            pytest.skip("scikit-learn not available")
            
    def test_summarize_with_max_length(self, summarizer):
        """Test summarization with max length constraint."""
        text = " ".join(["Long text content"] * 50)
        
        result = summarizer.summarize(
            text,
            mode=SummarizationMode.EXTRACTIVE,
            target_ratio=0.5,
            max_length=100
        )
        
        assert len(result.summary) <= 100
        
    def test_summarize_with_min_length(self, summarizer):
        """Test summarization with min length constraint."""
        text = "Short text."
        
        result = summarizer.summarize(
            text,
            mode=SummarizationMode.EXTRACTIVE,
            target_ratio=0.1,
            min_length=20
        )
        
        # Should maintain minimum length
        assert len(result.summary) >= len(text)  # Can't be shorter than original if min > original
        
    def test_summarize_file(self, summarizer, temp_files):
        """Test file summarization."""
        file_analysis = FileAnalysis(
            path=str(temp_files["python"]),
            content=temp_files["python"].read_text(),
            language="python",
            size=temp_files["python"].stat().st_size,
            lines=temp_files["python"].read_text().count("\n")
        )
        
        result = summarizer.summarize_file(
            file_analysis,
            target_ratio=0.5,
            preserve_structure=True
        )
        
        assert isinstance(result, SummarizationResult)
        assert result.strategy_used == "code-aware"
        assert "def" in result.summary or "class" in result.summary
        
    def test_auto_mode_selection(self, summarizer):
        """Test automatic mode selection."""
        # Short text should use extractive
        short_text = "Short text content."
        result = summarizer.summarize(short_text, mode=SummarizationMode.AUTO)
        assert result.strategy_used == "extractive"
        
        # Code-like content should use extractive
        code_text = "def function():\n    return True\nclass MyClass:\n    pass"
        result = summarizer.summarize(code_text, mode=SummarizationMode.AUTO)
        assert result.strategy_used == "extractive"
        
    def test_caching(self):
        """Test result caching."""
        summarizer = Summarizer(enable_cache=True)
        text = "Test text for caching."
        
        # First call
        result1 = summarizer.summarize(text, target_ratio=0.5)
        cache_misses = summarizer.stats["cache_misses"]
        
        # Second call should hit cache
        result2 = summarizer.summarize(text, target_ratio=0.5)
        cache_hits = summarizer.stats["cache_hits"]
        
        assert cache_hits > 0
        assert result1.summary == result2.summary
        
    def test_force_strategy(self, summarizer):
        """Test forcing specific strategy."""
        text = "Test text content."
        custom_strategy = Mock()
        custom_strategy.summarize.return_value = "Custom summary"
        custom_strategy.name = "custom"
        
        result = summarizer.summarize(
            text,
            force_strategy=custom_strategy
        )
        
        assert result.strategy_used == "custom"
        assert result.summary == "Custom summary"
        custom_strategy.summarize.assert_called_once()
        
    def test_error_handling(self, summarizer):
        """Test error handling with fallback."""
        text = "Test text content."
        
        # Mock strategy to fail
        with patch.object(summarizer.strategies[SummarizationMode.EXTRACTIVE], 'summarize') as mock_summarize:
            mock_summarize.side_effect = Exception("Strategy failed")
            
            result = summarizer.summarize(
                text,
                mode=SummarizationMode.EXTRACTIVE,
                target_ratio=0.5
            )
            
            # Should fall back to truncation
            assert result.strategy_used == "truncate"
            assert len(result.summary) <= len(text)
            
    def test_get_stats(self, summarizer):
        """Test getting statistics."""
        # Perform some summarizations
        summarizer.summarize("Test 1")
        summarizer.summarize("Test 2")
        
        stats = summarizer.get_stats()
        
        assert stats["total_summarized"] == 2
        assert "avg_time" in stats
        assert "cache_hit_rate" in stats
        assert "strategies_used" in stats


class TestFileSummarizer:
    """Test suite for FileSummarizer (backward compatibility)."""
    
    def test_initialization(self):
        """Test FileSummarizer initialization."""
        summarizer = FileSummarizer(model="gpt-4")
        assert summarizer.model == "gpt-4"
        
    def test_summarize_file_with_docstring(self, file_summarizer, temp_files):
        """Test summarizing Python file with docstring."""
        summary = file_summarizer.summarize_file(temp_files["python"])
        
        assert isinstance(summary, FileSummary)
        assert summary.path == str(temp_files["python"])
        assert "module docstring" in summary.summary
        assert summary.token_count > 0
        assert summary.metadata["strategy"] == "heuristic"
        
    def test_summarize_file_with_comments(self, file_summarizer, temp_files):
        """Test summarizing file with comments."""
        summary = file_summarizer.summarize_file(temp_files["comment"])
        
        assert "utility functions" in summary.summary
        assert "string manipulation" in summary.summary
        
    def test_summarize_nonexistent_file(self, file_summarizer, tmp_path):
        """Test summarizing non-existent file."""
        nonexistent = tmp_path / "does_not_exist.txt"
        
        summary = file_summarizer.summarize_file(nonexistent)
        
        assert summary.summary == ""
        assert summary.token_count == 0
        
    def test_max_lines_parameter(self, file_summarizer, temp_files):
        """Test max_lines parameter."""
        summary_10 = file_summarizer.summarize_file(temp_files["large_text"], max_lines=10)
        summary_50 = file_summarizer.summarize_file(temp_files["large_text"], max_lines=50)
        
        lines_10 = summary_10.summary.count("\n")
        lines_50 = summary_50.summary.count("\n")
        
        assert lines_10 <= 10
        assert lines_50 <= 50


class TestStrategiesWithNLP:
    """Test suite for summarization strategies with NLP integration."""
    
    @patch('tenets.core.summarizer.strategies.NLP_AVAILABLE', True)
    def test_extractive_with_nlp(self):
        """Test extractive strategy with NLP components."""
        strategy = ExtractiveStrategy(use_nlp=True)
        
        text = "Python programming is powerful. Machine learning with Python is popular. Data science uses Python extensively."
        summary = strategy.summarize(text, target_ratio=0.5)
        
        assert len(summary) < len(text)
        # Should preserve important sentences
        assert "Python" in summary
        
    @patch('tenets.core.summarizer.strategies.NLP_AVAILABLE', False)
    def test_extractive_without_nlp(self):
        """Test extractive strategy without NLP."""
        strategy = ExtractiveStrategy(use_nlp=False)
        
        text = "First sentence. Second sentence. Third sentence."
        summary = strategy.summarize(text, target_ratio=0.5)
        
        assert len(summary) < len(text)
        
    @patch('tenets.core.summarizer.strategies.NLP_AVAILABLE', True)
    def test_compressive_with_nlp(self):
        """Test compressive strategy with NLP."""
        strategy = CompressiveStrategy(use_nlp=True)
        
        text = "The quick brown fox jumps over the lazy dog. The fox was very quick."
        summary = strategy.summarize(text, target_ratio=0.5)
        
        assert len(summary) <= len(text)
        
    @patch('tenets.core.summarizer.strategies.NLP_AVAILABLE', True)
    @patch('tenets.core.summarizer.strategies.SKLEARN_AVAILABLE', True)
    def test_textrank_with_nlp(self):
        """Test TextRank strategy with NLP."""
        strategy = TextRankStrategy(use_nlp=True)
        
        text = " ".join([f"Sentence {i} contains information." for i in range(10)])
        summary = strategy.summarize(text, target_ratio=0.3)
        
        assert len(summary) < len(text)
        
    @patch('tenets.core.summarizer.strategies.NLP_AVAILABLE', True)
    @patch('tenets.core.nlp.embeddings.create_embedding_model')
    def test_nlp_enhanced_strategy(self, mock_create_model):
        """Test NLP-enhanced strategy."""
        # Mock embedding model
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        try:
            strategy = NLPEnhancedStrategy()
            
            text = "Machine learning is transforming industries. Deep learning models are powerful. Natural language processing enables text understanding."
            summary = strategy.summarize(text, target_ratio=0.5)
            
            assert len(summary) <= len(text)
        except ImportError:
            pytest.skip("NLP components not available")


class TestBatchSummarization:
    """Test suite for batch summarization."""
    
    def test_batch_summarize_texts(self, summarizer):
        """Test batch summarization of texts."""
        texts = [
            "First text content.",
            "Second text content.",
            "Third text content."
        ]
        
        result = summarizer.batch_summarize(texts, target_ratio=0.5)
        
        assert isinstance(result, BatchSummarizationResult)
        assert result.files_processed == 3
        assert result.files_failed == 0
        assert result.overall_compression_ratio < 1.0
        
    def test_batch_summarize_files(self, summarizer):
        """Test batch summarization of FileAnalysis objects."""
        files = [
            FileAnalysis(
                path=f"file{i}.txt",
                content=f"Content for file {i}",
                size=20,
                lines=1
            )
            for i in range(3)
        ]
        
        result = summarizer.batch_summarize(files, target_ratio=0.5)
        
        assert isinstance(result, BatchSummarizationResult)
        assert result.files_processed == 3
        
    def test_batch_summarize_with_failures(self, summarizer):
        """Test batch summarization with some failures."""
        texts = ["Valid text", None, "Another valid text"]
        
        with patch.object(summarizer, 'summarize') as mock_summarize:
            # Make None cause an error
            def side_effect(text, **kwargs):
                if text is None:
                    raise Exception("Invalid input")
                return Mock(
                    original_length=len(text),
                    summary_length=len(text) // 2,
                    summary=text[:len(text)//2]
                )
            
            mock_summarize.side_effect = side_effect
            
            result = summarizer.batch_summarize(texts)
            
            # One should fail
            assert mock_summarize.call_count == 3


class TestCodeSummarization:
    """Test suite for code-specific summarization."""
    
    def test_summarize_python_code(self, summarizer):
        """Test Python code summarization."""
        code = '''
import os
import sys

class MyClass:
    """A test class."""
    
    def __init__(self):
        self.value = 0
        
    def process(self, data):
        """Process the data."""
        # Complex implementation here
        result = data * 2
        for i in range(10):
            result += i
        return result

def main():
    """Main function."""
    obj = MyClass()
    return obj.process(10)
'''
        
        file_analysis = FileAnalysis(
            path="test.py",
            content=code,
            language="python",
            size=len(code),
            lines=code.count("\n"),
            structure=CodeStructure(
                classes=[Mock(name="MyClass", definition="class MyClass:")],
                functions=[Mock(name="main", signature="def main():")]
            )
        )
        
        result = summarizer.summarize_file(
            file_analysis,
            target_ratio=0.3,
            preserve_structure=True
        )
        
        assert "import" in result.summary
        assert "class" in result.summary or "def" in result.summary
        assert result.strategy_used == "code-aware"
        
    def test_code_detection(self, summarizer):
        """Test code detection in auto mode."""
        code = "def function():\n    return True\nif __name__ == '__main__':\n    function()"
        
        result = summarizer.summarize(code, mode=SummarizationMode.AUTO)
        
        # Should detect as code and use extractive
        assert result.strategy_used == "extractive"


class TestUtilityFunctions:
    """Test suite for utility functions in summarizer module."""
    
    def test_simple_truncate(self, summarizer):
        """Test simple truncation fallback."""
        text = "This is a long text that needs to be truncated at some point."
        
        truncated = summarizer._simple_truncate(text, target_ratio=0.5, max_length=None)
        
        assert len(truncated) <= len(text)
        assert truncated.endswith("...") or truncated.endswith(".")
        
    def test_is_import_line(self, summarizer):
        """Test import line detection."""
        assert summarizer._is_import_line("import os", "python")
        assert summarizer._is_import_line("from os import path", "python")
        assert summarizer._is_import_line("import React from 'react'", "javascript")
        assert summarizer._is_import_line("#include <stdio.h>", "c")
        assert summarizer._is_import_line("use std::io", "rust")
        assert not summarizer._is_import_line("# import comment", "python")
        
    def test_extract_comments(self, summarizer):
        """Test comment extraction."""
        python_code = '''
# This is a comment
def function():
    """This is a docstring."""
    # Another comment
    return True
'''
        
        comments = summarizer._extract_comments(python_code, "python")
        
        assert "comment" in comments.lower()
        assert "docstring" in comments.lower()
        
    def test_cache_key_generation(self, summarizer):
        """Test cache key generation."""
        key1 = summarizer._get_cache_key("text", 0.5, 100, 10)
        key2 = summarizer._get_cache_key("text", 0.5, 100, 10)
        key3 = summarizer._get_cache_key("different", 0.5, 100, 10)
        
        assert key1 == key2  # Same parameters
        assert key1 != key3  # Different text