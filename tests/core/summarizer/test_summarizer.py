"""Tests for file summarizer."""

import pytest
import tempfile
import ast
from pathlib import Path
from unittest.mock import Mock, patch

from tenets.core.summarizer.summarizer import FileSummarizer
from tenets.models.summary import FileSummary


@pytest.fixture
def summarizer():
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

    # File without docstring
    no_docstring = tmp_path / "no_doc.py"
    no_docstring.write_text(
        """
import json

def load_config(path):
    with open(path) as f:
        return json.load(f)
        
def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f)
"""
    )

    # JavaScript file
    js_file = tmp_path / "app.js"
    js_file.write_text(
        """
// Main application controller
// Handles user interactions and state management

class AppController {
    constructor() {
        this.state = {};
    }
    
    initialize() {
        // Initialize the application
        console.log("App initialized");
    }
}

export default AppController;
"""
    )

    # Text file
    text_file = tmp_path / "readme.txt"
    text_file.write_text(
        """
This is a README file for the project.

## Features
- Feature 1
- Feature 2
- Feature 3

## Installation
Run `pip install package`

## Usage
Import and use the module as needed.
"""
    )

    # Empty file
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("")

    # Large file
    large_file = tmp_path / "large.py"
    large_content = "\n".join([f"# Line {i}" for i in range(1000)])
    large_file.write_text(large_content)

    # Non-UTF8 file
    binary_file = tmp_path / "binary.dat"
    binary_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")

    return {
        "python": python_file,
        "comment": comment_file,
        "no_docstring": no_docstring,
        "javascript": js_file,
        "text": text_file,
        "empty": empty_file,
        "large": large_file,
        "binary": binary_file,
    }


class TestFileSummarizer:
    """Test suite for FileSummarizer."""

    def test_initialization(self):
        """Test FileSummarizer initialization."""
        summarizer = FileSummarizer(model="gpt-4")
        assert summarizer.model == "gpt-4"

        summarizer_no_model = FileSummarizer()
        assert summarizer_no_model.model is None

    def test_summarize_python_with_docstring(self, summarizer, temp_files):
        """Test summarizing Python file with module docstring."""
        summary = summarizer.summarize_file(temp_files["python"])

        assert isinstance(summary, FileSummary)
        assert summary.path == str(temp_files["python"])
        assert "module docstring" in summary.summary
        assert "utility functions" in summary.summary
        assert summary.token_count > 0
        assert summary.metadata["strategy"] == "heuristic"

    def test_summarize_python_with_comments(self, summarizer, temp_files):
        """Test summarizing Python file with leading comments."""
        summary = summarizer.summarize_file(temp_files["comment"])

        assert "utility functions" in summary.summary
        assert "string manipulation" in summary.summary
        # Comments should be extracted
        assert "Author" in summary.summary or "Test User" in summary.summary

    def test_summarize_no_docstring(self, summarizer, temp_files):
        """Test summarizing file without docstring."""
        summary = summarizer.summarize_file(temp_files["no_docstring"], max_lines=10)

        # Should fall back to first N lines
        assert "import json" in summary.summary
        assert "load_config" in summary.summary

    def test_summarize_javascript(self, summarizer, temp_files):
        """Test summarizing JavaScript file."""
        summary = summarizer.summarize_file(temp_files["javascript"])

        # Should extract leading comments
        assert (
            "application controller" in summary.summary.lower()
            or "Main application" in summary.summary
        )

    def test_summarize_text_file(self, summarizer, temp_files):
        """Test summarizing plain text file."""
        summary = summarizer.summarize_file(temp_files["text"], max_lines=5)

        assert "README" in summary.summary
        assert "Features" in summary.summary

    def test_summarize_empty_file(self, summarizer, temp_files):
        """Test summarizing empty file."""
        summary = summarizer.summarize_file(temp_files["empty"])

        assert summary.summary == ""
        assert summary.token_count == 0

    def test_summarize_large_file(self, summarizer, temp_files):
        """Test summarizing large file with max_lines limit."""
        summary = summarizer.summarize_file(temp_files["large"], max_lines=10)

        # Should only include first 10 lines
        lines = summary.summary.split("\n")
        assert len(lines) <= 10
        assert "Line 0" in summary.summary
        assert "Line 999" not in summary.summary

    def test_read_text_encoding(self, summarizer, tmp_path):
        """Test reading files with different encodings."""
        # UTF-8 file
        utf8_file = tmp_path / "utf8.txt"
        utf8_file.write_text("UTF-8 content: 你好", encoding="utf-8")

        text = summarizer._read_text(utf8_file)
        assert "你好" in text

        # Latin-1 file
        latin1_file = tmp_path / "latin1.txt"
        latin1_file.write_text("Latin-1: café", encoding="latin-1")

        text = summarizer._read_text(latin1_file)
        assert "café" in text or "caf" in text  # Might lose accent

    def test_read_text_binary_file(self, summarizer, temp_files):
        """Test reading binary file."""
        text = summarizer._read_text(temp_files["binary"])

        # Should handle gracefully, possibly empty or garbled
        assert isinstance(text, str)

    def test_read_text_nonexistent_file(self, summarizer, tmp_path):
        """Test reading non-existent file."""
        nonexistent = tmp_path / "does_not_exist.txt"

        text = summarizer._read_text(nonexistent)

        assert text == ""

    def test_extract_summary_with_docstring(self, summarizer):
        """Test extracting summary from Python code with docstring."""
        code = '''
"""This is the module docstring.

It has multiple lines and provides
a detailed description of the module.
"""

def function():
    """Function docstring."""
    pass
'''

        summary = summarizer._extract_summary(code, max_lines=100)

        # Should extract module docstring
        assert "module docstring" in summary
        assert "multiple lines" in summary
        assert "detailed description" in summary

    def test_extract_summary_class_docstring(self, summarizer):
        """Test extracting docstring from class."""
        code = '''
class MyClass:
    """This is a class docstring.
    
    It describes what the class does.
    """
    
    def method(self):
        pass
'''

        # AST parsing might fail for incomplete code
        summary = summarizer._extract_summary(code, max_lines=100)

        # Should include some content
        assert len(summary) > 0

    def test_extract_summary_malformed_python(self, summarizer):
        """Test extracting summary from malformed Python code."""
        code = """
This is not valid Python code {
    but it might be some other language
}

function test() {
    return true;
}
"""

        # Should fall back to head extraction
        summary = summarizer._extract_summary(code, max_lines=5)

        assert "not valid Python" in summary
        assert len(summary.split("\n")) <= 5

    def test_extract_summary_leading_comments(self, summarizer):
        """Test extracting leading comment block."""
        code = """
# This is a utility module
# It provides helper functions
# for data processing
#
# Author: Developer
# Date: 2024

import os

def helper():
    pass
"""

        summary = summarizer._extract_summary(code, max_lines=100)

        # Should extract comment block
        assert "utility module" in summary
        assert "helper functions" in summary

    def test_extract_summary_mixed_comments(self, summarizer):
        """Test extracting from mixed comment styles."""
        code = '''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module docstring here."""

# Additional comments
# More information

def main():
    pass
'''

        summary = summarizer._extract_summary(code, max_lines=100)

        # Should prioritize docstring
        assert "Module docstring" in summary

    def test_file_summary_model(self):
        """Test FileSummary model."""
        summary = FileSummary(
            path="/path/to/file.py",
            summary="This is the summary",
            token_count=42,
            metadata={"key": "value"},
        )

        assert summary.path == "/path/to/file.py"
        assert summary.summary == "This is the summary"
        assert summary.token_count == 42
        assert summary.metadata["key"] == "value"

    def test_token_counting(self, summarizer):
        """Test token counting in summaries."""
        # Mock token counting
        with patch("tenets.utils.tokens.count_tokens") as mock_count:
            mock_count.return_value = 100

            summary = FileSummary(
                path="test.py", summary="Test summary", token_count=100, metadata={}
            )

            assert summary.token_count == 100

    def test_max_lines_parameter(self, summarizer, tmp_path):
        """Test max_lines parameter effect."""
        # Create file with many lines
        long_file = tmp_path / "long.txt"
        lines = [f"Line {i}: Some content here" for i in range(200)]
        long_file.write_text("\n".join(lines))

        # Test different max_lines values
        summary_10 = summarizer.summarize_file(long_file, max_lines=10)
        summary_50 = summarizer.summarize_file(long_file, max_lines=50)
        summary_100 = summarizer.summarize_file(long_file, max_lines=100)

        # Summaries should have different lengths
        assert len(summary_10.summary) < len(summary_50.summary)
        assert len(summary_50.summary) < len(summary_100.summary)

        # Check line counts
        assert summary_10.summary.count("\n") <= 10
        assert summary_50.summary.count("\n") <= 50
        assert summary_100.summary.count("\n") <= 100

    def test_metadata_strategy(self, summarizer, temp_files):
        """Test that metadata includes strategy information."""
        summary = summarizer.summarize_file(temp_files["python"])

        assert "strategy" in summary.metadata
        assert summary.metadata["strategy"] == "heuristic"
        assert "max_lines" in summary.metadata

    def test_unicode_handling(self, summarizer, tmp_path):
        """Test handling of unicode content."""
        unicode_file = tmp_path / "unicode.py"
        unicode_file.write_text(
            '''
"""Module for handling 多语言 content.

Supports 中文, русский, العربية, and 🎉 emoji.
"""

def process_text(text: str) -> str:
    """Process multilingual text."""
    return text.upper()
''',
            encoding='utf-8'
        )

        summary = summarizer.summarize_file(unicode_file)

        # Should preserve unicode characters
        assert "多语言" in summary.summary or "Module for handling" in summary.summary

    def test_whitespace_preservation(self, summarizer, tmp_path):
        """Test that important whitespace is preserved."""
        indented_file = tmp_path / "indented.py"
        indented_file.write_text(
            '''
def outer():
    """Outer function."""
    
    def inner():
        """Inner function."""
        
        def deeply_nested():
            """Deeply nested."""
            pass
            
        return deeply_nested
        
    return inner
'''
        )

        summary = summarizer.summarize_file(indented_file, max_lines=20)

        # Should preserve some structure
        assert "def" in summary.summary
