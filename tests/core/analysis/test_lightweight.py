"""Tests for the lightweight file analyzer.

This module tests the LightweightAnalyzer which provides fast,
minimal file analysis for quick ranking operations.
"""

import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from tenets.core.analysis.lightweight import LightweightAnalyzer, LightweightAnalysis
from tenets.models.analysis import FileAnalysis


class TestLightweightAnalyzer:
    """Test the lightweight analyzer functionality."""
    
    def test_initialization(self):
        """Test analyzer initialization with different sample sizes."""
        # Default initialization
        analyzer = LightweightAnalyzer()
        assert analyzer.sample_size == LightweightAnalyzer.DEFAULT_SAMPLE_SIZE
        
        # Custom sample size
        analyzer = LightweightAnalyzer(sample_size=4096)
        assert analyzer.sample_size == 4096
        
        # Sample size capped at maximum
        analyzer = LightweightAnalyzer(sample_size=100000)
        assert analyzer.sample_size == LightweightAnalyzer.MAX_SAMPLE_SIZE
    
    def test_analyze_python_file(self, tmp_path):
        """Test analyzing a Python file."""
        # Create a test Python file
        test_file = tmp_path / "test_module.py"
        test_file.write_text('''
"""Test module for lightweight analysis."""

import os
import sys
from pathlib import Path

def test_function():
    """A test function."""
    return "hello"

class TestClass:
    """A test class."""
    def method(self):
        return 42

if __name__ == "__main__":
    test_function()
''')
        
        analyzer = LightweightAnalyzer()
        result = analyzer.analyze_file(test_file)
        
        assert result is not None
        assert isinstance(result, LightweightAnalysis)
        assert result.path == test_file
        assert result.extension == '.py'
        assert result._guess_language() == 'python'  # Language is guessed from extension
        assert result.size > 0
        assert result.line_count > 0
        assert 'test_function' in result.content_sample
        assert len(result.keywords) > 0
    
    def test_analyze_test_file_detection(self, tmp_path):
        """Test that test files are properly detected."""
        # Test file by name
        test_file = tmp_path / "test_something.py"
        test_file.write_text("def test_example(): pass")
        
        analyzer = LightweightAnalyzer()
        result = analyzer.analyze_file(test_file)
        
        assert result is not None
        assert result.has_tests is True
        
        # Test file by content
        spec_file = tmp_path / "something_spec.js"
        spec_file.write_text("describe('test suite', () => { it('should work', () => {}); });")
        
        result = analyzer.analyze_file(spec_file)
        assert result is not None
        assert result.has_tests is True
        
        # Non-test file
        regular_file = tmp_path / "module.py"
        regular_file.write_text("def regular_function(): pass")
        
        result = analyzer.analyze_file(regular_file)
        assert result is not None
        assert result.has_tests is False
    
    def test_to_file_analysis_conversion(self, tmp_path):
        """Test conversion to FileAnalysis object."""
        test_file = tmp_path / "module.py"
        test_file.write_text("import os\n\ndef main():\n    print('hello')\n")
        
        analyzer = LightweightAnalyzer()
        lightweight = analyzer.analyze_file(test_file)
        file_analysis = lightweight.to_file_analysis()
        
        assert isinstance(file_analysis, FileAnalysis)
        assert file_analysis.path == str(test_file)
        assert file_analysis.file_extension == '.py'
        assert file_analysis.language == 'python'
        assert file_analysis.lines == lightweight.line_count
        assert file_analysis.file_name == test_file.name
        assert isinstance(file_analysis.last_modified, datetime)
        assert file_analysis.content == lightweight.content_sample
        
        # Check that complex fields are empty/None for speed
        assert file_analysis.imports == []
        assert file_analysis.functions == []
        assert file_analysis.classes == []
        assert file_analysis.complexity is None
    
    def test_binary_file_detection(self, tmp_path):
        """Test that binary files are skipped."""
        # Create a binary file
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        analyzer = LightweightAnalyzer()
        result = analyzer.analyze_file(binary_file)
        
        assert result is None  # Binary files should be skipped
    
    def test_empty_file_handling(self, tmp_path):
        """Test that empty files are handled correctly."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        analyzer = LightweightAnalyzer()
        result = analyzer.analyze_file(empty_file)
        
        assert result is None  # Empty files should be skipped
    
    def test_cache_functionality(self, tmp_path):
        """Test that results are cached based on modification time."""
        test_file = tmp_path / "cached.py"
        test_file.write_text("def cached_function(): pass")
        
        analyzer = LightweightAnalyzer()
        
        # First analysis
        result1 = analyzer.analyze_file(test_file)
        assert result1 is not None
        
        # Second analysis should use cache
        result2 = analyzer.analyze_file(test_file)
        assert result2 is result1  # Same object from cache
        
        # Modify file
        test_file.write_text("def modified_function(): pass")
        
        # Third analysis should not use cache
        result3 = analyzer.analyze_file(test_file)
        assert result3 is not result1  # Different object, cache invalidated
        assert 'modified_function' in result3.content_sample
    
    def test_keyword_extraction(self, tmp_path):
        """Test keyword extraction from content."""
        test_file = tmp_path / "keywords.py"
        test_file.write_text('''
def process_user_authentication():
    """Handle user authentication flow."""
    user_token = generate_token()
    validate_token(user_token)
    return user_token

def generate_token():
    """Generate authentication token."""
    return "token123"
    
def validate_token(token):
    """Validate the given token."""
    return True
''')
        
        analyzer = LightweightAnalyzer()
        result = analyzer.analyze_file(test_file)
        
        # Check that relevant keywords are extracted
        keywords_lower = {k.lower() for k in result.keywords}
        assert 'user_token' in keywords_lower or 'token' in keywords_lower
        assert 'authentication' in keywords_lower or 'user_authentication' in keywords_lower
        
        # Check that common keywords are filtered
        assert 'def' not in keywords_lower
        assert 'return' not in keywords_lower
    
    def test_batch_analysis(self, tmp_path):
        """Test analyzing multiple files in batch."""
        # Create multiple test files
        files = []
        for i in range(5):
            file_path = tmp_path / f"file_{i}.py"
            file_path.write_text(f"def function_{i}(): return {i}")
            files.append(file_path)
        
        analyzer = LightweightAnalyzer()
        results = analyzer.analyze_files(files)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert isinstance(result, LightweightAnalysis)
            assert f"function_{i}" in result.content_sample
    
    def test_language_detection(self):
        """Test language detection based on file extension."""
        analyzer = LightweightAnalyzer()
        
        # Create lightweight analysis objects with different extensions
        test_cases = [
            ('.py', 'python'),
            ('.js', 'javascript'),
            ('.ts', 'typescript'),
            ('.java', 'java'),
            ('.go', 'go'),
            ('.rs', 'rust'),
            ('.cpp', 'cpp'),
            ('.rb', 'ruby'),
            ('.unknown', 'unknown'),
        ]
        
        for ext, expected_lang in test_cases:
            analysis = LightweightAnalysis(
                path=Path(f"test{ext}"),
                size=100,
                extension=ext,
                mime_type=None,
                content_sample="test",
                line_count=1,
                has_tests=False,
                last_modified=0,
                keywords=set(),
            )
            assert analysis._guess_language() == expected_lang


class TestLightweightAnalysisConversion:
    """Test the conversion of lightweight analysis to full FileAnalysis."""
    
    def test_test_file_structure_creation(self):
        """Test that test files get a proper CodeStructure."""
        # Create a test file analysis
        analysis = LightweightAnalysis(
            path=Path("test_file.py"),
            size=100,
            extension='.py',
            mime_type='text/x-python',
            content_sample="def test_something(): pass",
            line_count=1,
            has_tests=True,  # Mark as test file
            last_modified=0,
            keywords={'test', 'something'},
        )
        
        file_analysis = analysis.to_file_analysis()
        
        # Check that structure is created for test files
        assert file_analysis.structure is not None
        assert file_analysis.structure.is_test_file is True
    
    def test_non_test_file_structure(self):
        """Test that non-test files don't get a CodeStructure."""
        analysis = LightweightAnalysis(
            path=Path("regular_file.py"),
            size=100,
            extension='.py',
            mime_type='text/x-python',
            content_sample="def main(): pass",
            line_count=1,
            has_tests=False,  # Not a test file
            last_modified=0,
            keywords={'main'},
        )
        
        file_analysis = analysis.to_file_analysis()
        
        # Check that structure is None for non-test files
        assert file_analysis.structure is None