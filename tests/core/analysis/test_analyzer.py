"""Tests for the main CodeAnalyzer orchestrator."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.analysis.analyzer import CodeAnalyzer
from tenets.core.analysis.implementations.python_analyzer import PythonAnalyzer
from tenets.models.analysis import (
    CodeStructure,
    ComplexityMetrics,
    DependencyGraph,
    FileAnalysis,
    ImportInfo,
    ProjectAnalysis,
)


@pytest.fixture
def config():
    """Create a test configuration."""
    config = TenetsConfig()
    config.cache.enabled = False  # Disable caching for tests
    config.scanner.workers = 2
    config.scanner.timeout = 10
    config.scanner.encoding = "utf-8"
    return config


@pytest.fixture
def analyzer(config):
    """Create a CodeAnalyzer instance."""
    return CodeAnalyzer(config)


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure."""
    # Create project structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create Python files
    main_py = src_dir / "main.py"
    main_py.write_text(
        """
'''Main module for the application.'''
import os
import sys
from datetime import datetime
from .utils import helper_function

def main():
    '''Entry point.'''
    print("Hello, World!")
    result = helper_function()
    return result

class Application:
    '''Main application class.'''
    def __init__(self):
        self.name = "TestApp"
    
    def run(self):
        '''Run the application.'''
        return main()

if __name__ == "__main__":
    main()
"""
    )

    utils_py = src_dir / "utils.py"
    utils_py.write_text(
        """
'''Utility functions.'''

def helper_function():
    '''A helper function.'''
    return 42

def complex_function(a, b, c):
    '''A complex function with multiple branches.'''
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
            else:
                return a + b - c
        else:
            return a - b
    else:
        return 0
"""
    )

    # Create a JavaScript file
    js_file = src_dir / "app.js"
    js_file.write_text(
        """
// Main JavaScript application
import React from 'react';
import { render } from 'react-dom';

function App() {
    return <div>Hello World</div>;
}

export default App;
"""
    )

    # Create test file
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_py = test_dir / "test_main.py"
    test_py.write_text(
        """
import pytest
from src.main import main

def test_main():
    assert main() == 42
"""
    )

    # Create config file
    config_file = tmp_path / "config.json"
    config_file.write_text('{"name": "test", "version": "1.0.0"}')

    return tmp_path


class TestCodeAnalyzer:
    """Test suite for CodeAnalyzer."""

    def test_initialization(self, config):
        """Test analyzer initialization."""
        analyzer = CodeAnalyzer(config)

        assert analyzer.config == config
        assert analyzer.analyzers is not None
        assert len(analyzer.analyzers) > 0
        assert ".py" in analyzer.analyzers
        assert ".js" in analyzer.analyzers

    def test_get_analyzer(self, analyzer):
        """Test getting appropriate analyzer for file types."""
        # Python file
        py_analyzer = analyzer._get_analyzer(Path("test.py"))
        assert isinstance(py_analyzer, PythonAnalyzer)

        # JavaScript file
        js_analyzer = analyzer._get_analyzer(Path("test.js"))
        assert js_analyzer is not None
        assert js_analyzer.language_name == "javascript"

        # Dockerfile
        dockerfile_analyzer = analyzer._get_analyzer(Path("Dockerfile"))
        assert dockerfile_analyzer is not None

        # Unknown file - should get generic analyzer
        unknown_analyzer = analyzer._get_analyzer(Path("test.xyz"))
        assert unknown_analyzer is not None

    def test_detect_language(self, analyzer):
        """Test language detection."""
        assert analyzer._detect_language(Path("test.py")) == "python"
        assert analyzer._detect_language(Path("test.js")) == "javascript"
        assert analyzer._detect_language(Path("test.go")) == "go"
        assert analyzer._detect_language(Path("test.java")) == "java"
        assert analyzer._detect_language(Path("test.rs")) == "rust"
        assert analyzer._detect_language(Path("test.unknown")) == "unknown"

    def test_analyze_file_success(self, analyzer, temp_project):
        """Test successful file analysis."""
        main_py = temp_project / "src" / "main.py"

        analysis = analyzer.analyze_file(main_py, deep=True)

        assert analysis is not None
        assert analysis.path == str(main_py)
        assert analysis.language == "python"
        assert analysis.lines > 0
        assert analysis.size > 0
        assert analysis.file_name == "main.py"
        assert analysis.file_extension == ".py"

        # Check imports were extracted
        assert len(analysis.imports) > 0
        import_modules = [imp.module for imp in analysis.imports]
        assert "os" in import_modules
        assert "sys" in import_modules

        # Check structure was extracted
        assert analysis.structure is not None
        if analysis.functions:
            func_names = [f.name for f in analysis.functions]
            assert "main" in func_names

        if analysis.classes:
            class_names = [c.name for c in analysis.classes]
            assert "Application" in class_names

    def test_analyze_file_not_found(self, analyzer):
        """Test analyzing non-existent file."""
        with pytest.raises(FileNotFoundError):
            analyzer.analyze_file(Path("non_existent_file.py"))

    def test_analyze_file_with_cache(self, config, temp_project):
        """Test file analysis with caching enabled."""
        config.cache.enabled = True

        with tempfile.TemporaryDirectory() as cache_dir:
            config.cache.directory = cache_dir
            analyzer = CodeAnalyzer(config)

            main_py = temp_project / "src" / "main.py"

            # First analysis - cache miss
            analysis1 = analyzer.analyze_file(main_py, use_cache=True)
            assert analyzer.stats["cache_misses"] > 0

            # Second analysis - cache hit
            analysis2 = analyzer.analyze_file(main_py, use_cache=True)
            assert analyzer.stats["cache_hits"] > 0

            # Results should be equivalent
            assert analysis1.path == analysis2.path
            assert analysis1.lines == analysis2.lines

    def test_analyze_files_parallel(self, analyzer, temp_project):
        """Test parallel file analysis."""
        files = list(temp_project.glob("**/*.py"))

        results = analyzer.analyze_files(files, deep=True, parallel=True)

        assert len(results) == len(files)
        for result in results:
            assert isinstance(result, FileAnalysis)
            assert result.language == "python"

    def test_analyze_files_sequential(self, analyzer, temp_project):
        """Test sequential file analysis."""
        files = list(temp_project.glob("**/*.py"))

        results = analyzer.analyze_files(files, deep=True, parallel=False)

        assert len(results) == len(files)
        for result in results:
            assert isinstance(result, FileAnalysis)

    def test_analyze_files_with_timeout(self, analyzer):
        """Test file analysis with timeout."""
        with patch.object(analyzer, "analyze_file") as mock_analyze:
            # Make analyze_file hang
            import time

            mock_analyze.side_effect = lambda *args, **kwargs: time.sleep(100)

            files = [Path("test1.py"), Path("test2.py")]
            results = analyzer.analyze_files(files, parallel=True)

            # Should timeout and return error results
            assert len(results) == 2
            for result in results:
                assert result.error == "Analysis timeout"

    def test_analyze_project(self, analyzer, temp_project):
        """Test project analysis."""
        project = analyzer.analyze_project(
            temp_project, patterns=["*.py", "*.js"], deep=True, parallel=True
        )

        assert isinstance(project, ProjectAnalysis)
        assert project.path == str(temp_project)
        assert project.name == temp_project.name
        assert project.total_files > 0
        assert project.analyzed_files > 0
        assert len(project.files) > 0

        # Check project metrics
        assert hasattr(project, "total_lines")
        assert hasattr(project, "languages")
        assert "python" in project.languages

    def test_extract_keywords(self, analyzer):
        """Test keyword extraction."""
        content = """
        def authenticate_user(username, password):
            '''Authenticate user with OAuth2.'''
            token = generate_jwt_token(username)
            cache.set(token, user_data)
            return token
        """

        keywords = analyzer._extract_keywords(content, "python")

        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should extract meaningful identifiers
        assert any("authenticate" in kw.lower() for kw in keywords)

    def test_calculate_quality_score(self, analyzer):
        """Test quality score calculation."""
        analysis = FileAnalysis(
            path="test.py",
            complexity=ComplexityMetrics(
                cyclomatic=5, cognitive=10, max_depth=2, comment_ratio=0.2, maintainability_index=75
            ),
        )

        score = analyzer._calculate_quality_score(analysis)

        assert 0 <= score <= 100
        assert score > 50  # Should be decent for low complexity

        # High complexity should lower score
        analysis.complexity.cyclomatic = 25
        analysis.complexity.cognitive = 30
        low_score = analyzer._calculate_quality_score(analysis)
        assert low_score < score

    def test_build_dependency_graph(self, analyzer, temp_project):
        """Test dependency graph building."""
        files = list(temp_project.glob("**/*.py"))
        file_analyses = analyzer.analyze_files(files, deep=True)

        graph = analyzer._build_dependency_graph(file_analyses)

        assert isinstance(graph, DependencyGraph)
        # Should have nodes for each file
        assert len(graph.nodes) >= len(file_analyses)

    def test_detect_project_type(self, analyzer, temp_project):
        """Test project type detection."""
        # Create package.json for Node detection
        package_json = temp_project / "package.json"
        package_json.write_text('{"name": "test"}')

        project_type = analyzer._detect_project_type(temp_project, [])
        assert project_type == "node"

        # Remove package.json and add requirements.txt for Python
        package_json.unlink()
        requirements = temp_project / "requirements.txt"
        requirements.write_text("pytest\nrequests")

        project_type = analyzer._detect_project_type(temp_project, [])
        assert project_type == "python"

    def test_generate_json_report(self, analyzer, temp_project):
        """Test JSON report generation."""
        files = list(temp_project.glob("**/*.py"))[:1]
        analysis = analyzer.analyze_file(files[0])

        report = analyzer.generate_report(analysis, format="json")

        assert report.format == "json"
        assert report.content is not None

        # Should be valid JSON
        data = json.loads(report.content)
        assert "path" in data

    def test_generate_markdown_report(self, analyzer, temp_project):
        """Test Markdown report generation."""
        project = analyzer.analyze_project(temp_project)

        report = analyzer.generate_report(project, format="markdown")

        assert report.format == "markdown"
        assert "# Code Analysis Report" in report.content
        assert project.name in report.content

    def test_error_handling_in_analysis(self, analyzer):
        """Test error handling during analysis."""
        with patch.object(analyzer, "_read_file_content") as mock_read:
            mock_read.side_effect = Exception("Read error")

            analysis = analyzer.analyze_file(Path("test.py"))

            assert analysis.error is not None
            assert "Read error" in analysis.error
            assert analyzer.stats["errors"] > 0

    def test_collect_project_files(self, analyzer, temp_project):
        """Test project file collection."""
        files = analyzer._collect_project_files(
            temp_project, patterns=["*.py"], exclude_patterns=["test_*.py"]
        )

        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "test_main.py" not in file_names  # Excluded

    def test_cache_key_generation(self, analyzer, temp_project):
        """Test cache key generation."""
        file_path = temp_project / "src" / "main.py"

        key1 = analyzer._get_cache_key(file_path)
        key2 = analyzer._get_cache_key(file_path)

        assert key1 == key2  # Same file should generate same key
        assert len(key1) == 32  # MD5 hash length

    def test_parallel_processing_error_handling(self, analyzer):
        """Test error handling in parallel processing."""

        def failing_analysis(path, **kwargs):
            if "fail" in str(path):
                raise Exception("Simulated failure")
            return FileAnalysis(path=str(path))

        with patch.object(analyzer, "analyze_file", side_effect=failing_analysis):
            files = [Path("test.py"), Path("fail.py"), Path("other.py")]
            results = analyzer.analyze_files(files, parallel=True)

            assert len(results) == 3
            # One should have error
            errors = [r for r in results if r.error]
            assert len(errors) == 1

    def test_shutdown(self, analyzer):
        """Test analyzer shutdown."""
        analyzer.shutdown()

        # Executor should be shut down
        assert analyzer._executor._shutdown

        # Stats should be preserved
        assert "files_analyzed" in analyzer.stats
