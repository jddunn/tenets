"""
Shared pytest fixtures and configuration for the Tenets test suite.

This module provides common fixtures, mocks, and utilities used across
all test files. It's automatically loaded by pytest and makes these
fixtures available to all tests without explicit imports.

Fixtures provided:
    - Configuration fixtures (test configs, temp directories)
    - Mock fixtures (file system, git, ML models)
    - Sample data fixtures (code samples, analysis results)
    - Utility fixtures (loggers, temporary paths)
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

# Compatibility: pytest.Any was added in newer pytest versions.
# Provide a lightweight fallback so tests using pytest.Any(str) work.
if not hasattr(pytest, "Any"):

    class _Any:  # pragma: no cover - trivial shim
        def __init__(self, typ=object):
            self.typ = typ

        def __eq__(self, other):
            try:
                return isinstance(other, self.typ)
            except Exception:
                return True

        def __repr__(self) -> str:
            try:
                name = getattr(self.typ, "__name__", str(self.typ))
            except Exception:
                name = "object"
            return f"Any({name})"

    pytest.Any = _Any  # type: ignore[attr-defined]

# Set environment variables for testing
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Disable heavy ML features during tests to avoid network/model downloads
# Summarizer ML (transformers) and any LLM integrations are optional
os.environ.setdefault("TENETS_SUMMARIZER_ENABLE_ML_STRATEGIES", "false")
os.environ.setdefault("TENETS_LLM_ENABLED", "false")

# Import the modules we're testing
from tenets.config import TenetsConfig
from tenets.models.analysis import ComplexityMetrics, FileAnalysis, ImportInfo
from tenets.models.context import ContextResult, PromptContext
from tenets.models.tenet import Priority, Tenet, TenetStatus

# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def test_config() -> TenetsConfig:
    """
    Provide a test configuration with sensible defaults.

    This configuration is optimized for testing with smaller limits,
    disabled caching, and test-specific paths.

    Returns:
        TenetsConfig: A configuration suitable for testing
    """
    config = TenetsConfig()

    # Override with test-friendly settings
    config.max_tokens = 10000
    config.debug = True
    config.quiet = False

    # Disable caching for tests
    config.cache.enabled = False

    # Use smaller limits for faster tests
    config.scanner.max_files = 100
    config.scanner.max_file_size = 100_000

    # Use fast ranking for tests
    config.ranking.algorithm = "fast"
    config.ranking.threshold = 0.1

    # Reduce worker counts for predictable tests
    config.scanner.workers = 1
    config.ranking.workers = 1

    return config


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Provide a temporary directory that's cleaned up after the test.

    This is useful for tests that need to create temporary files
    or directories without polluting the file system.

    Yields:
        Path: Path to a temporary directory
    """
    temp_path = tmp_path / "test_workspace"
    temp_path.mkdir(exist_ok=True)

    yield temp_path

    # Cleanup is handled by pytest's tmp_path fixture


@pytest.fixture
def config_file(temp_dir: Path) -> Path:
    """
    Create a temporary configuration file for testing.

    Returns:
        Path: Path to a test configuration file
    """
    config_path = temp_dir / ".tenets.yml"
    config_data = {
        "max_tokens": 5000,
        "scanner": {"respect_gitignore": True, "max_file_size": 50000},
        "ranking": {"algorithm": "balanced", "threshold": 0.2},
        "tenet": {"auto_instill": True, "max_per_context": 3},
    }

    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    return config_path


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_git_repo():
    """
    Mock GitPython's Repo object for testing git operations.

    Returns:
        Mock: A mock Repo object with common git operations
    """
    mock_repo = Mock()

    # Mock basic repo properties
    mock_repo.head.commit.hexsha = "abc123def456"
    mock_repo.active_branch.name = "main"
    mock_repo.remotes.origin.url = "https://github.com/test/repo.git"

    # Mock commit history
    mock_commits = []
    for i in range(5):
        commit = Mock()
        commit.hexsha = f"commit{i:03d}"
        commit.author.name = f"Author {i}"
        commit.author.email = f"author{i}@test.com"
        commit.message = f"Test commit {i}"
        commit.committed_date = int((datetime.now() - timedelta(days=i)).timestamp())
        mock_commits.append(commit)

    mock_repo.iter_commits.return_value = mock_commits

    # Mock diff for changed files
    mock_repo.git.diff.return_value = "file1.py\nfile2.js\nfile3.go"

    return mock_repo


@pytest.fixture
def mock_ml_model():
    """
    Mock sentence-transformers model for semantic similarity testing.

    Returns:
        Mock: A mock SentenceTransformer model
    """
    mock_model = Mock()

    # Mock encode method to return fake embeddings
    def encode_side_effect(text, **kwargs):
        """Generate deterministic fake embeddings based on text length."""
        import numpy as np

        # Create a fake embedding based on text characteristics
        embedding_size = 384  # Standard size for all-MiniLM-L6-v2
        seed = len(text) % 1000  # Deterministic but varies with input
        np.random.seed(seed)
        return np.random.randn(embedding_size)

    mock_model.encode.side_effect = encode_side_effect

    return mock_model


@pytest.fixture
def mock_cache_manager():
    """Provide a simple mock cache manager usable across tests.

    This mirrors the per-module fixtures some tests define, but makes it
    globally available so classes without a local fixture (e.g.,
    TestConvenienceFunctions) can request it.
    """
    manager = MagicMock()
    # Ensure a .general namespace with common methods used by tests
    manager.general.get.return_value = None
    manager.general.put.return_value = None
    manager.general.clear.return_value = None
    return manager


@pytest.fixture(autouse=True)
def mock_external_dependencies(monkeypatch, request):
    """
    Automatically mock external dependencies that shouldn't be called in tests.

    This fixture runs automatically for all tests and prevents accidental
    calls to external services or slow operations.
    """
    # Mock file system operations that might be slow or destructive
    monkeypatch.setattr("shutil.rmtree", Mock())

    # Mock network requests
    mock_requests = Mock()
    mock_requests.get.return_value.status_code = 200
    mock_requests.get.return_value.json.return_value = {"data": "test"}
    monkeypatch.setattr("requests.get", mock_requests.get)
    monkeypatch.setattr("requests.post", mock_requests.get)

    # Decide whether to mock git based on test path (allow real git in git tests)
    test_path = str(getattr(request.node, "fspath", "")).replace("\\", "/")
    if "tests/core/git" not in test_path:
        mock_git = Mock()
        mock_git.Repo.return_value = Mock()
        monkeypatch.setattr("git.Repo", mock_git.Repo)

    # Mock expensive ML operations (graceful if dependency not installed)
    import importlib
    import sys
    import types

    try:
        monkeypatch.setattr("sentence_transformers.SentenceTransformer", Mock)
    except ModuleNotFoundError:
        # Create a minimal stub so code importing sentence_transformers works
        if importlib.util.find_spec("sentence_transformers") is None:
            stub = types.ModuleType("sentence_transformers")

            class SentenceTransformer:  # noqa: N801 (match real class name)
                def __init__(self, *_, **__):
                    pass

                def encode(self, texts, **kwargs):  # pragma: no cover - simple stub
                    if isinstance(texts, str):
                        texts = [texts]
                    # Return deterministic small vectors based on text length
                    return [[float(len(t) % 7)] * 16 for t in texts]

            stub.SentenceTransformer = SentenceTransformer
            sys.modules["sentence_transformers"] = stub


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_python_code() -> str:
    """
    Provide sample Python code for testing analyzers.

    Returns:
        str: Sample Python source code
    """
    return '''
"""Sample Python module for testing."""

import os
import sys
from datetime import datetime
from typing import List, Optional

class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, name: str):
        self.name = name
        
    def greet(self, greeting: str = "Hello") -> str:
        """Return a greeting message."""
        return f"{greeting}, {self.name}!"
        
def sample_function(items: List[str]) -> Optional[str]:
    """Process a list of items."""
    if not items:
        return None
    return ", ".join(items)

# Some variables
CONSTANT_VALUE = 42
variable = "test"
'''


@pytest.fixture
def sample_javascript_code() -> str:
    """
    Provide sample JavaScript code for testing analyzers.

    Returns:
        str: Sample JavaScript source code
    """
    return """
// Sample JavaScript module for testing

import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';

export default function SampleComponent({ name }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        fetchData();
    }, []);
    
    async function fetchData() {
        try {
            const response = await axios.get('/api/data');
            setData(response.data);
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            setLoading(false);
        }
    }
    
    return (
        <div className="sample-component">
            <h1>Hello, {name}!</h1>
            {loading ? <p>Loading...</p> : <p>Data: {data}</p>}
        </div>
    );
}

export const helperFunction = (x, y) => x + y;
"""


@pytest.fixture
def sample_file_analysis() -> FileAnalysis:
    """
    Create a sample FileAnalysis object for testing.

    Returns:
        FileAnalysis: A populated FileAnalysis object
    """
    return FileAnalysis(
        path="test/sample.py",
        content="# Test file\nprint('hello')",
        size=100,
        lines=2,
        language="python",
        file_name="sample.py",
        file_extension=".py",
        last_modified=datetime.now(),
        hash="abc123",
        imports=[ImportInfo(module="os", line=1, type="import", is_relative=False)],
        complexity=ComplexityMetrics(
            cyclomatic=5, cognitive=3, line_count=10, function_count=2, class_count=1
        ),
        keywords=["test", "sample", "hello"],
        relevance_score=0.75,
    )


@pytest.fixture
def sample_prompt_context() -> PromptContext:
    """
    Create a sample PromptContext for testing.

    Returns:
        PromptContext: A populated PromptContext object
    """
    return PromptContext(
        text="implement OAuth2 authentication",
        original="implement OAuth2 authentication",
        keywords=["oauth2", "authentication", "implement"],
        task_type="feature",
        intent="implement",
        entities=[],
        file_patterns=[],
        focus_areas=["authentication", "security"],
        temporal_context=None,
        scope={},
        external_context=None,
    )


@pytest.fixture
def sample_tenet() -> Tenet:
    """
    Create a sample Tenet for testing.

    Returns:
        Tenet: A sample tenet object
    """
    return Tenet(
        content="Always use type hints in Python code", priority=Priority.HIGH, author="test_user"
    )


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def create_test_file(temp_dir: Path):
    """
    Factory fixture for creating test files.

    Args:
        temp_dir: Temporary directory for test files

    Returns:
        Callable: Function to create test files
    """

    def _create_file(filename: str, content: str) -> Path:
        """
        Create a test file with given content.

        Args:
            filename: Name of the file to create
            content: Content to write to the file

        Returns:
            Path: Path to the created file
        """
        file_path = temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    return _create_file


@pytest.fixture
def create_test_repo(temp_dir: Path):
    """
    Create a test repository structure.

    Returns:
        Callable: Function to create a test repository
    """

    def _create_repo(files: Dict[str, str]) -> Path:
        """
        Create a repository with given files.

        Args:
            files: Dictionary mapping file paths to content

        Returns:
            Path: Path to the repository root
        """
        for file_path, content in files.items():
            full_path = temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

        # Initialize git repo
        import subprocess

        subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=temp_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=temp_dir,
            capture_output=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "Test",
                "GIT_AUTHOR_EMAIL": "test@test.com",
                "GIT_COMMITTER_NAME": "Test",
                "GIT_COMMITTER_EMAIL": "test@test.com",
            },
        )

        return temp_dir

    return _create_repo


@pytest.fixture
def capture_logs():
    """
    Capture log messages during tests.

    Returns:
        List: List that will contain captured log records
    """
    import logging

    logs = []

    class LogCapture(logging.Handler):
        def emit(self, record):
            logs.append(record)

    handler = LogCapture()
    logger = logging.getLogger("tenets")
    logger.addHandler(handler)

    yield logs

    logger.removeHandler(handler)


# ============================================================================
# Test Markers and Utilities
# ============================================================================


def requires_git(func):
    """Mark test as requiring git."""
    return pytest.mark.requires_git(func)


def requires_ml(func):
    """Mark test as requiring ML dependencies."""
    return pytest.mark.requires_ml(func)


def slow_test(func):
    """Mark test as slow."""
    return pytest.mark.slow(func)


# ============================================================================
# Session-scoped Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def sample_codebase(tmp_path_factory) -> Path:
    """
    Create a sample codebase that persists for the entire test session.

    This is useful for integration tests that need a realistic project structure.

    Returns:
        Path: Path to a sample codebase
    """
    root = tmp_path_factory.mktemp("sample_project")

    # Create a realistic project structure
    files = {
        "src/main.py": "# Main application\nfrom .auth import authenticate",
        "src/auth.py": "# Authentication module\ndef authenticate(user, password): pass",
        "src/api/routes.py": "# API routes\nfrom flask import Flask",
        "tests/test_main.py": "# Tests\nimport pytest",
        "README.md": "# Sample Project\nA test project for Tenets",
        ".gitignore": "*.pyc\n__pycache__/\n.env",
        "requirements.txt": "flask==2.0.0\npytest==7.0.0",
    }

    for path, content in files.items():
        file_path = root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    return root


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers if not already defined
    config.addinivalue_line("markers", "unit: Unit tests with mocked dependencies")
    config.addinivalue_line("markers", "integration: Integration tests with real components")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Auto-mark tests based on their path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
