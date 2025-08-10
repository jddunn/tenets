"""Tests for ContextFormatter."""

import pytest
import json
from datetime import datetime
from unittest.mock import patch

from tenets.core.distiller.formatter import ContextFormatter
from tenets.models.context import PromptContext
from tenets.models.analysis import FileAnalysis
from tenets.models.summary import FileSummary
from tenets.config import TenetsConfig


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    config.version = "1.0.0"
    return config


@pytest.fixture
def formatter(config):
    """Create ContextFormatter instance."""
    return ContextFormatter(config)


@pytest.fixture
def prompt_context():
    """Create sample PromptContext."""
    return PromptContext(
        text="implement authentication",
        keywords=["auth", "login", "user"],
        task_type="feature",
        focus_areas=["security", "api"],
    )


@pytest.fixture
def aggregated_data():
    """Create sample aggregated data."""
    return {
        "included_files": [
            {
                "file": FileAnalysis(
                    path="auth.py",
                    content="def authenticate():\n    pass",
                    language="python",
                    lines=2,
                    relevance_score=0.9,
                ),
                "content": "def authenticate():\n    pass",
                "summarized": False,
                "tokens": 10,
            },
            {
                "file": FileAnalysis(
                    path="models.py",
                    content="# User model summary",
                    language="python",
                    lines=100,
                    relevance_score=0.7,
                ),
                "content": "# User model summary",
                "summarized": True,
                "summary": FileSummary(
                    path="models.py",
                    content="# User model summary",
                    summary_tokens=5,
                    original_tokens=50,
                    instructions=["Full file had 100 lines"],
                ),
            },
        ],
        "total_tokens": 15,
        "available_tokens": 1000,
        "statistics": {
            "files_analyzed": 5,
            "files_included": 1,
            "files_summarized": 1,
            "files_skipped": 3,
            "token_utilization": 0.015,
        },
        "git_context": {
            "branch": "main",
            "recent_commits": [
                {"sha": "abc123", "message": "Add auth", "author": "dev", "date": "2024-01-01"}
            ],
            "contributors": [{"name": "dev", "commits": 10}],
        },
    }


class TestContextFormatter:
    """Test suite for ContextFormatter."""

    def test_initialization(self, config):
        """Test formatter initialization."""
        formatter = ContextFormatter(config)

        assert formatter.config == config

    def test_format_unsupported(self, formatter, aggregated_data, prompt_context):
        """Test formatting with unsupported format."""
        with pytest.raises(ValueError, match="Unknown format"):
            formatter.format(aggregated_data, "unsupported", prompt_context)

    def test_format_markdown(self, formatter, aggregated_data, prompt_context):
        """Test markdown formatting."""
        with patch("tenets.core.distiller.formatter.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2024-01-01 12:00:00"

            result = formatter._format_markdown(aggregated_data, prompt_context, "test-session")

            assert "# Context for: implement authentication" in result
            assert "*Session: test-session*" in result
            assert "## Task Analysis" in result
            assert "- **Type**: feature" in result
            assert "- **Keywords**: auth, login, user" in result
            assert "- **Focus Areas**: security, api" in result
            assert "## Context Summary" in result
            assert "## Git Context" in result
            assert "## Relevant Files" in result
            assert "### Complete Files" in result
            assert "### Summarized Files" in result
            assert "auth.py" in result
            assert "models.py" in result
            assert "```python" in result

    def test_format_markdown_no_session(self, formatter, aggregated_data, prompt_context):
        """Test markdown formatting without session."""
        result = formatter._format_markdown(aggregated_data, prompt_context, None)

        assert "*Session:" not in result

    def test_format_markdown_no_git(self, formatter, prompt_context):
        """Test markdown formatting without git context."""
        data = {
            "included_files": [],
            "total_tokens": 0,
            "available_tokens": 1000,
            "statistics": {
                "files_analyzed": 0,
                "files_included": 0,
                "files_summarized": 0,
                "files_skipped": 0,
                "token_utilization": 0,
            },
        }

        result = formatter._format_markdown(data, prompt_context, None)

        assert "## Git Context" not in result

    def test_format_xml(self, formatter, aggregated_data, prompt_context):
        """Test XML formatting."""
        result = formatter._format_xml(aggregated_data, prompt_context, "test-session")

        assert '<?xml version="1.0"' in result
        assert "<context>" in result
        assert "<metadata>" in result
        assert "<session>test-session</session>" in result
        assert "<task_type>feature</task_type>" in result
        assert "<files>" in result
        assert 'path="auth.py"' in result
        assert 'summarized="false"' in result
        assert 'summarized="true"' in result
        assert "<![CDATA[" in result
        assert "</context>" in result

    def test_format_xml_escaping(self, formatter, prompt_context):
        """Test XML special character escaping."""
        data = {
            "included_files": [
                {
                    "file": FileAnalysis(
                        path="test.py",
                        content="if a < b & c > d:",
                        language="python",
                        lines=1,
                        relevance_score=0.5,
                    ),
                    "content": "if a < b & c > d:",
                    "summarized": False,
                }
            ],
            "statistics": {
                "files_analyzed": 1,
                "files_included": 1,
                "files_summarized": 0,
                "files_skipped": 0,
                "token_utilization": 0.1,
            },
            "total_tokens": 10,
            "available_tokens": 100,
        }

        result = formatter._format_xml(data, prompt_context, None)

        # Content should be in CDATA, not escaped
        assert "<![CDATA[" in result
        assert "if a < b & c > d:" in result

    def test_format_json(self, formatter, aggregated_data, prompt_context):
        """Test JSON formatting."""
        result = formatter._format_json(aggregated_data, prompt_context, "test-session")

        data = json.loads(result)

        assert data["context"]["prompt"] == "implement authentication"
        assert data["context"]["session"] == "test-session"
        assert data["analysis"]["task_type"] == "feature"
        assert data["analysis"]["keywords"] == ["auth", "login", "user"]
        assert len(data["files"]) == 2
        assert data["files"][0]["path"] == "auth.py"
        assert data["files"][0]["summarized"] == False
        assert data["files"][1]["summarized"] == True
        assert "git_context" in data

    def test_format_git_context_markdown(self, formatter):
        """Test git context formatting for markdown."""
        git_context = {
            "branch": "feature/auth",
            "recent_commits": [
                {
                    "sha": "abc123def",
                    "message": "Add login",
                    "author": "John",
                    "date": "2024-01-01",
                },
                {"sha": "def456ghi", "message": "Fix bug", "author": "Jane", "date": "2024-01-02"},
            ],
            "contributors": [{"name": "John", "commits": 20}, {"name": "Jane", "commits": 15}],
        }

        lines = formatter._format_git_context_markdown(git_context)

        assert "## Git Context" in lines[0]
        assert "- **Current Branch**: feature/auth" in lines
        assert "### Recent Commits" in lines
        assert "`abc123de`" in " ".join(lines)
        assert "Add login" in " ".join(lines)
        assert "### Top Contributors" in lines
        assert "John: 20 commits" in " ".join(lines)

    def test_format_git_context_xml(self, formatter):
        """Test git context formatting for XML."""
        git_context = {
            "branch": "main",
            "recent_commits": [
                {
                    "sha": "abc123",
                    "message": "Test & fix",
                    "author": "Dev <dev@test.com>",
                    "date": "2024-01-01",
                }
            ],
        }

        lines = formatter._format_git_context_xml(git_context)

        assert "<git_context>" in lines[0]
        assert "<branch>main</branch>" in " ".join(lines)
        assert "<recent_commits>" in " ".join(lines)
        assert 'sha="abc123"' in " ".join(lines)
        assert "&lt;dev@test.com&gt;" in " ".join(lines)  # Escaped
        assert "Test &amp; fix" in " ".join(lines)  # Escaped

    def test_escape_xml(self, formatter):
        """Test XML escaping function."""
        text = "Test & <tag> \"quotes\" 'apostrophe'"
        escaped = formatter._escape_xml(text)

        assert escaped == "Test &amp; &lt;tag&gt; &quot;quotes&quot; &apos;apostrophe&apos;"

    def test_format_with_empty_files(self, formatter, prompt_context):
        """Test formatting with no files."""
        data = {
            "included_files": [],
            "total_tokens": 0,
            "available_tokens": 1000,
            "statistics": {
                "files_analyzed": 0,
                "files_included": 0,
                "files_summarized": 0,
                "files_skipped": 0,
                "token_utilization": 0,
            },
        }

        # Should not error
        result_md = formatter._format_markdown(data, prompt_context, None)
        result_xml = formatter._format_xml(data, prompt_context, None)
        result_json = formatter._format_json(data, prompt_context, None)

        assert "## Relevant Files" in result_md
        assert "<files>" in result_xml
        assert "</files>" in result_xml

        json_data = json.loads(result_json)
        assert json_data["files"] == []

    def test_format_instructions(self, formatter, aggregated_data, prompt_context):
        """Test that AI instructions are included."""
        result = formatter._format_markdown(aggregated_data, prompt_context, None)

        assert "## Instructions for AI Assistant" in result
        assert "Show me: path/to/file.py" in result
        assert "Find files related to:" in result

    def test_token_usage_formatting(self, formatter, aggregated_data, prompt_context):
        """Test token usage display."""
        result = formatter._format_markdown(aggregated_data, prompt_context, None)

        assert "**Token Usage**: 15 / 1,000 (1.5%)" in result
