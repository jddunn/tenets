"""Tests for HTML formatting in distiller."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.distiller.formatter import ContextFormatter
from tenets.models.analysis import FileAnalysis
from tenets.models.context import PromptContext


class TestHTMLFormatter:
    """Test suite for HTML formatting functionality."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return TenetsConfig()

    @pytest.fixture
    def formatter(self, config):
        """Create formatter instance."""
        return ContextFormatter(config)

    @pytest.fixture
    def prompt_context(self):
        """Create sample prompt context."""
        return PromptContext(
            text="review API implementation",
            task_type="review",
            intent="review",
            keywords=["api", "review", "implementation"],
            entities=["API"],
            focus_areas=["api"],
        )

    @pytest.fixture
    def sample_file(self):
        """Create sample file analysis."""
        return FileAnalysis(
            path=Path("api/endpoints.py"),
            content="def get_user(id): return User.find(id)",
            language="python",
            lines=100,
            size=2000,
            relevance_score=0.85,
        )

    @pytest.fixture
    def aggregated_data(self, sample_file):
        """Create sample aggregated data."""
        return {
            "included_files": [
                {
                    "file": sample_file,
                    "content": sample_file.content,
                    "tokens": 50,
                    "summarized": False,
                }
            ],
            "total_tokens": 50,
            "available_tokens": 1000,
            "statistics": {
                "files_analyzed": 10,
                "files_included": 1,
                "files_summarized": 0,
                "files_skipped": 9,
                "token_utilization": 0.05,
            },
            "git_context": {
                "current_branch": "main",
                "recent_commits": [
                    {
                        "sha": "abc123",
                        "author": "Test User",
                        "date": "2024-01-01",
                        "message": "Add API endpoints",
                    }
                ],
                "contributors": ["Test User"],
            },
        }

    def test_format_html_basic(self, formatter, aggregated_data, prompt_context):
        """Test basic HTML formatting."""
        result = formatter.format(
            aggregated=aggregated_data,
            format="html",
            prompt_context=prompt_context,
            session_name="test-session",
        )

        # Check it's valid HTML
        assert "<!DOCTYPE html>" in result
        assert "<html" in result
        assert "</html>" in result

        # Check header
        assert "Context Distillation Report" in result
        assert "test-session" in result

        # Check prompt section
        assert "review API implementation" in result
        assert "Task Type" in result

        # Check statistics
        assert "Files Analyzed" in result
        assert "10" in result  # files_analyzed value

        # Check file section - handle both forward and back slashes for cross-platform
        assert "endpoints.py" in result  # Just check filename, not full path
        assert "python" in result
        assert "0.85" in result  # relevance score

        # Check git context
        assert "Current Branch" in result
        assert "main" in result

    def test_format_html_escaping(self, formatter, prompt_context):
        """Test HTML escaping."""
        # Create file with HTML-like content
        file_with_html = FileAnalysis(
            path=Path("test.py"),
            content="<script>alert('xss')</script>",
            language="python",
            lines=1,
            size=30,
            relevance_score=0.9,
        )

        aggregated = {
            "included_files": [
                {
                    "file": file_with_html,
                    "content": file_with_html.content,
                    "tokens": 10,
                    "summarized": False,
                }
            ],
            "total_tokens": 10,
            "statistics": {},
        }

        result = formatter.format(
            aggregated=aggregated,
            format="html",
            prompt_context=prompt_context,
        )

        # Check that HTML is escaped
        assert "&lt;script&gt;" in result
        assert "<script>" not in result
        # The literal text "alert(" is escaped but still present in escaped form
        # We just need to ensure the actual script tag isn't executable
        assert "<script>alert" not in result

    def test_format_html_no_git_context(self, formatter, prompt_context):
        """Test HTML formatting without git context."""
        aggregated = {
            "included_files": [],
            "total_tokens": 0,
            "statistics": {
                "files_analyzed": 0,
                "files_included": 0,
                "files_summarized": 0,
                "files_skipped": 0,
                "token_utilization": 0,
            },
        }

        result = formatter.format(
            aggregated=aggregated,
            format="html",
            prompt_context=prompt_context,
        )

        # Should not have git section
        assert "Git Context" not in result
        assert "Current Branch" not in result

    def test_format_html_summarized_files(self, formatter, prompt_context, sample_file):
        """Test HTML formatting with summarized files."""
        aggregated = {
            "included_files": [
                {
                    "file": sample_file,
                    "content": "# Summary of file...",
                    "tokens": 20,
                    "summarized": True,
                    "summary": Mock(strategy_used="extractive"),
                }
            ],
            "total_tokens": 20,
            "statistics": {
                "files_analyzed": 5,
                "files_included": 0,
                "files_summarized": 1,
                "files_skipped": 4,
                "token_utilization": 0.02,
            },
        }

        result = formatter.format(
            aggregated=aggregated,
            format="html",
            prompt_context=prompt_context,
        )

        # Check summarized indicator
        assert "Summarized" in result
        assert "Summary of file" in result

    def test_format_html_multiple_files(self, formatter, prompt_context):
        """Test HTML formatting with multiple files."""
        files = [
            FileAnalysis(
                path=Path(f"file{i}.py"),
                content=f"content {i}",
                language="python",
                lines=i * 10,
                size=i * 100,
                relevance_score=0.9 - i * 0.1,
            )
            for i in range(3)
        ]

        aggregated = {
            "included_files": [
                {"file": f, "content": f.content, "tokens": 10, "summarized": False} for f in files
            ],
            "total_tokens": 30,
            "statistics": {
                "files_analyzed": 10,
                "files_included": 3,
                "files_summarized": 0,
                "files_skipped": 7,
                "token_utilization": 0.03,
            },
        }

        result = formatter.format(
            aggregated=aggregated,
            format="html",
            prompt_context=prompt_context,
        )

        # Check all files are present
        for i in range(3):
            assert f"file{i}.py" in result
            assert f"content {i}" in result

    def test_format_html_styles(self, formatter, prompt_context):
        """Test that HTML includes proper styles."""
        aggregated = {
            "included_files": [],
            "total_tokens": 0,
            "statistics": {},
        }

        result = formatter.format(
            aggregated=aggregated,
            format="html",
            prompt_context=prompt_context,
        )

        # Check for custom styles
        assert "<style>" in result
        assert ".distill-header" in result
        assert ".prompt-box" in result
        assert ".file-card" in result
        assert ".stats-grid" in result
        assert ".code-preview" in result

        # Check for responsive design
        assert "grid-template-columns" in result
        assert "@media" in result or "auto-fit" in result

    def test_format_html_large_content(self, formatter, prompt_context):
        """Test HTML formatting with large content (should truncate preview)."""
        large_content = "x" * 10000
        file_with_large_content = FileAnalysis(
            path=Path("large.py"),
            content=large_content,
            language="python",
            lines=1000,
            size=10000,
            relevance_score=0.9,
        )

        aggregated = {
            "included_files": [
                {
                    "file": file_with_large_content,
                    "content": large_content,
                    "tokens": 1000,
                    "summarized": False,
                }
            ],
            "total_tokens": 1000,
            "statistics": {},
        }

        result = formatter.format(
            aggregated=aggregated,
            format="html",
            prompt_context=prompt_context,
        )

        # Check that preview is truncated
        assert "..." in result
        # Should not include full content in preview
        assert result.count("x") < 1000  # Much less than 10000

    @patch("tenets.core.distiller.formatter.datetime")
    def test_format_html_timestamp(self, mock_datetime, formatter, prompt_context):
        """Test HTML includes timestamp."""
        mock_datetime.now.return_value.strftime.return_value = "2024-01-01 12:00:00"

        aggregated = {
            "included_files": [],
            "total_tokens": 0,
            "statistics": {},
        }

        result = formatter.format(
            aggregated=aggregated,
            format="html",
            prompt_context=prompt_context,
        )

        assert "2024-01-01 12:00:00" in result
        assert "Generated by Tenets" in result

    def test_format_html_token_stats(self, formatter, prompt_context, sample_file):
        """Test HTML formatting shows token statistics correctly."""
        aggregated = {
            "included_files": [
                {
                    "file": sample_file,
                    "content": sample_file.content,
                    "tokens": 150,
                    "summarized": False,
                }
            ],
            "total_tokens": 150,
            "available_tokens": 1000,
            "statistics": {
                "files_analyzed": 20,
                "files_included": 1,
                "files_summarized": 0,
                "files_skipped": 19,
                "token_utilization": 0.15,
            },
        }

        result = formatter.format(
            aggregated=aggregated,
            format="html",
            prompt_context=prompt_context,
        )

        # Check token stats are displayed
        assert "150" in result  # tokens for file
        assert "15.0%" in result  # utilization percentage
        assert "Total Tokens" in result
        assert "Token Utilization" in result
