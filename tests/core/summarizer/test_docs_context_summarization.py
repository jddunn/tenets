"""Tests for documentation context-aware summarization functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.summarizer import Summarizer
from tenets.models.analysis import FileAnalysis


class TestDocumentationContextSummarization:
    """Test documentation-specific context-aware summarization."""

    @pytest.fixture
    def config(self):
        """Create a test configuration with docs summarization enabled."""
        config = TenetsConfig()
        config.summarizer.docs_context_aware = True
        config.summarizer.docs_show_in_place_context = True
        config.summarizer.docs_context_search_depth = 2
        config.summarizer.docs_context_min_confidence = 0.6
        config.summarizer.docs_context_max_sections = 10
        config.summarizer.docs_context_preserve_examples = True
        return config

    @pytest.fixture
    def summarizer(self, config):
        """Create a summarizer instance."""
        return Summarizer(config)

    @pytest.fixture
    def markdown_file(self):
        """Create a sample markdown documentation file."""
        content = """# API Documentation

## Overview
This is the main API documentation for our service.

## Authentication
The API uses JWT tokens for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### User Management

#### GET /api/users
Retrieve a list of users.

Parameters:
- limit: Maximum number of users to return (default: 20)
- offset: Number of users to skip (default: 0)

```python
import requests

response = requests.get('/api/users?limit=10&offset=0')
users = response.json()
```

#### POST /api/users
Create a new user.

```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "secure_password"
}
```

### Configuration

#### Database Settings
Configure the database connection in `config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  name: myapp
  user: dbuser
  password: dbpass
```

## Error Handling
The API returns standard HTTP status codes and JSON error responses.

Example error response:
```json
{
  "error": "Authentication failed",
  "code": 401,
  "message": "Invalid or expired token"
}
```
"""
        return FileAnalysis(
            path="docs/api.md",
            content=content,
            language="markdown",
            lines=len(content.splitlines()),
            size=len(content),
            encoding="utf-8",
        )

    @pytest.fixture
    def config_file(self):
        """Create a sample configuration file."""
        content = """# Database Configuration
database:
  # Main database connection
  host: localhost
  port: 5432
  name: production_db

  # Connection pool settings
  pool_size: 10
  max_overflow: 20

  # Authentication
  username: admin
  password: ${DB_PASSWORD}

# API Configuration
api:
  # Server settings
  host: 0.0.0.0
  port: 8080
  debug: false

  # Rate limiting
  rate_limit: 1000
  burst: 100

# Logging configuration
logging:
  level: info
  format: json
  output: /var/log/app.log
"""
        return FileAnalysis(
            path="config/settings.yaml",
            content=content,
            language="yaml",
            lines=len(content.splitlines()),
            size=len(content),
            encoding="utf-8",
        )

    def test_is_documentation_file(self, summarizer):
        """Test documentation file detection."""
        # Test markdown files
        assert summarizer._is_documentation_file(Path("README.md"))
        assert summarizer._is_documentation_file(Path("docs/api.md"))
        assert summarizer._is_documentation_file(Path("CHANGELOG.markdown"))

        # Test configuration files
        assert summarizer._is_documentation_file(Path("config.yaml"))
        assert summarizer._is_documentation_file(Path("settings.json"))
        assert summarizer._is_documentation_file(Path("app.toml"))

        # Test documentation names
        assert summarizer._is_documentation_file(Path("README"))
        assert summarizer._is_documentation_file(Path("LICENSE"))
        assert summarizer._is_documentation_file(Path("INSTALL"))

        # Test docs directories
        assert summarizer._is_documentation_file(Path("docs/guide.txt"))
        assert summarizer._is_documentation_file(Path("documentation/api.rst"))

        # Test non-documentation files
        assert not summarizer._is_documentation_file(Path("main.py"))
        assert not summarizer._is_documentation_file(Path("test.js"))
        assert not summarizer._is_documentation_file(Path("src/utils.go"))

    def test_context_aware_summarization_api_keywords(self, summarizer, markdown_file):
        """Test context-aware summarization with API-related keywords."""
        prompt_keywords = ["api", "authentication", "users", "token"]

        result = summarizer.summarize_file(
            file=markdown_file, prompt_keywords=prompt_keywords, target_ratio=0.5
        )

        # Should use docs-context-aware strategy
        assert result.strategy_used == "docs-context-aware"
        assert result.metadata["is_documentation"] is True
        assert result.metadata["context_aware"] is True
        assert result.metadata["prompt_keywords"] == prompt_keywords

        # Summary should contain relevant sections
        summary = result.summary
        assert "# api.md" in summary
        assert "Authentication" in summary
        assert "User Management" in summary
        assert "GET /api/users" in summary
        assert "POST /api/users" in summary

        # Should preserve code examples
        assert "```http" in summary
        assert "Authorization: Bearer" in summary
        assert "```python" in summary
        assert "import requests" in summary

    def test_context_aware_summarization_config_keywords(self, summarizer, config_file):
        """Test context-aware summarization with configuration keywords."""
        prompt_keywords = ["database", "config", "host", "port"]

        result = summarizer.summarize_file(
            file=config_file, prompt_keywords=prompt_keywords, target_ratio=0.5
        )

        # Should use docs-context-aware strategy
        assert result.strategy_used == "docs-context-aware"

        # Summary should contain relevant sections
        summary = result.summary
        assert "settings.yaml" in summary
        assert "database" in summary.lower()
        assert "host:" in summary
        assert "port:" in summary

    def test_context_aware_disabled(self, config, markdown_file):
        """Test when context-aware documentation is disabled."""
        config.summarizer.docs_context_aware = False
        summarizer = Summarizer(config)

        prompt_keywords = ["api", "authentication"]

        result = summarizer.summarize_file(
            file=markdown_file, prompt_keywords=prompt_keywords, target_ratio=0.5
        )

        # Should NOT use docs-context-aware strategy
        assert result.strategy_used != "docs-context-aware"
        # For markdown files, it should fall back to regular text summarization
        assert result.strategy_used in ["extractive", "compressive", "textrank", "auto"]

    def test_no_prompt_keywords(self, summarizer, markdown_file):
        """Test behavior when no prompt keywords are provided."""
        result = summarizer.summarize_file(
            file=markdown_file, prompt_keywords=None, target_ratio=0.5
        )

        # Should NOT use docs-context-aware strategy without keywords
        assert result.strategy_used != "docs-context-aware"

    def test_non_documentation_file(self, summarizer):
        """Test that non-documentation files don't use context-aware strategy."""
        python_file = FileAnalysis(
            path="src/main.py",
            content="def main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()",
            language="python",
            lines=4,
            size=60,
            encoding="utf-8",
        )

        prompt_keywords = ["main", "function", "print"]

        result = summarizer.summarize_file(
            file=python_file, prompt_keywords=prompt_keywords, target_ratio=0.5
        )

        # Should use code-aware strategy, not docs-context-aware
        assert result.strategy_used == "code-aware"
        assert not result.metadata.get("is_documentation", False)

    def test_no_relevant_sections_fallback(self, summarizer):
        """Test fallback to generic summarization when no relevant sections found."""
        irrelevant_doc = FileAnalysis(
            path="docs/random.md",
            content="# Random Content\n\nThis document contains nothing related to the keywords.\n\nJust some random text about completely different topics.",
            language="markdown",
            lines=4,
            size=100,
            encoding="utf-8",
        )

        prompt_keywords = ["api", "database", "authentication"]

        result = summarizer.summarize_file(
            file=irrelevant_doc, prompt_keywords=prompt_keywords, target_ratio=0.5
        )

        # Should still use docs-context-aware strategy but fall back to generic summarization
        assert result.strategy_used == "docs-context-aware"
        summary = result.summary
        assert "random.md" in summary
        assert "## Summary" in summary  # Fallback section

    @patch(
        "tenets.core.analysis.implementations.generic_analyzer.GenericAnalyzer.extract_context_relevant_sections"
    )
    def test_context_extraction_parameters(self, mock_extract, summarizer, markdown_file):
        """Test that configuration parameters are passed correctly to context extraction."""
        mock_extract.return_value = {
            "relevant_sections": [],
            "metadata": {"total_sections": 0, "matched_sections": 0},
        }

        prompt_keywords = ["api", "test"]

        summarizer.summarize_file(
            file=markdown_file, prompt_keywords=prompt_keywords, target_ratio=0.5
        )

        # Verify the context extraction was called with correct parameters
        mock_extract.assert_called_once()
        args, kwargs = mock_extract.call_args

        assert args[0] == markdown_file.content  # content
        assert str(args[1]) == markdown_file.path  # file_path
        assert args[2] == prompt_keywords  # prompt_keywords
        assert kwargs.get("search_depth") == 2
        assert kwargs.get("min_confidence") == 0.6
        assert kwargs.get("max_sections") == 10

    def test_configuration_customization(self, config):
        """Test that configuration options can be customized."""
        # Modify configuration
        config.summarizer.docs_context_search_depth = 3
        config.summarizer.docs_context_min_confidence = 0.8
        config.summarizer.docs_context_max_sections = 5
        config.summarizer.docs_context_preserve_examples = False

        summarizer = Summarizer(config)

        # Create a simple doc file that should trigger context-aware summarization
        doc_file = FileAnalysis(
            path="test.md",
            content="# Test\n\nSome content with api keywords.",
            language="markdown",
            lines=3,
            size=40,
            encoding="utf-8",
        )

        with patch(
            "tenets.core.analysis.implementations.generic_analyzer.GenericAnalyzer.extract_context_relevant_sections"
        ) as mock_extract:
            mock_extract.return_value = {
                "relevant_sections": [],
                "metadata": {"total_sections": 0, "matched_sections": 0},
            }

            summarizer.summarize_file(file=doc_file, prompt_keywords=["api"], target_ratio=0.5)

            # Check that custom configuration was used
            args, kwargs = mock_extract.call_args
            assert kwargs.get("search_depth") == 3
            assert kwargs.get("min_confidence") == 0.8
            assert kwargs.get("max_sections") == 5

    def test_batch_summarize_with_context(self, summarizer, markdown_file, config_file):
        """Test batch summarization with documentation context."""
        files = [markdown_file, config_file]
        prompt_keywords = ["api", "database", "config"]

        result = summarizer.batch_summarize(
            texts=files, prompt_keywords=prompt_keywords, target_ratio=0.5
        )

        assert len(result.results) == 2

        # Both should use docs-context-aware strategy
        for summary_result in result.results:
            assert summary_result.strategy_used == "docs-context-aware"
            assert summary_result.metadata.get("is_documentation") is True
            assert summary_result.metadata.get("context_aware") is True
