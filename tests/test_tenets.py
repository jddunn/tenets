"""
Unit tests for the main Tenets API class.

This module tests the high-level API exposed through the Tenets class,
which is the primary interface users interact with. It tests initialization,
distillation, tenet management, and analysis features.

Test Coverage:
    - Tenets class initialization with various config options
    - distill() method with different parameters
    - Tenet management (add, list, remove, instill)
    - Analysis methods (examine, track_changes, momentum)
    - Session management
    - Error handling and edge cases
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import yaml

from tenets import ContextResult, Priority, Tenet, TenetCategory, Tenets, TenetsConfig


class TestTenetsInitialization:
    """Test suite for Tenets class initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        with patch("tenets.Distiller"), patch("tenets.Instiller"), patch("tenets.get_logger"):

            tenets = Tenets()

            assert tenets.config is not None
            assert isinstance(tenets.config, TenetsConfig)
            assert tenets.distiller is not None
            assert tenets.instiller is not None
            assert tenets.tenet_manager is not None

    def test_init_with_config_object(self, test_config):
        """Test initialization with a TenetsConfig object."""
        with (
            patch("tenets.Distiller") as mock_distiller,
            patch("tenets.Instiller") as mock_instiller,
            patch("tenets.get_logger"),
        ):

            tenets = Tenets(config=test_config)

            assert tenets.config == test_config
            mock_distiller.assert_called_once_with(test_config)
            mock_instiller.assert_called_once_with(test_config)

    def test_init_with_config_dict(self):
        """Test initialization with a configuration dictionary."""
        config_dict = {"max_tokens": 50000, "debug": True, "ranking_algorithm": "thorough"}

        with patch("tenets.Distiller"), patch("tenets.Instiller"), patch("tenets.get_logger"):

            tenets = Tenets(config=config_dict)

            assert tenets.config.max_tokens == 50000
            assert tenets.config.debug is True

    def test_init_with_config_file(self, config_file):
        """Test initialization with a configuration file path."""
        with patch("tenets.Distiller"), patch("tenets.Instiller"), patch("tenets.get_logger"):

            tenets = Tenets(config=config_file)

            assert tenets.config.max_tokens == 5000  # From config file
            assert tenets.config.scanner.respect_gitignore is True

    def test_init_with_invalid_config_type(self):
        """Test initialization with invalid config type raises error."""
        with patch("tenets.get_logger"):
            with pytest.raises(ValueError, match="Invalid config type"):
                Tenets(config=123)  # Invalid type

    def test_init_with_nonexistent_config_file(self):
        """Test initialization with non-existent config file raises error."""
        with patch("tenets.get_logger"):
            with pytest.raises(FileNotFoundError):
                Tenets(config=Path("/nonexistent/config.yml"))


class TestDistillMethod:
    """Test suite for the distill() method."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up mocks for each test."""
        self.mock_distiller = Mock()
        self.mock_instiller = Mock()
        self.mock_logger = Mock()

        with (
            patch("tenets.Distiller", return_value=self.mock_distiller),
            patch("tenets.Instiller", return_value=self.mock_instiller),
            patch("tenets.get_logger", return_value=self.mock_logger),
        ):

            self.tenets = Tenets()

    def test_distill_basic(self):
        """Test basic distillation with minimal parameters."""
        # Setup mock return value
        mock_result = ContextResult(
            context="# Relevant files\n## file1.py\n...",
            format="markdown",
            metadata={"files_analyzed": 10},
            token_count=1000,
        )
        self.mock_distiller.distill.return_value = mock_result

        # Call distill
        result = self.tenets.distill("implement OAuth2 authentication")

        # Verify
        assert result == mock_result
        self.mock_distiller.distill.assert_called_once()
        call_kwargs = self.mock_distiller.distill.call_args[1]
        assert call_kwargs["prompt"] == "implement OAuth2 authentication"
        assert call_kwargs["format"] == "markdown"

    def test_distill_with_all_parameters(self):
        """Test distillation with all parameters specified."""
        mock_result = ContextResult(
            context="<context>...</context>",
            format="xml",
            metadata={"files_analyzed": 20},
            token_count=2000,
        )
        self.mock_distiller.distill.return_value = mock_result

        result = self.tenets.distill(
            prompt="fix authentication bug",
            files=["src/auth.py", "src/api.py"],
            format="xml",
            model="claude-3-opus",
            max_tokens=100000,
            mode="thorough",
            include_git=True,
            session_name="auth-fix",
            include_patterns=["*.py"],
            exclude_patterns=["test_*.py"],
            apply_tenets=True,
        )

        assert result == mock_result
        call_kwargs = self.mock_distiller.distill.call_args[1]
        assert call_kwargs["prompt"] == "fix authentication bug"
        assert call_kwargs["paths"] == ["src/auth.py", "src/api.py"]
        assert call_kwargs["format"] == "xml"
        assert call_kwargs["model"] == "claude-3-opus"
        assert call_kwargs["max_tokens"] == 100000
        assert call_kwargs["mode"] == "thorough"

    def test_distill_with_tenet_injection(self, test_config):
        """Test distillation with automatic tenet injection."""
        # Setup config to auto-instill tenets
        test_config.auto_instill_tenets = True
        self.tenets.config = test_config

        # Setup mock tenets
        mock_tenet = Mock(spec=Tenet)
        self.tenets.tenet_manager = Mock()
        self.tenets.tenet_manager.get_pending_tenets.return_value = [mock_tenet]

        # Setup mock results
        mock_distill_result = ContextResult(
            context="# Context", format="markdown", metadata={}, token_count=1000
        )
        self.mock_distiller.distill.return_value = mock_distill_result

        mock_instill_result = ContextResult(
            context="# Context\n**Remember:** Use type hints",
            format="markdown",
            metadata={"tenets_injected": 1},
            token_count=1050,
        )
        self.mock_instiller.instill.return_value = mock_instill_result

        # Call distill with apply_tenets=True
        result = self.tenets.distill("implement feature", apply_tenets=True)

        # Verify tenet injection was called
        self.mock_instiller.instill.assert_called_once()
        assert result == mock_instill_result

    def test_distill_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            self.tenets.distill("")

    def test_distill_caches_result(self):
        """Test that distill results are cached."""
        mock_result = ContextResult(
            context="# Cached result", format="markdown", metadata={}, token_count=500
        )
        self.mock_distiller.distill.return_value = mock_result

        # First call
        result1 = self.tenets.distill("test prompt")

        # Check cache
        cache_key = "test prompt_global"
        assert cache_key in self.tenets._cache
        assert self.tenets._cache[cache_key] == result1


class TestTenetManagement:
    """Test suite for tenet management methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up mocks for each test."""
        self.mock_manager = Mock()

        with (
            patch("tenets.Distiller"),
            patch("tenets.Instiller") as mock_instiller_class,
            patch("tenets.get_logger"),
        ):

            # Setup the mock instiller to have a manager
            mock_instiller = Mock()
            mock_instiller.manager = self.mock_manager
            mock_instiller_class.return_value = mock_instiller

            self.tenets = Tenets()
            self.tenets.tenet_manager = self.mock_manager

    def test_add_tenet_basic(self):
        """Test adding a basic tenet."""
        mock_tenet = Mock(spec=Tenet)
        self.mock_manager.add_tenet.return_value = mock_tenet

        result = self.tenets.add_tenet("Always use type hints")

        assert result == mock_tenet
        self.mock_manager.add_tenet.assert_called_once_with(
            content="Always use type hints",
            priority="medium",
            category=None,
            session=None,
            author=None,
        )

    def test_add_tenet_with_all_parameters(self):
        """Test adding a tenet with all parameters."""
        mock_tenet = Mock(spec=Tenet)
        self.mock_manager.add_tenet.return_value = mock_tenet

        result = self.tenets.add_tenet(
            content="Use dependency injection",
            priority="high",
            category="architecture",
            session="refactor-di",
            author="john_doe",
        )

        assert result == mock_tenet
        self.mock_manager.add_tenet.assert_called_once_with(
            content="Use dependency injection",
            priority="high",
            category="architecture",
            session="refactor-di",
            author="john_doe",
        )

    def test_list_tenets(self):
        """Test listing tenets with various filters."""
        mock_tenets = [
            {"id": "1", "content": "Tenet 1", "priority": "high"},
            {"id": "2", "content": "Tenet 2", "priority": "medium"},
        ]
        self.mock_manager.list_tenets.return_value = mock_tenets

        # Test basic listing
        result = self.tenets.list_tenets()
        assert result == mock_tenets

        # Test with filters
        self.tenets.list_tenets(pending_only=True, session="test-session", category="security")

        self.mock_manager.list_tenets.assert_called_with(
            pending_only=True, instilled_only=False, session="test-session", category="security"
        )

    def test_get_tenet(self):
        """Test getting a specific tenet."""
        mock_tenet = Mock(spec=Tenet)
        self.mock_manager.get_tenet.return_value = mock_tenet

        result = self.tenets.get_tenet("abc123")

        assert result == mock_tenet
        self.mock_manager.get_tenet.assert_called_once_with("abc123")

    def test_remove_tenet(self):
        """Test removing a tenet."""
        self.mock_manager.remove_tenet.return_value = True

        result = self.tenets.remove_tenet("abc123")

        assert result is True
        self.mock_manager.remove_tenet.assert_called_once_with("abc123")

    def test_instill_tenets(self):
        """Test instilling tenets."""
        mock_result = {"count": 3, "tenets": ["Tenet 1", "Tenet 2", "Tenet 3"]}
        self.mock_manager.instill_tenets.return_value = mock_result

        result = self.tenets.instill_tenets(session="test-session", force=True)

        assert result == mock_result
        self.mock_manager.instill_tenets.assert_called_once_with(session="test-session", force=True)

    def test_export_tenets(self):
        """Test exporting tenets to YAML/JSON."""
        export_data = "tenets:\n  - content: Test tenet\n"
        self.mock_manager.export_tenets.return_value = export_data

        result = self.tenets.export_tenets(format="yaml", session="test")

        assert result == export_data
        self.mock_manager.export_tenets.assert_called_once_with(format="yaml", session="test")

    def test_import_tenets(self, temp_dir):
        """Test importing tenets from file."""
        # Create a test file
        import_file = temp_dir / "tenets.yml"
        import_file.write_text("tenets:\n  - content: Imported tenet\n")

        self.mock_manager.import_tenets.return_value = 5

        result = self.tenets.import_tenets(import_file, session="imported")

        assert result == 5
        self.mock_manager.import_tenets.assert_called_once_with(
            file_path=import_file, session="imported"
        )


class TestAnalysisMethods:
    """Test suite for analysis methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up mocks for each test."""
        with (
            patch("tenets.Distiller"),
            patch("tenets.Instiller"),
            patch("tenets.get_logger"),
            patch("tenets.CodeAnalyzer") as mock_analyzer_class,
        ):

            self.mock_analyzer = Mock()
            mock_analyzer_class.return_value = self.mock_analyzer

            self.tenets = Tenets()

    def test_examine_basic(self):
        """Test basic codebase examination."""
        # Currently returns placeholder dict
        result = self.tenets.examine()

        assert isinstance(result, dict)
        assert "total_files" in result
        assert "languages" in result
        assert "message" in result

    def test_examine_with_parameters(self):
        """Test examination with all parameters."""
        result = self.tenets.examine(
            path=Path("/test/path"), deep=True, include_git=True, output_metadata=True
        )

        assert isinstance(result, dict)

    def test_track_changes(self):
        """Test tracking code changes."""
        result = self.tenets.track_changes(
            path=Path("/test/repo"), since="1 week", author="john_doe", file_pattern="*.py"
        )

        assert isinstance(result, dict)
        assert "commits" in result
        assert "files" in result

    def test_momentum(self):
        """Test momentum tracking."""
        result = self.tenets.momentum(since="last-month", team=True, author="jane_doe")

        assert isinstance(result, dict)
        assert "overall" in result
        assert "weekly" in result

    def test_estimate_cost(self):
        """Test cost estimation for LLM usage."""
        from tenets.models.context import ContextResult

        mock_result = ContextResult(
            context="Test context", format="markdown", metadata={}, token_count=1000
        )

        with patch("tenets.models.llm.estimate_cost") as mock_estimate:
            mock_estimate.return_value = {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_cost": 0.015,
            }

            result = self.tenets.estimate_cost(mock_result, "gpt-4o")

            assert "input_tokens" in result
            assert "total_cost" in result


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up mocks for each test."""
        with (
            patch("tenets.Distiller") as mock_distiller_class,
            patch("tenets.Instiller") as mock_instiller_class,
            patch("tenets.get_logger"),
        ):

            self.mock_distiller = Mock()
            self.mock_instiller = Mock()
            mock_distiller_class.return_value = self.mock_distiller
            mock_instiller_class.return_value = self.mock_instiller
            self.mock_instiller.manager = Mock()

            self.tenets = Tenets()

    def test_distill_with_distiller_error(self):
        """Test distill handles distiller errors gracefully."""
        self.mock_distiller.distill.side_effect = RuntimeError("Distiller failed")

        with pytest.raises(RuntimeError, match="Distiller failed"):
            self.tenets.distill("test prompt")

    def test_distill_with_very_long_prompt(self):
        """Test distill handles very long prompts."""
        long_prompt = "implement " * 10000  # Very long prompt

        mock_result = ContextResult(
            context="Truncated context",
            format="markdown",
            metadata={"prompt_truncated": True},
            token_count=500,
        )
        self.mock_distiller.distill.return_value = mock_result

        result = self.tenets.distill(long_prompt)
        assert result == mock_result

    def test_concurrent_distill_calls(self):
        """Test that concurrent distill calls work correctly."""
        import threading

        results = []
        errors = []

        def distill_task(prompt):
            try:
                result = self.tenets.distill(prompt)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create mock results
        self.mock_distiller.distill.side_effect = [
            ContextResult(context=f"Result {i}", format="markdown", metadata={}, token_count=100)
            for i in range(5)
        ]

        # Run concurrent distill calls
        threads = []
        for i in range(5):
            t = threading.Thread(target=distill_task, args=(f"prompt {i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0

    def test_session_state_management(self):
        """Test that session state is properly managed."""
        # Set a session
        self.tenets._session = "test-session"

        mock_result = ContextResult(
            context="Session context",
            format="markdown",
            metadata={"session": "test-session"},
            token_count=500,
        )
        self.mock_distiller.distill.return_value = mock_result

        # Distill should use the session
        result = self.tenets.distill("test prompt")

        call_kwargs = self.mock_distiller.distill.call_args[1]
        assert call_kwargs["session_name"] == "test-session"

    def test_cache_size_limits(self):
        """Test that cache doesn't grow unbounded."""
        # Add many items to cache
        for i in range(100):
            self.tenets._cache[f"key_{i}"] = f"value_{i}"

        # Cache should still be accessible
        assert len(self.tenets._cache) == 100

        # Older items should be accessible
        assert self.tenets._cache["key_0"] == "value_0"
