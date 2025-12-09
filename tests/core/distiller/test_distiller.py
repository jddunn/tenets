"""Tests for main Distiller orchestrator."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.distiller.distiller import Distiller
from tenets.models.analysis import FileAnalysis
from tenets.models.context import ContextResult, PromptContext


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    config.max_tokens = 10000
    config.follow_symlinks = False
    config.respect_gitignore = True
    return config


@pytest.fixture
def distiller(config):
    """Create Distiller instance with mocked components."""
    with patch("tenets.core.distiller.distiller.FileScanner"):
        with patch("tenets.core.distiller.distiller.CodeAnalyzer"):
            with patch("tenets.core.distiller.distiller.RelevanceRanker"):
                with patch("tenets.core.distiller.distiller.PromptParser"):
                    with patch("tenets.core.distiller.distiller.GitAnalyzer"):
                        with patch("tenets.core.distiller.distiller.ContextAggregator"):
                            with patch("tenets.core.distiller.distiller.TokenOptimizer"):
                                with patch("tenets.core.distiller.distiller.ContextFormatter"):
                                    distiller = Distiller(config)
                                    return distiller


@pytest.fixture
def mock_components(distiller):
    """Setup mock behaviors for distiller components."""
    # Mock scanner
    distiller.scanner.scan.return_value = [Path("file1.py"), Path("file2.py"), Path("file3.py")]

    # Mock analyzer - now uses analyze_files instead of analyze_file
    def mock_analyze_files(file_paths, **kwargs):
        return [
            FileAnalysis(
                path=str(file_path),
                content=f"Content of {file_path.name}",
                language="python",
                lines=100,
            )
            for file_path in file_paths
        ]

    distiller.analyzer.analyze_files.side_effect = mock_analyze_files

    # Mock parser
    distiller.parser.parse.return_value = PromptContext(
        text="test prompt", keywords=["test", "auth"], task_type="feature", file_patterns=[]
    )

    # Mock ranker
    distiller.ranker.rank_files.side_effect = lambda files, **kwargs: files

    # Mock git
    distiller.git.is_git_repo.return_value = True
    distiller.git.get_recent_commits.return_value = []
    distiller.git.get_contributors.return_value = []
    distiller.git.get_current_branch.return_value = "main"

    # Mock aggregator
    distiller.aggregator.aggregate.return_value = {
        "included_files": [
            {
                "file": FileAnalysis(path="file1.py", content="content"),
                "content": "content",
                "summarized": False,
            }
        ],
        "total_tokens": 500,
        "available_tokens": 1000,
        "statistics": {
            "files_analyzed": 3,
            "files_included": 1,
            "files_summarized": 0,
            "files_skipped": 2,
            "token_utilization": 0.5,
        },
    }

    # Mock formatter
    distiller.formatter.format.return_value = "# Formatted Context\n\nContent here"

    return distiller


class TestDistiller:
    """Test suite for Distiller."""

    def test_initialization(self, config):
        """Test Distiller initialization."""
        with patch("tenets.core.distiller.distiller.FileScanner"):
            with patch("tenets.core.distiller.distiller.CodeAnalyzer"):
                distiller = Distiller(config)

                assert distiller.config == config
                assert distiller.scanner is not None
                assert distiller.analyzer is not None

    def test_distill_basic(self, mock_components):
        """Test basic distillation."""
        result = mock_components.distill("implement authentication")

        assert isinstance(result, ContextResult)
        assert result.context == "# Formatted Context\n\nContent here"
        assert result.metadata["prompt"] == "implement authentication"
        assert result.metadata["files_analyzed"] == 3

    def test_distill_with_paths(self, mock_components):
        """Test distillation with specific paths."""
        result = mock_components.distill("test prompt", paths=["./src", "./tests"])

        # Should normalize paths
        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        assert len(call_args[1]["paths"]) == 2

    def test_distill_with_single_path(self, mock_components):
        """Test distillation with single path."""
        result = mock_components.distill("test prompt", paths="./src")

        assert isinstance(result, ContextResult)

    def test_distill_with_format(self, mock_components):
        """Test distillation with different formats."""
        result = mock_components.distill("test prompt", format="xml")

        mock_components.formatter.format.assert_called_with(
            aggregated=mock_components.aggregator.aggregate.return_value,
            format="xml",
            prompt_context=mock_components.parser.parse.return_value,
            session_name=None,
        )

    def test_distill_with_model(self, mock_components):
        """Test distillation with specific model."""
        result = mock_components.distill("test prompt", model="gpt-4", max_tokens=8000)

        # Should pass model to aggregator
        mock_components.aggregator.aggregate.assert_called()
        call_args = mock_components.aggregator.aggregate.call_args
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["max_tokens"] == 8000

    def test_distill_with_mode(self, mock_components):
        """Test distillation with different modes."""
        result = mock_components.distill("test prompt", mode="thorough")

        # Should affect analysis depth - now uses analyze_files
        mock_components.analyzer.analyze_files.assert_called()
        call_args = mock_components.analyzer.analyze_files.call_args
        assert call_args[1]["deep"] == True

    def test_distill_fast_mode(self, mock_components):
        """Test distillation in fast mode."""
        result = mock_components.distill("test prompt", mode="fast")

        # Should not use deep analysis - now uses analyze_files
        mock_components.analyzer.analyze_files.assert_called()
        call_args = mock_components.analyzer.analyze_files.call_args
        assert call_args[1]["deep"] == False

    def test_distill_without_git(self, mock_components):
        """Test distillation without git context."""
        result = mock_components.distill("test prompt", include_git=False)

        # Should not call git methods
        mock_components.git.get_recent_commits.assert_not_called()

    def test_distill_with_git(self, mock_components):
        """Test distillation with git context."""
        result = mock_components.distill("test prompt", include_git=True)

        # Should call git methods
        mock_components.git.is_git_repo.assert_called()

    def test_distill_with_session(self, mock_components):
        """Test distillation with session."""
        result = mock_components.distill("test prompt", session_name="test-session")

        assert result.metadata["session"] == "test-session"

    def test_distill_with_patterns(self, mock_components):
        """Test distillation with file patterns."""
        result = mock_components.distill(
            "test prompt", include_patterns=["*.py", "*.js"], exclude_patterns=["test_*.py"]
        )

        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        assert call_args[1]["include_patterns"] == ["*.py", "*.js"]
        # Check that the user-provided pattern is in the exclude list
        # (other test patterns may be added automatically)
        assert "test_*.py" in call_args[1]["exclude_patterns"]

    def test_parse_prompt(self, mock_components):
        """Test prompt parsing."""
        context = mock_components._parse_prompt("implement OAuth2")

        mock_components.parser.parse.assert_called_with("implement OAuth2")
        assert isinstance(context, PromptContext)

    def test_normalize_paths_none(self, mock_components):
        """Test path normalization with None."""
        paths = mock_components._normalize_paths(None)

        assert len(paths) == 1
        assert paths[0] == Path.cwd()

    def test_normalize_paths_string(self, mock_components):
        """Test path normalization with string."""
        paths = mock_components._normalize_paths("./src")

        assert len(paths) == 1
        assert paths[0] == Path("./src")

    def test_normalize_paths_list(self, mock_components):
        """Test path normalization with list."""
        paths = mock_components._normalize_paths(["./src", "./tests"])

        assert len(paths) == 2
        assert paths[0] == Path("./src")
        assert paths[1] == Path("./tests")

    def test_discover_files(self, mock_components):
        """Test file discovery."""
        prompt_context = PromptContext(text="test", file_patterns=["*.py"])

        files = mock_components._discover_files(paths=[Path()], prompt_context=prompt_context)

        assert len(files) == 3
        mock_components.scanner.scan.assert_called()

    def test_analyze_files(self, mock_components):
        """Test file analysis."""
        files = [Path("file1.py"), Path("file2.py")]
        prompt_context = PromptContext(text="test")

        analyzed = mock_components._analyze_files(
            files=files, mode="balanced", prompt_context=prompt_context
        )

        assert len(analyzed) == 2
        assert all(isinstance(f, FileAnalysis) for f in analyzed)

    def test_analyze_files_with_errors(self, mock_components):
        """Test file analysis with errors."""
        mock_components.analyzer.analyze_files.side_effect = Exception("Analysis failed")

        files = [Path("file1.py")]
        prompt_context = PromptContext(text="test")

        analyzed = mock_components._analyze_files(
            files=files, mode="fast", prompt_context=prompt_context
        )

        # Should handle errors gracefully
        assert len(analyzed) == 0

    def test_rank_files(self, mock_components):
        """Test file ranking."""
        files = [
            FileAnalysis(path="file1.py", content="content"),
            FileAnalysis(path="file2.py", content="content"),
        ]
        prompt_context = PromptContext(text="test")

        ranked = mock_components._rank_files(
            files=files, prompt_context=prompt_context, mode="balanced"
        )

        # Verify rank_files was called with expected parameters
        mock_components.ranker.rank_files.assert_called_with(
            files=files,
            prompt_context=prompt_context,
            algorithm="balanced",
            deadline=None,  # deadline parameter now passed
        )

    def test_get_git_context(self, mock_components):
        """Test git context extraction."""
        paths = [Path()]
        prompt_context = PromptContext(text="recent changes")
        files = [FileAnalysis(path="file1.py", content="content")]

        git_context = mock_components._get_git_context(
            paths=paths, prompt_context=prompt_context, files=files
        )

        assert git_context is not None
        assert "recent_commits" in git_context
        assert "contributors" in git_context
        assert "branch" in git_context
        assert "recent_changes" in git_context  # Because "recent" in prompt

    def test_get_git_context_no_repo(self, mock_components):
        """Test git context when not a repo."""
        mock_components.git.is_git_repo.return_value = False

        git_context = mock_components._get_git_context(
            paths=[Path()], prompt_context=PromptContext(text="test"), files=[]
        )

        assert git_context is None

    def test_aggregate_files(self, mock_components):
        """Test file aggregation."""
        files = [FileAnalysis(path="file1.py", content="content")]
        prompt_context = PromptContext(text="test")

        aggregated = mock_components._aggregate_files(
            files=files,
            prompt_context=prompt_context,
            max_tokens=5000,
            model="gpt-4",
            git_context={"branch": "main"},
        )

        assert "included_files" in aggregated
        assert "total_tokens" in aggregated

    def test_format_output(self, mock_components):
        """Test output formatting."""
        aggregated = {"included_files": [], "statistics": {}}
        prompt_context = PromptContext(text="test")

        formatted = mock_components._format_output(
            aggregated=aggregated,
            format="markdown",
            prompt_context=prompt_context,
            session_name="test-session",
        )

        assert formatted == "# Formatted Context\n\nContent here"

    def test_build_result(self, mock_components):
        """Test result building."""
        result = mock_components._build_result(
            formatted="Formatted content", metadata={"key": "value"}
        )

        assert isinstance(result, ContextResult)
        assert result.context == "Formatted content"
        assert result.metadata["key"] == "value"

    def test_distill_end_to_end(self, mock_components):
        """Test complete distillation flow."""
        result = mock_components.distill(
            prompt="implement OAuth2 authentication for the API",
            paths="./src",
            format="markdown",
            model="gpt-4",
            max_tokens=8000,
            mode="thorough",
            include_git=True,
            session_name="oauth-impl",
            include_patterns=["*.py"],
            exclude_patterns=["test_*.py"],
        )

        # Verify all components were called
        assert mock_components.parser.parse.called
        assert mock_components.scanner.scan.called
        assert mock_components.analyzer.analyze_files.called
        assert mock_components.ranker.rank_files.called
        assert mock_components.aggregator.aggregate.called
        assert mock_components.formatter.format.called

        # Verify result
        assert isinstance(result, ContextResult)
        assert result.metadata["prompt"] == "implement OAuth2 authentication for the API"
        assert result.metadata["session"] == "oauth-impl"
        assert result.metadata["mode"] == "thorough"

    def test_full_mode_metadata(self, mock_components):
        """Full mode flag should propagate to metadata."""
        result = mock_components.distill("prompt", full=True)
        assert result.metadata.get("full_mode") is True

    def test_transform_flags_metadata(self, mock_components):
        """condense/remove_comments flags reflected in metadata."""
        result = mock_components.distill("prompt", condense=True, remove_comments=True)
        assert result.metadata.get("condense") is True
        assert result.metadata.get("remove_comments") is True

    def test_pinned_files_order(self, config):
        """Pinned files should be analyzed first."""
        # Fresh distiller with real method mocks
        with patch("tenets.core.distiller.distiller.FileScanner") as fs:
            with patch("tenets.core.distiller.distiller.CodeAnalyzer"):
                with patch("tenets.core.distiller.distiller.RelevanceRanker"):
                    with patch("tenets.core.distiller.distiller.PromptParser"):
                        with patch("tenets.core.distiller.distiller.GitAnalyzer"):
                            with patch("tenets.core.distiller.distiller.ContextAggregator"):
                                with patch("tenets.core.distiller.distiller.TokenOptimizer"):
                                    with patch("tenets.core.distiller.distiller.ContextFormatter"):
                                        dist = Distiller(config)
                                        # Configure scanner order different from pinned order
                                        fs.return_value.scan.return_value = [
                                            Path("b.py"),
                                            Path("a.py"),
                                            Path("c.py"),
                                        ]
                                        # Analyzer returns simple FileAnalysis objects
                                        inst_analyzer = dist.analyzer
                                        analyzed_files = []

                                        def analyze_files_side_effect(file_paths, **kwargs):
                                            analyzed_files.extend(
                                                [Path(p).name for p in file_paths]
                                            )
                                            return [
                                                FileAnalysis(path=str(path), content="x")
                                                for path in file_paths
                                            ]

                                        inst_analyzer.analyze_files.side_effect = (
                                            analyze_files_side_effect
                                        )
                                        dist.ranker.rank_files.side_effect = (
                                            lambda files, **k: files
                                        )
                                        dist.parser.parse.return_value = PromptContext(text="p")
                                        dist.aggregator.aggregate.return_value = {
                                            "included_files": [],
                                            "total_tokens": 0,
                                            "available_tokens": 1000,
                                            "statistics": {
                                                "files_analyzed": 3,
                                                "files_included": 0,
                                                "files_summarized": 0,
                                                "files_skipped": 3,
                                                "token_utilization": 0,
                                            },
                                        }
                                        dist.formatter.format.return_value = "ctx"
                                        # Distill with a.py pinned
                                        result = dist.distill("p", pinned_files=[Path("a.py")])
                                        # Pinned files should be first in the list passed to analyze
                                        assert analyzed_files[0] == "a.py"
                                        assert result.metadata.get("files_analyzed") == 3

    def test_test_exclusion_with_test_intent(self, mock_components):
        """Test that test files are included when prompt has test intent."""
        # Mock parser to return test intent
        mock_components.parser.parse.return_value = PromptContext(
            text="write unit tests", intent="test", include_tests=True
        )

        result = mock_components.distill("write unit tests for auth")

        # Verify scanner was called without test exclusions
        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        exclude_patterns = call_args[1].get("exclude_patterns", [])

        # Should not have test patterns in exclusions since include_tests=True
        test_pattern_found = any("test" in pattern.lower() for pattern in exclude_patterns)
        assert not test_pattern_found

    def test_test_exclusion_with_non_test_intent(self, mock_components):
        """Test that test files are excluded for non-test prompts."""
        # Mock parser to return non-test intent
        mock_components.parser.parse.return_value = PromptContext(
            text="explain auth flow", intent="understand", include_tests=False
        )

        # Set up config to exclude tests by default
        mock_components.config.scanner.exclude_tests_by_default = True
        mock_components.config.scanner.test_patterns = ["test_*.py", "*_test.py"]
        mock_components.config.scanner.test_directories = ["tests", "__tests__"]

        result = mock_components.distill("explain auth flow")

        # Verify scanner was called with test exclusions
        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        exclude_patterns = call_args[1].get("exclude_patterns", [])

        # Should have test patterns in exclusions
        assert "test_*.py" in exclude_patterns
        assert "*_test.py" in exclude_patterns
        assert "**/tests/**" in exclude_patterns

    def test_test_inclusion_override(self, mock_components):
        """Test explicit test inclusion parameter overrides prompt detection."""
        # Mock parser to return non-test intent
        mock_components.parser.parse.return_value = PromptContext(
            text="explain auth flow", intent="understand", include_tests=False
        )

        # Override with include_tests=True
        result = mock_components.distill("explain auth flow", include_tests=True)

        # Should call parser.parse but then override the include_tests flag
        mock_components.parser.parse.assert_called()

        # Verify scanner was called without test exclusions (due to override)
        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        exclude_patterns = call_args[1].get("exclude_patterns", [])

        # Should not have test patterns since include_tests was forced to True
        test_pattern_found = any("test" in pattern.lower() for pattern in exclude_patterns)
        assert not test_pattern_found

    def test_test_exclusion_override(self, mock_components):
        """Test explicit test exclusion parameter overrides prompt detection."""
        # Mock parser to return test intent
        mock_components.parser.parse.return_value = PromptContext(
            text="write unit tests", intent="test", include_tests=True
        )

        # Set up config
        mock_components.config.scanner.exclude_tests_by_default = True
        mock_components.config.scanner.test_patterns = ["test_*.py", "*_test.py"]
        mock_components.config.scanner.test_directories = ["tests"]

        # Override with include_tests=False
        result = mock_components.distill("write unit tests", include_tests=False)

        # Verify scanner was called with test exclusions (due to override)
        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        exclude_patterns = call_args[1].get("exclude_patterns", [])

        # Should have test patterns since include_tests was forced to False
        assert "test_*.py" in exclude_patterns
        assert "*_test.py" in exclude_patterns

    def test_discover_files_test_exclusion_logic(self, mock_components):
        """Test the _discover_files method test exclusion logic directly."""
        # Test with include_tests=False
        prompt_context = PromptContext(text="test", include_tests=False)
        mock_components.config.scanner.exclude_tests_by_default = True
        mock_components.config.scanner.test_patterns = ["test_*.py", "*.test.js"]
        mock_components.config.scanner.test_directories = ["tests", "spec"]

        files = mock_components._discover_files(
            paths=[Path()], prompt_context=prompt_context, exclude_patterns=["*.log"]
        )

        # Verify scanner was called with correct exclusion patterns
        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        exclude_patterns = call_args[1]["exclude_patterns"]

        # Should include original exclusions plus test patterns
        assert "*.log" in exclude_patterns
        assert "test_*.py" in exclude_patterns
        assert "*.test.js" in exclude_patterns
        assert "**/tests/**" in exclude_patterns
        assert "**/spec/**" in exclude_patterns

        # Test with include_tests=True
        prompt_context.include_tests = True

        files = mock_components._discover_files(
            paths=[Path()], prompt_context=prompt_context, exclude_patterns=["*.log"]
        )

        # Should only have original exclusions, no test patterns
        call_args = mock_components.scanner.scan.call_args
        exclude_patterns = call_args[1]["exclude_patterns"]
        assert "*.log" in exclude_patterns
        assert "test_*.py" not in exclude_patterns
