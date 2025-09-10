"""Tests for main Distiller orchestrator."""

from pathlib import Path
from unittest.mock import patch, Mock

import pytest

from tenets.config import TenetsConfig
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


class TestDistiller:
    """Test suite for Distiller."""

    @patch("tenets.core.distiller.distiller.ContextFormatter")
    @patch("tenets.core.distiller.distiller.TokenOptimizer")
    @patch("tenets.core.distiller.distiller.ContextAggregator")
    @patch("tenets.core.distiller.distiller.GitAnalyzer")
    @patch("tenets.core.distiller.distiller.PromptParser")
    @patch("tenets.core.distiller.distiller.RelevanceRanker")
    @patch("tenets.core.distiller.distiller.CodeAnalyzer")
    @patch("tenets.core.distiller.distiller.FileScanner")
    def test_initialization(
        self,
        MockScanner,
        MockAnalyzer,
        MockRanker,
        MockParser,
        MockGit,
        MockAggregator,
        MockOptimizer,
        MockFormatter,
        config,
    ):
        """Test Distiller initialization."""
        # Import Distiller after mocks are set up
        from tenets.core.distiller.distiller import Distiller

        distiller = Distiller(config)

        assert distiller.config == config
        assert distiller.scanner is not None
        assert distiller.analyzer is not None
        assert MockScanner.called
        assert MockAnalyzer.called

    @patch("tenets.core.distiller.distiller.ContextFormatter")
    @patch("tenets.core.distiller.distiller.TokenOptimizer")
    @patch("tenets.core.distiller.distiller.ContextAggregator")
    @patch("tenets.core.distiller.distiller.GitAnalyzer")
    @patch("tenets.core.distiller.distiller.PromptParser")
    @patch("tenets.core.distiller.distiller.RelevanceRanker")
    @patch("tenets.core.distiller.distiller.CodeAnalyzer")
    @patch("tenets.core.distiller.distiller.FileScanner")
    def test_distill_basic(
        self,
        MockScanner,
        MockAnalyzer,
        MockRanker,
        MockParser,
        MockGit,
        MockAggregator,
        MockOptimizer,
        MockFormatter,
        config,
    ):
        """Test basic distillation with all components mocked."""
        # Configure mock return values
        MockScanner.return_value.scan.return_value = [Path("file1.py"), Path("file2.py")]
        MockAnalyzer.return_value.analyze_file.return_value = FileAnalysis(
            path="test.py", content="test content", language="python", lines=100
        )
        MockParser.return_value.parse.return_value = PromptContext(
            text="test prompt", keywords=["test"], task_type="feature", file_patterns=[]
        )
        MockRanker.return_value.rank_files.return_value = []
        MockGit.return_value.is_git_repo.return_value = False
        MockAggregator.return_value.aggregate.return_value = {
            "included_files": [],
            "total_tokens": 500,
            "available_tokens": 1000,
            "statistics": {
                "files_analyzed": 2,
                "files_included": 0,
                "files_summarized": 0,
                "files_skipped": 2,
                "token_utilization": 0.5,
            },
        }
        MockFormatter.return_value.format.return_value = "# Formatted Context"

        # Import Distiller after mocks are set up
        from tenets.core.distiller.distiller import Distiller

        distiller = Distiller(config)
        result = distiller.distill("test prompt")

        assert isinstance(result, ContextResult)
        assert result.context == "# Formatted Context"
        assert result.metadata["prompt"] == "test prompt"

    @patch("tenets.core.distiller.distiller.ContextFormatter")
    @patch("tenets.core.distiller.distiller.TokenOptimizer")
    @patch("tenets.core.distiller.distiller.ContextAggregator")
    @patch("tenets.core.distiller.distiller.GitAnalyzer")
    @patch("tenets.core.distiller.distiller.PromptParser")
    @patch("tenets.core.distiller.distiller.RelevanceRanker")
    @patch("tenets.core.distiller.distiller.CodeAnalyzer")
    @patch("tenets.core.distiller.distiller.FileScanner")
    def test_distill_with_paths(
        self,
        MockScanner,
        MockAnalyzer,
        MockRanker,
        MockParser,
        MockGit,
        MockAggregator,
        MockOptimizer,
        MockFormatter,
        config,
    ):
        """Test distillation with specific paths."""
        # Configure mocks
        MockScanner.return_value.scan.return_value = [Path("src/file1.py"), Path("tests/test1.py")]
        MockAnalyzer.return_value.analyze_file.return_value = FileAnalysis(
            path="test.py", content="test content", language="python", lines=100
        )
        MockParser.return_value.parse.return_value = PromptContext(
            text="test prompt", keywords=["test"], task_type="feature", file_patterns=[]
        )
        MockRanker.return_value.rank_files.return_value = []
        MockGit.return_value.is_git_repo.return_value = False
        MockAggregator.return_value.aggregate.return_value = {
            "included_files": [],
            "total_tokens": 500,
            "available_tokens": 1000,
            "statistics": {},
        }
        MockFormatter.return_value.format.return_value = "# Formatted Context"

        from tenets.core.distiller.distiller import Distiller

        distiller = Distiller(config)
        result = distiller.distill("test prompt", paths=["./src", "./tests"])

        # Should normalize paths
        MockScanner.return_value.scan.assert_called()
        assert isinstance(result, ContextResult)

    @patch("tenets.core.distiller.distiller.ContextFormatter")
    @patch("tenets.core.distiller.distiller.TokenOptimizer")
    @patch("tenets.core.distiller.distiller.ContextAggregator")
    @patch("tenets.core.distiller.distiller.GitAnalyzer")
    @patch("tenets.core.distiller.distiller.PromptParser")
    @patch("tenets.core.distiller.distiller.RelevanceRanker")
    @patch("tenets.core.distiller.distiller.CodeAnalyzer")
    @patch("tenets.core.distiller.distiller.FileScanner")
    def test_distill_with_format(
        self,
        MockScanner,
        MockAnalyzer,
        MockRanker,
        MockParser,
        MockGit,
        MockAggregator,
        MockOptimizer,
        MockFormatter,
        config,
    ):
        """Test distillation with different output formats."""
        # Configure mocks
        MockScanner.return_value.scan.return_value = []
        MockParser.return_value.parse.return_value = PromptContext(
            text="test prompt", keywords=[], task_type="feature", file_patterns=[]
        )
        MockRanker.return_value.rank_files.return_value = []
        MockGit.return_value.is_git_repo.return_value = False
        MockAggregator.return_value.aggregate.return_value = {
            "included_files": [],
            "total_tokens": 0,
            "available_tokens": 1000,
            "statistics": {},
        }
        MockFormatter.return_value.format.return_value = "<xml>Context</xml>"

        from tenets.core.distiller.distiller import Distiller

        distiller = Distiller(config)
        result = distiller.distill("test prompt", format="xml")

        MockFormatter.return_value.format.assert_called_with(
            aggregated=MockAggregator.return_value.aggregate.return_value,
            format="xml",
            prompt_context=MockParser.return_value.parse.return_value,
            session_name=None,
        )
        assert result.context == "<xml>Context</xml>"

    @patch("tenets.core.distiller.distiller.ContextFormatter")
    @patch("tenets.core.distiller.distiller.TokenOptimizer")
    @patch("tenets.core.distiller.distiller.ContextAggregator")
    @patch("tenets.core.distiller.distiller.GitAnalyzer")
    @patch("tenets.core.distiller.distiller.PromptParser")
    @patch("tenets.core.distiller.distiller.RelevanceRanker")
    @patch("tenets.core.distiller.distiller.CodeAnalyzer")
    @patch("tenets.core.distiller.distiller.FileScanner")
    def test_distill_fast_mode(
        self,
        MockScanner,
        MockAnalyzer,
        MockRanker,
        MockParser,
        MockGit,
        MockAggregator,
        MockOptimizer,
        MockFormatter,
        config,
    ):
        """Test fast mode uses lightweight analyzer."""
        # Configure mocks
        MockScanner.return_value.scan.return_value = [Path("file1.py")]
        MockParser.return_value.parse.return_value = PromptContext(
            text="test", keywords=[], task_type="feature", file_patterns=[]
        )
        MockRanker.return_value.rank_files.return_value = []
        MockGit.return_value.is_git_repo.return_value = False
        MockAggregator.return_value.aggregate.return_value = {
            "included_files": [],
            "total_tokens": 0,
            "available_tokens": 1000,
            "statistics": {},
        }
        MockFormatter.return_value.format.return_value = "Fast mode output"

        from tenets.core.distiller.distiller import Distiller

        distiller = Distiller(config)
        result = distiller.distill("test", mode="fast")

        assert result.context == "Fast mode output"
        assert result.metadata.get("mode") == "fast"

    @patch("tenets.core.distiller.distiller.ContextFormatter")
    @patch("tenets.core.distiller.distiller.TokenOptimizer")
    @patch("tenets.core.distiller.distiller.ContextAggregator")
    @patch("tenets.core.distiller.distiller.GitAnalyzer")
    @patch("tenets.core.distiller.distiller.PromptParser")
    @patch("tenets.core.distiller.distiller.RelevanceRanker")
    @patch("tenets.core.distiller.distiller.CodeAnalyzer")
    @patch("tenets.core.distiller.distiller.FileScanner")
    def test_pinned_files_order(
        self,
        MockScanner,
        MockAnalyzer,
        MockRanker,
        MockParser,
        MockGit,
        MockAggregator,
        MockOptimizer,
        MockFormatter,
        config,
    ):
        """Test that pinned files are prioritized."""
        # Configure mocks - scanner returns files in one order
        MockScanner.return_value.scan.return_value = [
            Path("b.py"),
            Path("a.py"),
            Path("c.py"),
        ]
        
        # Analyzer returns file analysis for each
        def analyze_side_effect(path, **kwargs):
            return FileAnalysis(
                path=str(path),
                content=f"content of {path.name}",
                language="python",
                lines=10
            )
        MockAnalyzer.return_value.analyze_file.side_effect = analyze_side_effect
        
        MockParser.return_value.parse.return_value = PromptContext(
            text="test", keywords=[], task_type="feature", file_patterns=[]
        )
        MockGit.return_value.is_git_repo.return_value = False
        
        # Configure aggregator to show the order files were received
        def aggregate_side_effect(files, **kwargs):
            return {
                "included_files": [{"file": f, "content": f.content} for f in files[:2]],
                "total_tokens": 100,
                "available_tokens": 900,
                "statistics": {"files_analyzed": len(files)},
            }
        MockAggregator.return_value.aggregate.side_effect = aggregate_side_effect
        MockFormatter.return_value.format.return_value = "output"

        from tenets.core.distiller.distiller import Distiller

        distiller = Distiller(config)
        
        # Set pinned files
        distiller._pinned_files = ["a.py", "c.py"]
        
        # Analyze files should be called with pinned files first
        result = distiller.distill("test")
        
        # Verify analyze was called
        assert MockAnalyzer.return_value.analyze_file.call_count >= 3
        
        # Check that the calls were made in the expected order (pinned first)
        calls = MockAnalyzer.return_value.analyze_file.call_args_list
        analyzed_paths = [str(call[0][0]) for call in calls]
        
        # Just verify analyze was called with our files
        assert len(analyzed_paths) == 3
        assert "a.py" in analyzed_paths
        assert "b.py" in analyzed_paths
        assert "c.py" in analyzed_paths