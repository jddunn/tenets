"""Tests for ContextAggregator."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tenets.core.distiller.aggregator import ContextAggregator, AggregationStrategy
from tenets.models.analysis import FileAnalysis
from tenets.models.context import PromptContext
from tenets.models.summary import FileSummary
from tenets.config import TenetsConfig


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    config.max_tokens = 10000
    return config


@pytest.fixture
def aggregator(config):
    """Create ContextAggregator instance."""
    return ContextAggregator(config)


@pytest.fixture
def sample_files():
    """Create sample FileAnalysis objects."""
    files = []
    for i in range(5):
        file = FileAnalysis(
            path=f"file{i}.py",
            content=f"# File {i}\n" + "x" * (100 * (i + 1)),  # Varying sizes
            language="python",
            lines=10 + i * 5,
            relevance_score=0.9 - i * 0.15,  # Decreasing relevance
        )
        files.append(file)
    return files


@pytest.fixture
def prompt_context():
    """Create sample PromptContext."""
    return PromptContext(
        text="implement authentication", keywords=["auth", "login", "user"], task_type="feature"
    )


@pytest.fixture
def mock_summarizer():
    """Create mock Summarizer."""
    summarizer = Mock()
    summarizer.summarize_file.return_value = FileSummary(
        path="test.py",
        content="Summarized content",
        summary_tokens=50,
        original_tokens=200,
        compression_ratio=0.25,
        instructions=["This is a summary"],
    )
    return summarizer


class TestAggregationStrategy:
    """Test suite for AggregationStrategy."""

    def test_default_strategy(self):
        """Test default strategy settings."""
        strategy = AggregationStrategy(name="test")

        assert strategy.name == "test"
        assert strategy.max_full_files == 10
        assert strategy.summarize_threshold == 0.7
        assert strategy.min_relevance == 0.3
        assert strategy.preserve_structure == True

    def test_custom_strategy(self):
        """Test custom strategy settings."""
        strategy = AggregationStrategy(
            name="custom",
            max_full_files=5,
            summarize_threshold=0.8,
            min_relevance=0.5,
            preserve_structure=False,
        )

        assert strategy.max_full_files == 5
        assert strategy.summarize_threshold == 0.8
        assert strategy.min_relevance == 0.5
        assert strategy.preserve_structure == False


class TestContextAggregator:
    """Test suite for ContextAggregator."""

    def test_initialization(self, config):
        """Test aggregator initialization."""
        aggregator = ContextAggregator(config)

        assert aggregator.config == config
        assert aggregator.summarizer is not None
        assert "greedy" in aggregator.strategies
        assert "balanced" in aggregator.strategies
        assert "conservative" in aggregator.strategies

    def test_aggregate_empty_files(self, aggregator, prompt_context):
        """Test aggregating with no files."""
        result = aggregator.aggregate(files=[], prompt_context=prompt_context, max_tokens=1000)

        assert result["included_files"] == []
        assert result["total_tokens"] == 0
        assert result["statistics"]["files_analyzed"] == 0

    def test_aggregate_greedy_strategy(self, aggregator, sample_files, prompt_context):
        """Test aggregation with greedy strategy."""
        with patch.object(aggregator.summarizer, "summarize_file") as mock_summarize:
            mock_summarize.return_value = FileSummary(
                path="test.py", content="Summary", summary_tokens=30, original_tokens=100
            )

            result = aggregator.aggregate(
                files=sample_files,
                prompt_context=prompt_context,
                max_tokens=1000,
                strategy="greedy",
            )

            assert len(result["included_files"]) > 0
            assert result["statistics"]["files_analyzed"] == 5
            assert result["total_tokens"] <= 1000

    def test_aggregate_balanced_strategy(self, aggregator, sample_files, prompt_context):
        """Test aggregation with balanced strategy."""
        result = aggregator.aggregate(
            files=sample_files, prompt_context=prompt_context, max_tokens=1000, strategy="balanced"
        )

        assert len(result["included_files"]) > 0
        assert result["total_tokens"] <= 1000

    def test_aggregate_conservative_strategy(self, aggregator, sample_files, prompt_context):
        """Test aggregation with conservative strategy."""
        result = aggregator.aggregate(
            files=sample_files,
            prompt_context=prompt_context,
            max_tokens=1000,
            strategy="conservative",
        )

        # Conservative should include fewer files
        assert len(result["included_files"]) <= 5

    def test_aggregate_respects_min_relevance(self, aggregator, prompt_context):
        """Test that files below min relevance are skipped."""
        files = [
            FileAnalysis(path="high.py", content="content", relevance_score=0.8),
            FileAnalysis(path="low.py", content="content", relevance_score=0.1),  # Below threshold
        ]

        result = aggregator.aggregate(
            files=files,
            prompt_context=prompt_context,
            max_tokens=1000,
            strategy="balanced",  # min_relevance=0.3
        )

        included_paths = [f["file"].path for f in result["included_files"]]
        assert "high.py" in included_paths
        assert "low.py" not in included_paths

    def test_aggregate_with_git_context(self, aggregator, sample_files, prompt_context):
        """Test aggregation with git context."""
        git_context = {
            "recent_commits": [{"sha": "abc123", "message": "test"}] * 5,
            "contributors": [{"name": "dev", "commits": 10}] * 3,
        }

        result = aggregator.aggregate(
            files=sample_files,
            prompt_context=prompt_context,
            max_tokens=2000,
            git_context=git_context,
        )

        assert result["git_context"] == git_context
        # Should reserve tokens for git context
        assert result["available_tokens"] < 2000

    def test_aggregate_summarization(self, aggregator, prompt_context):
        """Test file summarization when needed."""
        # Create large files that won't all fit
        large_files = [
            FileAnalysis(
                path=f"large{i}.py",
                content="x" * 5000,  # Large content
                relevance_score=0.7 - i * 0.1,
            )
            for i in range(3)
        ]

        with patch.object(aggregator.summarizer, "summarize_file") as mock_summarize:
            mock_summarize.return_value = FileSummary(
                path="test.py", content="Summarized", summary_tokens=100, original_tokens=1000
            )

            result = aggregator.aggregate(
                files=large_files, prompt_context=prompt_context, max_tokens=1000
            )

            # Should have called summarizer for some files
            assert mock_summarize.called
            assert result["statistics"]["files_summarized"] > 0

    def test_aggregate_sorting(self, aggregator, sample_files, prompt_context):
        """Test that files are sorted by relevance in output."""
        result = aggregator.aggregate(
            files=sample_files, prompt_context=prompt_context, max_tokens=5000
        )

        # Check files are sorted by relevance
        relevance_scores = [f["file"].relevance_score for f in result["included_files"]]
        assert relevance_scores == sorted(relevance_scores, reverse=True)

    def test_estimate_git_tokens(self, aggregator):
        """Test git token estimation."""
        git_context = {
            "recent_commits": [{}] * 10,  # 10 commits
            "contributors": [{}] * 5,  # 5 contributors
            "recent_changes": [{}] * 3,  # 3 changes
        }

        tokens = aggregator._estimate_git_tokens(git_context)

        # Should be roughly 50*10 + 20*5 + 100 = 700
        assert 600 <= tokens <= 800

    def test_estimate_git_tokens_empty(self, aggregator):
        """Test git token estimation with no context."""
        assert aggregator._estimate_git_tokens(None) == 0
        assert aggregator._estimate_git_tokens({}) == 0

    def test_optimize_packing(self, aggregator):
        """Test optimal file packing algorithm."""
        files = [
            FileAnalysis(
                path=f"file{i}.py", content="x" * (100 * (i + 1)), relevance_score=0.5 + i * 0.1
            )
            for i in range(4)
        ]

        with patch("tenets.utils.tokens.count_tokens") as mock_count:
            # Mock token counts
            mock_count.side_effect = lambda content, model: len(content)

            result = aggregator.optimize_packing(files=files, max_tokens=500, model=None)

            assert len(result) > 0
            # Each result should be (file, should_summarize)
            for file, should_summarize in result:
                assert isinstance(file, FileAnalysis)
                assert isinstance(should_summarize, bool)

    def test_optimize_packing_empty(self, aggregator):
        """Test packing with no files."""
        result = aggregator.optimize_packing([], max_tokens=1000)
        assert result == []

    def test_token_utilization(self, aggregator, sample_files, prompt_context):
        """Test token utilization calculation."""
        result = aggregator.aggregate(
            files=sample_files, prompt_context=prompt_context, max_tokens=1000
        )

        utilization = result["statistics"]["token_utilization"]
        assert 0 <= utilization <= 1.0

    def test_aggregate_with_model(self, aggregator, sample_files, prompt_context):
        """Test aggregation with specific model."""
        with patch("tenets.core.distiller.aggregator.count_tokens") as mock_count:
            mock_count.return_value = 100

            result = aggregator.aggregate(
                files=sample_files, prompt_context=prompt_context, max_tokens=1000, model="gpt-4"
            )

            # Should pass model to token counting
            assert mock_count.called, f"count_tokens should have been called, calls: {mock_count.call_args_list}"
            # Check that it was called with the model parameter
            assert any(
                len(call[0]) >= 2 and call[0][1] == "gpt-4" 
                for call in mock_count.call_args_list
            ), f"Expected model 'gpt-4' in calls, got: {mock_count.call_args_list}"
