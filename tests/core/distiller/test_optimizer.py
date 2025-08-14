"""Tests for TokenOptimizer."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.distiller.optimizer import TokenBudget, TokenOptimizer
from tenets.models.analysis import FileAnalysis


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    config.max_tokens = 10000
    return config


@pytest.fixture
def optimizer(config):
    """Create TokenOptimizer instance."""
    return TokenOptimizer(config)


@pytest.fixture
def sample_files():
    """Create sample FileAnalysis objects with varying sizes."""
    files = []
    for i in range(5):
        file = FileAnalysis(
            path=f"file{i}.py",
            content="x" * (100 * (i + 1)),  # 100, 200, 300, 400, 500 chars
            language="python",
            lines=10 + i * 5,
            relevance_score=0.9 - i * 0.1,  # 0.9, 0.8, 0.7, 0.6, 0.5
        )
        files.append(file)
    return files


class TestTokenBudget:
    """Test suite for TokenBudget."""

    def test_default_budget(self):
        """Test default token budget initialization."""
        budget = TokenBudget(total_limit=10000)
        
        assert budget.total_limit == 10000
        assert budget.model is None
        assert budget.prompt_tokens == 0
        assert budget.response_reserve == 4000
        assert budget.structure_tokens == 1000
        assert budget.git_tokens == 0
        assert budget.tenet_tokens == 0

    def test_available_for_files(self):
        """Test calculation of available tokens for files."""
        budget = TokenBudget(total_limit=10000)
        available = budget.available_for_files
        
        # 10000 - 4000 (response) - 1000 (structure) = 5000
        assert available == 5000

    def test_available_with_allocations(self):
        """Test available tokens with various allocations."""
        budget = TokenBudget(total_limit=10000)
        budget.prompt_tokens = 500
        budget.git_tokens = 300
        budget.tenet_tokens = 200
        
        available = budget.available_for_files
        # 10000 - 500 - 4000 - 1000 - 300 - 200 = 4000
        assert available == 4000

    def test_utilization(self):
        """Test budget utilization calculation."""
        budget = TokenBudget(total_limit=10000)
        budget.prompt_tokens = 1000
        budget.structure_tokens = 1000
        budget.git_tokens = 500
        
        utilization = budget.utilization
        # (1000 + 1000 + 500) / 10000 = 0.25
        assert utilization == 0.25

    def test_utilization_zero_total(self):
        """Test utilization with zero total limit."""
        budget = TokenBudget(total_limit=0)
        assert budget.utilization == 0

    def test_negative_available_tokens(self):
        """Test that available tokens don't go negative."""
        budget = TokenBudget(total_limit=1000)
        budget.response_reserve = 2000  # More than total
        
        assert budget.available_for_files == 0  # Should be 0, not negative


class TestTokenOptimizer:
    """Test suite for TokenOptimizer."""

    def test_initialization(self, config):
        """Test optimizer initialization."""
        optimizer = TokenOptimizer(config)
        assert optimizer.config == config

    def test_create_budget_with_max_tokens(self, optimizer):
        """Test budget creation with explicit max_tokens."""
        budget = optimizer.create_budget(
            model=None,
            max_tokens=5000,
            prompt_tokens=100,
            has_git_context=False,
            has_tenets=False
        )
        
        assert budget.total_limit == 5000
        assert budget.prompt_tokens == 100
        assert budget.response_reserve == 2000  # Default for unknown model

    def test_create_budget_with_model(self, optimizer):
        """Test budget creation with model."""
        with patch("tenets.core.distiller.optimizer.get_model_limits") as mock_limits:
            mock_limits.return_value = MagicMock(max_context=8000)
            
            budget = optimizer.create_budget(
                model="gpt-4",
                max_tokens=None,
                prompt_tokens=200
            )
            
            assert budget.total_limit == 8000
            assert budget.model == "gpt-4"
            assert budget.response_reserve == 4000  # GPT-4 gets more reserve

    def test_create_budget_claude_model(self, optimizer):
        """Test budget creation with Claude model."""
        with patch("tenets.core.distiller.optimizer.get_model_limits") as mock_limits:
            mock_limits.return_value = MagicMock(max_context=100000)
            
            budget = optimizer.create_budget(
                model="claude-3",
                max_tokens=None,
                prompt_tokens=500
            )
            
            assert budget.model == "claude-3"
            assert budget.response_reserve == 4000  # Claude gets more reserve

    def test_create_budget_with_git_context(self, optimizer):
        """Test budget creation with git context."""
        budget = optimizer.create_budget(
            model=None,
            max_tokens=5000,
            prompt_tokens=100,
            has_git_context=True
        )
        
        assert budget.git_tokens == 500  # Reserved for git

    def test_create_budget_with_tenets(self, optimizer):
        """Test budget creation with tenets."""
        budget = optimizer.create_budget(
            model=None,
            max_tokens=5000,
            prompt_tokens=100,
            has_tenets=True
        )
        
        assert budget.tenet_tokens == 300  # Reserved for tenets

    def test_greedy_selection(self, optimizer, sample_files):
        """Test greedy file selection strategy."""
        budget = TokenBudget(total_limit=10000)
        budget.available_for_files = 1000
        
        with patch("tenets.core.distiller.optimizer.count_tokens") as mock_count:
            # Mock token counts based on content length
            mock_count.side_effect = lambda content, model: len(content)
            
            selected = optimizer._greedy_selection(sample_files, budget)
            
            # Should select files greedily by relevance until budget exhausted
            assert len(selected) > 0
            total_tokens = sum(
                len(f.content) if action == "full" else len(f.content) // 4
                for f, action in selected
            )
            assert total_tokens <= 1000

    def test_balanced_selection(self, optimizer, sample_files):
        """Test balanced file selection strategy."""
        budget = TokenBudget(total_limit=10000)
        budget.available_for_files = 2000
        
        with patch("tenets.core.distiller.optimizer.count_tokens") as mock_count:
            mock_count.side_effect = lambda content, model: len(content)
            
            selected = optimizer._balanced_selection(sample_files, budget)
            
            # Should have mix of full and summarized files
            full_files = [f for f, action in selected if action == "full"]
            summary_files = [f for f, action in selected if action == "summary"]
            
            assert len(selected) > 0
            # Balanced should limit full files
            assert len(full_files) <= 10

    def test_diverse_selection(self, optimizer):
        """Test diverse file selection strategy."""
        # Create files in different directories
        files = [
            FileAnalysis(path="dir1/file1.py", content="x" * 100, relevance_score=0.9),
            FileAnalysis(path="dir1/file2.py", content="x" * 100, relevance_score=0.8),
            FileAnalysis(path="dir2/file3.py", content="x" * 100, relevance_score=0.7),
            FileAnalysis(path="dir2/file4.js", content="x" * 100, relevance_score=0.6),
            FileAnalysis(path="dir3/file5.py", content="x" * 100, relevance_score=0.5),
        ]
        
        budget = TokenBudget(total_limit=10000)
        budget.available_for_files = 1000
        
        with patch("tenets.core.distiller.optimizer.count_tokens") as mock_count:
            mock_count.side_effect = lambda content, model: len(content)
            
            selected = optimizer._diverse_selection(files, budget)
            
            # Should select from different directories
            selected_dirs = set()
            for file, _ in selected:
                dir_path = str(Path(file.path).parent)
                selected_dirs.add(dir_path)
            
            assert len(selected) > 0
            # Should have files from multiple directories
            assert len(selected_dirs) > 1

    def test_optimize_file_selection_strategies(self, optimizer, sample_files):
        """Test optimize_file_selection with different strategies."""
        budget = TokenBudget(total_limit=10000)
        budget.available_for_files = 1000
        
        with patch("tenets.core.distiller.optimizer.count_tokens") as mock_count:
            mock_count.side_effect = lambda content, model: len(content)
            
            # Test each strategy
            greedy = optimizer.optimize_file_selection(sample_files, budget, "greedy")
            balanced = optimizer.optimize_file_selection(sample_files, budget, "balanced")
            diverse = optimizer.optimize_file_selection(sample_files, budget, "diverse")
            
            assert all(isinstance(r, tuple) for r in greedy)
            assert all(isinstance(r, tuple) for r in balanced)
            assert all(isinstance(r, tuple) for r in diverse)

    def test_optimize_file_selection_unknown_strategy(self, optimizer, sample_files):
        """Test optimize_file_selection with unknown strategy defaults to balanced."""
        budget = TokenBudget(total_limit=10000)
        budget.available_for_files = 1000
        
        with patch("tenets.core.distiller.optimizer.count_tokens") as mock_count:
            mock_count.side_effect = lambda content, model: len(content)
            
            result = optimizer.optimize_file_selection(sample_files, budget, "unknown")
            
            # Should default to balanced
            assert len(result) > 0

    def test_estimate_tokens_for_git(self, optimizer):
        """Test git token estimation."""
        git_context = {
            "recent_commits": [{"sha": "abc"}] * 10,
            "contributors": [{"name": "dev"}] * 5,
            "recent_changes": [{"file": "test.py"}] * 3
        }
        
        tokens = optimizer.estimate_tokens_for_git(git_context)
        
        # 100 base + 10*50 + 5*20 + 3*30 = 100 + 500 + 100 + 90 = 790
        assert 700 <= tokens <= 900

    def test_estimate_tokens_for_git_empty(self, optimizer):
        """Test git token estimation with no context."""
        assert optimizer.estimate_tokens_for_git(None) == 0
        assert optimizer.estimate_tokens_for_git({}) == 100  # Base overhead

    def test_estimate_tokens_for_tenets(self, optimizer):
        """Test tenet token estimation."""
        # Without reinforcement
        tokens = optimizer.estimate_tokens_for_tenets(5, with_reinforcement=False)
        assert tokens == 5 * 30  # 150
        
        # With reinforcement (>3 tenets)
        tokens = optimizer.estimate_tokens_for_tenets(5, with_reinforcement=True)
        assert tokens == 5 * 30 + 100  # 250

    def test_estimate_tokens_for_tenets_no_reinforcement(self, optimizer):
        """Test tenet token estimation without reinforcement for few tenets."""
        # With reinforcement but only 2 tenets (no reinforcement added)
        tokens = optimizer.estimate_tokens_for_tenets(2, with_reinforcement=True)
        assert tokens == 2 * 30  # 60, no reinforcement

    def test_selection_respects_token_limit(self, optimizer, sample_files):
        """Test that selection strategies respect token limits."""
        budget = TokenBudget(total_limit=10000)
        budget.available_for_files = 200  # Very limited budget
        
        with patch("tenets.core.distiller.optimizer.count_tokens") as mock_count:
            mock_count.side_effect = lambda content, model: len(content)
            
            selected = optimizer._greedy_selection(sample_files, budget)
            
            # Calculate total tokens
            total = 0
            for file, action in selected:
                if action == "full":
                    total += len(file.content)
                else:  # summary
                    total += min(len(file.content) // 4, 200 - total)
            
            # Should not exceed budget
            assert total <= 200 * 1.05  # Allow 5% overflow for rounding

    def test_balanced_selection_phases(self, optimizer):
        """Test that balanced selection has distinct phases."""
        # Create many files
        files = [
            FileAnalysis(
                path=f"file{i}.py",
                content="x" * 200,
                relevance_score=1.0 - i * 0.05
            )
            for i in range(20)
        ]
        
        budget = TokenBudget(total_limit=10000)
        budget.available_for_files = 5000
        
        with patch("tenets.core.distiller.optimizer.count_tokens") as mock_count:
            mock_count.side_effect = lambda content, model: len(content)
            
            selected = optimizer._balanced_selection(files, budget)
            
            # Check phase 1: full files (should be limited)
            full_files = [(f, a) for f, a in selected if a == "full"]
            assert len(full_files) <= 10  # max_full_files limit
            
            # Check phase 2: summaries
            summary_files = [(f, a) for f, a in selected if a == "summary"]
            assert len(summary_files) > 0  # Should have some summaries

    def test_diverse_selection_file_types(self, optimizer):
        """Test diverse selection with different file types."""
        files = [
            FileAnalysis(path="app.py", content="x" * 100, relevance_score=0.9),
            FileAnalysis(path="app.js", content="x" * 100, relevance_score=0.8),
            FileAnalysis(path="style.css", content="x" * 100, relevance_score=0.7),
            FileAnalysis(path="test.py", content="x" * 100, relevance_score=0.6),
            FileAnalysis(path="config.json", content="x" * 100, relevance_score=0.5),
        ]
        
        budget = TokenBudget(total_limit=10000)
        budget.available_for_files = 1000
        
        with patch("tenets.core.distiller.optimizer.count_tokens") as mock_count:
            mock_count.side_effect = lambda content, model: len(content)
            
            selected = optimizer._diverse_selection(files, budget)
            
            # Should select diverse file types
            extensions = set()
            for file, _ in selected:
                ext = Path(file.path).suffix
                extensions.add(ext)
            
            assert len(selected) > 0
            # Should have multiple file types
            assert len(extensions) > 1