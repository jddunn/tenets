"""Tests for main Distiller orchestrator."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from tenets.core.distiller.distiller import Distiller
from tenets.models.context import ContextResult, PromptContext
from tenets.models.analysis import FileAnalysis
from tenets.config import TenetsConfig


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
    with patch('tenets.core.distiller.distiller.FileScanner'):
        with patch('tenets.core.distiller.distiller.CodeAnalyzer'):
            with patch('tenets.core.distiller.distiller.RelevanceRanker'):
                with patch('tenets.core.distiller.distiller.PromptParser'):
                    with patch('tenets.core.distiller.distiller.GitAnalyzer'):
                        with patch('tenets.core.distiller.distiller.ContextAggregator'):
                            with patch('tenets.core.distiller.distiller.TokenOptimizer'):
                                with patch('tenets.core.distiller.distiller.ContextFormatter'):
                                    distiller = Distiller(config)
                                    return distiller


@pytest.fixture
def mock_components(distiller):
    """Setup mock behaviors for distiller components."""
    # Mock scanner
    distiller.scanner.scan.return_value = [
        Path("file1.py"),
        Path("file2.py"),
        Path("file3.py")
    ]
    
    # Mock analyzer
    distiller.analyzer.analyze_file.side_effect = lambda path, **kwargs: FileAnalysis(
        path=str(path),
        content=f"Content of {path.name}",
        language="python",
        lines=100
    )
    
    # Mock parser
    distiller.parser.parse.return_value = PromptContext(
        text="test prompt",
        keywords=["test", "auth"],
        task_type="feature",
        file_patterns=[]
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
        'included_files': [
            {'file': FileAnalysis(path="file1.py", content="content"), 'content': "content", 'summarized': False}
        ],
        'total_tokens': 500,
        'available_tokens': 1000,
        'statistics': {
            'files_analyzed': 3,
            'files_included': 1,
            'files_summarized': 0,
            'files_skipped': 2,
            'token_utilization': 0.5
        }
    }
    
    # Mock formatter
    distiller.formatter.format.return_value = "# Formatted Context\n\nContent here"
    
    return distiller


class TestDistiller:
    """Test suite for Distiller."""
    
    def test_initialization(self, config):
        """Test Distiller initialization."""
        with patch('tenets.core.distiller.distiller.FileScanner'):
            with patch('tenets.core.distiller.distiller.CodeAnalyzer'):
                distiller = Distiller(config)
                
                assert distiller.config == config
                assert distiller.scanner is not None
                assert distiller.analyzer is not None
                
    def test_distill_basic(self, mock_components):
        """Test basic distillation."""
        result = mock_components.distill("implement authentication")
        
        assert isinstance(result, ContextResult)
        assert result.context == "# Formatted Context\n\nContent here"
        assert result.metadata['prompt'] == "implement authentication"
        assert result.metadata['files_analyzed'] == 3
        
    def test_distill_with_paths(self, mock_components):
        """Test distillation with specific paths."""
        result = mock_components.distill(
            "test prompt",
            paths=["./src", "./tests"]
        )
        
        # Should normalize paths
        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        assert len(call_args[1]['paths']) == 2
        
    def test_distill_with_single_path(self, mock_components):
        """Test distillation with single path."""
        result = mock_components.distill(
            "test prompt",
            paths="./src"
        )
        
        assert isinstance(result, ContextResult)
        
    def test_distill_with_format(self, mock_components):
        """Test distillation with different formats."""
        result = mock_components.distill(
            "test prompt",
            format="xml"
        )
        
        mock_components.formatter.format.assert_called_with(
            aggregated=mock_components.aggregator.aggregate.return_value,
            format="xml",
            prompt_context=mock_components.parser.parse.return_value,
            session_name=None
        )
        
    def test_distill_with_model(self, mock_components):
        """Test distillation with specific model."""
        result = mock_components.distill(
            "test prompt",
            model="gpt-4",
            max_tokens=8000
        )
        
        # Should pass model to aggregator
        mock_components.aggregator.aggregate.assert_called()
        call_args = mock_components.aggregator.aggregate.call_args
        assert call_args[1]['model'] == "gpt-4"
        assert call_args[1]['max_tokens'] == 8000
        
    def test_distill_with_mode(self, mock_components):
        """Test distillation with different modes."""
        result = mock_components.distill(
            "test prompt",
            mode="thorough"
        )
        
        # Should affect analysis depth
        mock_components.analyzer.analyze_file.assert_called()
        call_args = mock_components.analyzer.analyze_file.call_args
        assert call_args[1]['deep'] == True
        
    def test_distill_fast_mode(self, mock_components):
        """Test distillation in fast mode."""
        result = mock_components.distill(
            "test prompt",
            mode="fast"
        )
        
        # Should not use deep analysis
        mock_components.analyzer.analyze_file.assert_called()
        call_args = mock_components.analyzer.analyze_file.call_args
        assert call_args[1]['deep'] == False
        
    def test_distill_without_git(self, mock_components):
        """Test distillation without git context."""
        result = mock_components.distill(
            "test prompt",
            include_git=False
        )
        
        # Should not call git methods
        mock_components.git.get_recent_commits.assert_not_called()
        
    def test_distill_with_git(self, mock_components):
        """Test distillation with git context."""
        result = mock_components.distill(
            "test prompt",
            include_git=True
        )
        
        # Should call git methods
        mock_components.git.is_git_repo.assert_called()
        
    def test_distill_with_session(self, mock_components):
        """Test distillation with session."""
        result = mock_components.distill(
            "test prompt",
            session_name="test-session"
        )
        
        assert result.metadata['session'] == "test-session"
        
    def test_distill_with_patterns(self, mock_components):
        """Test distillation with file patterns."""
        result = mock_components.distill(
            "test prompt",
            include_patterns=["*.py", "*.js"],
            exclude_patterns=["test_*.py"]
        )
        
        mock_components.scanner.scan.assert_called()
        call_args = mock_components.scanner.scan.call_args
        assert call_args[1]['include_patterns'] == ["*.py", "*.js"]
        assert call_args[1]['exclude_patterns'] == ["test_*.py"]
        
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
        prompt_context = PromptContext(
            text="test",
            file_patterns=["*.py"]
        )
        
        files = mock_components._discover_files(
            paths=[Path(".")],
            prompt_context=prompt_context
        )
        
        assert len(files) == 3
        mock_components.scanner.scan.assert_called()
        
    def test_analyze_files(self, mock_components):
        """Test file analysis."""
        files = [Path("file1.py"), Path("file2.py")]
        prompt_context = PromptContext(text="test")
        
        analyzed = mock_components._analyze_files(
            files=files,
            mode="balanced",
            prompt_context=prompt_context
        )
        
        assert len(analyzed) == 2
        assert all(isinstance(f, FileAnalysis) for f in analyzed)
        
    def test_analyze_files_with_errors(self, mock_components):
        """Test file analysis with errors."""
        mock_components.analyzer.analyze_file.side_effect = Exception("Analysis failed")
        
        files = [Path("file1.py")]
        prompt_context = PromptContext(text="test")
        
        analyzed = mock_components._analyze_files(
            files=files,
            mode="fast",
            prompt_context=prompt_context
        )
        
        # Should handle errors gracefully
        assert len(analyzed) == 0
        
    def test_rank_files(self, mock_components):
        """Test file ranking."""
        files = [
            FileAnalysis(path="file1.py", content="content"),
            FileAnalysis(path="file2.py", content="content")
        ]
        prompt_context = PromptContext(text="test")
        
        ranked = mock_components._rank_files(
            files=files,
            prompt_context=prompt_context,
            mode="balanced"
        )
        
        mock_components.ranker.rank_files.assert_called_with(
            files=files,
            prompt_context=prompt_context,
            algorithm="balanced"
        )
        
    def test_get_git_context(self, mock_components):
        """Test git context extraction."""
        paths = [Path(".")]
        prompt_context = PromptContext(text="recent changes")
        files = [FileAnalysis(path="file1.py", content="content")]
        
        git_context = mock_components._get_git_context(
            paths=paths,
            prompt_context=prompt_context,
            files=files
        )
        
        assert git_context is not None
        assert 'recent_commits' in git_context
        assert 'contributors' in git_context
        assert 'branch' in git_context
        assert 'recent_changes' in git_context  # Because "recent" in prompt
        
    def test_get_git_context_no_repo(self, mock_components):
        """Test git context when not a repo."""
        mock_components.git.is_git_repo.return_value = False
        
        git_context = mock_components._get_git_context(
            paths=[Path(".")],
            prompt_context=PromptContext(text="test"),
            files=[]
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
            git_context={'branch': 'main'}
        )
        
        assert 'included_files' in aggregated
        assert 'total_tokens' in aggregated
        
    def test_format_output(self, mock_components):
        """Test output formatting."""
        aggregated = {'included_files': [], 'statistics': {}}
        prompt_context = PromptContext(text="test")
        
        formatted = mock_components._format_output(
            aggregated=aggregated,
            format="markdown",
            prompt_context=prompt_context,
            session_name="test-session"
        )
        
        assert formatted == "# Formatted Context\n\nContent here"
        
    def test_build_result(self, mock_components):
        """Test result building."""
        result = mock_components._build_result(
            formatted="Formatted content",
            metadata={'key': 'value'}
        )
        
        assert isinstance(result, ContextResult)
        assert result.context == "Formatted content"
        assert result.metadata['key'] == 'value'
        
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
            exclude_patterns=["test_*.py"]
        )
        
        # Verify all components were called
        assert mock_components.parser.parse.called
        assert mock_components.scanner.scan.called
        assert mock_components.analyzer.analyze_file.called
        assert mock_components.ranker.rank_files.called
        assert mock_components.aggregator.aggregate.called
        assert mock_components.formatter.format.called
        
        # Verify result
        assert isinstance(result, ContextResult)
        assert result.metadata['prompt'] == "implement OAuth2 authentication for the API"
        assert result.metadata['session'] == "oauth-impl"
        assert result.metadata['mode'] == "thorough"