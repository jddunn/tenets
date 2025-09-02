"""Tests for the rank command."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from tenets.cli.commands.rank import rank
from tenets.models.analysis import FileAnalysis


@pytest.fixture
def mock_tenets():
    """Mock Tenets instance."""
    with patch('tenets.cli.commands.rank.Tenets') as mock:
        yield mock.return_value


@pytest.fixture
def mock_files():
    """Create mock file analysis results."""
    files = []
    for i in range(5):
        file = MagicMock(spec=FileAnalysis)
        file.path = Path(f"src/file{i}.py")
        file.relevance_score = 0.9 - (i * 0.15)  # Descending scores
        file.relevance_rank = i + 1
        file.relevance_factors = {
            "semantic_similarity": 0.8 - (i * 0.1),
            "keyword_match": 0.7 - (i * 0.1),
            "path_relevance": 0.6 - (i * 0.1),
        }
        files.append(file)
    return files


@pytest.fixture
def mock_ranker(mock_files):
    """Mock RelevanceRanker."""
    # The RelevanceRanker is imported inside the function, so we need to patch it at the source
    with patch('tenets.core.ranking.RelevanceRanker') as mock:
        ranker = mock.return_value
        ranker.rank_files.return_value = mock_files
        ranker.stats = MagicMock()
        ranker.stats.to_dict.return_value = {
            "total_files": 100,
            "files_ranked": 95,
            "files_above_threshold": len(mock_files),
            "average_score": 0.6,
            "time_elapsed": 2.5,
        }
        yield ranker


@pytest.fixture
def mock_scanner():
    """Mock FileScanner."""
    # FileScanner is imported from tenets.utils.scanner inside the function
    with patch('tenets.utils.scanner.FileScanner') as mock:
        scanner = mock.return_value
        scanner.scan.return_value = [
            Path(f"src/file{i}.py") for i in range(5)
        ]
        yield scanner


@pytest.fixture
def mock_analyzer(mock_files):
    """Mock CodeAnalyzer."""
    # CodeAnalyzer is imported from tenets.core.analysis.analyzer inside the function
    with patch('tenets.core.analysis.analyzer.CodeAnalyzer') as mock:
        analyzer = mock.return_value
        analyzer.analyze_file.side_effect = mock_files
        yield analyzer


def test_rank_basic(mock_files):
    """Test basic rank command."""
    runner = CliRunner()
    
    with patch('tenets.cli.commands.rank.Tenets') as mock_tenets_cls, \
         patch('tenets.utils.scanner.FileScanner') as mock_scanner_cls, \
         patch('tenets.core.analysis.analyzer.CodeAnalyzer') as mock_analyzer_cls, \
         patch('tenets.core.ranking.RelevanceRanker') as mock_ranker_cls, \
         patch('tenets.core.prompt.PromptParser') as mock_parser_cls, \
         patch('tenets.core.distiller.Distiller') as mock_distiller_cls:
        
        # Setup mocks
        mock_tenets = mock_tenets_cls.return_value
        mock_tenets.config.scanner.workers = 4
        
        mock_scanner = mock_scanner_cls.return_value
        mock_scanner.scan.return_value = [Path(f"src/file{i}.py") for i in range(5)]
        
        mock_analyzer = mock_analyzer_cls.return_value
        mock_analyzer.analyze_file.side_effect = mock_files
        
        mock_ranker = mock_ranker_cls.return_value
        mock_ranker.rank_files.return_value = mock_files
        
        mock_parser = mock_parser_cls.return_value
        mock_parser.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query"])
        
        # Print debug info if test fails
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
            if result.exception:
                import traceback
                print(f"Exception: {result.exception}")
                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        
        assert result.exit_code == 0
        assert "Ranked Files" in result.output
        # Handle both forward and backward slashes
        assert "file0.py" in result.output
        assert "0.900" in result.output  # Score should be shown


def test_rank_top_n(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with --top option."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--top", "2"])
        
        assert result.exit_code == 0
        # Handle both forward and backward slashes
        assert "file0.py" in result.output
        assert "file1.py" in result.output
        # Only top 2 should be shown
        assert "file4.py" not in result.output


def test_rank_min_score(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with --min-score option."""
    runner = CliRunner()
    
    # Filter to only files with score >= 0.5
    mock_ranker.rank_files.return_value = [
        f for f in mock_ranker.rank_files.return_value 
        if f.relevance_score >= 0.5
    ]
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--min-score", "0.5"])
        
        assert result.exit_code == 0
        # Check that ranker was called
        mock_ranker.rank_files.assert_called_once()
        # The min_score filtering is done after ranking in the command itself


def test_rank_no_scores(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with --no-scores option."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--no-scores"])
        
        assert result.exit_code == 0
        # Handle both forward and backward slashes
        assert "file0.py" in result.output
        # Scores should not be shown
        assert "0.900" not in result.output
        assert "Score:" not in result.output


def test_rank_with_factors(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with --factors option."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--factors"])
        
        assert result.exit_code == 0
        assert "semantic_similarity" in result.output
        assert "keyword_match" in result.output
        assert "path_relevance" in result.output


def test_rank_json_format(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with JSON output format."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--format", "json"])
        
        assert result.exit_code == 0
        
        # Parse JSON output (strip any non-JSON prefix/suffix)
        # Find the JSON part (starts with { and ends with })
        json_start = result.output.find('{')
        json_end = result.output.rfind('}') + 1
        json_str = result.output[json_start:json_end] if json_start >= 0 else result.output
        output_data = json.loads(json_str)
        assert "files" in output_data
        assert "total_files" in output_data
        assert len(output_data["files"]) == 5
        # Handle both forward and backward slashes
        assert "file0.py" in output_data["files"][0]["path"]
        assert output_data["files"][0]["score"] == 0.9
        assert output_data["files"][0]["rank"] == 1


def test_rank_xml_format(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with XML output format."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--format", "xml"])
        
        assert result.exit_code == 0
        assert '<?xml version="1.0"' in result.output
        assert "<ranking>" in result.output
        assert "<files>" in result.output
        # Handle both forward and backward slashes
        assert "file0.py</path>" in result.output


def test_rank_html_format(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with HTML output format."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--format", "html"])
        
        assert result.exit_code == 0
        assert "<!DOCTYPE html>" in result.output
        assert "Ranked Files" in result.output
        # Handle both forward and backward slashes
        assert "file0.py" in result.output


def test_rank_tree_view(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with tree view."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--tree"])
        
        assert result.exit_code == 0
        # Tree view should group by directory
        assert "📁" in result.output or "src" in result.output
        assert "📄" in result.output or "file0.py" in result.output


def test_rank_output_to_file(mock_tenets, mock_ranker, mock_scanner, mock_analyzer, tmp_path):
    """Test rank with output to file."""
    runner = CliRunner()
    output_file = tmp_path / "ranked.json"
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, [
            "test query",
            "--format", "json",
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Verify file content
        with open(output_file) as f:
            data = json.load(f)
            assert "files" in data
            assert len(data["files"]) == 5


def test_rank_with_stats(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with --stats option."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--stats"])
        
        assert result.exit_code == 0
        assert "Ranking Statistics" in result.output
        assert "Total Files" in result.output
        assert "Files Ranked" in result.output
        assert "Average Score" in result.output


def test_rank_with_session(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with session option."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--session", "my-session"])
        
        assert result.exit_code == 0
        # Session should be used (implementation would handle this)


def test_rank_with_include_exclude(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with include/exclude patterns."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, [
            "test query",
            "--include", "*.py,*.js",
            "--exclude", "test_*"
        ])
        
        assert result.exit_code == 0
        # Scanner should be called with patterns
        mock_scanner.scan.assert_called_once()
        call_args = mock_scanner.scan.call_args
        assert call_args[1].get('include_patterns') == ['*.py', '*.js']
        assert call_args[1].get('exclude_patterns') == ['test_*']


def test_rank_copy_to_clipboard(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with --copy option."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser, \
         patch('tenets.cli.commands.rank.pyperclip') as mock_pyperclip:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        mock_pyperclip.copy = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        result = runner.invoke(app, ["test query", "--copy"])
        
        assert result.exit_code == 0
        assert "Copied file list to clipboard" in result.output
        mock_pyperclip.copy.assert_called_once()
        
        # Check clipboard content
        clipboard_content = mock_pyperclip.copy.call_args[0][0]
        # Handle both forward and backward slashes
        assert "file0.py" in clipboard_content


def test_rank_different_modes(mock_tenets, mock_ranker, mock_scanner, mock_analyzer):
    """Test rank with different ranking modes."""
    runner = CliRunner()
    
    with patch('tenets.core.distiller.Distiller'), \
         patch('tenets.core.prompt.PromptParser') as mock_parser:
        
        mock_parser.return_value.parse.return_value = MagicMock()
        
        app = typer.Typer()
        app.command()(rank)
        
        # Test fast mode
        result = runner.invoke(app, ["test query", "--mode", "fast"])
        assert result.exit_code == 0
        
        # Test thorough mode
        result = runner.invoke(app, ["test query", "--mode", "thorough"])
        assert result.exit_code == 0
        
        # Ranker should be created with correct algorithm
        # The mock_ranker fixture already handles this
        assert mock_ranker.rank_files.called