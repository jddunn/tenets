"""Tests for main examiner module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.examiner.examiner import (
    ExaminationResult,
    Examiner,
    examine_directory,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return TenetsConfig()


@pytest.fixture
def examiner(config):
    """Create Examiner instance."""
    return Examiner(config)


@pytest.fixture
def sample_analyzed_files():
    """Create sample analyzed file objects."""
    files = []

    for i in range(5):
        file = Mock()
        file.path = f"file{i}.py"
        file.language = "Python"
        file.lines = 100 + i * 50
        file.functions = [Mock() for _ in range(i + 1)]
        file.classes = [Mock() for _ in range(i)]
        file.complexity = Mock(cyclomatic=5 + i * 2)
        files.append(file)

    return files


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure."""
    # Create source files
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    (src_dir / "main.py").write_text(
        """
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
"""
    )

    (src_dir / "utils.py").write_text(
        """
def helper(x, y):
    return x + y

class Utility:
    def process(self, data):
        return data * 2
"""
    )

    # Create test files
    test_dir = tmp_path / "tests"
    test_dir.mkdir()

    (test_dir / "test_main.py").write_text(
        """
import pytest
from src.main import main

def test_main():
    assert main() is None
"""
    )

    # Create config file
    (tmp_path / ".tenets.yml").write_text(
        """
ranking:
  threshold: 0.1
"""
    )

    return tmp_path


class TestExaminationResult:
    """Test suite for ExaminationResult dataclass."""

    def test_examination_result_creation(self, tmp_path):
        """Test creating ExaminationResult."""
        result = ExaminationResult(root_path=tmp_path, total_files=10, total_lines=1500)

        assert result.root_path == tmp_path
        assert result.total_files == 10
        assert result.total_lines == 1500
        assert isinstance(result.timestamp, datetime)

    def test_has_issues(self):
        """Test has_issues property."""
        # No issues
        result = ExaminationResult(root_path=Path())
        assert result.has_issues == False

        # Has errors
        result.errors = ["Error 1", "Error 2"]
        assert result.has_issues == True

        # Has high complexity
        result.errors = []
        result.complexity = Mock(high_complexity_count=5)
        assert result.has_issues == True

        # Has critical hotspots
        result.complexity = None
        result.hotspots = Mock(critical_count=2)
        assert result.has_issues == True

    def test_health_score(self):
        """Test health_score calculation."""
        result = ExaminationResult(root_path=Path())

        # Base health score (no analysis done yet)
        # Should start at 85.0 when no hotspot analysis is available
        assert result.health_score == 85.0

        # Add complexity issues (only applies when no hotspots)
        result.complexity = Mock(high_complexity_count=10)
        score_with_complexity = result.health_score
        assert score_with_complexity < 85.0

        # Add hotspots with their own health score
        # When hotspots exist, we use their health score as base
        result.hotspots = Mock(critical_count=3, health_score=75.0)
        score_with_hotspots = result.health_score
        assert score_with_hotspots == 75.0  # Uses hotspot's health score as base

        # Add errors (these always apply)
        result.errors = ["Error 1", "Error 2"]
        score_with_errors = result.health_score
        assert score_with_errors < score_with_hotspots

        # Add good metrics
        result.metrics = Mock(test_coverage=0.85, documentation_ratio=0.4)
        score_with_good_metrics = result.health_score
        assert score_with_good_metrics > score_with_errors

    def test_to_dict(self, tmp_path):
        """Test to_dict conversion."""
        result = ExaminationResult(
            root_path=tmp_path, total_files=5, total_lines=500, languages=["Python", "JavaScript"]
        )

        result.metrics = Mock()
        result.metrics.to_dict.return_value = {"total_functions": 20}

        result.complexity = Mock()
        result.complexity.to_dict.return_value = {"avg_complexity": 8.5}

        dict_result = result.to_dict()

        assert dict_result["root_path"] == str(tmp_path)
        assert dict_result["total_files"] == 5
        assert dict_result["languages"] == ["Python", "JavaScript"]
        assert dict_result["metrics"]["total_functions"] == 20
        assert dict_result["complexity"]["avg_complexity"] == 8.5

    def test_to_json(self, tmp_path):
        """Test to_json conversion."""
        result = ExaminationResult(root_path=tmp_path, total_files=3, total_lines=300)

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["total_files"] == 3
        assert parsed["total_lines"] == 300


class TestExaminer:
    """Test suite for Examiner class."""

    def test_initialization(self, config):
        """Test Examiner initialization."""
        examiner = Examiner(config)

        assert examiner.config == config
        assert examiner.analyzer is not None
        assert examiner.scanner is not None
        assert examiner.metrics_calculator is not None
        assert examiner.complexity_analyzer is not None
        assert examiner.ownership_tracker is not None
        assert examiner.hotspot_detector is not None

    def test_examine_project_invalid_path(self, examiner):
        """Test examining non-existent path."""
        with pytest.raises(ValueError, match="Path does not exist"):
            examiner.examine_project(Path("/nonexistent/path"))

    def test_examine_project_not_directory(self, examiner, tmp_path):
        """Test examining a file instead of directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="Path is not a directory"):
            examiner.examine_project(file_path)

    @patch.object(Examiner, "_discover_files")
    def test_examine_project_no_files(self, mock_discover, examiner, tmp_path):
        """Test examining project with no files."""
        mock_discover.return_value = []

        result = examiner.examine_project(tmp_path)

        assert result.root_path == tmp_path
        assert result.total_files == 0
        assert "No files found" in result.errors[0]

    @patch.object(Examiner, "_discover_files")
    @patch.object(Examiner, "_analyze_files")
    def test_examine_project_basic(
        self, mock_analyze, mock_discover, examiner, tmp_path, sample_analyzed_files
    ):
        """Test basic project examination."""
        mock_discover.return_value = [Path(f"file{i}.py") for i in range(5)]
        mock_analyze.return_value = sample_analyzed_files

        result = examiner.examine_project(
            tmp_path,
            include_metrics=True,
            include_complexity=True,
            include_git=False,
            include_ownership=False,
            include_hotspots=False,
        )

        assert result.root_path == tmp_path
        assert result.total_files == 5
        assert result.total_lines > 0
        assert len(result.languages) > 0
        assert result.metrics is not None
        assert result.complexity is not None

    @patch.object(Examiner, "_is_git_repo")
    @patch.object(Examiner, "_analyze_git")
    @patch.object(Examiner, "_discover_files")
    @patch.object(Examiner, "_analyze_files")
    def test_examine_project_with_git(
        self,
        mock_analyze_files,
        mock_discover,
        mock_git_analyze,
        mock_is_git,
        examiner,
        tmp_path,
        sample_analyzed_files,
    ):
        """Test project examination with git analysis."""
        mock_discover.return_value = [Path(f"file{i}.py") for i in range(3)]
        mock_analyze_files.return_value = sample_analyzed_files[:3]
        mock_is_git.return_value = True
        mock_git_analyze.return_value = {"current_branch": "main", "total_commits": 100}

        result = examiner.examine_project(
            tmp_path, include_git=True, include_ownership=False, include_hotspots=False
        )

        assert result.git_analysis is not None
        assert result.git_analysis["current_branch"] == "main"

    def test_examine_file_invalid(self, examiner):
        """Test examining non-existent file."""
        with pytest.raises(ValueError, match="File does not exist"):
            examiner.examine_file(Path("/nonexistent/file.py"))

    def test_examine_file_directory(self, examiner, tmp_path):
        """Test examining directory instead of file."""
        with pytest.raises(ValueError, match="Path is not a file"):
            examiner.examine_file(tmp_path)

    @patch.object(Examiner, "analyzer")
    def test_examine_file_basic(self, mock_analyzer, examiner, tmp_path):
        """Test basic file examination."""
        file_path = tmp_path / "test.py"
        file_path.write_text("print('test')")

        mock_analysis = Mock()
        mock_analysis.lines = 1
        mock_analysis.language = "Python"
        mock_analysis.functions = []
        mock_analysis.classes = []
        mock_analysis.imports = []
        mock_analysis.complexity = Mock(cyclomatic=1)

        mock_analyzer.analyze_file.return_value = mock_analysis

        result = examiner.examine_file(file_path)

        assert result["path"] == str(file_path)
        assert result["name"] == "test.py"
        assert result["lines"] == 1
        assert result["language"] == "Python"

    def test_discover_files(self, examiner, temp_project):
        """Test file discovery."""
        files = examiner._discover_files(temp_project)

        assert len(files) > 0
        assert any("main.py" in str(f) for f in files)
        assert any("utils.py" in str(f) for f in files)

    def test_discover_files_with_patterns(self, examiner, temp_project):
        """Test file discovery with patterns."""
        # Include only Python files
        files = examiner._discover_files(temp_project, include_patterns=["*.py"])
        assert all(str(f).endswith(".py") for f in files)

        # Exclude test files
        files = examiner._discover_files(temp_project, exclude_patterns=["test_*"])
        assert not any("test_" in str(f) for f in files)

    def test_discover_files_max_files(self, examiner, temp_project):
        """Test file discovery with max files limit."""
        files = examiner._discover_files(temp_project, max_files=2)
        assert len(files) == 2

    def test_extract_languages(self, examiner, sample_analyzed_files):
        """Test language extraction."""
        languages = examiner._extract_languages(sample_analyzed_files)

        assert "Python" in languages
        assert isinstance(languages, list)
        assert len(languages) > 0

    @patch("tenets.core.examiner.examiner.GitAnalyzer")
    def test_is_git_repo(self, mock_git_class, examiner, tmp_path):
        """Test git repository detection."""
        mock_git = Mock()
        mock_git.is_repo.return_value = True
        mock_git_class.return_value = mock_git

        assert examiner._is_git_repo(tmp_path) == True

        mock_git.is_repo.return_value = False
        assert examiner._is_git_repo(tmp_path) == False

        # Test exception handling
        mock_git.is_repo.side_effect = Exception("Git error")
        assert examiner._is_git_repo(tmp_path) == False

    def test_generate_summary(self, examiner):
        """Test summary generation."""
        result = ExaminationResult(root_path=Path(), total_files=10, total_lines=1000)
        result.languages = ["Python", "JavaScript"]

        result.metrics = Mock(avg_file_size=100, total_functions=50, total_classes=10)

        result.complexity = Mock(high_complexity_count=3, avg_complexity=8.5)

        result.ownership = Mock(total_contributors=5, bus_factor=2)

        result.hotspots = Mock(total_count=7, critical_count=2)

        summary = examiner._generate_summary(result)

        assert summary["total_files"] == 10
        assert summary["total_lines"] == 1000
        assert summary["language_count"] == 2
        assert summary["avg_file_size"] == 100
        assert summary["high_complexity_files"] == 3
        assert summary["bus_factor"] == 2
        assert summary["hotspot_count"] == 7

    @patch.object(Examiner, "analyzer")
    def test_analyze_files(self, mock_analyzer, examiner, tmp_path):
        """Test file analysis."""
        files = [tmp_path / "file1.py", tmp_path / "file2.py", tmp_path / "file3.py"]

        # Create files
        for f in files:
            f.write_text("print('test')")

        mock_analysis = Mock()
        mock_analyzer.analyze_file.return_value = mock_analysis

        analyzed = examiner._analyze_files(files, deep=True)

        assert len(analyzed) == 3
        assert mock_analyzer.analyze_file.call_count == 3

    @patch.object(Examiner, "analyzer")
    def test_analyze_files_with_error(self, mock_analyzer, examiner, tmp_path):
        """Test file analysis with errors."""
        files = [tmp_path / "good.py", tmp_path / "bad.py"]

        for f in files:
            f.write_text("print('test')")

        # First file succeeds, second fails
        mock_analyzer.analyze_file.side_effect = [Mock(), Exception("Analysis failed")]

        analyzed = examiner._analyze_files(files)

        # Should still get one successful analysis
        assert len(analyzed) == 1

    def test_examine_project_integration(self, examiner, temp_project):
        """Test full project examination integration."""
        result = examiner.examine_project(
            temp_project,
            deep=False,
            include_git=False,  # Avoid git dependency
            include_metrics=True,
            include_complexity=True,
            include_ownership=False,
            include_hotspots=False,
        )

        assert result.root_path == temp_project
        assert result.total_files > 0
        assert result.total_lines > 0
        assert len(result.errors) == 0
        assert result.duration > 0


class TestExamineDirectoryFunction:
    """Test the examine_directory convenience function."""

    @patch("tenets.core.examiner.examiner.Examiner")
    def test_examine_directory_basic(self, mock_examiner_class, tmp_path):
        """Test basic directory examination."""
        mock_examiner = Mock()
        mock_result = ExaminationResult(root_path=tmp_path, total_files=5)
        mock_examiner.examine_project.return_value = mock_result
        mock_examiner_class.return_value = mock_examiner

        result = examine_directory(tmp_path)

        assert isinstance(result, ExaminationResult)
        assert result.total_files == 5
        mock_examiner.examine_project.assert_called_once()

    @patch("tenets.core.examiner.examiner.Examiner")
    def test_examine_directory_with_config(self, mock_examiner_class, tmp_path, config):
        """Test with custom config."""
        mock_examiner = Mock()
        mock_examiner.examine_project.return_value = ExaminationResult(root_path=tmp_path)
        mock_examiner_class.return_value = mock_examiner

        result = examine_directory(tmp_path, config=config)

        assert isinstance(result, ExaminationResult)
        mock_examiner_class.assert_called_with(config)

    @patch("tenets.core.examiner.examiner.Examiner")
    def test_examine_directory_with_options(self, mock_examiner_class, tmp_path):
        """Test with various options."""
        mock_examiner = Mock()
        mock_examiner.examine_project.return_value = ExaminationResult(root_path=tmp_path)
        mock_examiner_class.return_value = mock_examiner

        result = examine_directory(tmp_path, deep=True, include_git=False, max_files=100)

        assert isinstance(result, ExaminationResult)
        mock_examiner.examine_project.assert_called_with(
            tmp_path, deep=True, include_git=False, max_files=100
        )
