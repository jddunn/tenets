"""Tests for Git blame analyzer."""

import subprocess
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.git.blame import (
    BlameAnalyzer,
    BlameLine,
    BlameReport,
    FileBlame,
    analyze_blame,
)

# Check Git availability
try:
    from git import Repo

    result = subprocess.run(["git", "--version"], capture_output=True, check=True)
    _HAS_GIT = True
except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
    _HAS_GIT = False


@pytest.fixture
def temp_git_repo_with_history(tmp_path):
    """Create a git repo with rich history for blame testing."""
    if not _HAS_GIT:
        pytest.skip("Git not available")

    from git import Repo

    repo = Repo.init(tmp_path)

    # Configure multiple users
    with repo.config_writer() as config:
        config.set_value("user", "name", "Alice")
        config.set_value("user", "email", "alice@example.com")

    # Create initial file
    main_file = tmp_path / "main.py"
    main_file.write_text(
        """def hello():
    print("Hello")

def world():
    print("World")
"""
    )
    repo.index.add(["main.py"])
    repo.index.commit("Initial commit by Alice")

    # Change user to Bob
    with repo.config_writer() as config:
        config.set_value("user", "name", "Bob")
        config.set_value("user", "email", "bob@example.com")

    # Bob modifies the file
    main_file.write_text(
        """def hello():
    print("Hello, everyone!")  # Bob's change

def world():
    print("World")

def goodbye():
    print("Goodbye")  # Bob added this
"""
    )
    repo.index.add(["main.py"])
    repo.index.commit("Bob's modifications")

    # Change user to Charlie
    with repo.config_writer() as config:
        config.set_value("user", "name", "Charlie")
        config.set_value("user", "email", "charlie@example.com")

    # Charlie adds more
    main_file.write_text(
        """import sys  # Charlie added import

def hello():
    print("Hello, everyone!")  # Bob's change

def world():
    print("World")

def goodbye():
    print("Goodbye")  # Bob added this

# Charlie's addition
def main():
    hello()
    world()
    goodbye()
"""
    )
    repo.index.add(["main.py"])
    repo.index.commit("Charlie adds main function")

    # Create another file for directory analysis
    utils_file = tmp_path / "utils.py"
    utils_file.write_text(
        """# Utility functions
def helper():
    pass
"""
    )
    repo.index.add(["utils.py"])
    repo.index.commit("Add utils by Charlie")

    return tmp_path


@pytest.fixture
def config():
    """Create test configuration."""
    return TenetsConfig()


@pytest.fixture
def blame_analyzer(config):
    """Create BlameAnalyzer instance."""
    return BlameAnalyzer(config)


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestBlameLine:
    """Test suite for BlameLine dataclass."""

    def test_blame_line_creation(self):
        """Test creating a BlameLine."""
        line = BlameLine(
            line_number=10,
            content="    print('test')",
            author="Alice",
            author_email="alice@example.com",
            commit_sha="abc123",
            commit_date=datetime.now() - timedelta(days=5),
            commit_message="Add test print",
            age_days=5,
        )

        assert line.line_number == 10
        assert line.content == "    print('test')"
        assert line.author == "Alice"
        assert line.age_days == 5

    def test_blame_line_is_recent(self):
        """Test is_recent property."""
        recent_line = BlameLine(
            line_number=1,
            content="code",
            author="Test",
            author_email="test@example.com",
            commit_sha="abc",
            commit_date=datetime.now() - timedelta(days=10),
            commit_message="msg",
            age_days=10,
        )

        old_line = BlameLine(
            line_number=2,
            content="old code",
            author="Test",
            author_email="test@example.com",
            commit_sha="def",
            commit_date=datetime.now() - timedelta(days=100),
            commit_message="old msg",
            age_days=100,
        )

        assert recent_line.is_recent == True
        assert old_line.is_recent == False

    def test_blame_line_is_old(self):
        """Test is_old property."""
        new_line = BlameLine(
            line_number=1,
            content="new",
            author="Test",
            author_email="test@example.com",
            commit_sha="abc",
            commit_date=datetime.now() - timedelta(days=30),
            commit_message="msg",
            age_days=30,
        )

        old_line = BlameLine(
            line_number=2,
            content="old",
            author="Test",
            author_email="test@example.com",
            commit_sha="def",
            commit_date=datetime.now() - timedelta(days=200),
            commit_message="old msg",
            age_days=200,
        )

        assert new_line.is_old == False
        assert old_line.is_old == True

    def test_blame_line_is_documentation(self):
        """Test is_documentation property."""
        doc_lines = [
            "# This is a comment",
            "// JavaScript comment",
            "/* C-style comment */",
            "* Javadoc style",
            '"""Python docstring"""',
            "'''Another docstring'''",
            "    # TODO: fix this",
            "// FIXME: broken",
            "/* NOTE: important */",
        ]

        code_lines = ["print('hello')", "function test() {", "return value"]

        for content in doc_lines:
            line = BlameLine(
                line_number=1,
                content=content,
                author="Test",
                author_email="test@example.com",
                commit_sha="abc",
                commit_date=datetime.now(),
                commit_message="msg",
            )
            assert line.is_documentation == True

        for content in code_lines:
            line = BlameLine(
                line_number=1,
                content=content,
                author="Test",
                author_email="test@example.com",
                commit_sha="abc",
                commit_date=datetime.now(),
                commit_message="msg",
            )
            assert line.is_documentation == False

    def test_blame_line_is_empty(self):
        """Test is_empty property."""
        empty_lines = ["", "   ", "\t", "\n", "  \t  "]
        non_empty_lines = ["code", "  # comment", "\tprint()"]

        for content in empty_lines:
            line = BlameLine(
                line_number=1,
                content=content,
                author="Test",
                author_email="test@example.com",
                commit_sha="abc",
                commit_date=datetime.now(),
                commit_message="msg",
            )
            assert line.is_empty == True

        for content in non_empty_lines:
            line = BlameLine(
                line_number=1,
                content=content,
                author="Test",
                author_email="test@example.com",
                commit_sha="abc",
                commit_date=datetime.now(),
                commit_message="msg",
            )
            assert line.is_empty == False


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestFileBlame:
    """Test suite for FileBlame dataclass."""

    def test_file_blame_creation(self):
        """Test creating FileBlame."""
        file_blame = FileBlame(file_path="test.py", total_lines=100)

        assert file_blame.file_path == "test.py"
        assert file_blame.total_lines == 100
        assert file_blame.blame_lines == []
        assert len(file_blame.authors) == 0

    def test_primary_author(self):
        """Test primary_author property."""
        file_blame = FileBlame(file_path="test.py")

        # No authors yet
        assert file_blame.primary_author is None

        # Add author stats
        file_blame.author_stats = {
            "Alice": {"lines": 50},
            "Bob": {"lines": 30},
            "Charlie": {"lines": 20},
        }

        assert file_blame.primary_author == "Alice"

    def test_author_diversity(self):
        """Test author_diversity calculation."""
        file_blame = FileBlame(file_path="test.py", total_lines=100)

        # Single author - no diversity
        file_blame.author_stats = {"Alice": {"lines": 100}}
        assert file_blame.author_diversity == 0.0

        # Multiple authors with equal distribution - high diversity
        file_blame.author_stats = {
            "Alice": {"lines": 33},
            "Bob": {"lines": 33},
            "Charlie": {"lines": 34},
        }
        diversity = file_blame.author_diversity
        assert diversity > 0.9  # Should be close to 1.0

    def test_average_age_days(self):
        """Test average_age_days calculation."""
        file_blame = FileBlame(file_path="test.py")

        # No lines
        assert file_blame.average_age_days == 0.0

        # Add lines with different ages
        file_blame.blame_lines = [
            BlameLine(
                line_number=i,
                content=f"line{i}",
                author="Test",
                author_email="test@example.com",
                commit_sha="abc",
                commit_date=datetime.now(),
                commit_message="msg",
                age_days=age,
            )
            for i, age in enumerate([10, 20, 30], 1)
        ]

        assert file_blame.average_age_days == 20.0

    def test_freshness_score(self):
        """Test freshness_score calculation."""
        file_blame = FileBlame(file_path="test.py")

        # No lines
        assert file_blame.freshness_score == 0.0

        # All recent lines
        file_blame.blame_lines = [
            BlameLine(
                line_number=i,
                content=f"line{i}",
                author="Test",
                author_email="test@example.com",
                commit_sha="abc",
                commit_date=datetime.now() - timedelta(days=5),
                commit_message="msg",
                age_days=5,
            )
            for i in range(1, 11)
        ]

        assert file_blame.freshness_score == 100.0

        # Half recent, half old
        for i in range(5, 10):
            file_blame.blame_lines[i].age_days = 100

        assert file_blame.freshness_score == 50.0


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestBlameReport:
    """Test suite for BlameReport dataclass."""

    def test_blame_report_creation(self):
        """Test creating BlameReport."""
        report = BlameReport(files_analyzed=5, total_lines=500, total_authors=3)

        assert report.files_analyzed == 5
        assert report.total_lines == 500
        assert report.total_authors == 3

    def test_bus_factor(self):
        """Test bus_factor calculation."""
        report = BlameReport(total_lines=1000)

        # No authors
        assert report.bus_factor == 0

        # Single dominant author
        report.author_summary = {"Alice": {"total_lines": 900}, "Bob": {"total_lines": 100}}

        assert report.bus_factor == 1  # Alice owns >10%

        # Multiple critical authors
        report.author_summary = {
            "Alice": {"total_lines": 400},
            "Bob": {"total_lines": 300},
            "Charlie": {"total_lines": 200},
            "Dave": {"total_lines": 100},
        }

        assert report.bus_factor == 3  # Alice, Bob, Charlie each own >10%

    def test_collaboration_score(self):
        """Test collaboration_score calculation."""
        report = BlameReport()

        # No files
        assert report.collaboration_score == 0.0

        # All single-author files
        report.files_analyzed = 10
        for i in range(10):
            file_blame = FileBlame(file_path=f"file{i}.py")
            file_blame.authors = {"Alice"}
            report.file_blames[f"file{i}.py"] = file_blame

        assert report.collaboration_score == 0.0

        # All multi-author files
        for i in range(10):
            report.file_blames[f"file{i}.py"].authors = {"Alice", "Bob"}

        assert report.collaboration_score == 100.0

    def test_to_dict(self):
        """Test to_dict conversion."""
        report = BlameReport(files_analyzed=2, total_lines=100, total_authors=2)

        report.author_summary = {"Alice": {"total_lines": 60}, "Bob": {"total_lines": 40}}

        result = report.to_dict()

        assert result["summary"]["files_analyzed"] == 2
        assert result["summary"]["total_lines"] == 100
        assert result["summary"]["total_authors"] == 2
        assert len(result["top_authors"]) == 2


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestBlameAnalyzer:
    """Test suite for BlameAnalyzer."""

    def test_initialization(self, config):
        """Test BlameAnalyzer initialization."""
        analyzer = BlameAnalyzer(config)

        assert analyzer.config == config
        assert analyzer._blame_cache == {}

    def test_analyze_file(self, temp_git_repo_with_history, blame_analyzer):
        """Test analyzing a single file."""
        file_blame = blame_analyzer.analyze_file(temp_git_repo_with_history, "main.py")

        assert file_blame.file_path == "main.py"
        assert file_blame.total_lines > 0
        assert len(file_blame.authors) > 0
        assert "Charlie" in file_blame.authors  # Last committer

        # Check caching
        cached = blame_analyzer.analyze_file(temp_git_repo_with_history, "main.py")
        assert cached == file_blame

    def test_analyze_file_with_options(self, temp_git_repo_with_history, blame_analyzer):
        """Test analyze_file with different options."""
        file_blame = blame_analyzer.analyze_file(
            temp_git_repo_with_history, "main.py", ignore_whitespace=False, follow_renames=False
        )

        assert file_blame.total_lines > 0

    @patch("subprocess.run")
    def test_analyze_file_error_handling(
        self, mock_run, temp_git_repo_with_history, blame_analyzer
    ):
        """Test error handling in analyze_file."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git blame")

        file_blame = blame_analyzer.analyze_file(temp_git_repo_with_history, "main.py")

        # Should return empty FileBlame on error
        assert file_blame.file_path == "main.py"
        assert file_blame.total_lines == 0

    def test_analyze_directory(self, temp_git_repo_with_history, blame_analyzer):
        """Test analyzing a directory."""
        report = blame_analyzer.analyze_directory(
            temp_git_repo_with_history, directory=".", file_pattern="*.py", max_files=10
        )

        assert report.files_analyzed > 0
        assert report.total_authors > 0
        assert report.total_lines > 0
        assert len(report.file_blames) > 0

        # Should have recommendations
        assert len(report.recommendations) > 0

    def test_analyze_directory_recursive(self, temp_git_repo_with_history, blame_analyzer):
        """Test recursive directory analysis."""
        # Create subdirectory with file
        subdir = temp_git_repo_with_history / "subdir"
        subdir.mkdir()
        (subdir / "sub.py").write_text("# subfile")

        report = blame_analyzer.analyze_directory(
            temp_git_repo_with_history, directory=".", recursive=True
        )

        assert report.files_analyzed > 0

    def test_analyze_directory_non_recursive(self, temp_git_repo_with_history, blame_analyzer):
        """Test non-recursive directory analysis."""
        report = blame_analyzer.analyze_directory(
            temp_git_repo_with_history, directory=".", recursive=False
        )

        assert report.files_analyzed > 0

    def test_get_line_history(self, temp_git_repo_with_history, blame_analyzer):
        """Test getting line history."""
        history = blame_analyzer.get_line_history(
            temp_git_repo_with_history, "main.py", line_number=1
        )

        assert isinstance(history, list)
        # Should have at least one entry
        if history:
            assert "author" in history[0]
            assert "date" in history[0]

    @patch("subprocess.run")
    def test_get_line_history_error(self, mock_run, temp_git_repo_with_history, blame_analyzer):
        """Test error handling in get_line_history."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git blame")

        history = blame_analyzer.get_line_history(temp_git_repo_with_history, "main.py", 1)

        assert history == []

    def test_should_skip_file(self, blame_analyzer):
        """Test file skipping logic."""
        skip_files = [
            ".git/config",
            "__pycache__/test.pyc",
            "file.pyo",
            "lib.so",
            "node_modules/package.json",
            "vendor/lib.php",
            "script.min.js",
            "styles.min.css",
            "package-lock.json",
            "image.png",
            "archive.zip",
        ]

        keep_files = ["main.py", "src/utils.js", "README.md", "config.yaml"]

        for file_path in skip_files:
            assert blame_analyzer._should_skip_file(file_path) == True

        for file_path in keep_files:
            assert blame_analyzer._should_skip_file(file_path) == False

    def test_parse_blame_output(self, blame_analyzer):
        """Test parsing git blame porcelain output."""
        # Sample git blame porcelain output
        output = """abc123def456789012345678901234567890abcd 1 1 1
author Alice
author-mail <alice@example.com>
author-time 1234567890
summary Initial commit
	print("Hello")
def456789012345678901234567890abcdef1234 2 2 1
author Bob
author-mail <bob@example.com>
author-time 1234567900
summary Bob's change
	print("World")"""

        file_blame = FileBlame(file_path="test.py")
        blame_analyzer._parse_blame_output(output, file_blame)

        assert file_blame.total_lines == 2
        assert len(file_blame.blame_lines) == 2
        assert "Alice" in file_blame.authors
        assert "Bob" in file_blame.authors

    def test_calculate_file_stats(self, blame_analyzer):
        """Test calculating file statistics."""
        file_blame = FileBlame(file_path="test.py")

        # Add some blame lines
        now = datetime.now()
        file_blame.blame_lines = [
            BlameLine(
                line_number=1,
                content="# comment",
                author="Alice",
                author_email="alice@example.com",
                commit_sha="abc123",
                commit_date=now - timedelta(days=100),
                commit_message="Old commit",
                age_days=100,
            ),
            BlameLine(
                line_number=2,
                content="print('hello')",
                author="Bob",
                author_email="bob@example.com",
                commit_sha="def456",
                commit_date=now - timedelta(days=10),
                commit_message="Recent commit",
                age_days=10,
            ),
        ]

        blame_analyzer._calculate_file_stats(file_blame)

        assert file_blame.oldest_line.author == "Alice"
        assert file_blame.newest_line.author == "Bob"
        assert "Alice" in file_blame.author_stats
        assert "Bob" in file_blame.author_stats
        assert file_blame.age_distribution["recent"] == 1
        assert file_blame.age_distribution["old"] == 1

    def test_generate_recommendations(self, blame_analyzer):
        """Test generating recommendations."""
        report = BlameReport()

        # Low bus factor
        report.bus_factor = 1
        recommendations = blame_analyzer._generate_recommendations(report)
        assert any("bus factor" in r.lower() for r in recommendations)

        # Many single-author files
        report.files_analyzed = 10
        report.single_author_files = ["file1.py", "file2.py", "file3.py", "file4.py"]
        recommendations = blame_analyzer._generate_recommendations(report)
        assert any("single author" in r.lower() for r in recommendations)

        # Low collaboration
        report.collaboration_score = 30.0
        recommendations = blame_analyzer._generate_recommendations(report)
        assert any("collaboration" in r.lower() for r in recommendations)


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestAnalyzeBlameFunction:
    """Test the analyze_blame convenience function."""

    def test_analyze_blame_file(self, temp_git_repo_with_history):
        """Test analyzing a single file."""
        report = analyze_blame(temp_git_repo_with_history, target="main.py")

        assert report.files_analyzed == 1
        assert "main.py" in report.file_blames

    def test_analyze_blame_directory(self, temp_git_repo_with_history):
        """Test analyzing a directory."""
        report = analyze_blame(temp_git_repo_with_history, target=".")

        assert report.files_analyzed > 0
        assert report.total_lines > 0

    def test_analyze_blame_with_config(self, temp_git_repo_with_history, config):
        """Test with custom config."""
        report = analyze_blame(temp_git_repo_with_history, config=config)

        assert isinstance(report, BlameReport)

    def test_analyze_blame_kwargs(self, temp_git_repo_with_history):
        """Test passing kwargs."""
        report = analyze_blame(temp_git_repo_with_history, target=".", max_files=1)

        # Should respect max_files
        assert report.files_analyzed <= 1
