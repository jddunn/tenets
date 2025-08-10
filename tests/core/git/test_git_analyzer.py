"""Tests for Git analyzer."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import os

from tenets.core.git.analyzer import GitAnalyzer, CommitInfo
from tenets.config import TenetsConfig

# Check Git availability
try:
    from git import Repo
    import subprocess
    result = subprocess.run(["git", "--version"], capture_output=True, check=True)
    _HAS_GIT = True
except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
    _HAS_GIT = False


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repository."""
    if not _HAS_GIT:
        pytest.skip("Git not available")
    
    from git import Repo

    # Initialize repo
    repo = Repo.init(tmp_path)

    # Configure git user
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test User")
        config.set_value("user", "email", "test@example.com")

    # Create some files
    file1 = tmp_path / "file1.py"
    file1.write_text("print('hello')")
    file2 = tmp_path / "file2.py"
    file2.write_text("print('world')")

    # Add and commit
    repo.index.add(["file1.py", "file2.py"])
    repo.index.commit("Initial commit")

    # Make changes
    file1.write_text("print('hello world')")
    repo.index.add(["file1.py"])
    repo.index.commit("Update file1")

    # Create a new file
    file3 = tmp_path / "file3.py"
    file3.write_text("# new file")
    repo.index.add(["file3.py"])
    repo.index.commit("Add file3")

    return tmp_path


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    return config


@pytest.fixture
def non_git_dir(tmp_path):
    """Create a directory that's not a git repo."""
    (tmp_path / "file.txt").write_text("content")
    return tmp_path


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestGitAnalyzer:
    """Test suite for GitAnalyzer."""

    def test_initialization_with_git_repo(self, temp_git_repo):
        """Test initialization with a valid git repository."""
        analyzer = GitAnalyzer(temp_git_repo)

        assert analyzer.root == temp_git_repo
        assert analyzer.repo is not None
        assert analyzer.is_repo() == True

    def test_initialization_with_non_git_dir(self, non_git_dir):
        """Test initialization with non-git directory."""
        analyzer = GitAnalyzer(non_git_dir)

        assert analyzer.root == non_git_dir
        assert analyzer.repo is None
        assert analyzer.is_repo() == False

    def test_initialization_with_config(self, temp_git_repo, config):
        """Test initialization with TenetsConfig."""
        config.project_root = temp_git_repo
        analyzer = GitAnalyzer(config)

        assert analyzer.root == temp_git_repo
        assert analyzer.is_repo() == True

    def test_initialization_with_nested_git(self, temp_git_repo):
        """Test initialization from subdirectory of git repo."""
        subdir = temp_git_repo / "subdir"
        subdir.mkdir()

        analyzer = GitAnalyzer(subdir)

        # Should find parent git repo
        assert analyzer.is_repo() == True

    def test_changed_files_head(self, temp_git_repo):
        """Test getting changed files from HEAD."""
        analyzer = GitAnalyzer(temp_git_repo)

        # Make uncommitted changes
        file1 = temp_git_repo / "file1.py"
        file1.write_text("print('modified')")

        changed = analyzer.changed_files(ref="HEAD")

        # Should detect uncommitted change
        assert len(changed) > 0
        assert any("file1.py" in str(f) for f in changed)

    def test_changed_files_between_refs(self, temp_git_repo):
        """Test getting changed files between two refs."""
        from git import Repo

        repo = Repo(temp_git_repo)

        # Get commit hashes
        commits = list(repo.iter_commits())
        older_commit = commits[-1].hexsha  # Initial commit
        newer_commit = commits[0].hexsha  # Latest commit

        analyzer = GitAnalyzer(temp_git_repo)
        changed = analyzer.changed_files(ref=newer_commit, diff_with=older_commit)

        # Should show files changed between commits
        assert len(changed) > 0
        file_names = [f.name for f in changed]
        assert "file1.py" in file_names or "file3.py" in file_names

    def test_changed_files_no_repo(self, non_git_dir):
        """Test changed_files when not in a git repo."""
        analyzer = GitAnalyzer(non_git_dir)

        changed = analyzer.changed_files()

        assert changed == []

    def test_changed_files_error_handling(self, temp_git_repo):
        """Test error handling in changed_files."""
        analyzer = GitAnalyzer(temp_git_repo)

        # Invalid ref should return empty list
        changed = analyzer.changed_files(ref="invalid-ref")
        assert changed == []

    def test_recent_commits(self, temp_git_repo):
        """Test getting recent commits."""
        analyzer = GitAnalyzer(temp_git_repo)

        commits = analyzer.recent_commits(limit=10)

        assert len(commits) == 3  # We made 3 commits
        assert all(isinstance(c, CommitInfo) for c in commits)

        # Check commit order (most recent first)
        assert "Add file3" in commits[0].message
        assert "Update file1" in commits[1].message
        assert "Initial commit" in commits[2].message

        # Check commit info
        for commit in commits:
            assert commit.hexsha is not None
            assert commit.author == "Test User"
            assert commit.email == "test@example.com"
            assert commit.committed_date > 0

    def test_recent_commits_with_paths(self, temp_git_repo):
        """Test getting commits for specific paths."""
        analyzer = GitAnalyzer(temp_git_repo)

        # Get commits only for file1.py
        commits = analyzer.recent_commits(limit=10, paths=[temp_git_repo / "file1.py"])

        # Should only show commits that touched file1.py
        assert len(commits) == 2  # Initial and Update commits
        assert "Update file1" in commits[0].message
        assert "Initial commit" in commits[1].message

    def test_recent_commits_limit(self, temp_git_repo):
        """Test commit limit."""
        analyzer = GitAnalyzer(temp_git_repo)

        commits = analyzer.recent_commits(limit=2)

        assert len(commits) == 2
        assert "Add file3" in commits[0].message
        assert "Update file1" in commits[1].message

    def test_recent_commits_no_repo(self, non_git_dir):
        """Test recent_commits when not in a git repo."""
        analyzer = GitAnalyzer(non_git_dir)

        commits = analyzer.recent_commits()

        assert commits == []

    def test_recent_commits_error_handling(self, temp_git_repo):
        """Test error handling in recent_commits."""
        analyzer = GitAnalyzer(temp_git_repo)

        with patch.object(analyzer.repo, "iter_commits", side_effect=Exception("Git error")):
            commits = analyzer.recent_commits()
            assert commits == []

    def test_blame(self, temp_git_repo):
        """Test git blame functionality."""
        analyzer = GitAnalyzer(temp_git_repo)

        file1 = temp_git_repo / "file1.py"

        blame_info = analyzer.blame(file1)

        assert len(blame_info) > 0

        # Each entry should be (author, line)
        for author, line in blame_info:
            assert author == "Test User"
            assert isinstance(line, str)

        # Should have one line
        assert len(blame_info) == 1
        assert blame_info[0][1] == "print('hello world')"

    def test_blame_multi_line_file(self, temp_git_repo):
        """Test blame on multi-line file."""
        from git import Repo

        repo = Repo(temp_git_repo)

        # Create multi-line file
        multiline = temp_git_repo / "multiline.py"
        multiline.write_text("line1\nline2\nline3")
        repo.index.add(["multiline.py"])
        repo.index.commit("Add multiline")

        # Modify middle line
        multiline.write_text("line1\nmodified\nline3")
        repo.index.add(["multiline.py"])
        repo.index.commit("Modify line2")

        analyzer = GitAnalyzer(temp_git_repo)
        blame_info = analyzer.blame(multiline)

        assert len(blame_info) == 3

        # All by same author
        authors = [author for author, _ in blame_info]
        assert all(a == "Test User" for a in authors)

        # Check lines
        lines = [line for _, line in blame_info]
        assert lines == ["line1", "modified", "line3"]

    def test_blame_no_repo(self, non_git_dir):
        """Test blame when not in a git repo."""
        analyzer = GitAnalyzer(non_git_dir)

        file_path = non_git_dir / "file.txt"
        blame_info = analyzer.blame(file_path)

        assert blame_info == []

    def test_blame_nonexistent_file(self, temp_git_repo):
        """Test blame on non-existent file."""
        analyzer = GitAnalyzer(temp_git_repo)

        blame_info = analyzer.blame(temp_git_repo / "nonexistent.py")

        assert blame_info == []

    def test_blame_error_handling(self, temp_git_repo):
        """Test error handling in blame."""
        analyzer = GitAnalyzer(temp_git_repo)

        with patch.object(analyzer.repo, "blame", side_effect=Exception("Blame error")):
            blame_info = analyzer.blame(temp_git_repo / "file1.py")
            assert blame_info == []

    def test_relative_paths(self, temp_git_repo):
        """Test handling of relative paths."""
        analyzer = GitAnalyzer(temp_git_repo)

        # Create file in subdirectory
        subdir = temp_git_repo / "subdir"
        subdir.mkdir()
        subfile = subdir / "subfile.py"
        subfile.write_text("# subfile")

        from git import Repo

        repo = Repo(temp_git_repo)
        repo.index.add(["subdir/subfile.py"])
        repo.index.commit("Add subfile")

        # Test with relative path
        blame_info = analyzer.blame(subfile)

        assert len(blame_info) == 1
        assert blame_info[0][0] == "Test User"
        assert blame_info[0][1] == "# subfile"

    def test_commit_info_dataclass(self):
        """Test CommitInfo dataclass."""
        commit = CommitInfo(
            hexsha="abc123",
            author="John Doe",
            email="john@example.com",
            message="Test commit",
            committed_date=1234567890,
        )

        assert commit.hexsha == "abc123"
        assert commit.author == "John Doe"
        assert commit.email == "john@example.com"
        assert commit.message == "Test commit"
        assert commit.committed_date == 1234567890

    def test_analyzer_with_none_root(self):
        """Test analyzer with None as root."""
        with patch("pathlib.Path.cwd", return_value=Path("/tmp")):
            analyzer = GitAnalyzer(None)
            assert analyzer.root == Path("/tmp")

    def test_search_parent_directories(self, tmp_path):
        """Test that GitAnalyzer searches parent directories for .git."""
        from git import Repo

        # Create git repo at root
        repo = Repo.init(tmp_path)

        # Create nested directory structure
        deep_dir = tmp_path / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)

        # Initialize analyzer from deep directory
        analyzer = GitAnalyzer(deep_dir)

        # Should find the git repo in parent
        assert analyzer.is_repo() == True

    def test_git_operations_with_binary_files(self, temp_git_repo):
        """Test git operations with binary files."""
        from git import Repo

        repo = Repo(temp_git_repo)

        # Add binary file
        binary_file = temp_git_repo / "image.png"
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        repo.index.add(["image.png"])
        repo.index.commit("Add binary file")

        analyzer = GitAnalyzer(temp_git_repo)

        # Should handle binary files in commits
        commits = analyzer.recent_commits(limit=1)
        assert len(commits) == 1
        assert "Add binary file" in commits[0].message
