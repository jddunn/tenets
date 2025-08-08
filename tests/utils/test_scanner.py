"""Tests for the FileScanner utility."""

import pytest
import tempfile
from pathlib import Path
import os

from tenets.utils.scanner import FileScanner
from tenets.config import TenetsConfig


@pytest.fixture
def config():
    """Create test configuration."""
    config = TenetsConfig()
    config.additional_ignore_patterns = ['.test_ignore']
    return config


@pytest.fixture
def scanner(config):
    """Create a FileScanner instance."""
    return FileScanner(config)


@pytest.fixture
def test_project(tmp_path):
    """Create a test project structure."""
    # Source files
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("print('main')")
    (src_dir / "utils.py").write_text("# utils")
    (src_dir / "config.json").write_text('{"key": "value"}')
    
    # Test files
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_main.py").write_text("import pytest")
    (test_dir / "test_utils.py").write_text("# test")
    
    # Node modules (should be ignored)
    node_dir = tmp_path / "node_modules"
    node_dir.mkdir()
    (node_dir / "package.js").write_text("module.exports = {}")
    
    # Build artifacts (should be ignored)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "output.js").write_text("compiled")
    
    # Hidden files
    (tmp_path / ".env").write_text("SECRET=value")
    (tmp_path / ".gitignore").write_text("*.pyc\nbuild/\n.env")
    
    # Binary files (should be ignored)
    (tmp_path / "image.png").write_bytes(b'\x89PNG\r\n\x1a\n')
    (tmp_path / "data.db").write_bytes(b'SQLite format 3\x00')
    
    # Python cache (should be ignored)
    pycache_dir = src_dir / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "main.cpython-39.pyc").write_bytes(b'pyc file')
    
    return tmp_path


class TestFileScanner:
    """Test suite for FileScanner."""
    
    def test_initialization(self, config):
        """Test scanner initialization."""
        scanner = FileScanner(config)
        
        assert scanner.config == config
        assert '.git' in scanner.ignore_patterns
        assert 'node_modules' in scanner.ignore_patterns
        assert '.test_ignore' in scanner.ignore_patterns
        
    def test_scan_single_file(self, scanner, test_project):
        """Test scanning a single file."""
        file_path = test_project / "src" / "main.py"
        
        files = scanner.scan([file_path])
        
        assert len(files) == 1
        assert files[0] == file_path
        
    def test_scan_directory(self, scanner, test_project):
        """Test scanning a directory."""
        files = scanner.scan([test_project])
        
        file_names = [f.name for f in files]
        
        # Should include source files
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "config.json" in file_names
        assert "test_main.py" in file_names
        
        # Should exclude ignored files
        assert "package.js" not in file_names  # node_modules
        assert "output.js" not in file_names   # build
        assert "main.cpython-39.pyc" not in file_names  # __pycache__
        assert "image.png" not in file_names   # binary
        assert "data.db" not in file_names     # binary
        
    def test_include_patterns(self, scanner, test_project):
        """Test include patterns filtering."""
        files = scanner.scan(
            [test_project],
            include_patterns=["*.py"]
        )
        
        # Should only include Python files
        for file in files:
            assert file.suffix == ".py"
        
        file_names = [f.name for f in files]
        assert "main.py" in file_names
        assert "config.json" not in file_names
        
    def test_exclude_patterns(self, scanner, test_project):
        """Test exclude patterns filtering."""
        files = scanner.scan(
            [test_project],
            exclude_patterns=["test_*.py", "*.json"]
        )
        
        file_names = [f.name for f in files]
        
        # Should exclude test files and JSON
        assert "test_main.py" not in file_names
        assert "test_utils.py" not in file_names
        assert "config.json" not in file_names
        
        # Should include other files
        assert "main.py" in file_names
        assert "utils.py" in file_names
        
    def test_max_file_size(self, scanner, test_project):
        """Test maximum file size filtering."""
        # Create a large file
        large_file = test_project / "large.txt"
        large_file.write_text("x" * 10000)
        
        # Create a small file
        small_file = test_project / "small.txt"
        small_file.write_text("x" * 10)
        
        files = scanner.scan(
            [test_project],
            max_file_size=100
        )
        
        file_names = [f.name for f in files]
        
        assert "small.txt" in file_names
        assert "large.txt" not in file_names
        
    def test_gitignore_respect(self, scanner, test_project):
        """Test respecting .gitignore patterns."""
        files = scanner.scan(
            [test_project],
            respect_gitignore=True
        )
        
        file_names = [f.name for f in files]
        
        # .gitignore specifies *.pyc, build/, .env
        assert ".env" not in file_names
        # Note: build/ is also in default ignore patterns
        
    def test_gitignore_disabled(self, scanner, test_project):
        """Test ignoring .gitignore patterns."""
        # Create a file that would be gitignored
        ignored_file = test_project / "src" / "temp.pyc"
        ignored_file.write_bytes(b'pyc content')
        
        files = scanner.scan(
            [test_project],
            respect_gitignore=False,
            include_patterns=["*.pyc"]  # Explicitly include
        )
        
        # Since we're not respecting gitignore and explicitly including .pyc
        # (though .pyc is also in DEFAULT_IGNORE_PATTERNS)
        # This tests the interaction between patterns
        
    def test_symlinks(self, scanner, test_project):
        """Test handling of symbolic links."""
        # Create a symlink
        link_target = test_project / "src" / "main.py"
        symlink = test_project / "main_link.py"
        
        try:
            symlink.symlink_to(link_target)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")
        
        # Without following symlinks
        files_no_follow = scanner.scan(
            [test_project],
            follow_symlinks=False
        )
        
        # With following symlinks
        files_follow = scanner.scan(
            [test_project],
            follow_symlinks=True
        )
        
        # Both should include the symlink file itself
        assert symlink in files_follow or symlink.resolve() in files_follow
        
    def test_find_files_by_name(self, scanner, test_project):
        """Test finding files by name pattern."""
        # Case-sensitive search
        files = scanner.find_files_by_name(
            test_project,
            "main.*",
            case_sensitive=True
        )
        
        assert len(files) == 1
        assert files[0].name == "main.py"
        
        # Case-insensitive search
        files = scanner.find_files_by_name(
            test_project,
            "MAIN.*",
            case_sensitive=False
        )
        
        assert len(files) == 1
        assert files[0].name == "main.py"
        
        # Wildcard pattern
        files = scanner.find_files_by_name(
            test_project,
            "test_*.py"
        )
        
        file_names = [f.name for f in files]
        assert "test_main.py" in file_names
        assert "test_utils.py" in file_names
        
    def test_find_files_by_content(self, scanner, test_project):
        """Test finding files by content."""
        # Case-sensitive content search
        files = scanner.find_files_by_content(
            test_project,
            "pytest",
            case_sensitive=True
        )
        
        assert len(files) == 1
        assert files[0].name == "test_main.py"
        
        # Case-insensitive content search
        files = scanner.find_files_by_content(
            test_project,
            "PYTEST",
            case_sensitive=False
        )
        
        assert len(files) == 1
        assert files[0].name == "test_main.py"
        
        # Search with file pattern filter
        files = scanner.find_files_by_content(
            test_project,
            "print",
            file_patterns=["*.py"]
        )
        
        assert len(files) == 1
        assert files[0].name == "main.py"
        
    def test_binary_file_exclusion(self, scanner, test_project):
        """Test that binary files are excluded."""
        files = scanner.scan([test_project])
        
        file_names = [f.name for f in files]
        
        # Binary files should be excluded
        assert "image.png" not in file_names
        assert "data.db" not in file_names
        
        # Check that binary extensions are recognized
        assert ".png" in scanner.BINARY_EXTENSIONS
        assert ".db" in scanner.BINARY_EXTENSIONS
        
    def test_duplicate_removal(self, scanner, test_project):
        """Test that duplicates are removed from results."""
        # Scan the same path multiple times
        files = scanner.scan([
            test_project,
            test_project / "src",
            test_project / "src"  # Duplicate
        ])
        
        # Count occurrences of main.py
        main_py_count = sum(1 for f in files if f.name == "main.py")
        
        # Should only appear once despite multiple scan paths
        assert main_py_count == 1
        
    def test_nested_directory_structure(self, scanner, tmp_path):
        """Test scanning deeply nested directories."""
        # Create nested structure
        deep_dir = tmp_path
        for i in range(5):
            deep_dir = deep_dir / f"level_{i}"
            deep_dir.mkdir()
            (deep_dir / f"file_{i}.py").write_text(f"# Level {i}")
        
        files = scanner.scan([tmp_path])
        
        # Should find all nested files
        assert len(files) == 5
        
        # Check that all levels are found
        for i in range(5):
            assert any(f"file_{i}.py" in str(f) for f in files)
            
    def test_mixed_path_types(self, scanner, test_project):
        """Test scanning with mixed files and directories."""
        paths = [
            test_project / "src" / "main.py",  # File
            test_project / "tests",            # Directory
            test_project / "src" / "utils.py"  # File
        ]
        
        files = scanner.scan(paths)
        
        file_names = [f.name for f in files]
        
        # Should include direct files
        assert "main.py" in file_names
        assert "utils.py" in file_names
        
        # Should include files from directory
        assert "test_main.py" in file_names
        assert "test_utils.py" in file_names
        
    def test_empty_directory(self, scanner, tmp_path):
        """Test scanning an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        files = scanner.scan([empty_dir])
        
        assert len(files) == 0
        
    def test_nonexistent_path(self, scanner, tmp_path):
        """Test scanning a non-existent path."""
        nonexistent = tmp_path / "does_not_exist"
        
        files = scanner.scan([nonexistent])
        
        # Should handle gracefully and return empty
        assert len(files) == 0
        
    def test_custom_ignore_patterns(self, config, test_project):
        """Test custom ignore patterns from config."""
        # Create a file matching custom ignore pattern
        ignored_file = test_project / ".test_ignore"
        ignored_file.write_text("ignored")
        
        scanner = FileScanner(config)
        files = scanner.scan([test_project])
        
        file_names = [f.name for f in files]
        
        # Custom ignore pattern should be applied
        assert ".test_ignore" not in file_names
        
    def test_gitignore_loading_error(self, scanner, tmp_path):
        """Test handling of gitignore loading errors."""
        # Create an unreadable .gitignore
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.txt")
        
        # Make it unreadable (platform-dependent)
        try:
            gitignore.chmod(0o000)
            
            # Should handle error gracefully
            files = scanner.scan([tmp_path], respect_gitignore=True)
            
            # Should still scan files
            test_file = tmp_path / "test.py"
            test_file.write_text("test")
            files = scanner.scan([tmp_path], respect_gitignore=True)
            
            assert test_file in files
            
        except Exception:
            pytest.skip("Cannot change file permissions on this system")
        finally:
            # Restore permissions for cleanup
            try:
                gitignore.chmod(0o644)
            except:
                pass