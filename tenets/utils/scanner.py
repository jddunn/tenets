"""File scanning utilities.

This module provides functionality for discovering files in a codebase,
respecting ignore patterns and filtering rules.
"""

import fnmatch
import os
from pathlib import Path
from typing import Generator, List, Optional, Set

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger


class FileScanner:
    """Scans directories for files matching criteria."""

    # Common ignore patterns
    DEFAULT_IGNORE_PATTERNS = {
        # Version control
        ".git",
        ".svn",
        ".hg",
        ".bzr",
        # Dependencies
        "node_modules",
        "vendor",
        "bower_components",
        # Python
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".mypy_cache",
        "venv",
        ".venv",
        "env",
        ".env",
        # Build artifacts
        "build",
        "dist",
        "out",
        "target",
        "*.egg-info",
        # IDE
        ".idea",
        ".vscode",
        "*.swp",
        "*.swo",
        ".DS_Store",
        # Other
        "*.log",
        "*.tmp",
        "*.temp",
        "*.cache",
    }

    # Binary file extensions to skip
    BINARY_EXTENSIONS = {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".a",
        ".o",
        ".jar",
        ".class",
        ".pyc",
        ".pyo",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".db",
        ".sqlite",
        ".bin",
        ".dat",
    }

    def __init__(self, config: TenetsConfig):
        """Initialize the scanner.

        Args:
            config: Tenets configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Log multiprocessing configuration
        from tenets.utils.multiprocessing import get_scanner_workers, log_worker_info

        self.workers = get_scanner_workers(config)
        parallel_mode = getattr(config.scanner, "parallel_mode", "auto") if config else "auto"
        log_worker_info(self.logger, "FileScanner", self.workers)
        self.logger.info(f"FileScanner initialized (parallel_mode: {parallel_mode})")

        # Build ignore patterns
        self.ignore_patterns = set(self.DEFAULT_IGNORE_PATTERNS)
        if (
            config
            and hasattr(config, "additional_ignore_patterns")
            and config.additional_ignore_patterns
        ):
            self.ignore_patterns.update(config.additional_ignore_patterns)

        # Add minified file patterns if exclude_minified is True (default)
        self.exclude_minified = getattr(config, "exclude_minified", True) if config else True
        if self.exclude_minified:
            # Add minified patterns
            minified_patterns = getattr(config, "minified_patterns", []) if config else []
            if minified_patterns:
                self.ignore_patterns.update(minified_patterns)
            else:
                # Default minified patterns
                self.ignore_patterns.update(
                    [
                        "*.min.js",
                        "*.min.css",
                        "bundle.js",
                        "*.bundle.js",
                        "*.bundle.css",
                        "*.production.js",
                        "*.prod.js",
                        "vendor.prod.js",
                        "*.dist.js",
                        "*.compiled.js",
                    ]
                )

            # Add build directory patterns
            build_dirs = getattr(config, "build_directory_patterns", []) if config else []
            if build_dirs:
                # Remove trailing slashes for directory name matching
                self.ignore_patterns.update(d.rstrip("/") for d in build_dirs)
            else:
                # Default build directories (without trailing slashes)
                self.ignore_patterns.update(["dist", "build", "out", "output", "node_modules"])

    def scan(
        self,
        paths: List[Path],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        follow_symlinks: bool = False,
        respect_gitignore: bool = True,
        max_file_size: Optional[int] = None,
    ) -> List[Path]:
        """Scan paths for files matching criteria.

        Args:
            paths: Paths to scan (files or directories)
            include_patterns: Patterns of files to include (e.g., "*.py")
            exclude_patterns: Additional patterns to exclude
            follow_symlinks: Whether to follow symbolic links
            respect_gitignore: Whether to respect .gitignore files
            max_file_size: Maximum file size in bytes

        Returns:
            List of file paths found
        """
        files = []

        for path in paths:
            if path.is_file():
                # Direct file reference
                if self._should_include_file(
                    path, include_patterns, exclude_patterns, max_file_size
                ):
                    files.append(path)
            elif path.is_dir():
                # Scan directory
                files.extend(
                    self._scan_directory(
                        path,
                        include_patterns,
                        exclude_patterns,
                        follow_symlinks,
                        respect_gitignore,
                        max_file_size,
                    )
                )

        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for file in files:
            if file not in seen:
                seen.add(file)
                unique_files.append(file)

        self.logger.info(f"Scanned {len(paths)} paths, found {len(unique_files)} files")
        return unique_files

    def _scan_directory(
        self,
        directory: Path,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
        follow_symlinks: bool,
        respect_gitignore: bool,
        max_file_size: Optional[int],
    ) -> Generator[Path, None, None]:
        """Scan a directory recursively."""
        # Load .gitignore if needed
        gitignore_patterns = set()
        if respect_gitignore:
            gitignore_path = directory / ".gitignore"
            if gitignore_path.exists():
                gitignore_patterns = self._load_gitignore(gitignore_path)

        # Walk directory
        for root, dirs, files in os.walk(directory, followlinks=follow_symlinks):
            root_path = Path(root)

            # Filter directories to avoid scanning ignored ones
            dirs[:] = [
                d
                for d in dirs
                if not self._should_ignore_directory(root_path / d, directory, gitignore_patterns)
            ]

            # Check files
            for filename in files:
                file_path = root_path / filename

                # Skip if ignored by gitignore
                if respect_gitignore and self._matches_gitignore(
                    file_path, directory, gitignore_patterns
                ):
                    continue

                # Check if file should be included
                if self._should_include_file(
                    file_path, include_patterns, exclude_patterns, max_file_size
                ):
                    yield file_path

    def _should_include_file(
        self,
        file_path: Path,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
        max_file_size: Optional[int],
    ) -> bool:
        """Check if a file should be included."""
        # Skip binary files
        if file_path.suffix.lower() in self.BINARY_EXTENSIONS:
            return False

        # Check file size
        if max_file_size:
            try:
                if file_path.stat().st_size > max_file_size:
                    return False
            except OSError:
                return False

        # Check include patterns (match against both full path and filename)
        if include_patterns:
            full = str(file_path)
            base = file_path.name
            if not any(
                fnmatch.fnmatch(full, pattern) or fnmatch.fnmatch(base, pattern)
                for pattern in include_patterns
            ):
                return False

        # Check exclude patterns (match against both full path and filename)
        if exclude_patterns:
            full = str(file_path)
            base = file_path.name
            if any(
                fnmatch.fnmatch(full, pattern) or fnmatch.fnmatch(base, pattern)
                for pattern in exclude_patterns
            ):
                return False

        # Check default ignore patterns
        filename = file_path.name
        if any(fnmatch.fnmatch(filename, pattern) for pattern in self.ignore_patterns):
            return False

        return True

    def _should_ignore_directory(
        self, dir_path: Path, root: Path, gitignore_patterns: Set[str]
    ) -> bool:
        """Check if a directory should be ignored."""
        dir_name = dir_path.name

        # Check default ignore patterns (exact match and glob patterns)
        if dir_name in self.ignore_patterns:
            return True

        # Check if directory name matches any glob patterns
        if any(fnmatch.fnmatch(dir_name, pattern) for pattern in self.ignore_patterns):
            return True

        # Check gitignore
        if gitignore_patterns:
            rel_path = str(dir_path.relative_to(root))
            for pattern in gitignore_patterns:
                if pattern.endswith("/"):
                    # Directory pattern
                    if fnmatch.fnmatch(rel_path + "/", pattern):
                        return True
                elif fnmatch.fnmatch(rel_path, pattern):
                    return True

        return False

    def _load_gitignore(self, gitignore_path: Path) -> Set[str]:
        """Load patterns from a .gitignore file."""
        patterns = set()

        try:
            with open(gitignore_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith("#"):
                        patterns.add(line)
        except Exception as e:
            self.logger.warning(f"Failed to read .gitignore: {e}")

        return patterns

    def _matches_gitignore(self, file_path: Path, root: Path, patterns: Set[str]) -> bool:
        """Check if a file matches gitignore patterns."""
        try:
            rel_path = str(file_path.relative_to(root))
        except ValueError:
            return False

        for pattern in patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True

        return False

    def find_files_by_name(
        self, root: Path, name_pattern: str, case_sensitive: bool = False
    ) -> List[Path]:
        """Find files matching a name pattern.

        Args:
            root: Root directory to search
            name_pattern: Pattern to match (supports wildcards)
            case_sensitive: Whether to match case-sensitively

        Returns:
            List of matching file paths
        """
        matches = []

        if not case_sensitive:
            name_pattern = name_pattern.lower()

        for file_path in self.scan([root]):
            filename = file_path.name
            if not case_sensitive:
                filename = filename.lower()

            if fnmatch.fnmatch(filename, name_pattern):
                matches.append(file_path)

        return matches

    def find_files_by_content(
        self,
        root: Path,
        content_pattern: str,
        file_patterns: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ) -> List[Path]:
        """Find files containing specific content.

        Args:
            root: Root directory to search
            content_pattern: Text to search for
            file_patterns: File patterns to search in
            case_sensitive: Whether to match case-sensitively

        Returns:
            List of files containing the pattern
        """
        matches = []

        if not case_sensitive:
            content_pattern = content_pattern.lower()

        for file_path in self.scan([root], include_patterns=file_patterns):
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    if not case_sensitive:
                        content = content.lower()

                    if content_pattern in content:
                        matches.append(file_path)
            except Exception:
                # Skip files that can't be read as text
                continue

        return matches
