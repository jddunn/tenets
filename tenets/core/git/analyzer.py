"""Git analyzer using GitPython.

Provides helpers to extract recent context, changed files, and authorship.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Lazy import check for GitPython
def _check_git_available():
    """Check if GitPython is available without importing."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("git")
        return spec is not None
    except (ImportError, AttributeError):
        return False


GIT_AVAILABLE = _check_git_available()

# These will be imported lazily when needed
Repo = None
InvalidGitRepositoryError = None
NoSuchPathError = None


def _ensure_git_imported():
    """Import git modules when actually needed."""
    global Repo, InvalidGitRepositoryError, NoSuchPathError
    if Repo is None:
        try:
            from git import InvalidGitRepositoryError as _InvalidGit
            from git import NoSuchPathError as _NoPath
            from git import Repo as _Repo

            Repo = _Repo
            InvalidGitRepositoryError = _InvalidGit
            NoSuchPathError = _NoPath
        except ImportError as e:
            raise ImportError(f"GitPython is required but not installed: {e}")


from tenets.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CommitInfo:
    hexsha: str
    author: str
    email: str
    message: str
    committed_date: int


class GitAnalyzer:
    def __init__(self, root: Any) -> None:
        # Allow passing a TenetsConfig or a Path
        try:
            from tenets.config import TenetsConfig  # local import to avoid cycles
        except Exception:
            TenetsConfig = None  # type: ignore
        if TenetsConfig is not None and isinstance(root, TenetsConfig):
            base = root.project_root or Path.cwd()
        else:
            base = Path(root) if root is not None else Path.cwd()
        self.root = Path(base)
        self.repo: Optional[Repo] = None
        self._repo_initialized = False
        # Don't call _ensure_repo() here - lazy load on first use

    def _ensure_repo(self) -> None:
        if self._repo_initialized:
            return
        self._repo_initialized = True

        if not GIT_AVAILABLE:
            self.repo = None
            logger.warning(
                "Git is not available. Install git and ensure it's in your PATH. "
                "Git-related features will be disabled. "
                "Non-git features will continue to work normally."
            )
            return

        _ensure_git_imported()  # Import GitPython modules when needed
        try:
            self.repo = Repo(self.root, search_parent_directories=True)
            # Normalize root to the repo working tree directory
            try:
                self.root = Path(self.repo.working_tree_dir or self.root)
            except Exception:
                pass
        except (InvalidGitRepositoryError, NoSuchPathError):
            self.repo = None
            logger.debug("No git repository detected at %s", self.root)
        except Exception as e:
            self.repo = None
            logger.debug("Failed to initialize git repository: %s", str(e))

    def is_repo(self) -> bool:
        self._ensure_repo()  # Lazy load git repo
        return self.repo is not None

    # New: method expected by Distiller
    def is_git_repo(self, path: Optional[Path] = None) -> bool:
        """Return True if the given path (or current root) is inside a git repo.

        If a path is provided, update internal root and repo accordingly.
        """
        if path is not None:
            self.root = Path(path)
        self._ensure_repo()
        return self.repo is not None

    # Compatibility helper used by CLI chronicle already
    def changed_files(self, ref: str = "HEAD", diff_with: Optional[str] = None) -> List[Path]:
        self._ensure_repo()  # Ensure repo is initialized
        if not self.repo:
            return []
        repo = self.repo
        try:
            if diff_with:
                diff = repo.git.diff("--name-only", f"{diff_with}..{ref}")
            else:
                diff = repo.git.diff("--name-only", ref)
            files = [self.root / Path(p.strip()) for p in diff.splitlines() if p.strip()]
            return files
        except Exception:
            return []

    # New: methods used by Distiller
    def get_recent_commits(
        self, path: Optional[Path] = None, limit: int = 10, files: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Return recent commits as dictionaries suitable for formatting.

        Each item contains: sha, author, email, message, date (ISO date string).
        """
        if path is not None:
            self.root = Path(path)
            self._ensure_repo()
        if not self.repo:
            return []
        results: List[Dict[str, Any]] = []
        try:
            iter_commits = (
                self.repo.iter_commits(paths=files, max_count=limit)
                if files
                else self.repo.iter_commits(max_count=limit)
            )
            for c in iter_commits:
                dt = datetime.fromtimestamp(getattr(c, "committed_date", 0))
                results.append(
                    {
                        "sha": c.hexsha,
                        "author": getattr(c.author, "name", ""),
                        "email": getattr(c.author, "email", ""),
                        "message": (c.message or "").strip().splitlines()[0],
                        "date": dt.strftime("%Y-%m-%d"),
                    }
                )
        except Exception:
            return []
        return results

    def get_contributors(
        self, path: Optional[Path] = None, files: Optional[List[str]] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return contributors with commit counts.

        Returns a list of dicts: { name, email, commits } sorted by commits desc.
        """
        if path is not None:
            self.root = Path(path)
            self._ensure_repo()
        if not self.repo:
            return []
        counts: Dict[str, Dict[str, Any]] = {}
        try:
            iter_commits = (
                self.repo.iter_commits(paths=files) if files else self.repo.iter_commits()
            )
            for c in iter_commits:
                name = getattr(c.author, "name", "") or "Unknown"
                email = getattr(c.author, "email", "") or ""
                key = f"{name}<{email}>"
                if key not in counts:
                    counts[key] = {"name": name, "email": email, "commits": 0}
                counts[key]["commits"] += 1
        except Exception:
            return []
        # Sort and limit
        contributors = sorted(counts.values(), key=lambda x: x["commits"], reverse=True)
        return contributors[:limit]

    def get_current_branch(self, path: Optional[Path] = None) -> str:
        """Return current branch name, or 'HEAD' when detached/unknown."""
        if path is not None:
            self.root = Path(path)
            self._ensure_repo()
        if not self.repo:
            return ""
        try:
            return getattr(self.repo.active_branch, "name", "HEAD")
        except Exception:
            # Detached HEAD or other issue
            return "HEAD"

    def current_branch(self) -> str:
        """Alias for get_current_branch() for backward compatibility."""
        return self.get_current_branch()

    def get_tracked_files(self) -> List[str]:
        """Return list of tracked files in the repository."""
        if not self.repo:
            return []
        try:
            # Get all tracked files from git ls-files
            tracked = self.repo.git.ls_files().splitlines()
            return [f for f in tracked if f.strip()]
        except Exception:
            return []

    def get_file_history(self, file_path: str) -> List[Any]:
        """Return commit history for a specific file."""
        if not self.repo:
            return []
        try:
            # Get commits that touched this file
            commits = list(self.repo.iter_commits(paths=file_path))
            return commits
        except Exception:
            return []

    def commit_count(self) -> int:
        """Return total number of commits in the repository."""
        if not self.repo:
            return 0
        try:
            return len(list(self.repo.iter_commits()))
        except Exception:
            return 0

    def list_authors(self) -> List[str]:
        """Return list of unique authors in the repository."""
        if not self.repo:
            return []
        try:
            authors = set()
            for commit in self.repo.iter_commits():
                author = getattr(commit.author, "name", "")
                if author:
                    authors.add(author)
            return list(authors)
        except Exception:
            return []

    def author_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return statistics by author."""
        if not self.repo:
            return {}
        try:
            stats: Dict[str, Dict[str, Any]] = {}
            for commit in self.repo.iter_commits():
                author = getattr(commit.author, "name", "")
                if not author:
                    continue

                if author not in stats:
                    stats[author] = {
                        "commits": 0,
                        "lines_added": 0,
                        "lines_removed": 0,
                        "files_touched": set(),
                    }

                stats[author]["commits"] += 1

                # Try to get stats if available
                if hasattr(commit, "stats") and commit.stats:
                    try:
                        stats[author]["lines_added"] += commit.stats.total.get("insertions", 0)
                        stats[author]["lines_removed"] += commit.stats.total.get("deletions", 0)
                        if hasattr(commit.stats, "files"):
                            stats[author]["files_touched"].update(commit.stats.files.keys())
                    except Exception:
                        pass

            # Convert sets to counts for serialization
            for author_stats in stats.values():
                if "files_touched" in author_stats:
                    author_stats["files_count"] = len(author_stats["files_touched"])
                    del author_stats["files_touched"]

            return stats
        except Exception:
            return {}

    def get_changes_since(
        self,
        path: Optional[Path] = None,
        since: str = "1 week ago",
        files: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return a lightweight list of changes since a given time.

        Each item contains: sha, message, date.
        """
        if path is not None:
            self.root = Path(path)
            self._ensure_repo()
        if not self.repo:
            return []
        results: List[Dict[str, Any]] = []
        try:
            kwargs: Dict[str, Any] = {"since": since}
            if files:
                kwargs["paths"] = files
            for c in self.repo.iter_commits(**kwargs):
                dt = datetime.fromtimestamp(getattr(c, "committed_date", 0))
                results.append(
                    {
                        "sha": c.hexsha,
                        "message": (c.message or "").strip().splitlines()[0],
                        "date": dt.strftime("%Y-%m-%d"),
                    }
                )
        except Exception:
            return []
        return results

    # Added to support Chronicle expectations
    def get_commits_since(
        self,
        since: datetime,
        max_count: int = 1000,
        author: Optional[str] = None,
        branch: Optional[str] = None,
        include_merges: bool = True,
    ) -> List[Any]:
        """Return raw commit objects since a given datetime.

        Args:
            since: Start datetime (inclusive)
            max_count: Maximum number of commits
            author: Optional author filter
            branch: Optional branch name
            include_merges: Whether to include merge commits

        Returns:
            List of GitPython commit objects
        """
        self._ensure_repo()  # Lazy load git repo
        if not self.repo:
            return []

        # Try using subprocess as a fallback for Windows performance issues
        try:
            import subprocess
            import time

            from tenets.utils.logger import get_logger

            logger = get_logger(__name__)

            start = time.time()

            # Limit max_count for performance
            max_count = min(max_count, 200)  # Hard limit for performance

            # Build git log command
            cmd = [
                "git",
                "log",
                f"--since={since.isoformat()}",
                f"--max-count={max_count}",
                "--format=%H",
            ]
            if author:
                cmd.append(f"--author={author}")
            if not include_merges:
                cmd.append("--no-merges")
            if branch:
                cmd.append(branch)

            logger.debug(f"Running git command: {' '.join(cmd)}")

            # Run git command with timeout
            try:
                result = subprocess.run(
                    cmd, cwd=str(self.root), capture_output=True, text=True, timeout=5, check=False
                )

                if result.returncode != 0:
                    logger.debug(f"Git command failed: {result.stderr}")
                    return []

                # Parse commit hashes
                commit_hashes = [h.strip() for h in result.stdout.strip().split("\n") if h.strip()]

                # Convert to GitPython commit objects if possible
                commits = []
                for hash in commit_hashes[:max_count]:  # Extra safety limit
                    try:
                        commit = self.repo.commit(hash)
                        commits.append(commit)
                    except:
                        pass  # Skip commits that can't be loaded

                elapsed = time.time() - start
                logger.debug(f"Fetched {len(commits)} commits in {elapsed:.2f}s")

                return commits

            except subprocess.TimeoutExpired:
                logger.warning("Git command timed out after 5 seconds")
                return []

        except Exception as e:
            from tenets.utils.logger import get_logger

            logger = get_logger(__name__)
            logger.debug(f"Error fetching commits: {e}")

            # Fall back to empty list if subprocess fails
            return []

    def get_commits(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        max_count: int = 1000,
        author: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return commits between two dates.

        This method was missing and called by momentum.py.

        Args:
            since: Start datetime (inclusive)
            until: End datetime (exclusive)
            max_count: Maximum number of commits
            author: Optional author filter
            branch: Optional branch name

        Returns:
            List of commit dictionaries with standard fields
        """
        if not since:
            since = datetime(1970, 1, 1)

        # Use existing get_commits_since
        commits = self.get_commits_since(since, max_count, author, branch)

        # Filter by until date if provided
        if until:
            filtered = []
            for commit in commits:
                commit_date = datetime.fromtimestamp(commit.committed_date)
                if commit_date <= until:
                    filtered.append(commit)
            return filtered

        return commits

    # Existing APIs retained
    def recent_commits(
        self, limit: int = 50, paths: Optional[List[Path]] = None
    ) -> List[CommitInfo]:
        self._ensure_repo()  # Ensure repo is initialized
        if not self.repo:
            return []
        commits = []
        try:
            iter_commits = self.repo.iter_commits(
                paths=[str(p) for p in paths] if paths else None, max_count=limit
            )
            for c in iter_commits:
                commits.append(
                    CommitInfo(
                        hexsha=c.hexsha,
                        author=getattr(c.author, "name", ""),
                        email=getattr(c.author, "email", ""),
                        message=c.message.strip(),
                        committed_date=c.committed_date,
                    )
                )
        except Exception:
            return []
        return commits

    def blame(self, file_path: Path) -> List[Tuple[str, str]]:
        """Return list of (author, line) for a file using git blame."""
        self._ensure_repo()  # Ensure repo is initialized
        if not self.repo:
            return []
        try:
            rel = str(Path(file_path))
            blame = self.repo.blame("HEAD", rel)
            result: List[Tuple[str, str]] = []
            for commit, lines in blame:
                author = getattr(commit.author, "name", "")
                for line in lines:
                    result.append((author, line))
            return result
        except Exception:
            return []
