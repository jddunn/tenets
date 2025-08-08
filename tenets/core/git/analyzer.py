"""Git analyzer using GitPython.

Provides helpers to extract recent context, changed files, and authorship.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from git import Repo, InvalidGitRepositoryError, NoSuchPathError  # type: ignore

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
        self._ensure_repo()

    def _ensure_repo(self) -> None:
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

    def is_repo(self) -> bool:
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
            iter_commits = self.repo.iter_commits(
                paths=files, max_count=limit
            ) if files else self.repo.iter_commits(max_count=limit)
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
            iter_commits = self.repo.iter_commits(paths=files) if files else self.repo.iter_commits()
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

    def get_changes_since(
        self, path: Optional[Path] = None, since: str = "1 week ago", files: Optional[List[str]] = None
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

    # Existing APIs retained
    def recent_commits(
        self, limit: int = 50, paths: Optional[List[Path]] = None
    ) -> List[CommitInfo]:
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
