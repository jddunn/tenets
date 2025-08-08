"""Git analyzer using GitPython.

Provides helpers to extract recent context, changed files, and authorship.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
        except (InvalidGitRepositoryError, NoSuchPathError):
            self.repo = None
            logger.debug("No git repository detected at %s", self.root)

    def is_repo(self) -> bool:
        return self.repo is not None

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
