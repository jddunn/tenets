"""SQLite storage utilities for Tenets.

This module centralizes SQLite database path resolution, connection
management, and pragmas. All persistent storage (sessions, tenets,
config state) should use this utility to open connections inside the
configured cache directory.

By default, the cache directory is resolved by TenetsConfig. Do not
write inside the installed package directory. When Tenets is installed
via pip, the package location may be read-only; the cache directory will
be user- or project-local and writable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import sqlite3

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger


@dataclass
class SQLitePaths:
    """Resolved paths for SQLite databases.

    Attributes:
        root: The cache directory root where DB files live.
        main_db: Path to the main Tenets database file.
    """
    root: Path
    main_db: Path


class Database:
    """SQLite database manager applying Tenets pragmas.

    Use this to obtain connections to the main Tenets DB file located in
    the configured cache directory.
    """

    def __init__(self, config: TenetsConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.paths = self._resolve_paths(config)
        self._ensure_dirs()

    @staticmethod
    def _resolve_paths(config: TenetsConfig) -> SQLitePaths:
        root = Path(config.cache.directory)
        main_db = root / "tenets.db"
        return SQLitePaths(root=root, main_db=main_db)

    def _ensure_dirs(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)

    def connect(self, db_path: Optional[Path] = None) -> sqlite3.Connection:
        """Open a SQLite connection with configured PRAGMAs applied.

        Args:
            db_path: Optional custom DB path; defaults to main DB path.
        Returns:
            sqlite3.Connection ready for use.
        """
        path = Path(db_path) if db_path else self.paths.main_db
        conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        self._apply_pragmas(conn, self.config.cache.sqlite_pragmas)
        return conn

    def _apply_pragmas(self, conn: sqlite3.Connection, pragmas: Dict[str, str]) -> None:
        cur = conn.cursor()
        for key, value in pragmas.items():
            try:
                cur.execute(f"PRAGMA {key}={value}")
            except Exception as exc:
                self.logger.debug(f"Failed to apply PRAGMA {key}={value}: {exc}")
        cur.close()
