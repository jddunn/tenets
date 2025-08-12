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

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger


# Register explicit adapters/converters for datetime to avoid Python 3.12+
# deprecation warnings about default adapters/converters.
# This ensures consistent behavior across all connections in this process.
def _register_datetime_adapters_and_converters() -> None:
    # Adapt datetime -> ISO-8601 string with space separator for readability.
    sqlite3.register_adapter(datetime, lambda v: v.isoformat(sep=" ", timespec="microseconds"))

    def _convert_timestamp(val: bytes) -> datetime:
        s = val.decode("utf-8")
        # Try fromisoformat first (supports both 'T' and space separators, with or without TZ)
        try:
            return datetime.fromisoformat(s)
        except Exception:
            # Fallback common formats
            for fmt in (
                "%Y-%m-%d %H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    return datetime.strptime(s, fmt)
                except Exception:
                    continue
            # As a last resort, return naive datetime without parsing microseconds/tz
            try:
                date_part, time_part = s.split(" ", 1)
                return datetime.fromisoformat(f"{date_part}T{time_part}")
            except Exception:
                # Give up and return original string converted via fromtimestamp if possible
                raise

    # Bind to common declared types (case-insensitive)
    sqlite3.register_converter("TIMESTAMP", _convert_timestamp)
    sqlite3.register_converter("DATETIME", _convert_timestamp)


# Ensure registration happens on import so even raw sqlite3.connect in tests
# benefits from the custom adapters/converters.
_register_datetime_adapters_and_converters()


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
        # Enable declared-type and column-name based conversions and allow
        # cross-thread usage for tests that access the same connection across threads.
        conn = sqlite3.connect(
            path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            check_same_thread=False,
        )
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
