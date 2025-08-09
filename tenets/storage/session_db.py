"""Session storage using SQLite.

Persists session metadata and context chunks into the main Tenets DB
located in the cache directory resolved by TenetsConfig.

This module centralizes all persistence for interactive sessions. It is
safe to use in environments where the installed package directory may be
read-only (e.g., pip installs) because the SQLite database lives under
Tenets' cache directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import sqlite3

from tenets.config import TenetsConfig
from tenets.storage.sqlite import Database
from tenets.utils.logger import get_logger


@dataclass
class SessionRecord:
    id: int
    name: str
    created_at: datetime
    metadata: Dict[str, Any]


class SessionDB:
    """SQLite-backed session storage.

    Manages two tables:
      - sessions(id, name, created_at, metadata)
      - session_context(id, session_id, kind, content, created_at)

    Note: Foreign keys are not declared with ON DELETE CASCADE, so this
    class explicitly removes child rows where appropriate.
    """

    def __init__(self, config: TenetsConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.db = Database(config)
        self._init_schema()

    def _init_schema(self) -> None:
        conn = self.db.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    metadata TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_context (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def create_session(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> SessionRecord:
        conn = self.db.connect()
        try:
            cur = conn.cursor()
            now = datetime.utcnow()
            cur.execute(
                "INSERT INTO sessions (name, created_at, metadata) VALUES (?, ?, ?)",
                (name, now, json.dumps(metadata or {})),
            )
            conn.commit()
            session_id = cur.lastrowid
            return SessionRecord(id=session_id, name=name, created_at=now, metadata=metadata or {})
        finally:
            conn.close()

    def get_session(self, name: str) -> Optional[SessionRecord]:
        conn = self.db.connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name, created_at, metadata FROM sessions WHERE name=?", (name,))
            row = cur.fetchone()
            if not row:
                return None
            meta = json.loads(row[3]) if row[3] else {}
            return SessionRecord(id=row[0], name=row[1], created_at=row[2], metadata=meta)
        finally:
            conn.close()

    def list_sessions(self) -> List[SessionRecord]:
        conn = self.db.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, name, created_at, metadata FROM sessions ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
            records: List[SessionRecord] = []
            for row in rows:
                meta = json.loads(row[3]) if row[3] else {}
                records.append(
                    SessionRecord(id=row[0], name=row[1], created_at=row[2], metadata=meta)
                )
            return records
        finally:
            conn.close()

    def get_active_session(self) -> Optional[SessionRecord]:
        """Return the currently active session, if any.

        Chooses the most recently created active session if multiple are marked active.
        """
        for s in self.list_sessions():  # list_sessions is newest-first
            if s.metadata.get("active"):
                return s
        return None

    def add_context(self, session_name: str, kind: str, content: str) -> None:
        """Append a context artifact to a session.

        Args:
            session_name: Friendly name of the session.
            kind: Type tag for the content (e.g., "context_result").
            content: Serialized content (JSON string or text).
        """
        sess = self.get_session(session_name)
        if not sess:
            sess = self.create_session(session_name)
        conn = self.db.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO session_context (session_id, kind, content, created_at) VALUES (?, ?, ?, ?)",
                (sess.id, kind, content, datetime.utcnow()),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_session(self, name: str, purge_context: bool = True) -> bool:
        """Delete a session record by name.

        This removes the session row and, by default, all related entries
        from ``session_context``.

        Args:
            name: Session name to delete.
            purge_context: When True (default), also remove all associated
                rows from ``session_context``.

        Returns:
            True if a session row was deleted; False if no session matched.
        """
        conn = self.db.connect()
        try:
            cur = conn.cursor()
            # Lookup session id
            cur.execute("SELECT id FROM sessions WHERE name=?", (name,))
            row = cur.fetchone()
            if not row:
                return False
            session_id = row[0]
            # Optionally delete related context first (no ON DELETE CASCADE in schema)
            if purge_context:
                cur.execute("DELETE FROM session_context WHERE session_id=?", (session_id,))
            # Delete the session
            cur.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def delete_all_sessions(self, purge_context: bool = True) -> int:
        """Delete all sessions. Returns the number of sessions removed.

        If purge_context is True, also clears all session_context rows.
        """
        conn = self.db.connect()
        try:
            cur = conn.cursor()
            if purge_context:
                cur.execute("DELETE FROM session_context")
            cur.execute("SELECT COUNT(*) FROM sessions")
            (count_before,) = cur.fetchone() or (0,)
            cur.execute("DELETE FROM sessions")
            conn.commit()
            return count_before
        finally:
            conn.close()

    def update_session_metadata(self, name: str, updates: Dict[str, Any]) -> bool:
        """Merge ``updates`` into the session's metadata JSON.

        Returns True if the session exists and was updated.
        """
        conn = self.db.connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, metadata FROM sessions WHERE name=?", (name,))
            row = cur.fetchone()
            if not row:
                return False
            session_id, metadata_text = row
            meta = json.loads(metadata_text) if metadata_text else {}
            meta.update(updates or {})
            cur.execute(
                "UPDATE sessions SET metadata=? WHERE id=?",
                (json.dumps(meta), session_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def set_active(self, name: str, active: bool) -> bool:
        """Mark a session as active/inactive via metadata.

        When activating a session, all other sessions are marked inactive to
        guarantee there is at most one active session at a time.
        """
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        updates: Dict[str, Any] = {"active": active, "updated_at": timestamp}
        if active:
            updates["resumed_at"] = timestamp
        else:
            updates["ended_at"] = timestamp
        ok = self.update_session_metadata(name, updates)
        if active and ok:
            # Deactivate any other active sessions
            for other in self.list_sessions():
                if other.name != name and other.metadata.get("active"):
                    self.update_session_metadata(
                        other.name,
                        {"active": False, "updated_at": timestamp, "ended_at": timestamp},
                    )
        return ok
