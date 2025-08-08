"""Session manager with optional SQLite persistence.

Uses an in-memory dict by default. When provided a TenetsConfig, it will
persist sessions and context entries via storage.SessionDB while keeping
an in-memory mirror for fast access.

This layer is intentionally thin: persistent semantics live in
``tenets.storage.session_db.SessionDB``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tenets.models.context import SessionContext, ContextResult
from tenets.config import TenetsConfig
from tenets.storage.session_db import SessionDB
from tenets.utils.logger import get_logger


@dataclass
class SessionManager:
    """High-level session manager used by the CLI and core flows."""
    sessions: Dict[str, SessionContext] = field(default_factory=dict)
    _db: Optional[SessionDB] = field(default=None, init=False, repr=False)
    _logger: any = field(default=None, init=False, repr=False)

    def __init__(self, config: Optional[TenetsConfig] = None):
        self.sessions = {}
        self._logger = get_logger(__name__)
        self._db = SessionDB(config) if config else None

    def create(self, name: str) -> SessionContext:
        if name in self.sessions:
            return self.sessions[name]
        if self._db:
            try:
                # Ensure exists in DB
                if not self._db.get_session(name):
                    self._db.create_session(name)
            except Exception as e:
                self._logger.debug(f"SessionDB create failed for {name}: {e}")
        sc = SessionContext(session_id=name, name=name)
        self.sessions[name] = sc
        return sc

    def list(self) -> List[SessionContext]:  # noqa: A003 - shadow builtin
        if self._db:
            try:
                records = self._db.list_sessions()
                for r in records:
                    if r.name not in self.sessions:
                        self.sessions[r.name] = SessionContext(session_id=r.name, name=r.name)
            except Exception as e:
                self._logger.debug(f"SessionDB list failed: {e}")
        return list(self.sessions.values())

    def get(self, name: str) -> Optional[SessionContext]:
        sc = self.sessions.get(name)
        if sc:
            return sc
        if self._db:
            try:
                rec = self._db.get_session(name)
                if rec:
                    sc = SessionContext(session_id=rec.name, name=rec.name)
                    self.sessions[name] = sc
                    return sc
            except Exception as e:
                self._logger.debug(f"SessionDB get failed for {name}: {e}")
        return None

    def delete(self, name: str) -> bool:
        """Delete a session by name from persistence (if configured) and memory."""
        db_deleted = False
        if self._db:
            try:
                db_deleted = self._db.delete_session(name, purge_context=True)
            except Exception as e:
                self._logger.debug(f"SessionDB delete failed for {name}: {e}")
        mem_deleted = self.sessions.pop(name, None) is not None
        return db_deleted or mem_deleted

    def add_context(self, name: str, context: ContextResult) -> None:
        sc = self.create(name)
        sc.add_context(context)
        if self._db:
            try:
                # Persist a JSON snapshot of the ContextResult
                import json
                self._db.add_context(name, kind="context_result", content=json.dumps(context.to_dict()))
            except Exception as e:
                self._logger.debug(f"SessionDB add_context failed for {name}: {e}")
