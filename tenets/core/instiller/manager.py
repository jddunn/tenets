"""Tenet management system.

This module manages the lifecycle of tenets (guiding principles) and handles
their storage, retrieval, and application to contexts.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from tenets.config import TenetsConfig
from tenets.models.tenet import Priority, Tenet, TenetCategory, TenetCollection, TenetStatus
from tenets.utils.logger import get_logger


class TenetManager:
    """Manages tenets throughout their lifecycle."""

    def __init__(self, config: TenetsConfig):
        """Initialize the tenet manager.

        Args:
            config: Tenets configuration
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize storage
        self.storage_path = Path(config.cache_dir) / "tenets"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.storage_path / "tenets.db"
        self._init_database()

        # Cache for active tenets
        self._tenet_cache: Dict[str, Tenet] = {}
        self._load_active_tenets()

    def _init_database(self) -> None:
        """Initialize the tenet database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenets (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    category TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    instilled_at TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL,
                    author TEXT,
                    data JSON NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenet_sessions (
                    tenet_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    bound_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (tenet_id, session_id),
                    FOREIGN KEY (tenet_id) REFERENCES tenets(id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenet_metrics (
                    tenet_id TEXT PRIMARY KEY,
                    injection_count INTEGER DEFAULT 0,
                    last_injected TIMESTAMP,
                    contexts_appeared_in INTEGER DEFAULT 0,
                    compliance_score REAL DEFAULT 0.0,
                    reinforcement_needed BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (tenet_id) REFERENCES tenets(id)
                )
            """
            )

            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tenets_status ON tenets(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tenets_priority ON tenets(priority)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_session ON tenet_sessions(session_id)"
            )

    def _load_active_tenets(self) -> None:
        """Load active tenets into cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM tenets 
                WHERE status IN ('pending', 'instilled')
                ORDER BY created_at DESC
            """
            )

            for row in cursor:
                tenet_data = json.loads(row["data"])
                tenet = Tenet.from_dict(tenet_data)
                self._tenet_cache[tenet.id] = tenet

    def add_tenet(
        self,
        content: Union[str, Tenet],
        priority: Union[str, Priority] = "medium",
        category: Optional[Union[str, TenetCategory]] = None,
        session: Optional[str] = None,
        author: Optional[str] = None,
    ) -> Tenet:
        """Add a new tenet.

        Args:
            content: The guiding principle text or a Tenet object
            priority: Priority level (low, medium, high, critical)
            category: Category for organization
            session: Bind to specific session
            author: Who created the tenet

        Returns:
            The created Tenet
        """
        # Check if content is already a Tenet object
        if isinstance(content, Tenet):
            tenet = content
            # Update session bindings if a session was specified
            if session and session not in (tenet.session_bindings or []):
                if tenet.session_bindings:
                    tenet.session_bindings.append(session)
                else:
                    tenet.session_bindings = [session]
        else:
            # Create tenet from string content
            # Ensure content is a string before calling strip()
            if not isinstance(content, str):
                raise TypeError(f"Expected string or Tenet, got {type(content).__name__}: {content}")
            tenet = Tenet(
                content=content.strip(),
                priority=priority if isinstance(priority, Priority) else Priority(priority),
                category=(
                    category
                    if isinstance(category, TenetCategory)
                    else (TenetCategory(category) if category else None)
                ),
                author=author,
            )

        # Bind to session if specified
        if session:
            tenet.bind_to_session(session)

        # Save to database
        self._save_tenet(tenet)

        # Add to cache
        self._tenet_cache[tenet.id] = tenet

        self.logger.info(f"Added tenet: {tenet.id} - {tenet.content[:50]}...")

        return tenet

    def _save_tenet(self, tenet: Tenet) -> None:
        """Save tenet to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Save main tenet data
            conn.execute(
                """
                INSERT OR REPLACE INTO tenets 
                (id, content, priority, category, status, created_at, instilled_at, updated_at, author, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    tenet.id,
                    tenet.content,
                    tenet.priority.value,
                    tenet.category.value if tenet.category else None,
                    tenet.status.value,
                    tenet.created_at,
                    tenet.instilled_at,
                    tenet.updated_at,
                    tenet.author,
                    json.dumps(tenet.to_dict()),
                ),
            )

            # Save session bindings
            conn.execute("DELETE FROM tenet_sessions WHERE tenet_id = ?", (tenet.id,))
            for session_id in tenet.session_bindings:
                conn.execute(
                    """
                    INSERT INTO tenet_sessions (tenet_id, session_id, bound_at)
                    VALUES (?, ?, ?)
                """,
                    (tenet.id, session_id, datetime.now()),
                )

            # Save metrics
            conn.execute(
                """
                INSERT OR REPLACE INTO tenet_metrics
                (tenet_id, injection_count, last_injected, contexts_appeared_in, 
                 compliance_score, reinforcement_needed)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    tenet.id,
                    tenet.metrics.injection_count,
                    tenet.metrics.last_injected,
                    tenet.metrics.contexts_appeared_in,
                    tenet.metrics.compliance_score,
                    tenet.metrics.reinforcement_needed,
                ),
            )

    def get_tenet(self, tenet_id: str) -> Optional[Tenet]:
        """Get a specific tenet by ID.

        Args:
            tenet_id: Tenet ID (can be partial)

        Returns:
            The Tenet or None if not found
        """
        # Try cache first
        if tenet_id in self._tenet_cache:
            return self._tenet_cache[tenet_id]

        # Try partial match
        for tid, tenet in self._tenet_cache.items():
            if tid.startswith(tenet_id):
                return tenet

        # Try database
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT data FROM tenets WHERE id LIKE ?", (f"{tenet_id}%",))
            row = cursor.fetchone()

            if row:
                tenet = Tenet.from_dict(json.loads(row["data"]))
                self._tenet_cache[tenet.id] = tenet
                return tenet

        return None

    def list_tenets(
        self,
        pending_only: bool = False,
        instilled_only: bool = False,
        session: Optional[str] = None,
        category: Optional[Union[str, TenetCategory]] = None,
    ) -> List[Dict[str, Any]]:
        """List tenets with filtering.

        Args:
            pending_only: Only show pending tenets
            instilled_only: Only show instilled tenets
            session: Filter by session binding
            category: Filter by category

        Returns:
            List of tenet dictionaries
        """
        tenets = []

        # Build query
        query = "SELECT data FROM tenets WHERE 1=1"
        params = []

        if pending_only:
            query += " AND status = ?"
            params.append(TenetStatus.PENDING.value)
        elif instilled_only:
            query += " AND status = ?"
            params.append(TenetStatus.INSTILLED.value)
        else:
            query += " AND status != ?"
            params.append(TenetStatus.ARCHIVED.value)

        if category:
            cat_value = category if isinstance(category, str) else category.value
            query += " AND category = ?"
            params.append(cat_value)

        query += " ORDER BY created_at DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            for row in cursor:
                tenet = Tenet.from_dict(json.loads(row["data"]))

                # Filter by session if specified
                if session and not tenet.applies_to_session(session):
                    continue

                tenet_dict = tenet.to_dict()
                tenet_dict["instilled"] = tenet.status == TenetStatus.INSTILLED
                tenets.append(tenet_dict)

        return tenets

    def get_pending_tenets(self, session: Optional[str] = None) -> List[Tenet]:
        """Get all pending tenets.

        Args:
            session: Filter by session

        Returns:
            List of pending Tenet objects
        """
        pending = []

        for tenet in self._tenet_cache.values():
            if tenet.status == TenetStatus.PENDING:
                if not session or tenet.applies_to_session(session):
                    pending.append(tenet)

        return sorted(pending, key=lambda t: (t.priority.weight, t.created_at), reverse=True)

    def remove_tenet(self, tenet_id: str) -> bool:
        """Remove a tenet.

        Args:
            tenet_id: Tenet ID (can be partial)

        Returns:
            True if removed, False if not found
        """
        tenet = self.get_tenet(tenet_id)
        if not tenet:
            return False

        # Archive instead of delete
        tenet.archive()
        self._save_tenet(tenet)

        # Remove from cache
        if tenet.id in self._tenet_cache:
            del self._tenet_cache[tenet.id]

        self.logger.info(f"Archived tenet: {tenet.id}")
        return True

    def instill_tenets(self, session: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
        """Instill pending tenets.

        Args:
            session: Target session
            force: Re-instill even if already instilled

        Returns:
            Dictionary with results
        """
        tenets_to_instill = []

        if force:
            # Get all non-archived tenets
            for tenet in self._tenet_cache.values():
                if tenet.status != TenetStatus.ARCHIVED:
                    if not session or tenet.applies_to_session(session):
                        tenets_to_instill.append(tenet)
        else:
            # Get only pending tenets
            tenets_to_instill = self.get_pending_tenets(session)

        # Sort by priority and creation date
        tenets_to_instill.sort(key=lambda t: (t.priority.weight, t.created_at), reverse=True)

        # Mark as instilled
        instilled = []
        for tenet in tenets_to_instill:
            tenet.instill()
            self._save_tenet(tenet)
            instilled.append(tenet.content)

        self.logger.info(f"Instilled {len(instilled)} tenets")

        return {
            "count": len(instilled),
            "tenets": instilled,
            "session": session,
            "strategy": "priority-based",
        }

    def get_tenets_for_injection(
        self, context_length: int, session: Optional[str] = None, max_tenets: int = 5
    ) -> List[Tenet]:
        """Get tenets ready for injection into context.

        Args:
            context_length: Current context length in tokens
            session: Current session
            max_tenets: Maximum number of tenets to return

        Returns:
            List of tenets to inject
        """
        candidates = []

        # Get applicable tenets
        for tenet in self._tenet_cache.values():
            if tenet.status == TenetStatus.INSTILLED:
                if not session or tenet.applies_to_session(session):
                    candidates.append(tenet)

        # Sort by priority and need for reinforcement
        candidates.sort(
            key=lambda t: (
                t.priority.weight,
                t.metrics.reinforcement_needed,
                -t.metrics.injection_count,  # Prefer less frequently injected
            ),
            reverse=True,
        )

        # Select tenets based on injection strategy
        selected = []
        for tenet in candidates:
            if len(selected) >= max_tenets:
                break

            if tenet.should_inject(context_length, len(selected)):
                selected.append(tenet)

                # Update metrics
                tenet.metrics.update_injection()
                self._save_tenet(tenet)

        return selected

    def export_tenets(
        self, format: str = "yaml", session: Optional[str] = None, include_archived: bool = False
    ) -> str:
        """Export tenets to YAML or JSON.

        Args:
            format: Export format (yaml or json)
            session: Filter by session
            include_archived: Include archived tenets

        Returns:
            Serialized tenets
        """
        tenets_data = []

        for tenet in self._tenet_cache.values():
            if not include_archived and tenet.status == TenetStatus.ARCHIVED:
                continue

            if session and not tenet.applies_to_session(session):
                continue

            tenets_data.append(tenet.to_dict())

        # Sort by creation date
        tenets_data.sort(key=lambda t: t["created_at"])

        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "tenets": tenets_data,
        }

        if format == "yaml":
            return yaml.dump(export_data, default_flow_style=False, sort_keys=False)
        else:
            return json.dumps(export_data, indent=2)

    def import_tenets(
        self,
        file_path: Union[str, Path],
        session: Optional[str] = None,
        override_priority: Optional[Priority] = None,
    ) -> int:
        """Import tenets from file.

        Args:
            file_path: Path to import file
            session: Bind imported tenets to session
            override_priority: Override priority for all imported tenets

        Returns:
            Number of tenets imported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Import file not found: {file_path}")

        # Load data
        with open(file_path, "r") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Import tenets
        imported = 0
        tenets = data.get("tenets", [])

        for tenet_data in tenets:
            # Skip if already exists
            if self.get_tenet(tenet_data.get("id", "")):
                continue

            # Create new tenet
            tenet = Tenet.from_dict(tenet_data)

            # Override priority if requested
            if override_priority:
                tenet.priority = override_priority

            # Bind to session if specified
            if session:
                tenet.bind_to_session(session)

            # Reset status to pending
            tenet.status = TenetStatus.PENDING
            tenet.instilled_at = None

            # Save
            self._save_tenet(tenet)
            self._tenet_cache[tenet.id] = tenet

            imported += 1

        self.logger.info(f"Imported {imported} tenets from {file_path}")
        return imported

    def create_collection(
        self, name: str, description: str = "", tenet_ids: Optional[List[str]] = None
    ) -> TenetCollection:
        """Create a collection of related tenets.

        Args:
            name: Collection name
            description: Collection description
            tenet_ids: IDs of tenets to include

        Returns:
            The created TenetCollection
        """
        collection = TenetCollection(name=name, description=description)

        if tenet_ids:
            for tenet_id in tenet_ids:
                if tenet := self.get_tenet(tenet_id):
                    collection.add_tenet(tenet)

        # Save collection
        collection_path = self.storage_path / f"collection_{name.lower().replace(' ', '_')}.json"
        with open(collection_path, "w") as f:
            json.dump(collection.to_dict(), f, indent=2)

        return collection

    def analyze_tenet_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of tenets.

        Returns:
            Analysis of tenet usage and effectiveness
        """
        total_tenets = len(self._tenet_cache)

        if total_tenets == 0:
            return {"total_tenets": 0, "status": "No tenets configured"}

        # Gather statistics
        stats = {
            "total_tenets": total_tenets,
            "by_status": {},
            "by_priority": {},
            "by_category": {},
            "most_injected": [],
            "least_effective": [],
            "need_reinforcement": [],
        }

        # Count by status
        for status in TenetStatus:
            count = sum(1 for t in self._tenet_cache.values() if t.status == status)
            stats["by_status"][status.value] = count

        # Count by priority
        for priority in Priority:
            count = sum(1 for t in self._tenet_cache.values() if t.priority == priority)
            stats["by_priority"][priority.value] = count

        # Count by category
        category_counts = {}
        for tenet in self._tenet_cache.values():
            if tenet.category:
                cat = tenet.category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1
        stats["by_category"] = category_counts

        # Find most injected
        sorted_by_injection = sorted(
            self._tenet_cache.values(), key=lambda t: t.metrics.injection_count, reverse=True
        )
        stats["most_injected"] = [
            {
                "id": t.id[:8],
                "content": t.content[:50] + "..." if len(t.content) > 50 else t.content,
                "count": t.metrics.injection_count,
            }
            for t in sorted_by_injection[:5]
        ]

        # Find least effective
        sorted_by_compliance = sorted(
            [t for t in self._tenet_cache.values() if t.metrics.injection_count > 0],
            key=lambda t: t.metrics.compliance_score,
        )
        stats["least_effective"] = [
            {
                "id": t.id[:8],
                "content": t.content[:50] + "..." if len(t.content) > 50 else t.content,
                "score": t.metrics.compliance_score,
            }
            for t in sorted_by_compliance[:5]
        ]

        # Find those needing reinforcement
        stats["need_reinforcement"] = [
            {
                "id": t.id[:8],
                "content": t.content[:50] + "..." if len(t.content) > 50 else t.content,
                "priority": t.priority.value,
            }
            for t in self._tenet_cache.values()
            if t.metrics.reinforcement_needed
        ]

        return stats
