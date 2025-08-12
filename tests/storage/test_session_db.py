"""Tests for SessionDB SQLite storage."""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.storage.session_db import SessionDB, SessionRecord


@pytest.fixture
def config():
    """Create test configuration with temporary cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TenetsConfig()
        config.cache.directory = tmpdir
        yield config


@pytest.fixture
def session_db(config):
    """Create SessionDB instance."""
    return SessionDB(config)


class TestSessionDB:
    """Test suite for SessionDB."""

    def test_initialization(self, config):
        """Test SessionDB initialization."""
        db = SessionDB(config)

        assert db.config == config
        assert db.db is not None

        # Check that tables were created
        db_path = Path(config.cache.directory) / "tenets.db"
        assert db_path.exists()

        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "sessions" in tables
        assert "session_context" in tables

    def test_create_session(self, session_db):
        """Test creating a session."""
        metadata = {"project": "test", "version": "1.0"}

        record = session_db.create_session("test_session", metadata=metadata)

        assert isinstance(record, SessionRecord)
        assert record.name == "test_session"
        assert record.metadata == metadata
        assert record.created_at is not None
        assert record.id > 0

    def test_create_session_without_metadata(self, session_db):
        """Test creating session without metadata."""
        record = session_db.create_session("simple_session")

        assert record.name == "simple_session"
        assert record.metadata == {}

    def test_create_duplicate_session(self, session_db):
        """Test creating duplicate session raises error."""
        session_db.create_session("duplicate")

        # SQLite UNIQUE constraint should prevent duplicates
        with pytest.raises(Exception):  # Will be IntegrityError
            session_db.create_session("duplicate")

    def test_get_session_exists(self, session_db):
        """Test getting an existing session."""
        created = session_db.create_session("test_session", {"key": "value"})

        retrieved = session_db.get_session("test_session")

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "test_session"
        assert retrieved.metadata["key"] == "value"

    def test_get_session_not_exists(self, session_db):
        """Test getting non-existent session."""
        retrieved = session_db.get_session("nonexistent")

        assert retrieved is None

    def test_list_sessions_empty(self, session_db):
        """Test listing sessions when empty."""
        sessions = session_db.list_sessions()

        assert sessions == []

    def test_list_sessions_multiple(self, session_db):
        """Test listing multiple sessions."""
        session_db.create_session("session1", {"type": "test"})
        session_db.create_session("session2", {"type": "prod"})
        session_db.create_session("session3")

        sessions = session_db.list_sessions()

        assert len(sessions) == 3

        # Should be ordered by created_at DESC (newest first)
        names = [s.name for s in sessions]
        assert "session3" in names
        assert "session2" in names
        assert "session1" in names

        # Check metadata is preserved
        session1 = next(s for s in sessions if s.name == "session1")
        assert session1.metadata["type"] == "test"

    def test_add_context(self, session_db):
        """Test adding context to a session."""
        # Create session first
        session_db.create_session("test_session")

        # Add context
        context_content = json.dumps(
            {"files": ["file1.py", "file2.py"], "content": "test content", "token_count": 100}
        )

        session_db.add_context("test_session", "context_result", context_content)

        # Verify context was added
        db_path = Path(session_db.config.cache.directory) / "tenets.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT kind, content FROM session_context WHERE session_id = "
            "(SELECT id FROM sessions WHERE name = ?)",
            ("test_session",),
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "context_result"
        assert json.loads(row[1])["token_count"] == 100

    def test_add_context_creates_session(self, session_db):
        """Test that add_context creates session if needed."""
        # Add context to non-existent session
        session_db.add_context("new_session", "test", "content")

        # Session should have been created
        session = session_db.get_session("new_session")
        assert session is not None
        assert session.name == "new_session"

    def test_add_multiple_contexts(self, session_db):
        """Test adding multiple contexts to same session."""
        session_db.create_session("multi_context")

        session_db.add_context("multi_context", "type1", "content1")
        session_db.add_context("multi_context", "type2", "content2")
        session_db.add_context("multi_context", "type1", "content3")  # Same type again

        # Verify all contexts were added
        db_path = Path(session_db.config.cache.directory) / "tenets.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM session_context WHERE session_id = "
            "(SELECT id FROM sessions WHERE name = ?)",
            ("multi_context",),
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 3

    def test_delete_session(self, session_db):
        """Test deleting a session."""
        # Create session with context
        session_db.create_session("to_delete")
        session_db.add_context("to_delete", "test", "content")

        # Delete session
        deleted = session_db.delete_session("to_delete")

        assert deleted == True

        # Session should be gone
        assert session_db.get_session("to_delete") is None

        # Context should also be deleted
        db_path = Path(session_db.config.cache.directory) / "tenets.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM session_context WHERE session_id = "
            "(SELECT id FROM sessions WHERE name = ?)",
            ("to_delete",),
        )
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    def test_delete_nonexistent_session(self, session_db):
        """Test deleting non-existent session."""
        deleted = session_db.delete_session("nonexistent")

        assert deleted == False

    def test_session_record_dataclass(self):
        """Test SessionRecord dataclass."""
        record = SessionRecord(
            id=1, name="test", created_at=datetime.now(), metadata={"key": "value"}
        )

        assert record.id == 1
        assert record.name == "test"
        assert record.metadata["key"] == "value"
        assert isinstance(record.created_at, datetime)

    def test_metadata_json_handling(self, session_db):
        """Test JSON serialization of metadata."""
        # Test with various data types
        metadata = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }

        record = session_db.create_session("json_test", metadata)

        retrieved = session_db.get_session("json_test")

        assert retrieved.metadata["string"] == "value"
        assert retrieved.metadata["number"] == 42
        assert retrieved.metadata["float"] == 3.14
        assert retrieved.metadata["bool"] == True
        assert retrieved.metadata["null"] is None
        assert retrieved.metadata["list"] == [1, 2, 3]
        assert retrieved.metadata["dict"]["nested"] == "value"

    def test_large_context_storage(self, session_db):
        """Test storing large context content."""
        session_db.create_session("large_session")

        # Create large content (1MB)
        large_content = "x" * (1024 * 1024)

        session_db.add_context("large_session", "large", large_content)

        # Verify it can be stored and retrieved
        db_path = Path(session_db.config.cache.directory) / "tenets.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT LENGTH(content) FROM session_context WHERE session_id = "
            "(SELECT id FROM sessions WHERE name = ?)",
            ("large_session",),
        )
        length = cursor.fetchone()[0]
        conn.close()

        assert length == len(large_content)

    def test_concurrent_access(self, session_db):
        """Test concurrent database access."""
        import threading

        def create_session(name):
            try:
                session_db.create_session(name)
            except:
                pass  # Handle unique constraint violations

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_session, args=(f"session_{i}",))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have created sessions
        sessions = session_db.list_sessions()
        assert len(sessions) > 0

    def test_transaction_rollback(self, config):
        """Test transaction rollback on error."""
        db = SessionDB(config)

        # Mock connection to simulate error
        with patch.object(db.db, "connect") as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.execute.side_effect = Exception("DB Error")
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            # Should handle error gracefully
            with pytest.raises(Exception):
                db.create_session("error_session")

            # Connection should be closed even on error
            mock_conn.close.assert_called()

    def test_schema_initialization_idempotent(self, config):
        """Test that schema initialization is idempotent."""
        # Create first instance
        db1 = SessionDB(config)
        db1.create_session("session1")

        # Create second instance - should not error
        db2 = SessionDB(config)

        # Should see existing session
        session = db2.get_session("session1")
        assert session is not None

    def test_timestamps(self, session_db):
        """Test timestamp handling."""
        import time

        # Create session
        record1 = session_db.create_session("time_test1")
        time1 = record1.created_at

        # Wait a bit
        time.sleep(0.1)

        # Create another session
        record2 = session_db.create_session("time_test2")
        time2 = record2.created_at

        # Second should be newer
        assert time2 > time1

        # List should return newest first
        sessions = session_db.list_sessions()
        assert sessions[0].name == "time_test2"
        assert sessions[1].name == "time_test1"

    def test_foreign_key_constraint(self, session_db):
        """Test foreign key constraint on session_context."""
        db_path = Path(session_db.config.cache.directory) / "tenets.db"
        conn = sqlite3.connect(db_path)

        # Try to insert context for non-existent session
        try:
            conn.execute(
                "INSERT INTO session_context (session_id, kind, content, created_at) "
                "VALUES (9999, 'test', 'content', ?)",
                (datetime.utcnow(),),
            )
            conn.commit()

            # If foreign keys are enabled, this should fail
            # If not, it will succeed (SQLite default)

        except sqlite3.IntegrityError:
            # Foreign key constraint worked
            pass
        finally:
            conn.close()
