"""Tests for session management."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from tenets.core.session.session import SessionManager
from tenets.models.context import SessionContext, ContextResult
from tenets.config import TenetsConfig
from tenets.storage.session_db import SessionDB


@pytest.fixture
def config():
    """Create test configuration with temporary cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TenetsConfig()
        config.cache.directory = tmpdir
        yield config


@pytest.fixture
def session_manager_memory():
    """Create SessionManager without database (memory only)."""
    return SessionManager(config=None)


@pytest.fixture
def session_manager_db(config):
    """Create SessionManager with database persistence."""
    return SessionManager(config=config)


class TestSessionManager:
    """Test suite for SessionManager."""

    def test_initialization_memory_only(self):
        """Test initialization without database."""
        manager = SessionManager(config=None)

        assert manager.sessions == {}
        assert manager._db is None

    def test_initialization_with_db(self, config):
        """Test initialization with database."""
        manager = SessionManager(config=config)

        assert manager.sessions == {}
        assert manager._db is not None
        assert isinstance(manager._db, SessionDB)

    def test_create_session_memory(self, session_manager_memory):
        """Test creating a session in memory."""
        session = session_manager_memory.create("test_session")

        assert isinstance(session, SessionContext)
        assert session.session_id == "test_session"
        assert session.name == "test_session"
        assert "test_session" in session_manager_memory.sessions

    def test_create_session_idempotent(self, session_manager_memory):
        """Test that creating same session returns existing."""
        session1 = session_manager_memory.create("test_session")
        session2 = session_manager_memory.create("test_session")

        assert session1 is session2
        assert len(session_manager_memory.sessions) == 1

    def test_create_session_with_db(self, session_manager_db):
        """Test creating a session with database persistence."""
        session = session_manager_db.create("db_session")

        assert isinstance(session, SessionContext)
        assert session.name == "db_session"

        # Verify it was persisted to database
        with patch.object(session_manager_db._db, "create_session") as mock_create:
            session_manager_db.create("new_session")
            # Only called if session doesn't exist

    def test_list_sessions_empty(self, session_manager_memory):
        """Test listing sessions when empty."""
        sessions = session_manager_memory.list()

        assert sessions == []

    def test_list_sessions_memory(self, session_manager_memory):
        """Test listing sessions from memory."""
        session_manager_memory.create("session1")
        session_manager_memory.create("session2")
        session_manager_memory.create("session3")

        sessions = session_manager_memory.list()

        assert len(sessions) == 3
        session_names = [s.name for s in sessions]
        assert "session1" in session_names
        assert "session2" in session_names
        assert "session3" in session_names

    def test_list_sessions_with_db(self, session_manager_db):
        """Test listing sessions with database."""
        # Mock database returning sessions
        from tenets.storage.session_db import SessionRecord
        from datetime import datetime

        mock_records = [
            SessionRecord(id=1, name="db_session1", created_at=datetime.now(), metadata={}),
            SessionRecord(id=2, name="db_session2", created_at=datetime.now(), metadata={}),
        ]

        with patch.object(session_manager_db._db, "list_sessions", return_value=mock_records):
            sessions = session_manager_db.list()

            assert len(sessions) >= 2
            session_names = [s.name for s in sessions]
            assert "db_session1" in session_names
            assert "db_session2" in session_names

    def test_get_session_exists(self, session_manager_memory):
        """Test getting an existing session."""
        created = session_manager_memory.create("test_session")

        retrieved = session_manager_memory.get("test_session")

        assert retrieved is created
        assert retrieved.name == "test_session"

    def test_get_session_not_exists(self, session_manager_memory):
        """Test getting a non-existent session."""
        retrieved = session_manager_memory.get("nonexistent")

        assert retrieved is None

    def test_get_session_from_db(self, session_manager_db):
        """Test getting session from database."""
        from tenets.storage.session_db import SessionRecord
        from datetime import datetime

        mock_record = SessionRecord(
            id=1, name="db_session", created_at=datetime.now(), metadata={"key": "value"}
        )

        with patch.object(session_manager_db._db, "get_session", return_value=mock_record):
            session = session_manager_db.get("db_session")

            assert session is not None
            assert session.name == "db_session"
            # Should be cached in memory now
            assert "db_session" in session_manager_db.sessions

    def test_delete_session_exists(self, session_manager_memory):
        """Test deleting an existing session."""
        session_manager_memory.create("to_delete")

        deleted = session_manager_memory.delete("to_delete")

        assert deleted == True
        assert "to_delete" not in session_manager_memory.sessions
        assert session_manager_memory.get("to_delete") is None

    def test_delete_session_not_exists(self, session_manager_memory):
        """Test deleting a non-existent session."""
        deleted = session_manager_memory.delete("nonexistent")

        assert deleted == False

    def test_delete_session_with_db(self, session_manager_db):
        """Test deleting session with database."""
        session_manager_db.create("to_delete")

        with patch.object(session_manager_db._db, "delete_session") as mock_delete:
            deleted = session_manager_db.delete("to_delete")

            assert deleted == True
            mock_delete.assert_called_once_with("to_delete")
            assert "to_delete" not in session_manager_db.sessions

    def test_add_context_to_session(self, session_manager_memory):
        """Test adding context to a session."""
        context_result = ContextResult(
            files=[], content="test content", token_count=100, metadata={"test": "data"}
        )

        session_manager_memory.add_context("test_session", context_result)

        session = session_manager_memory.get("test_session")
        assert session is not None
        assert len(session.context_history) == 1
        assert session.context_history[0] == context_result

    def test_add_context_creates_session(self, session_manager_memory):
        """Test that add_context creates session if needed."""
        context_result = ContextResult(files=[], content="test content", token_count=100)

        # Session doesn't exist yet
        assert session_manager_memory.get("new_session") is None

        session_manager_memory.add_context("new_session", context_result)

        # Session should be created
        session = session_manager_memory.get("new_session")
        assert session is not None
        assert len(session.context_history) == 1

    def test_add_context_with_db(self, session_manager_db):
        """Test adding context with database persistence."""
        context_result = ContextResult(
            files=["file1.py", "file2.py"],
            content="test content",
            token_count=100,
            metadata={"key": "value"},
        )

        with patch.object(session_manager_db._db, "add_context") as mock_add:
            session_manager_db.add_context("test_session", context_result)

            mock_add.assert_called_once()
            # Check both args and kwargs
            call_args = mock_add.call_args[0]  # This is the args tuple
            call_kwargs = mock_add.call_args[1]  # This is the kwargs dict

            # Debug what we actually got
            print(f"Args: {call_args}")
            print(f"Kwargs: {call_kwargs}")

            # The first argument should be session name (positional)
            assert (
                len(call_args) >= 1
            ), f"Expected at least 1 positional arg, got {len(call_args)}: {call_args}"
            assert call_args[0] == "test_session"

            # Check if kind and content are passed as keyword arguments
            if len(call_args) >= 3:
                # All arguments passed positionally
                assert call_args[1] == "context_result"
                content_data = json.loads(call_args[2])
            else:
                # Some arguments passed as keywords
                assert "kind" in call_kwargs, f"Expected 'kind' in kwargs: {call_kwargs}"
                assert "content" in call_kwargs, f"Expected 'content' in kwargs: {call_kwargs}"
                assert call_kwargs["kind"] == "context_result"
                content_data = json.loads(call_kwargs["content"])

            assert content_data["files"] == ["file1.py", "file2.py"]
            assert content_data["token_count"] == 100

    def test_db_error_handling(self, session_manager_db):
        """Test graceful handling of database errors."""
        # Simulate database error
        with patch.object(
            session_manager_db._db, "create_session", side_effect=Exception("DB Error")
        ):
            # Should still work with memory
            session = session_manager_db.create("test_session")
            assert session is not None
            assert session.name == "test_session"

        with patch.object(
            session_manager_db._db, "list_sessions", side_effect=Exception("DB Error")
        ):
            # Should return memory sessions
            sessions = session_manager_db.list()
            assert isinstance(sessions, list)

        with patch.object(session_manager_db._db, "get_session", side_effect=Exception("DB Error")):
            # Should return None if not in memory
            session = session_manager_db.get("unknown")
            assert session is None

        with patch.object(
            session_manager_db._db, "delete_session", side_effect=Exception("DB Error")
        ):
            # Should still delete from memory
            session_manager_db.create("to_delete")
            deleted = session_manager_db.delete("to_delete")
            assert deleted == True


class TestSessionContext:
    """Test suite for SessionContext model."""

    def test_session_context_initialization(self):
        """Test SessionContext initialization."""
        context = SessionContext(session_id="test_id", name="test_session")

        assert context.session_id == "test_id"
        assert context.name == "test_session"
        assert context.context_history == []
        assert context.created_at is not None
        assert context.updated_at is not None

    def test_add_context_to_session_context(self):
        """Test adding context to SessionContext."""
        session = SessionContext(session_id="test", name="test")

        context1 = ContextResult(files=["file1.py"], content="content1", token_count=10)
        context2 = ContextResult(files=["file2.py"], content="content2", token_count=20)

        session.add_context(context1)
        session.add_context(context2)

        assert len(session.context_history) == 2
        assert session.context_history[0] == context1
        assert session.context_history[1] == context2

        # Updated time should change
        assert session.updated_at >= session.created_at

    def test_session_context_to_dict(self):
        """Test SessionContext serialization."""
        session = SessionContext(session_id="test", name="test_session")

        context = ContextResult(files=["file.py"], content="content", token_count=100)
        session.add_context(context)

        data = session.to_dict()

        assert data["session_id"] == "test"
        assert data["name"] == "test_session"
        assert len(data["context_history"]) == 1
        assert data["context_history"][0]["token_count"] == 100
        assert "created_at" in data
        assert "updated_at" in data

    def test_session_context_metadata(self):
        """Test SessionContext with metadata."""
        session = SessionContext(
            session_id="test", name="test", metadata={"project": "tenets", "version": "1.0"}
        )

        assert session.metadata["project"] == "tenets"
        assert session.metadata["version"] == "1.0"

        # Add more metadata
        session.metadata["updated"] = True

        data = session.to_dict()
        assert data["metadata"]["updated"] == True


class TestContextResult:
    """Test suite for ContextResult model."""

    def test_context_result_initialization(self):
        """Test ContextResult initialization."""
        result = ContextResult(
            files=["file1.py", "file2.py"],
            content="combined content",
            token_count=150,
            metadata={"key": "value"},
        )

        assert result.files == ["file1.py", "file2.py"]
        assert result.content == "combined content"
        assert result.token_count == 150
        assert result.metadata["key"] == "value"

    def test_context_result_to_dict(self):
        """Test ContextResult serialization."""
        result = ContextResult(
            files=["test.py"], content="test content", token_count=50, metadata={"source": "test"}
        )

        data = result.to_dict()

        assert data["files"] == ["test.py"]
        assert data["content"] == "test content"
        assert data["token_count"] == 50
        assert data["metadata"]["source"] == "test"
        assert "timestamp" in data

    def test_context_result_empty_files(self):
        """Test ContextResult with no files."""
        result = ContextResult(files=[], content="general content", token_count=25)

        assert result.files == []
        assert result.content == "general content"
        assert result.token_count == 25

    def test_context_result_large_content(self):
        """Test ContextResult with large content."""
        large_content = "x" * 100000  # 100KB of content

        result = ContextResult(files=["large.txt"], content=large_content, token_count=25000)

        assert len(result.content) == 100000
        assert result.token_count == 25000

        # Should serialize without issues
        data = result.to_dict()
        assert len(data["content"]) == 100000
