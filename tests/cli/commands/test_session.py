"""Unit tests for the session CLI command.

Tests cover all session management functionality including:
- Creating and starting sessions
- Listing and showing sessions
- Deleting sessions
- Session context management
- Resuming and exiting sessions
- Session reset
- Error handling
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tenets.cli.commands.session import session_app


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_session_db():
    """Create a mock SessionDB."""
    mock_db = MagicMock()

    # Mock session object
    mock_session = MagicMock()
    mock_session.name = "test-session"
    mock_session.created_at = datetime.now()
    mock_session.metadata = {"active": True}

    # Mock methods
    mock_db.get_session.return_value = mock_session
    mock_db.create_session.return_value = mock_session
    mock_db.list_sessions.return_value = [mock_session]
    mock_db.delete_session.return_value = True
    mock_db.delete_all_sessions.return_value = 2
    mock_db.set_active.return_value = True
    mock_db.get_active_session.return_value = mock_session
    mock_db.add_context.return_value = True

    return mock_db


class TestSessionCreate:
    """Test session creation commands."""

    def test_create_new_session(self, runner, mock_session_db):
        """Test creating a new session."""
        mock_session_db.get_session.return_value = None  # Session doesn't exist

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["create", "new-feature"])

            assert result.exit_code == 0
            assert "Created session: new-feature" in result.stdout
            mock_session_db.create_session.assert_called_once_with("new-feature")
            mock_session_db.set_active.assert_called_once_with("new-feature", True)

    def test_create_existing_session(self, runner, mock_session_db):
        """Test creating/activating an existing session."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["create", "test-session"])

            assert result.exit_code == 0
            assert "Activated session: test-session" in result.stdout
            mock_session_db.create_session.assert_not_called()
            mock_session_db.set_active.assert_called_once_with("test-session", True)

    def test_start_session_alias(self, runner, mock_session_db):
        """Test 'start' as an alias for 'create'."""
        mock_session_db.get_session.return_value = None

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["start", "new-session"])

            assert result.exit_code == 0
            assert "Created session: new-session" in result.stdout
            mock_session_db.create_session.assert_called_once()


class TestSessionList:
    """Test session listing commands."""

    def test_list_sessions(self, runner, mock_session_db):
        """Test listing all sessions."""
        mock_session_db.list_sessions.return_value = [
            MagicMock(name="session1", created_at=datetime.now(), metadata={"active": True}),
            MagicMock(name="session2", created_at=datetime.now(), metadata={"active": False}),
        ]

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["list"])

            assert result.exit_code == 0
            assert "Sessions" in result.stdout
            assert "session1" in result.stdout
            assert "session2" in result.stdout
            assert "yes" in result.stdout  # Active status

    def test_list_no_sessions(self, runner, mock_session_db):
        """Test listing when no sessions exist."""
        mock_session_db.list_sessions.return_value = []

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["list"])

            assert result.exit_code == 0
            assert "No sessions found" in result.stdout


class TestSessionShow:
    """Test showing session details."""

    def test_show_session(self, runner, mock_session_db):
        """Test showing session details."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["show", "test-session"])

            assert result.exit_code == 0
            assert "Session: test-session" in result.stdout
            assert "Active: True" in result.stdout
            mock_session_db.get_session.assert_called_once_with("test-session")

    def test_show_nonexistent_session(self, runner, mock_session_db):
        """Test showing non-existent session."""
        mock_session_db.get_session.return_value = None

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["show", "nonexistent"])

            assert result.exit_code == 1
            assert "Session not found: nonexistent" in result.stdout


class TestSessionDelete:
    """Test session deletion commands."""

    def test_delete_session(self, runner, mock_session_db):
        """Test deleting a session."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["delete", "old-session"])

            assert result.exit_code == 0
            assert "Deleted session: old-session" in result.stdout
            mock_session_db.delete_session.assert_called_once_with(
                "old-session", purge_context=True
            )

    def test_delete_session_keep_context(self, runner, mock_session_db):
        """Test deleting session but keeping context."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["delete", "old-session", "--keep-context"])

            assert result.exit_code == 0
            mock_session_db.delete_session.assert_called_once_with(
                "old-session", purge_context=False
            )

    def test_delete_nonexistent_session(self, runner, mock_session_db):
        """Test deleting non-existent session."""
        mock_session_db.delete_session.return_value = False

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["delete", "nonexistent"])

            assert result.exit_code == 0
            assert "No such session: nonexistent" in result.stdout

    def test_clear_all_sessions(self, runner, mock_session_db):
        """Test clearing all sessions."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["clear"])

            assert result.exit_code == 0
            assert "Deleted 2 session(s)" in result.stdout
            mock_session_db.delete_all_sessions.assert_called_once_with(purge_context=True)

    def test_clear_all_keep_context(self, runner, mock_session_db):
        """Test clearing all sessions but keeping artifacts."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["clear", "--keep-context"])

            assert result.exit_code == 0
            mock_session_db.delete_all_sessions.assert_called_once_with(purge_context=False)

    def test_clear_no_sessions(self, runner, mock_session_db):
        """Test clearing when no sessions exist."""
        mock_session_db.delete_all_sessions.return_value = 0

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["clear"])

            assert result.exit_code == 0
            assert "No sessions to delete" in result.stdout


class TestSessionContext:
    """Test session context management."""

    def test_add_context(self, runner, mock_session_db, tmp_path):
        """Test adding context to a session."""
        # Create a temporary file with content
        context_file = tmp_path / "context.txt"
        context_file.write_text("This is test context")

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["add", "test-session", "note", str(context_file)])

            assert result.exit_code == 0
            assert "Added note to session: test-session" in result.stdout
            mock_session_db.add_context.assert_called_once()
            call_args = mock_session_db.add_context.call_args
            assert call_args[0][0] == "test-session"
            assert call_args[1]["kind"] == "note"
            assert call_args[1]["content"] == "This is test context"


class TestSessionReset:
    """Test session reset functionality."""

    def test_reset_session(self, runner, mock_session_db):
        """Test resetting a session."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["reset", "test-session"])

            assert result.exit_code == 0
            assert "Reset session: test-session" in result.stdout
            # Should delete then recreate
            mock_session_db.delete_session.assert_called_once_with(
                "test-session", purge_context=True
            )
            mock_session_db.create_session.assert_called_once_with("test-session")
            mock_session_db.set_active.assert_called_once_with("test-session", True)


class TestSessionResumeExit:
    """Test session resume and exit functionality."""

    def test_resume_specific_session(self, runner, mock_session_db):
        """Test resuming a specific session."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["resume", "test-session"])

            assert result.exit_code == 0
            assert "Resumed session: test-session" in result.stdout
            mock_session_db.get_session.assert_called_once_with("test-session")
            mock_session_db.set_active.assert_called_once_with("test-session", True)

    def test_resume_no_name(self, runner, mock_session_db):
        """Test resuming most recent active session."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["resume"])

            assert result.exit_code == 0
            assert "Resumed session: test-session" in result.stdout
            mock_session_db.get_active_session.assert_called_once()

    def test_resume_no_active_session(self, runner, mock_session_db):
        """Test resume when no active session exists."""
        mock_session_db.get_active_session.return_value = None

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["resume"])

            assert result.exit_code == 1
            assert "No active session" in result.stdout

    def test_resume_nonexistent_session(self, runner, mock_session_db):
        """Test resuming non-existent session."""
        mock_session_db.get_session.return_value = None

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["resume", "nonexistent"])

            assert result.exit_code == 1
            assert "Session not found: nonexistent" in result.stdout

    def test_exit_specific_session(self, runner, mock_session_db):
        """Test exiting a specific session."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["exit", "test-session"])

            assert result.exit_code == 0
            assert "Exited session: test-session" in result.stdout
            mock_session_db.set_active.assert_called_once_with("test-session", False)

    def test_exit_current_session(self, runner, mock_session_db):
        """Test exiting current active session."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["exit"])

            assert result.exit_code == 0
            assert "Exited session: test-session" in result.stdout
            mock_session_db.get_active_session.assert_called_once()
            mock_session_db.set_active.assert_called_once_with("test-session", False)

    def test_exit_no_active_session(self, runner, mock_session_db):
        """Test exit when no active session."""
        mock_session_db.get_active_session.return_value = None

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["exit"])

            assert result.exit_code == 1
            assert "No active session to exit" in result.stdout

    def test_exit_nonexistent_session(self, runner, mock_session_db):
        """Test exiting non-existent session."""
        mock_session_db.get_session.return_value = None

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["exit", "nonexistent"])

            assert result.exit_code == 1
            assert "Session not found: nonexistent" in result.stdout


class TestSessionEdgeCases:
    """Test edge cases and error scenarios."""

    def test_session_with_special_characters(self, runner, mock_session_db):
        """Test session names with special characters."""
        mock_session_db.get_session.return_value = None

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["create", "feature-123_test"])

            assert result.exit_code == 0
            mock_session_db.create_session.assert_called_once_with("feature-123_test")

    def test_session_with_spaces(self, runner, mock_session_db):
        """Test handling session names with spaces."""
        mock_session_db.get_session.return_value = None

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["create", "my feature"])

            assert result.exit_code == 0
            mock_session_db.create_session.assert_called_once_with("my feature")

    def test_empty_session_name(self, runner, mock_session_db):
        """Test handling empty session name."""
        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["create"])

            # Should show help or error due to missing argument
            assert result.exit_code != 0

    def test_session_metadata_display(self, runner, mock_session_db):
        """Test proper display of session metadata."""
        mock_session = MagicMock()
        mock_session.name = "complex-session"
        mock_session.created_at = datetime.now()
        mock_session.metadata = {
            "active": True,
            "files_pinned": 5,
            "last_distill": "2024-01-15T10:30:00",
            "custom_data": {"key": "value"},
        }
        mock_session_db.list_sessions.return_value = [mock_session]

        with patch("tenets.cli.commands.session.SessionDB", return_value=mock_session_db):
            result = runner.invoke(session_app, ["list"])

            assert result.exit_code == 0
            assert "complex-session" in result.stdout
            # Metadata should be displayed as JSON
            assert "files_pinned" in result.stdout
