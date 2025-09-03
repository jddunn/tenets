"""Unit tests for the tenet CLI command.

Tests cover all tenet management functionality including:
- Adding tenets with various priorities and categories
- Listing and filtering tenets
- Removing tenets
- Showing tenet details
- Importing and exporting tenets
- Session bindings
- Error handling
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tenets.cli.commands.tenet import tenet_app


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_tenets():
    """Create a mock tenet manager instance."""
    mock = MagicMock()

    # Mock tenet object
    mock_tenet = MagicMock()
    mock_tenet.id = "abc123def456"
    mock_tenet.content = "Always use type hints in Python"
    mock_tenet.priority = MagicMock(value="high")
    mock_tenet.status = MagicMock(value="pending")
    mock_tenet.category = MagicMock(value="style")
    mock_tenet.created_at = datetime.now()
    mock_tenet.instilled_at = None
    mock_tenet.session_bindings = []
    mock_tenet.metrics = MagicMock(
        injection_count=5, contexts_appeared_in=3, reinforcement_needed=False
    )

    # Mock manager methods - add_tenet doesn't return anything in the actual implementation
    # The actual implementation calls add_tenet with keyword argument tenet=
    def mock_add_tenet(**kwargs):
        if "tenet" in kwargs:
            tenet = kwargs["tenet"]
            tenet.id = "abc123def456"

    mock.add_tenet = MagicMock(side_effect=mock_add_tenet)
    mock.get_all_tenets.return_value = [mock_tenet]
    mock.get_tenet.return_value = mock_tenet
    mock.remove_tenet.return_value = True
    # Export/import methods that the CLI expects
    mock.export_tenets = MagicMock(return_value="---\ntenets:\n  - content: Always use type hints")
    mock.import_tenets = MagicMock(return_value=2)
    # mock.get_pending_tenets.return_value = [mock_tenet]

    return mock


class TestTenetAdd:
    """Test adding tenets."""

    def test_add_basic_tenet(self, runner, mock_tenets):
        """Test adding a basic tenet."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["add", "Always use type hints"])

            assert result.exit_code == 0
            assert "Added tenet: Always use type hints" in result.stdout
            assert "ID: abc123de..." in result.stdout
            assert "Priority: medium" in result.stdout  # default priority is medium
            assert "Use 'tenets instill' to apply" in result.stdout

            # Check that add_tenet was called with a Tenet object
            mock_tenets.add_tenet.assert_called_once()
            # The actual implementation calls add_tenet with keyword argument tenet=
            tenet_arg = mock_tenets.add_tenet.call_args.kwargs["tenet"]
            assert tenet_arg.content == "Always use type hints"
            assert tenet_arg.priority.value == "medium"  # default
            assert tenet_arg.category is None
            assert tenet_arg.session_bindings == []

    def test_add_tenet_with_priority(self, runner, mock_tenets):
        """Test adding tenet with priority."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(
                tenet_app, ["add", "Validate all inputs", "--priority", "critical"]
            )

            assert result.exit_code == 0
            # Check that add_tenet was called with a Tenet object
            mock_tenets.add_tenet.assert_called_once()
            # The actual implementation calls add_tenet with keyword argument tenet=
            tenet_arg = mock_tenets.add_tenet.call_args.kwargs["tenet"]
            assert tenet_arg.content == "Validate all inputs"
            assert tenet_arg.priority.value == "critical"
            assert tenet_arg.category is None
            assert tenet_arg.session_bindings == []

    def test_add_tenet_with_category(self, runner, mock_tenets):
        """Test adding tenet with category."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(
                tenet_app, ["add", "Use async for I/O", "--category", "performance"]
            )

            assert result.exit_code == 0
            assert "Category: performance" in result.stdout
            # Check that add_tenet was called with a Tenet object
            mock_tenets.add_tenet.assert_called_once()
            # The actual implementation calls add_tenet with keyword argument tenet=
            tenet_arg = mock_tenets.add_tenet.call_args.kwargs["tenet"]
            assert tenet_arg.content == "Use async for I/O"
            assert tenet_arg.priority.value == "medium"
            assert (
                tenet_arg.category.value == "performance"
                if hasattr(tenet_arg.category, "value")
                else str(tenet_arg.category) == "performance"
            )

    def test_add_tenet_with_session(self, runner, mock_tenets):
        """Test adding tenet bound to session."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(
                tenet_app, ["add", "Feature-specific rule", "--session", "feature-x"]
            )

            assert result.exit_code == 0
            assert "Bound to session: feature-x" in result.stdout
            # Check that add_tenet was called with a Tenet object
            mock_tenets.add_tenet.assert_called_once()
            # The actual implementation calls add_tenet with keyword argument tenet=
            tenet_arg = mock_tenets.add_tenet.call_args.kwargs["tenet"]
            assert tenet_arg.content == "Feature-specific rule"
            assert tenet_arg.priority.value == "medium"
            assert tenet_arg.category is None
            assert tenet_arg.session_bindings == ["feature-x"]

    def test_add_tenet_all_options(self, runner, mock_tenets):
        """Test adding tenet with all options."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(
                tenet_app,
                [
                    "add",
                    "Sanitize user input",
                    "--priority",
                    "critical",
                    "--category",
                    "security",
                    "--session",
                    "auth-work",
                ],
            )

            assert result.exit_code == 0
            # Check that add_tenet was called with a Tenet object
            mock_tenets.add_tenet.assert_called_once()
            # The actual implementation calls add_tenet with keyword argument tenet=
            tenet_arg = mock_tenets.add_tenet.call_args.kwargs["tenet"]
            assert tenet_arg.content == "Sanitize user input"
            assert tenet_arg.priority.value == "critical"
            assert (
                tenet_arg.category.value == "security"
                if hasattr(tenet_arg.category, "value")
                else str(tenet_arg.category) == "security"
            )
            assert tenet_arg.session_bindings == ["auth-work"]

    def test_add_tenet_no_manager(self, runner):
        """Test error when tenet manager unavailable."""
        # Simulate manager error by making add_tenet raise an exception
        with patch(
            "tenets.cli.commands.tenet.get_tenet_manager",
            side_effect=Exception("Tenet system is not available"),
        ):
            result = runner.invoke(tenet_app, ["add", "Test"])

            assert result.exit_code == 1
            assert (
                "Error:" in result.stdout
            )  # The CLI prints Error: followed by the exception message


class TestTenetList:
    """Test listing tenets."""

    def test_list_all_tenets(self, runner, mock_tenets):
        """Test listing all tenets."""
        mock_tenets.list_tenets.return_value = [
            {
                "id": "abc123def456",
                "content": "Always use type hints",
                "priority": "high",
                "category": "style",
                "instilled": False,
                "created_at": "2024-01-15T10:00:00",
            },
            {
                "id": "xyz789ghi012",
                "content": "Validate inputs",
                "priority": "critical",
                "category": "security",
                "instilled": True,
                "created_at": "2024-01-14T09:00:00",
            },
        ]

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["list"])

            assert result.exit_code == 0
            assert "Guiding Principles (Tenets)" in result.stdout
            assert "Always use type hints" in result.stdout
            assert "Validate inputs" in result.stdout
            assert "✓ Instilled" in result.stdout
            assert "⏳ Pending" in result.stdout
            assert "Total: 2 | Pending: 1 | Instilled: 1" in result.stdout

    def test_list_pending_only(self, runner, mock_tenets):
        """Test listing only pending tenets."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["list", "--pending"])

            assert result.exit_code == 0
            assert "Pending Only" in result.stdout
            mock_tenets.list_tenets.assert_called_once_with(
                pending_only=True, instilled_only=False, session=None
            )

    def test_list_instilled_only(self, runner, mock_tenets):
        """Test listing only instilled tenets."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["list", "--instilled"])

            assert result.exit_code == 0
            assert "Instilled Only" in result.stdout
            mock_tenets.list_tenets.assert_called_once_with(
                pending_only=False, instilled_only=True, session=None
            )

    def test_list_by_session(self, runner, mock_tenets):
        """Test listing tenets filtered by session."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["list", "--session", "oauth"])

            assert result.exit_code == 0
            assert "Session: oauth" in result.stdout
            mock_tenets.list_tenets.assert_called_once_with(
                pending_only=False, instilled_only=False, session="oauth"
            )

    def test_list_by_category(self, runner, mock_tenets):
        """Test listing tenets filtered by category."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["list", "--category", "security"])

            assert result.exit_code == 0
            assert "Category: security" in result.stdout

    def test_list_verbose(self, runner, mock_tenets):
        """Test verbose listing with full content."""
        mock_tenets.list_tenets.return_value = [
            {
                "id": "abc123def456",
                "content": "This is a very long tenet content that would normally be truncated but should be shown in full in verbose mode",
                "priority": "high",
                "category": "style",
                "instilled": False,
                "created_at": "2024-01-15T10:00:00",
                "session_bindings": ["session1", "session2"],
            }
        ]

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["list", "--verbose"])

            assert result.exit_code == 0
            assert "shown in full in verbose mode" in result.stdout
            assert "session1, session2" in result.stdout

    def test_list_no_tenets(self, runner, mock_tenets):
        """Test listing when no tenets exist."""
        mock_tenets.list_tenets.return_value = []

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["list"])

            assert result.exit_code == 0
            assert "No tenets found" in result.stdout
            assert 'tenets tenet add "Your principle"' in result.stdout


class TestTenetRemove:
    """Test removing tenets."""

    def test_remove_tenet_with_confirmation(self, runner, mock_tenets):
        """Test removing tenet with confirmation."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            with patch("rich.prompt.Confirm.ask", return_value=True):
                result = runner.invoke(tenet_app, ["remove", "abc123"])

                assert result.exit_code == 0
                assert "Removed tenet: Always use type hints" in result.stdout
                mock_tenets.get_tenet.assert_called_once_with("abc123")
                mock_tenets.remove_tenet.assert_called_once_with("abc123")

    def test_remove_tenet_cancelled(self, runner, mock_tenets):
        """Test cancelling tenet removal."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            with patch("rich.prompt.Confirm.ask", return_value=False):
                result = runner.invoke(tenet_app, ["remove", "abc123"])

                assert result.exit_code == 0
                assert "Cancelled" in result.stdout
                mock_tenets.remove_tenet.assert_not_called()

    def test_remove_tenet_forced(self, runner, mock_tenets):
        """Test forced tenet removal without confirmation."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["remove", "abc123", "--force"])

            assert result.exit_code == 0
            assert "Removed tenet" in result.stdout
            mock_tenets.remove_tenet.assert_called_once_with("abc123")

    def test_remove_nonexistent_tenet(self, runner, mock_tenets):
        """Test removing non-existent tenet."""
        mock_tenets.get_tenet.return_value = None

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["remove", "nonexistent"])

            assert result.exit_code == 1
            assert "Tenet not found: nonexistent" in result.stdout

    def test_remove_tenet_failed(self, runner, mock_tenets):
        """Test failed tenet removal."""
        mock_tenets.remove_tenet.return_value = False

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["remove", "abc123", "--force"])

            assert result.exit_code == 1
            assert "Failed to remove tenet" in result.stdout


class TestTenetShow:
    """Test showing tenet details."""

    def test_show_tenet(self, runner, mock_tenets):
        """Test showing tenet details."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["show", "abc123"])

            assert result.exit_code == 0
            assert "Tenet Details" in result.stdout
            assert "Always use type hints" in result.stdout
            assert "Priority: high" in result.stdout
            assert "Status: pending" in result.stdout
            assert "Category: style" in result.stdout
            assert "Injections: 5" in result.stdout
            assert "Contexts appeared in: 3" in result.stdout
            mock_tenets.get_tenet.assert_called_once_with("abc123")

    def test_show_tenet_with_session_bindings(self, runner, mock_tenets):
        """Test showing tenet with session bindings."""
        mock_tenets.get_tenet.return_value.session_bindings = ["session1", "session2"]

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["show", "abc123"])

            assert result.exit_code == 0
            assert "Session Bindings: session1, session2" in result.stdout

    def test_show_nonexistent_tenet(self, runner, mock_tenets):
        """Test showing non-existent tenet."""
        mock_tenets.get_tenet.return_value = None

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["show", "nonexistent"])

            assert result.exit_code == 1
            assert "Tenet not found: nonexistent" in result.stdout


class TestTenetExportImport:
    """Test tenet export and import functionality."""

    def test_export_tenets_stdout(self, runner, mock_tenets):
        """Test exporting tenets to stdout."""
        mock_tenets.export_tenets.return_value = "---\ntenets:\n  - content: Always use type hints"

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["export"])

            assert result.exit_code == 0
            # The exported content should be printed to stdout
            assert "---" in result.stdout or "tenets:" in result.stdout  # Either YAML format
            mock_tenets.export_tenets.assert_called_once_with(format="yaml", session=None)

    def test_export_tenets_file(self, runner, mock_tenets, tmp_path):
        """Test exporting tenets to file."""
        output_file = tmp_path / "tenets.yml"
        mock_tenets.export_tenets.return_value = "---\ntenets:\n  - content: Test"

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["export", "--output", str(output_file)])

            assert result.exit_code == 0
            assert "Exported tenets to" in result.stdout
            # Note: file might not exist because mock doesn't actually write it

    def test_export_tenets_json(self, runner, mock_tenets, tmp_path):
        """Test exporting tenets as JSON."""
        output_file = tmp_path / "tenets.json"
        mock_tenets.export_tenets.return_value = '{"tenets": []}'

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(
                tenet_app, ["export", "--format", "json", "--output", str(output_file)]
            )

            assert result.exit_code == 0
            mock_tenets.export_tenets.assert_called_once_with(format="json", session=None)

    def test_export_session_tenets(self, runner, mock_tenets):
        """Test exporting session-specific tenets."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["export", "--session", "oauth"])

            assert result.exit_code == 0
            mock_tenets.export_tenets.assert_called_once_with(format="yaml", session="oauth")

    def test_import_tenets(self, runner, mock_tenets, tmp_path):
        """Test importing tenets from file."""
        import_file = tmp_path / "tenets.yml"
        import_file.write_text("tenets:\n  - content: Test tenet")

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["import", str(import_file)])

            assert result.exit_code == 0
            # The import_tenets returns 2 from the mock
            assert "Imported" in result.stdout and "2" in result.stdout
            assert "Use 'tenets instill' to apply" in result.stdout
            mock_tenets.import_tenets.assert_called_once_with(import_file, session=None)

    def test_import_tenets_to_session(self, runner, mock_tenets, tmp_path):
        """Test importing tenets to specific session."""
        import_file = tmp_path / "tenets.yml"
        import_file.write_text("tenets:\n  - content: Test")

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(
                tenet_app, ["import", str(import_file), "--session", "feature-x"]
            )

            assert result.exit_code == 0
            assert "Imported into session: feature-x" in result.stdout
            mock_tenets.import_tenets.assert_called_once_with(import_file, session="feature-x")

    def test_import_dry_run(self, runner, mock_tenets, tmp_path):
        """Test dry run import preview."""
        import_file = tmp_path / "tenets.yml"
        content = "tenets:\n  - content: Test tenet\n    priority: high"
        import_file.write_text(content)

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["import", str(import_file), "--dry-run"])

            assert result.exit_code == 0
            assert "Would import tenets from" in result.stdout
            assert "Test tenet" in result.stdout
            mock_tenets.import_tenets.assert_not_called()

    def test_import_nonexistent_file(self, runner, mock_tenets):
        """Test importing from non-existent file."""
        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["import", "nonexistent.yml"])

            assert result.exit_code == 1
            assert "File not found: nonexistent.yml" in result.stdout


class TestTenetErrorHandling:
    """Test error handling scenarios."""

    def test_add_tenet_error(self, runner, mock_tenets):
        """Test error when adding tenet."""
        mock_tenets.add_tenet.side_effect = Exception("Database error")

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["add", "Test"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Database error" in result.stdout

    def test_list_tenets_error(self, runner, mock_tenets):
        """Test error when listing tenets."""
        mock_tenets.list_tenets.side_effect = Exception("Query failed")

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["list"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Query failed" in result.stdout

    def test_export_error(self, runner, mock_tenets):
        """Test error during export."""
        mock_tenets.export_tenets.side_effect = Exception("Export failed")

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["export"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Export failed" in result.stdout

    def test_import_error(self, runner, mock_tenets, tmp_path):
        """Test error during import."""
        import_file = tmp_path / "tenets.yml"
        import_file.write_text("invalid: yaml")
        mock_tenets.import_tenets.side_effect = Exception("Invalid format")

        with patch("tenets.cli.commands.tenet.get_tenet_manager", return_value=mock_tenets):
            result = runner.invoke(tenet_app, ["import", str(import_file)])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Invalid format" in result.stdout
