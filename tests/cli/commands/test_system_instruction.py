"""Unit tests for the system_instruction CLI command.

Tests cover all system instruction management functionality including:
- Setting system instruction from text or file
- Showing current instruction
- Clearing instruction
- Testing injection
- Exporting and importing
- Validation
- Editing
- Error handling
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from typer.testing import CliRunner

from tenets.cli.commands.system_instruction import app
from tenets.cli.commands.system_instruction import system_app
from tenets.cli.commands.system_instruction import system_app as sys_app


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Create a mock TenetsConfig."""
    config = MagicMock()
    config.tenet = MagicMock()
    config.tenet.system_instruction = "You are a helpful coding assistant"
    config.tenet.system_instruction_enabled = True
    config.tenet.system_instruction_position = "top"
    config.tenet.system_instruction_format = "markdown"
    config.tenet.system_instruction_once_per_session = True
    config.config_file = Path(".tenets.yml")
    config.save = MagicMock()
    return config


class TestSystemInstructionSet:
    """Test setting system instruction."""

    def test_set_instruction_direct(self, runner, mock_config):
        """Test setting instruction directly from text."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["set", "You are a helpful assistant"])

            assert result.exit_code == 0
            assert "System instruction enabled" in result.stdout
            assert "Configuration saved" in result.stdout
            assert mock_config.tenet.system_instruction == "You are a helpful assistant"
            assert mock_config.tenet.system_instruction_enabled is True
            mock_config.save.assert_called_once()

    def test_set_instruction_from_file(self, runner, mock_config, tmp_path):
        """Test setting instruction from file."""
        instruction_file = tmp_path / "prompt.md"
        instruction_file.write_text("# System Prompt\n\nYou are an expert developer.")

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["set", "--file", str(instruction_file)])

            assert result.exit_code == 0
            assert "System instruction enabled" in result.stdout
            assert (
                mock_config.tenet.system_instruction
                == "# System Prompt\n\nYou are an expert developer."
            )

    def test_set_instruction_with_options(self, runner, mock_config):
        """Test setting instruction with position and format options."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(
                system_app,
                ["set", "Test instruction", "--position", "after_header", "--format", "xml"],
            )

            assert result.exit_code == 0
            assert mock_config.tenet.system_instruction_position == "after_header"
            assert mock_config.tenet.system_instruction_format == "xml"

    def test_set_instruction_disabled(self, runner, mock_config):
        """Test setting instruction but disabling it."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["set", "Test", "--disable"])

            assert result.exit_code == 0
            assert "System instruction disabled" in result.stdout
            assert mock_config.tenet.system_instruction_enabled is False

    def test_set_no_save(self, runner, mock_config):
        """Test setting without saving to config."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["set", "Test", "--no-save"])

            assert result.exit_code == 0
            mock_config.save.assert_not_called()

    def test_set_instruction_update_only_settings(self, runner, mock_config):
        """Test updating only settings without changing instruction text."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["set", "--position", "bottom", "--enable"])

            assert result.exit_code == 0
            assert mock_config.tenet.system_instruction == "You are a helpful coding assistant"
            assert mock_config.tenet.system_instruction_position == "bottom"

    def test_set_file_not_found(self, runner, mock_config):
        """Test error when file not found."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["set", "--file", "nonexistent.md"])

            assert result.exit_code == 1
            assert "File not found: nonexistent.md" in result.stdout


class TestSystemInstructionShow:
    """Test showing system instruction."""

    def test_show_instruction(self, runner, mock_config):
        """Test showing current instruction."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["show"])

            assert result.exit_code == 0
            assert "System Instruction Configuration" in result.stdout
            assert "Status: Enabled" in result.stdout
            assert "Position: top" in result.stdout
            assert "Format: markdown" in result.stdout
            assert "You are a helpful coding assistant" in result.stdout

    def test_show_instruction_raw(self, runner, mock_config):
        """Test showing raw instruction text."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["show", "--raw"])

            assert result.exit_code == 0
            assert result.stdout.strip() == "You are a helpful coding assistant"

    def test_show_no_instruction(self, runner, mock_config):
        """Test showing when no instruction configured."""
        mock_config.tenet.system_instruction = None

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["show"])

            assert result.exit_code == 0
            assert "No system instruction configured" in result.stdout
            assert 'tenets system-instruction set "Your instruction"' in result.stdout

    def test_show_code_highlighted(self, runner, mock_config):
        """Test syntax highlighting for code-like instruction."""
        mock_config.tenet.system_instruction = "def helper():\n    return True"

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["show"])

            assert result.exit_code == 0
            assert "def helper()" in result.stdout


class TestSystemInstructionClear:
    """Test clearing system instruction."""

    def test_clear_with_confirmation(self, runner, mock_config):
        """Test clearing with confirmation."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            with patch("typer.confirm", return_value=True):
                result = runner.invoke(system_app, ["clear"])

                assert result.exit_code == 0
                assert "System instruction cleared" in result.stdout
                assert mock_config.tenet.system_instruction is None
                assert mock_config.tenet.system_instruction_enabled is False
                mock_config.save.assert_called_once()

    def test_clear_cancelled(self, runner, mock_config):
        """Test cancelling clear."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            with patch("typer.confirm", return_value=False):
                result = runner.invoke(system_app, ["clear"])

                assert result.exit_code == 0
                assert "Cancelled" in result.stdout
                assert mock_config.tenet.system_instruction == "You are a helpful coding assistant"

    def test_clear_forced(self, runner, mock_config):
        """Test forced clear without confirmation."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["clear", "--yes"])

            assert result.exit_code == 0
            assert "System instruction cleared" in result.stdout
            assert mock_config.tenet.system_instruction is None


class TestSystemInstructionTest:
    """Test testing system instruction injection."""

    def test_test_injection(self, runner, mock_config):
        """Test injection on sample content."""
        mock_tenets = MagicMock()
        mock_tenets.instiller = MagicMock()
        mock_tenets.instiller.inject_system_instruction.return_value = (
            "# System Instruction\n\nYou are a helpful coding assistant\n\n# Sample Content",
            {
                "system_instruction_injected": True,
                "system_instruction_position": "top",
                "token_increase": 20,
            },
        )

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.system_instruction.Tenets", return_value=mock_tenets):
                result = runner.invoke(system_app, ["test"])

                assert result.exit_code == 0
                assert "System instruction injected" in result.stdout
                assert "Position: top" in result.stdout
                assert "Token increase: 20" in result.stdout
                assert "Modified Content:" in result.stdout

    def test_test_with_session(self, runner, mock_config):
        """Test injection with specific session."""
        mock_tenets = MagicMock()
        mock_tenets.instiller = MagicMock()
        mock_tenets.instiller.inject_system_instruction.return_value = (
            "Modified content",
            {"system_instruction_injected": True},
        )

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.system_instruction.Tenets", return_value=mock_tenets):
                result = runner.invoke(system_app, ["test", "--session", "test-session"])

                assert result.exit_code == 0
                mock_tenets.instiller.inject_system_instruction.assert_called_once()
                call_args = mock_tenets.instiller.inject_system_instruction.call_args[1]
                assert call_args["session"] == "test-session"

    def test_test_no_instruction(self, runner, mock_config):
        """Test when no instruction configured."""
        mock_config.tenet.system_instruction = None

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["test"])

            assert result.exit_code == 0
            assert "No system instruction configured" in result.stdout

    def test_test_injection_skipped(self, runner, mock_config):
        """Test when injection is skipped."""
        mock_tenets = MagicMock()
        mock_tenets.instiller = MagicMock()
        mock_tenets.instiller.inject_system_instruction.return_value = (
            "Unchanged content",
            {"system_instruction_injected": False, "reason": "Already injected in session"},
        )

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            with patch("tenets.cli.commands.system_instruction.Tenets", return_value=mock_tenets):
                result = runner.invoke(system_app, ["test"])

                assert result.exit_code == 0
                assert (
                    "System instruction not injected: Already injected in session" in result.stdout
                )


class TestSystemInstructionExport:
    """Test exporting system instruction."""

    def test_export_to_file(self, runner, mock_config, tmp_path):
        """Test exporting to file."""
        output_file = tmp_path / "system_prompt.txt"

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["export", str(output_file)])

            assert result.exit_code == 0
            assert f"Exported to {output_file}" in result.stdout
            assert "Size: 34 characters" in result.stdout
            assert output_file.exists()
            assert output_file.read_text() == "You are a helpful coding assistant"

    def test_export_creates_parent_dirs(self, runner, mock_config, tmp_path):
        """Test export creates parent directories."""
        output_file = tmp_path / "prompts" / "system" / "main.txt"

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["export", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()
            assert output_file.parent.exists()

    def test_export_no_instruction(self, runner, mock_config, tmp_path):
        """Test export when no instruction exists."""
        mock_config.tenet.system_instruction = None
        output_file = tmp_path / "prompt.txt"

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["export", str(output_file)])

            assert result.exit_code == 0
            assert "No system instruction to export" in result.stdout
            assert not output_file.exists()


class TestSystemInstructionValidate:
    """Test validating system instruction."""

    def test_validate_valid_instruction(self, runner, mock_config):
        """Test validating a valid instruction."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["validate"])

            assert result.exit_code == 0
            assert "System instruction is valid" in result.stdout
            assert "Length: 31 characters" in result.stdout
            assert "Lines: 1" in result.stdout
            assert "Format: markdown" in result.stdout

    def test_validate_with_token_check(self, runner, mock_config):
        """Test validation with token count check."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["validate", "--tokens", "--max-tokens", "100"])

            assert result.exit_code == 0
            assert "Token Estimate:" in result.stdout

    def test_validate_too_long(self, runner, mock_config):
        """Test validation when instruction is too long."""
        mock_config.tenet.system_instruction = "x" * 6000

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["validate"])

            assert result.exit_code == 0
            assert "Warnings:" in result.stdout
            assert "Instruction is quite long" in result.stdout

    def test_validate_too_short(self, runner, mock_config):
        """Test validation when instruction is too short."""
        mock_config.tenet.system_instruction = "Hi"

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["validate"])

            assert result.exit_code == 1
            assert "Issues found:" in result.stdout
            assert "Instruction seems too short" in result.stdout

    def test_validate_format_mismatch(self, runner, mock_config):
        """Test validation when format doesn't match content."""
        mock_config.tenet.system_instruction = "Plain text instruction"
        mock_config.tenet.system_instruction_format = "xml"

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["validate"])

            assert result.exit_code == 0
            assert "Warnings:" in result.stdout
            assert "Format is 'xml' but instruction doesn't contain XML tags" in result.stdout

    def test_validate_no_instruction(self, runner, mock_config):
        """Test validation when no instruction exists."""
        mock_config.tenet.system_instruction = None

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["validate"])

            assert result.exit_code == 0
            assert "No system instruction to validate" in result.stdout


class TestSystemInstructionEdit:
    """Test editing system instruction."""

    @patch("subprocess.call")
    @patch("tempfile.NamedTemporaryFile")
    def test_edit_instruction(self, mock_tempfile, mock_subprocess, runner, mock_config):
        """Test editing instruction in editor."""
        # Mock the temporary file
        mock_file = MagicMock()
        mock_file.name = "/tmp/test.md"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        # Mock reading edited content
        with patch("builtins.open", mock_open(read_data="Edited instruction")):
            with patch("os.unlink"):
                with patch(
                    "tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config
                ):
                    result = runner.invoke(system_app, ["edit"])

                    assert result.exit_code == 0
                    assert "System instruction updated" in result.stdout
                    assert mock_config.tenet.system_instruction == "Edited instruction"
                    mock_subprocess.assert_called_once()

    @patch("subprocess.call")
    @patch("tempfile.NamedTemporaryFile")
    def test_edit_with_custom_editor(self, mock_tempfile, mock_subprocess, runner, mock_config):
        """Test editing with custom editor."""
        mock_file = MagicMock()
        mock_file.name = "/tmp/test.md"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        with patch("builtins.open", mock_open(read_data="New")):
            with patch("os.unlink"):
                with patch(
                    "tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config
                ):
                    result = runner.invoke(system_app, ["edit", "--editor", "vim"])

                    assert result.exit_code == 0
                    mock_subprocess.assert_called_once_with(["vim", "/tmp/test.md"])

    @patch("subprocess.call")
    @patch("tempfile.NamedTemporaryFile")
    def test_edit_no_changes(self, mock_tempfile, mock_subprocess, runner, mock_config):
        """Test when no changes made during edit."""
        mock_file = MagicMock()
        mock_file.name = "/tmp/test.md"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        # Return same content
        with patch("builtins.open", mock_open(read_data="You are a helpful coding assistant")):
            with patch("os.unlink"):
                with patch(
                    "tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config
                ):
                    result = runner.invoke(system_app, ["edit"])

                    assert result.exit_code == 0
                    assert "No changes made" in result.stdout
                    mock_config.save.assert_not_called()


class TestSystemInstructionErrorHandling:
    """Test error handling scenarios."""

    def test_set_error(self, runner, mock_config):
        """Test error during set operation."""
        mock_config.save.side_effect = Exception("Save failed")

        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            result = runner.invoke(system_app, ["set", "Test"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Save failed" in result.stdout

    def test_show_error(self, runner):
        """Test error during show operation."""
        with patch(
            "tenets.cli.commands.system_instruction.TenetsConfig",
            side_effect=Exception("Config error"),
        ):
            result = runner.invoke(system_app, ["show"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Config error" in result.stdout

    def test_export_error(self, runner, mock_config):
        """Test error during export."""
        with patch("tenets.cli.commands.system_instruction.TenetsConfig", return_value=mock_config):
            with patch("pathlib.Path.write_text", side_effect=Exception("Write failed")):
                result = runner.invoke(system_app, ["export", "output.txt"])

                assert result.exit_code == 1
                assert "Error:" in result.stdout
                assert "Write failed" in result.stdout

    def test_validate_error(self, runner):
        """Test error during validation."""
        with patch(
            "tenets.cli.commands.system_instruction.TenetsConfig",
            side_effect=Exception("Validation error"),
        ):
            result = runner.invoke(system_app, ["validate"])

            assert result.exit_code == 1
            assert "Error:" in result.stdout
            assert "Validation error" in result.stdout

            class TestSystemInstructionAppExport:
                def test_app_alias_exports_system_app(self):
                    # Import locally to avoid modifying top-level imports
                    assert app is sys_app

            class TestSystemInstructionCLIRegistration:
                def test_help_lists_all_commands(self, runner):
                    result = runner.invoke(system_app, ["--help"])
                    assert result.exit_code == 0
                    # Ensure all subcommands are present in help
                    for cmd in ["set", "show", "clear", "test", "export", "validate", "edit"]:
                        assert cmd in result.stdout

            class TestSystemInstructionTestCommandPatchedTenetsModule:
                def test_test_injection_with_patching_tenets_module(self, runner, mock_config):
                    # Patch tenets.Tenets because function does a local 'from tenets import Tenets'
                    mock_tenets = MagicMock()
                    mock_tenets.instiller = MagicMock()
                    mock_tenets.instiller.inject_system_instruction.return_value = (
                        "# System Instruction\n\nYou are a helpful coding assistant\n\n# Sample Content",
                        {
                            "system_instruction_injected": True,
                            "system_instruction_position": "top",
                            "token_increase": 20,
                        },
                    )
                    with patch(
                        "tenets.cli.commands.system_instruction.TenetsConfig",
                        return_value=mock_config,
                    ):
                        with patch("tenets.Tenets", return_value=mock_tenets):
                            result = runner.invoke(system_app, ["test"])
                            assert result.exit_code == 0
                            assert "System instruction injected" in result.stdout
                            assert "Position: top" in result.stdout
                            assert "Token increase: 20" in result.stdout

                def test_test_injection_skipped_with_patching_tenets_module(
                    self, runner, mock_config
                ):
                    mock_tenets = MagicMock()
                    mock_tenets.instiller = MagicMock()
                    mock_tenets.instiller.inject_system_instruction.return_value = (
                        "Unchanged content",
                        {
                            "system_instruction_injected": False,
                            "reason": "Already injected in session",
                        },
                    )
                    with patch(
                        "tenets.cli.commands.system_instruction.TenetsConfig",
                        return_value=mock_config,
                    ):
                        with patch("tenets.Tenets", return_value=mock_tenets):
                            result = runner.invoke(system_app, ["test"])
                            assert result.exit_code == 0
                            assert (
                                "System instruction not injected: Already injected in session"
                                in result.stdout
                            )

                def test_test_with_session_argument_patched(self, runner, mock_config):
                    mock_tenets = MagicMock()
                    mock_tenets.instiller = MagicMock()
                    mock_tenets.instiller.inject_system_instruction.return_value = (
                        "Modified content",
                        {"system_instruction_injected": True},
                    )
                    with patch(
                        "tenets.cli.commands.system_instruction.TenetsConfig",
                        return_value=mock_config,
                    ):
                        with patch("tenets.Tenets", return_value=mock_tenets):
                            result = runner.invoke(system_app, ["test", "--session", "sess-123"])
                            assert result.exit_code == 0
                            mock_tenets.instiller.inject_system_instruction.assert_called_once()
                            kwargs = mock_tenets.instiller.inject_system_instruction.call_args[1]
                            assert kwargs.get("session") == "sess-123"

            class TestSystemInstructionOutputFlex:
                def test_export_size_uses_actual_length(self, runner, mock_config, tmp_path):
                    # Compute true length of the configured instruction to avoid brittle expectations
                    expected_len = len(mock_config.tenet.system_instruction or "")
                    out = tmp_path / "prompt.txt"
                    with patch(
                        "tenets.cli.commands.system_instruction.TenetsConfig",
                        return_value=mock_config,
                    ):
                        result = runner.invoke(system_app, ["export", str(out)])
                        assert result.exit_code == 0
                        assert out.exists()
                        # Look for the size line with the computed length
                        assert f"Size: {expected_len} characters" in result.stdout

                def test_validate_reports_length_and_lines(self, runner, mock_config):
                    # Validate dynamic length/lines against current config content
                    expected_len = len(mock_config.tenet.system_instruction or "")
                    expected_lines = (
                        (mock_config.tenet.system_instruction or "").count("\n") + 1
                        if mock_config.tenet.system_instruction
                        else 1
                    )
                    with patch(
                        "tenets.cli.commands.system_instruction.TenetsConfig",
                        return_value=mock_config,
                    ):
                        result = runner.invoke(system_app, ["validate"])
                        assert result.exit_code == 0
                        # Ensure the validation panel includes computed metrics
                        assert f"Length: {expected_len} characters" in result.stdout

                        class TestSystemInstructionAppAndHelp:
                            def test_app_alias_exports_system_app(self):
                                assert app is sys_app

                            def test_help_lists_all_commands(self, runner):
                                result = runner.invoke(system_app, ["--help"])
                                assert result.exit_code == 0
                                for cmd in [
                                    "set",
                                    "show",
                                    "clear",
                                    "test",
                                    "export",
                                    "validate",
                                    "edit",
                                ]:
                                    assert cmd in result.stdout

                        class TestSystemInstructionTestCommandPatchedTenetsModule:
                            def test_test_injection_with_patching_tenets_module(
                                self, runner, mock_config
                            ):
                                mock_tenets = MagicMock()
                                mock_tenets.instiller = MagicMock()
                                mock_tenets.instiller.inject_system_instruction.return_value = (
                                    "# System Instruction\n\nYou are a helpful coding assistant\n\n# Sample Content",
                                    {
                                        "system_instruction_injected": True,
                                        "system_instruction_position": "top",
                                        "token_increase": 20,
                                    },
                                )
                                with patch(
                                    "tenets.cli.commands.system_instruction.TenetsConfig",
                                    return_value=mock_config,
                                ):
                                    # Patch where the function imports it: from tenets import Tenets
                                    with patch("tenets.Tenets", return_value=mock_tenets):
                                        result = runner.invoke(system_app, ["test"])
                                        assert result.exit_code == 0
                                        assert "System instruction injected" in result.stdout
                                        assert "Position: top" in result.stdout
                                        assert "Token increase: 20" in result.stdout
                                        assert "Modified Content:" in result.stdout

                            def test_test_injection_skipped_with_patching_tenets_module(
                                self, runner, mock_config
                            ):
                                mock_tenets = MagicMock()
                                mock_tenets.instiller = MagicMock()
                                mock_tenets.instiller.inject_system_instruction.return_value = (
                                    "Unchanged content",
                                    {
                                        "system_instruction_injected": False,
                                        "reason": "Already injected in session",
                                    },
                                )
                                with patch(
                                    "tenets.cli.commands.system_instruction.TenetsConfig",
                                    return_value=mock_config,
                                ):
                                    with patch("tenets.Tenets", return_value=mock_tenets):
                                        result = runner.invoke(system_app, ["test"])
                                        assert result.exit_code == 0
                                        assert (
                                            "System instruction not injected: Already injected in session"
                                            in result.stdout
                                        )

                            def test_test_with_session_argument_patched(self, runner, mock_config):
                                mock_tenets = MagicMock()
                                mock_tenets.instiller = MagicMock()
                                mock_tenets.instiller.inject_system_instruction.return_value = (
                                    "Modified content",
                                    {"system_instruction_injected": True},
                                )
                                with patch(
                                    "tenets.cli.commands.system_instruction.TenetsConfig",
                                    return_value=mock_config,
                                ):
                                    with patch("tenets.Tenets", return_value=mock_tenets):
                                        result = runner.invoke(
                                            system_app, ["test", "--session", "sess-123"]
                                        )
                                        assert result.exit_code == 0
                                        mock_tenets.instiller.inject_system_instruction.assert_called_once()
                                        kwargs = mock_tenets.instiller.inject_system_instruction.call_args[
                                            1
                                        ]
                                        assert kwargs.get("session") == "sess-123"

                        class TestSystemInstructionOutputFlex:
                            def test_export_size_uses_actual_length(
                                self, runner, mock_config, tmp_path
                            ):
                                expected_len = len(mock_config.tenet.system_instruction or "")
                                out = tmp_path / "prompt.txt"
                                with patch(
                                    "tenets.cli.commands.system_instruction.TenetsConfig",
                                    return_value=mock_config,
                                ):
                                    result = runner.invoke(system_app, ["export", str(out)])
                                    assert result.exit_code == 0
                                    assert out.exists()
                                    assert f"Size: {expected_len} characters" in result.stdout

                            def test_validate_reports_length_and_lines(self, runner, mock_config):
                                text = mock_config.tenet.system_instruction or ""
                                expected_len = len(text)
                                expected_lines = text.count("\n") + 1 if text else 1
                                with patch(
                                    "tenets.cli.commands.system_instruction.TenetsConfig",
                                    return_value=mock_config,
                                ):
                                    result = runner.invoke(system_app, ["validate"])
                                    assert result.exit_code == 0
                                    assert f"Length: {expected_len} characters" in result.stdout
                                    assert f"Lines: {expected_lines}" in result.stdout
