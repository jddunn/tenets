"""Tests for the main CLI app, including --version flag and app initialization."""

import pytest
from typer.testing import CliRunner

from tenets import __version__
from tenets.cli.app import app

# Skip tests if typer is not available
pytest.importorskip("typer")


def test_version_flag():
    """Test that --version flag shows version and exits."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])

    # Print debug info if test fails
    if result.exit_code != 0:
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
        if result.exception:
            print(f"Exception: {result.exception}")

    assert result.exit_code == 0
    assert f"tenets v{__version__}" in result.output
    # Should exit immediately, not show help
    assert "Usage:" not in result.output


def test_version_command():
    """Test that version command still works."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert f"tenets v{__version__}" in result.output or "Tenets" in result.output


def test_version_flag_with_other_commands():
    """Test that --version takes precedence over other commands."""
    runner = CliRunner()
    result = runner.invoke(app, ["--version", "distill", "test"])

    assert result.exit_code == 0
    assert f"tenets v{__version__}" in result.output
    # Should not execute distill command
    assert "distill" not in result.output.lower() or "tenets v" in result.output


def test_version_verbose():
    """Test version command with --verbose flag."""
    runner = CliRunner()
    result = runner.invoke(app, ["version", "--verbose"])

    assert result.exit_code == 0
    # Verbose version should show more info
    assert "Tenets" in result.output
    assert __version__ in result.output
    # Should show features
    assert "Features:" in result.output or "Context" in result.output
