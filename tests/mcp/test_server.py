"""Tests for the Tenets MCP server.

These tests verify the MCP server functionality without requiring the
actual MCP SDK to be installed. Tests mock the MCP dependencies to
ensure the server logic works correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class MockFastMCP:
    """Mock FastMCP server for testing without MCP SDK installed."""

    def __init__(self, name: str):
        self.name = name
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self):
        """Decorator to register a tool."""

        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator

    def resource(self, uri: str):
        """Decorator to register a resource."""

        def decorator(func):
            self.resources[uri] = func
            return func

        return decorator

    def prompt(self):
        """Decorator to register a prompt."""

        def decorator(func):
            self.prompts[func.__name__] = func
            return func

        return decorator

    def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8080):
        """Mock run method."""
        pass


@pytest.fixture
def mock_mcp_module():
    """Mock the MCP module for testing."""
    mock_mcp = MagicMock()
    mock_mcp.server.fastmcp.FastMCP = MockFastMCP

    with patch.dict(sys.modules, {"mcp": mock_mcp, "mcp.server": mock_mcp.server, "mcp.server.fastmcp": mock_mcp.server.fastmcp}):
        # Reset the global availability check
        import tenets.mcp.server as server_module
        server_module._mcp_available = None
        yield mock_mcp


@pytest.fixture
def mcp_server(mock_mcp_module, tmp_path):
    """Create a TenetsMCP server instance for testing."""
    from tenets.config import TenetsConfig
    from tenets.mcp.server import TenetsMCP

    # Create config with temp cache directory
    config = TenetsConfig()
    config.cache.directory = tmp_path / "cache"
    config.cache.directory.mkdir(parents=True, exist_ok=True)
    config.tenet.storage_path = tmp_path / "tenets"
    config.tenet.storage_path.mkdir(parents=True, exist_ok=True)

    server = TenetsMCP(name="test-tenets", config=config, project_path=tmp_path)
    return server


class TestTenetsMCPInitialization:
    """Tests for TenetsMCP server initialization."""

    def test_server_creation(self, mcp_server):
        """Test server can be created with default settings."""
        assert mcp_server.name == "test-tenets"
        assert mcp_server._mcp is not None

    def test_server_has_tools_registered(self, mcp_server):
        """Test that tools are registered on the server."""
        tools = mcp_server._mcp.tools
        assert "distill" in tools
        assert "rank_files" in tools
        assert "examine" in tools
        assert "chronicle" in tools
        assert "momentum" in tools
        assert "session_create" in tools
        assert "session_list" in tools
        assert "session_pin_file" in tools
        assert "session_pin_folder" in tools
        assert "tenet_add" in tools
        assert "tenet_list" in tools
        assert "tenet_instill" in tools
        assert "set_system_instruction" in tools

    def test_server_has_resources_registered(self, mcp_server):
        """Test that resources are registered on the server."""
        resources = mcp_server._mcp.resources
        assert "tenets://sessions/list" in resources
        assert "tenets://tenets/list" in resources
        assert "tenets://config/current" in resources
        # Dynamic resource patterns
        assert any("sessions" in uri and "state" in uri for uri in resources)

    def test_server_has_prompts_registered(self, mcp_server):
        """Test that prompts are registered on the server."""
        prompts = mcp_server._mcp.prompts
        assert "build_context_for_task" in prompts
        assert "code_review_context" in prompts
        assert "understand_codebase" in prompts

    def test_tenets_lazy_loading(self, mcp_server):
        """Test that Tenets instance is lazily loaded."""
        # Initially should be None
        assert mcp_server._tenets is None

        # Accessing the property should load it
        tenets_instance = mcp_server.tenets
        assert tenets_instance is not None
        assert mcp_server._tenets is tenets_instance


class TestMCPTools:
    """Tests for MCP tool implementations."""

    @pytest.mark.asyncio
    async def test_distill_tool(self, mcp_server, tmp_path):
        """Test the distill tool."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        distill_tool = mcp_server._mcp.tools["distill"]
        result = await distill_tool(
            prompt="find hello function",
            path=str(tmp_path),
            mode="fast",
            max_tokens=1000,
        )

        assert isinstance(result, dict)
        assert "context" in result or "content" in result

    @pytest.mark.asyncio
    async def test_rank_files_tool(self, mcp_server, tmp_path):
        """Test the rank_files tool."""
        # Create test files
        (tmp_path / "auth.py").write_text("def authenticate(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        rank_tool = mcp_server._mcp.tools["rank_files"]
        result = await rank_tool(
            prompt="authentication",
            path=str(tmp_path),
            mode="fast",
            top_n=10,
        )

        assert isinstance(result, dict)
        assert "files" in result
        assert "total_scanned" in result

    @pytest.mark.asyncio
    async def test_session_create_tool(self, mcp_server):
        """Test the session_create tool."""
        session_tool = mcp_server._mcp.tools["session_create"]
        result = await session_tool(
            name="test-session",
            description="A test session",
        )

        assert isinstance(result, dict)
        assert "name" in result
        assert result["name"] == "test-session"

    @pytest.mark.asyncio
    async def test_session_list_tool(self, mcp_server):
        """Test the session_list tool."""
        # First create a session
        create_tool = mcp_server._mcp.tools["session_create"]
        await create_tool(name="list-test-session")

        # Then list sessions
        list_tool = mcp_server._mcp.tools["session_list"]
        result = await list_tool()

        assert isinstance(result, dict)
        assert "sessions" in result
        assert isinstance(result["sessions"], list)

    @pytest.mark.asyncio
    async def test_tenet_add_tool(self, mcp_server):
        """Test the tenet_add tool."""
        tenet_tool = mcp_server._mcp.tools["tenet_add"]
        result = await tenet_tool(
            content="Always validate user input",
            priority="high",
            category="security",
        )

        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert result["content"] == "Always validate user input"

    @pytest.mark.asyncio
    async def test_tenet_list_tool(self, mcp_server):
        """Test the tenet_list tool."""
        # First add a tenet
        add_tool = mcp_server._mcp.tools["tenet_add"]
        await add_tool(content="Test tenet")

        # Then list tenets
        list_tool = mcp_server._mcp.tools["tenet_list"]
        result = await list_tool()

        assert isinstance(result, dict)
        assert "tenets" in result

    @pytest.mark.asyncio
    async def test_set_system_instruction_tool(self, mcp_server):
        """Test the set_system_instruction tool."""
        instruction_tool = mcp_server._mcp.tools["set_system_instruction"]
        result = await instruction_tool(
            instruction="You are a helpful coding assistant.",
            position="top",
        )

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["position"] == "top"


class TestMCPResources:
    """Tests for MCP resource implementations."""

    @pytest.mark.asyncio
    async def test_sessions_list_resource(self, mcp_server):
        """Test the sessions list resource."""
        # Create a session first
        create_tool = mcp_server._mcp.tools["session_create"]
        await create_tool(name="resource-test-session")

        resource_func = mcp_server._mcp.resources["tenets://sessions/list"]
        result = await resource_func()

        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_tenets_list_resource(self, mcp_server):
        """Test the tenets list resource."""
        resource_func = mcp_server._mcp.resources["tenets://tenets/list"]
        result = await resource_func()

        assert isinstance(result, str)
        # Should be valid JSON
        json.loads(result)

    @pytest.mark.asyncio
    async def test_config_current_resource(self, mcp_server):
        """Test the config current resource."""
        resource_func = mcp_server._mcp.resources["tenets://config/current"]
        result = await resource_func()

        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)
        # API keys should be masked
        if "llm" in data and "api_keys" in data["llm"]:
            for key in data["llm"]["api_keys"].values():
                assert key == "***"


class TestMCPPrompts:
    """Tests for MCP prompt implementations."""

    def test_build_context_for_task_prompt(self, mcp_server):
        """Test the build_context_for_task prompt."""
        prompt_func = mcp_server._mcp.prompts["build_context_for_task"]
        result = prompt_func(
            task="implement user authentication",
            focus_areas="auth, security",
        )

        assert isinstance(result, str)
        assert "implement user authentication" in result
        assert "auth, security" in result

    def test_code_review_context_prompt(self, mcp_server):
        """Test the code_review_context prompt."""
        prompt_func = mcp_server._mcp.prompts["code_review_context"]
        result = prompt_func(
            scope="recent",
            focus="security",
        )

        assert isinstance(result, str)
        assert "security" in result

    def test_understand_codebase_prompt(self, mcp_server):
        """Test the understand_codebase prompt."""
        prompt_func = mcp_server._mcp.prompts["understand_codebase"]
        result = prompt_func(
            depth="overview",
            area="authentication",
        )

        assert isinstance(result, str)
        assert "authentication" in result


class TestMCPServerRun:
    """Tests for MCP server run functionality."""

    def test_run_stdio_transport(self, mcp_server):
        """Test running with stdio transport."""
        # Should not raise
        mcp_server.run(transport="stdio")

    def test_run_sse_transport(self, mcp_server):
        """Test running with SSE transport."""
        # Should not raise
        mcp_server.run(transport="sse", host="127.0.0.1", port=8080)

    def test_run_http_transport(self, mcp_server):
        """Test running with HTTP transport."""
        # Should not raise
        mcp_server.run(transport="http", host="127.0.0.1", port=8081)

    def test_run_invalid_transport(self, mcp_server):
        """Test running with invalid transport raises error."""
        with pytest.raises(ValueError, match="Unknown transport"):
            mcp_server.run(transport="invalid")


class TestMCPServerFactory:
    """Tests for MCP server factory functions."""

    def test_create_server(self, mock_mcp_module, tmp_path):
        """Test create_server factory function."""
        from tenets.config import TenetsConfig
        from tenets.mcp.server import create_server

        config = TenetsConfig()
        config.cache.directory = tmp_path / "cache"
        config.cache.directory.mkdir(parents=True, exist_ok=True)

        server = create_server(name="factory-test", config=config)
        assert server.name == "factory-test"


class TestMCPWithoutDependencies:
    """Tests for behavior when MCP dependencies are not installed."""

    def test_import_error_without_mcp(self, tmp_path):
        """Test that ImportError is raised when MCP is not installed."""
        import tenets.mcp.server as server_module

        # Reset availability check
        server_module._mcp_available = None

        # Mock ImportError for mcp
        with patch.dict(sys.modules, {"mcp": None}):
            # Force re-check
            server_module._mcp_available = None

            def raise_import_error(*args, **kwargs):
                raise ImportError("No module named 'mcp'")

            with patch("builtins.__import__", side_effect=raise_import_error):
                server_module._mcp_available = None
                assert server_module._check_mcp_available() is False


class TestMCPToolDocstrings:
    """Tests to verify tool docstrings are present and informative."""

    def test_distill_has_docstring(self, mcp_server):
        """Test that distill tool has a proper docstring."""
        tool = mcp_server._mcp.tools["distill"]
        assert tool.__doc__ is not None
        assert "Distill" in tool.__doc__
        assert "context" in tool.__doc__.lower()

    def test_rank_files_has_docstring(self, mcp_server):
        """Test that rank_files tool has a proper docstring."""
        tool = mcp_server._mcp.tools["rank_files"]
        assert tool.__doc__ is not None
        assert "Rank" in tool.__doc__

    def test_all_tools_have_docstrings(self, mcp_server):
        """Test that all tools have docstrings."""
        for name, tool in mcp_server._mcp.tools.items():
            assert tool.__doc__ is not None, f"Tool {name} missing docstring"
            assert len(tool.__doc__) > 20, f"Tool {name} has too short docstring"

