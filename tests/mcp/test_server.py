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

# Python 3.14+ has an issue with importlib.util.find_spec for some packages
# Skip distill/rank tests that trigger the deep import chain on 3.14
PYTHON_314_PLUS = sys.version_info >= (3, 14)
skip_on_314 = pytest.mark.skipif(
    PYTHON_314_PLUS, reason="Python 3.14 has importlib.util.find_spec compatibility issues"
)


class MockFastMCP:
    """Mock FastMCP server for testing without MCP SDK installed."""

    def __init__(self, name: str):
        self.name = name
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}
        self.last_run: dict[str, Any] | None = None
        self.runs: list[dict[str, Any]] = []

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
        """Mock run method that records invocation."""
        self.last_run = {"transport": transport, "host": host, "port": port}
        self.runs.append(self.last_run)


@pytest.fixture
def mock_mcp_module():
    """Mock the MCP module for testing."""
    mock_mcp = MagicMock()
    mock_mcp.server.fastmcp.FastMCP = MockFastMCP

    with patch.dict(
        sys.modules,
        {
            "mcp": mock_mcp,
            "mcp.server": mock_mcp.server,
            "mcp.server.fastmcp": mock_mcp.server.fastmcp,
        },
    ):
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

    @skip_on_314
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

    @skip_on_314
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
    async def test_session_state_resource(self, mcp_server, tmp_path):
        """Test the dynamic session state resource returns expected fields."""
        # Create a session
        create_tool = mcp_server._mcp.tools["session_create"]
        await create_tool(name="state-session", description="State test")

        # Access the resource with path parameter
        resource_func = mcp_server._mcp.resources["tenets://sessions/{name}/state"]
        result = await resource_func(name="state-session")

        assert isinstance(result, str)
        data = json.loads(result)
        assert data["name"] == "state-session"
        assert "created_at" in data
        assert "metadata" in data

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
        mcp_server.run(transport="stdio")
        assert mcp_server._mcp.last_run == {"transport": "stdio", "host": "127.0.0.1", "port": 8080}

    def test_run_sse_transport(self, mcp_server):
        """Test running with SSE transport."""
        mcp_server.run(transport="sse", host="127.0.0.1", port=8080)
        assert mcp_server._mcp.last_run == {"transport": "sse", "host": "127.0.0.1", "port": 8080}

    def test_run_http_transport(self, mcp_server):
        """Test running with HTTP transport."""
        mcp_server.run(transport="http", host="127.0.0.1", port=8081)
        assert mcp_server._mcp.last_run == {
            "transport": "streamable-http",
            "host": "127.0.0.1",
            "port": 8081,
        }

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
        # Updated docstring check
        assert "context" in tool.__doc__.lower()

    def test_rank_files_has_docstring(self, mcp_server):
        """Test that rank_files tool has a proper docstring."""
        tool = mcp_server._mcp.tools["rank_files"]
        assert tool.__doc__ is not None
        assert "relevant" in tool.__doc__.lower()

    def test_all_tools_have_docstrings(self, mcp_server):
        """Test that all tools have docstrings."""
        for name, tool in mcp_server._mcp.tools.items():
            assert tool.__doc__ is not None, f"Tool {name} missing docstring"
            assert len(tool.__doc__) > 20, f"Tool {name} has too short docstring"

    def test_docstrings_have_args_section(self, mcp_server):
        """Test that tool docstrings include Args documentation."""
        key_tools = ["distill", "rank_files", "tenet_add", "session_create"]
        for name in key_tools:
            tool = mcp_server._mcp.tools[name]
            assert "Args:" in tool.__doc__, f"Tool {name} missing Args section"

    def test_docstrings_have_returns_section(self, mcp_server):
        """Test that tool docstrings include Returns documentation."""
        key_tools = ["distill", "rank_files", "tenet_add"]
        for name in key_tools:
            tool = mcp_server._mcp.tools[name]
            # Returns can be in docstring or as Return:
            assert "return" in tool.__doc__.lower(), f"Tool {name} missing Returns section"


class TestMCPSessionPersistence:
    """Tests for session persistence across operations."""

    @pytest.mark.asyncio
    async def test_session_persists_across_operations(self, mcp_server, tmp_path):
        """Test that session data persists correctly."""
        # Create session
        create_tool = mcp_server._mcp.tools["session_create"]
        result = await create_tool(name="persist-test", description="Persistence test")
        session_id = result["id"]

        # Verify it appears in list
        list_tool = mcp_server._mcp.tools["session_list"]
        sessions = await list_tool()
        session_names = [s["name"] for s in sessions["sessions"]]
        assert "persist-test" in session_names

    @pytest.mark.asyncio
    async def test_pinned_files_persist(self, mcp_server, tmp_path):
        """Test that pinned files are remembered."""
        # Create a test file
        test_file = tmp_path / "pinned.py"
        test_file.write_text("# Important file")

        # Create session
        create_tool = mcp_server._mcp.tools["session_create"]
        await create_tool(name="pin-test")

        # Pin file
        pin_tool = mcp_server._mcp.tools["session_pin_file"]
        result = await pin_tool(session="pin-test", file_path=str(test_file))
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self, mcp_server):
        """Test that multiple sessions are isolated from each other."""
        create_tool = mcp_server._mcp.tools["session_create"]

        # Create two sessions
        await create_tool(name="session-a", description="Session A")
        await create_tool(name="session-b", description="Session B")

        # Verify both exist
        list_tool = mcp_server._mcp.tools["session_list"]
        sessions = await list_tool()
        session_names = [s["name"] for s in sessions["sessions"]]
        assert "session-a" in session_names
        assert "session-b" in session_names


class TestMCPTenetOperations:
    """Tests for tenet CRUD operations."""

    @pytest.mark.asyncio
    async def test_tenet_priority_levels(self, mcp_server):
        """Test that all priority levels work."""
        add_tool = mcp_server._mcp.tools["tenet_add"]

        for priority in ["low", "medium", "high", "critical"]:
            result = await add_tool(
                content=f"Test tenet with {priority} priority",
                priority=priority,
            )
            assert "id" in result

    @pytest.mark.asyncio
    async def test_tenet_categories(self, mcp_server):
        """Test that categories are stored correctly."""
        add_tool = mcp_server._mcp.tools["tenet_add"]

        result = await add_tool(
            content="Always use HTTPS",
            priority="high",
            category="security",
        )

        assert result["category"] == "security"

    @pytest.mark.asyncio
    async def test_tenet_instill(self, mcp_server):
        """Test the tenet instill functionality."""
        # Add a pending tenet
        add_tool = mcp_server._mcp.tools["tenet_add"]
        await add_tool(content="Pending tenet for instill test")

        # Instill it
        instill_tool = mcp_server._mcp.tools["tenet_instill"]
        result = await instill_tool()

        assert isinstance(result, dict)


class TestMCPErrorHandling:
    """Tests for error handling in MCP tools."""

    @skip_on_314
    @pytest.mark.asyncio
    async def test_distill_with_nonexistent_path(self, mcp_server):
        """Test distill handles nonexistent paths gracefully."""
        distill_tool = mcp_server._mcp.tools["distill"]

        # Should not crash, but may return empty or error
        try:
            result = await distill_tool(
                prompt="test",
                path="/nonexistent/path/that/does/not/exist",
            )
            # If it doesn't raise, result should indicate no files
            assert isinstance(result, dict)
        except Exception as e:
            # Some error handling is acceptable
            assert "not found" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.asyncio
    async def test_session_pin_nonexistent_file(self, mcp_server):
        """Test pinning a nonexistent file."""
        create_tool = mcp_server._mcp.tools["session_create"]
        await create_tool(name="error-test-session")

        pin_tool = mcp_server._mcp.tools["session_pin_file"]
        result = await pin_tool(
            session="error-test-session",
            file_path="/nonexistent/file.py",
        )

        # Should return failure indicator
        assert isinstance(result, dict)


class TestMCPParameterValidation:
    """Tests for parameter validation in MCP tools."""

    @skip_on_314
    @pytest.mark.asyncio
    async def test_distill_mode_values(self, mcp_server, tmp_path):
        """Test that distill accepts valid mode values."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        distill_tool = mcp_server._mcp.tools["distill"]

        for mode in ["fast", "balanced", "thorough"]:
            result = await distill_tool(
                prompt="test",
                path=str(tmp_path),
                mode=mode,
                max_tokens=100,
            )
            assert isinstance(result, dict)

    @skip_on_314
    @pytest.mark.asyncio
    async def test_distill_format_values(self, mcp_server, tmp_path):
        """Test that distill accepts valid format values."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        distill_tool = mcp_server._mcp.tools["distill"]

        for fmt in ["markdown", "xml", "json"]:
            result = await distill_tool(
                prompt="test",
                path=str(tmp_path),
                format=fmt,
                max_tokens=100,
            )
            assert isinstance(result, dict)

    @skip_on_314
    @pytest.mark.asyncio
    async def test_rank_files_top_n_parameter(self, mcp_server, tmp_path):
        """Test that rank_files respects top_n parameter."""
        # Create multiple test files
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"def func{i}(): pass")

        rank_tool = mcp_server._mcp.tools["rank_files"]
        result = await rank_tool(
            prompt="function",
            path=str(tmp_path),
            top_n=3,
        )

        assert len(result["files"]) <= 3


class TestMCPIntegration:
    """Integration tests for MCP tool workflows."""

    @skip_on_314
    @pytest.mark.asyncio
    async def test_full_workflow_session_to_distill(self, mcp_server, tmp_path):
        """Test a complete workflow: create session, pin files, distill."""
        # Create test files
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def util(): pass")

        # Create session
        create_tool = mcp_server._mcp.tools["session_create"]
        session_result = await create_tool(name="workflow-test")
        assert session_result["name"] == "workflow-test"

        # Pin a file
        pin_tool = mcp_server._mcp.tools["session_pin_file"]
        pin_result = await pin_tool(
            session="workflow-test",
            file_path=str(tmp_path / "main.py"),
        )
        assert pin_result["success"] is True

        # Distill with session
        distill_tool = mcp_server._mcp.tools["distill"]
        distill_result = await distill_tool(
            prompt="find main function",
            path=str(tmp_path),
            session="workflow-test",
            max_tokens=1000,
        )
        assert isinstance(distill_result, dict)
        # Pinned file should appear in files list
        files = distill_result.get("files", [])
        assert any("main.py" in str(p) for p in files)

    @pytest.mark.asyncio
    async def test_tenet_workflow(self, mcp_server):
        """Test tenet workflow: add, list, instill."""
        # Add tenet
        add_tool = mcp_server._mcp.tools["tenet_add"]
        add_result = await add_tool(
            content="Always write tests",
            priority="high",
            category="quality",
        )
        tenet_id = add_result["id"]
        assert tenet_id is not None

        # List tenets
        list_tool = mcp_server._mcp.tools["tenet_list"]
        list_result = await list_tool()
        assert isinstance(list_result, dict)

        # Instill
        instill_tool = mcp_server._mcp.tools["tenet_instill"]
        instill_result = await instill_tool()
        assert isinstance(instill_result, dict)

    @skip_on_314
    @pytest.mark.asyncio
    async def test_include_exclude_patterns(self, mcp_server, tmp_path):
        """Test include/exclude patterns are respected by distill."""
        (tmp_path / "keep.md").write_text("# Doc")
        (tmp_path / "skip.py").write_text("print('skip')")

        distill_tool = mcp_server._mcp.tools["distill"]
        result = await distill_tool(
            prompt="docs",
            path=str(tmp_path),
            include_patterns=["*.md"],
            exclude_patterns=["*.py"],
            max_tokens=500,
        )
        files = [str(p) for p in result.get("files", [])]
        assert any("keep.md" in f for f in files)
        assert all("skip.py" not in f for f in files)


class TestMCPResourceContent:
    """Tests for MCP resource content format."""

    @pytest.mark.asyncio
    async def test_sessions_resource_json_format(self, mcp_server):
        """Test that sessions resource returns valid JSON."""
        # Create a session
        create_tool = mcp_server._mcp.tools["session_create"]
        await create_tool(name="json-test")

        resource_func = mcp_server._mcp.resources["tenets://sessions/list"]
        result = await resource_func()

        # Should be valid JSON
        data = json.loads(result)
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_config_resource_masks_secrets(self, mcp_server):
        """Test that config resource masks sensitive data."""
        resource_func = mcp_server._mcp.resources["tenets://config/current"]
        result = await resource_func()

        data = json.loads(result)

        # If there are API keys, they should be masked
        if "llm" in data and "api_keys" in data["llm"]:
            for key, value in data["llm"]["api_keys"].items():
                assert value == "***", f"API key {key} not masked"
