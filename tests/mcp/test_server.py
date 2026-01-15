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
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Python 3.14+ has an issue with importlib.util.find_spec for some packages
PYTHON_314_PLUS = sys.version_info >= (3, 14)


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


class MockContextResult:
    """Mock ContextResult for testing."""

    def __init__(self, context: str = "", files: list = None, metadata: dict = None):
        self.context = context
        self.files = files or []
        self.files_summarized = []
        self.token_count = len(context) // 4
        self.metadata = metadata or {"mode": "fast", "total_scanned": 10}

    def to_dict(self) -> dict:
        return {
            "context": self.context,
            "files": [str(f) for f in self.files],
            "files_summarized": self.files_summarized,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


class MockRankResult:
    """Mock RankResult for testing."""

    def __init__(self, files: list = None, mode: str = "fast", total_scanned: int = 10):
        self.files = files or []
        self.mode = mode
        self.total_scanned = total_scanned


class MockRankedFile:
    """Mock ranked file for testing."""

    def __init__(self, path: str, score: float = 0.5, factors: dict = None):
        self.path = Path(path)
        self.relevance_score = score
        self.ranking_factors = factors or {"keyword": 0.3, "path": 0.2}


@pytest.fixture(autouse=True)
def reset_mcp_singleton():
    """Reset the MCP singleton before and after each test for isolation."""
    try:
        import tenets.mcp.server as server_module

        # Reset before test
        server_module._mcp_instance = None
        yield
        # Reset after test
        server_module._mcp_instance = None
    except ImportError:
        yield


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

    # Reset singleton before creating server for test isolation
    TenetsMCP.reset_instance()

    # Create config with temp cache directory
    config = TenetsConfig()
    config.cache.directory = tmp_path / "cache"
    config.cache.directory.mkdir(parents=True, exist_ok=True)
    config.tenet.storage_path = tmp_path / "tenets"
    config.tenet.storage_path.mkdir(parents=True, exist_ok=True)

    server = TenetsMCP(name="test-tenets", config=config, project_path=tmp_path)
    yield server

    # Clean up singleton after test
    TenetsMCP.reset_instance()


@pytest.fixture
def mcp_server_with_mock_tenets(mock_mcp_module, tmp_path):
    """Create a TenetsMCP server with mocked Tenets instance for Py3.14 compat."""
    from tenets.config import TenetsConfig
    from tenets.mcp.server import TenetsMCP

    # Reset singleton before creating server for test isolation
    TenetsMCP.reset_instance()

    config = TenetsConfig()
    config.cache.directory = tmp_path / "cache"
    config.cache.directory.mkdir(parents=True, exist_ok=True)
    config.tenet.storage_path = tmp_path / "tenets"
    config.tenet.storage_path.mkdir(parents=True, exist_ok=True)

    server = TenetsMCP(name="test-tenets", config=config, project_path=tmp_path)

    # Create mock Tenets instance
    mock_tenets = MagicMock()
    mock_tenets.config = config

    # Track calls for assertions
    distill_calls: list[dict[str, Any]] = []

    # Mock distill to return MockContextResult
    def mock_distill(**kwargs):
        distill_calls.append(kwargs)
        return MockContextResult(
            context="# Test context\ndef hello(): pass",
            files=[tmp_path / "test.py"],
            metadata={"mode": kwargs.get("mode", "fast"), "total_scanned": 5},
        )

    mock_tenets.distill = mock_distill

    # Mock rank_files to return MockRankResult
    def mock_rank_files(**kwargs):
        return MockRankResult(
            files=[
                MockRankedFile(str(tmp_path / "auth.py"), 0.85),
                MockRankedFile(str(tmp_path / "utils.py"), 0.65),
            ],
            mode=kwargs.get("mode", "fast"),
            total_scanned=10,
        )

    mock_tenets.rank_files = mock_rank_files

    # Mock session operations
    mock_tenets.add_file_to_session = MagicMock(return_value=True)
    mock_tenets.add_folder_to_session = MagicMock(return_value=2)

    # Inject mock
    server._tenets = mock_tenets
    server._tenets.distill_calls = distill_calls

    yield server

    # Clean up singleton after test
    from tenets.mcp.server import TenetsMCP

    TenetsMCP.reset_instance()


class TestTenetsMCPInitialization:
    """Tests for TenetsMCP server initialization."""

    def test_server_creation(self, mcp_server):
        """Test server can be created with default settings."""
        assert mcp_server.name == "test-tenets"
        assert mcp_server._mcp is not None

    def test_server_has_tools_registered(self, mcp_server):
        """Test that tools are registered on the server."""
        tools = mcp_server._mcp.tools
        # Context tools
        assert "tenets_distill" in tools
        assert "tenets_rank_files" in tools
        # Analysis tools
        assert "tenets_examine" in tools
        assert "tenets_chronicle" in tools
        assert "tenets_momentum" in tools
        # Consolidated session tool
        assert "tenets_session" in tools
        # Consolidated tenet tool
        assert "tenets_tenet" in tools
        # System instruction tool
        assert "tenets_system_instruction" in tools
        # Verify old tool names are NOT present (consolidation)
        assert "session_create" not in tools
        assert "session_list" not in tools
        assert "tenet_add" not in tools

    def test_server_has_resources_registered(self, mcp_server):
        """Test that resources are registered on the server."""
        resources = mcp_server._mcp.resources
        assert "tenets://sessions/list" in resources
        assert "tenets://tenets/list" in resources
        assert "tenets://config/current" in resources
        # New resources
        assert "tenets://ranking/factors" in resources
        assert "tenets://sessions/active" in resources
        assert "tenets://analysis/hotspots" in resources
        assert "tenets://analysis/summary" in resources
        # Dynamic resource patterns
        assert any("sessions" in uri and "state" in uri for uri in resources)

    def test_server_has_prompts_registered(self, mcp_server):
        """Test that prompts are registered on the server."""
        prompts = mcp_server._mcp.prompts
        assert "build_context_for_task" in prompts
        assert "understand_codebase" in prompts
        # New workflow prompts
        assert "refactoring_guide" in prompts
        assert "bug_investigation" in prompts
        assert "code_review" in prompts
        assert "onboarding" in prompts

    def test_tenets_lazy_loading(self, mcp_server):
        """Test that Tenets instance is lazily loaded."""
        # Initially should be None
        assert mcp_server._tenets is None

        # Accessing the property should load it
        tenets_instance = mcp_server.tenets
        assert tenets_instance is not None
        assert mcp_server._tenets is tenets_instance

    def test_server_name_customization(self, mock_mcp_module, tmp_path):
        """Test server name can be customized."""
        from tenets.config import TenetsConfig
        from tenets.mcp.server import TenetsMCP

        config = TenetsConfig()
        config.cache.directory = tmp_path / "cache"
        config.cache.directory.mkdir(parents=True, exist_ok=True)

        server = TenetsMCP(name="custom-server-name", config=config)
        assert server.name == "custom-server-name"

    def test_server_project_path_default(self, mock_mcp_module, tmp_path):
        """Test server uses cwd as default project path."""
        from tenets.config import TenetsConfig
        from tenets.mcp.server import TenetsMCP

        config = TenetsConfig()
        config.cache.directory = tmp_path / "cache"
        config.cache.directory.mkdir(parents=True, exist_ok=True)

        server = TenetsMCP(config=config)
        assert server._project_path == Path.cwd()


class TestMCPTools:
    """Tests for MCP tool implementations."""

    @pytest.mark.asyncio
    async def test_distill_tool_with_mock(self, mcp_server_with_mock_tenets, tmp_path):
        """Test the tenets_distill tool with mocked Tenets (Py3.14 compatible)."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        result = await distill_tool(
            prompt="find hello function",
            path=str(tmp_path),
            mode="fast",
            max_tokens=1000,
        )

        assert isinstance(result, dict)
        assert "context" in result
        assert "files" in result
        assert "token_count" in result

    @pytest.mark.asyncio
    async def test_rank_files_tool_with_mock(self, mcp_server_with_mock_tenets, tmp_path):
        """Test the tenets_rank_files tool with mocked Tenets (Py3.14 compatible)."""
        # Create test files
        (tmp_path / "auth.py").write_text("def authenticate(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        rank_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_rank_files"]
        result = await rank_tool(
            prompt="authentication",
            path=str(tmp_path),
            mode="fast",
            top_n=10,
        )

        assert isinstance(result, dict)
        assert "files" in result
        assert "total_scanned" in result
        assert len(result["files"]) <= 10

    @pytest.mark.asyncio
    async def test_distill_tool_modes(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill accepts all valid mode values."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]

        for mode in ["fast", "balanced", "thorough"]:
            result = await distill_tool(
                prompt="test",
                path=str(tmp_path),
                mode=mode,
                max_tokens=100,
            )
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_distill_tool_formats(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill accepts all valid format values."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]

        for fmt in ["markdown", "xml", "json", "html"]:
            result = await distill_tool(
                prompt="test",
                path=str(tmp_path),
                format=fmt,
                max_tokens=100,
            )
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_distill_tool_timeout_param(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill propagates timeout parameter."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        await distill_tool(
            prompt="test timeout",
            path=str(tmp_path),
            timeout=7,
        )

        calls = getattr(mcp_server_with_mock_tenets._tenets, "distill_calls", [])
        assert calls, "distill was not called"
        assert calls[-1].get("timeout") == 7

    @pytest.mark.asyncio
    async def test_distill_timeout_zero_disables(self, mcp_server_with_mock_tenets, tmp_path):
        """Test timeout=0 is passed through to disable timeout."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        await distill_tool(
            prompt="test timeout zero",
            path=str(tmp_path),
            timeout=0,
        )

        calls = getattr(mcp_server_with_mock_tenets._tenets, "distill_calls", [])
        assert calls, "distill was not called"
        # timeout=0 should be passed through (disables timeout)
        assert calls[-1].get("timeout") == 0

    @pytest.mark.asyncio
    async def test_distill_default_timeout(self, mcp_server_with_mock_tenets, tmp_path):
        """Test default timeout value (120) when not specified."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        await distill_tool(
            prompt="test default timeout",
            path=str(tmp_path),
            # Don't specify timeout - should use default
        )

        calls = getattr(mcp_server_with_mock_tenets._tenets, "distill_calls", [])
        assert calls, "distill was not called"
        # Default timeout is 120 seconds
        assert calls[-1].get("timeout") == 120

    @pytest.mark.asyncio
    async def test_distill_with_patterns(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill with include/exclude patterns."""
        (tmp_path / "keep.py").write_text("# keep")
        (tmp_path / "skip.log").write_text("# skip")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        result = await distill_tool(
            prompt="test",
            path=str(tmp_path),
            include_patterns=["*.py"],
            exclude_patterns=["*.log"],
            max_tokens=100,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_distill_with_session(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill with session parameter."""
        # Create session first using consolidated tool
        session_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_session"]
        await session_tool(action="create", name="distill-session")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        result = await distill_tool(
            prompt="test",
            path=str(tmp_path),
            session="distill-session",
            max_tokens=100,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_rank_files_with_explain(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_rank_files with explain flag."""
        (tmp_path / "test.py").write_text("def test(): pass")

        rank_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_rank_files"]
        result = await rank_tool(
            prompt="test",
            path=str(tmp_path),
            explain=True,
            top_n=5,
        )

        assert isinstance(result, dict)
        assert "files" in result
        # With explain=True, files should have factors
        if result["files"]:
            assert "factors" in result["files"][0]

    @pytest.mark.asyncio
    async def test_rank_files_top_n_respected(self, mcp_server_with_mock_tenets, tmp_path):
        """Test that tenets_rank_files respects top_n parameter."""
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(f"def func{i}(): pass")

        rank_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_rank_files"]
        result = await rank_tool(
            prompt="function",
            path=str(tmp_path),
            top_n=3,
        )

        assert len(result["files"]) <= 3

    @pytest.mark.asyncio
    async def test_session_create_action(self, mcp_server):
        """Test the tenets_session tool with create action."""
        session_tool = mcp_server._mcp.tools["tenets_session"]
        result = await session_tool(
            action="create",
            name="test-session",
            description="A test session",
        )

        assert isinstance(result, dict)
        assert result["action"] == "create"
        assert "name" in result
        assert result["name"] == "test-session"

    @pytest.mark.asyncio
    async def test_session_list_action(self, mcp_server):
        """Test the tenets_session tool with list action."""
        session_tool = mcp_server._mcp.tools["tenets_session"]

        # First create a session
        await session_tool(action="create", name="list-test-session")

        # Then list sessions
        result = await session_tool(action="list")

        assert isinstance(result, dict)
        assert result["action"] == "list"
        assert "sessions" in result
        assert isinstance(result["sessions"], list)

    @pytest.mark.asyncio
    async def test_tenet_add_action(self, mcp_server):
        """Test the tenets_tenet tool with add action."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]
        result = await tenet_tool(
            action="add",
            content="Always validate user input",
            priority="high",
            category="security",
        )

        assert isinstance(result, dict)
        assert result["action"] == "add"
        assert "id" in result
        assert "content" in result
        assert result["content"] == "Always validate user input"

    @pytest.mark.asyncio
    async def test_tenet_list_action(self, mcp_server):
        """Test the tenets_tenet tool with list action."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]

        # First add a tenet
        await tenet_tool(action="add", content="Test tenet")

        # Then list tenets
        result = await tenet_tool(action="list")

        assert isinstance(result, dict)
        assert result["action"] == "list"
        assert "tenets" in result

    @pytest.mark.asyncio
    async def test_system_instruction_tool(self, mcp_server):
        """Test the tenets_system_instruction tool."""
        instruction_tool = mcp_server._mcp.tools["tenets_system_instruction"]
        result = await instruction_tool(
            instruction="You are a helpful coding assistant.",
            position="top",
        )

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["position"] == "top"

    @pytest.mark.asyncio
    async def test_examine_tool(self, mcp_server, tmp_path):
        """Test the tenets_examine tool."""
        # Create some test files
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def util(): pass")

        examine_tool = mcp_server._mcp.tools["tenets_examine"]
        result = await examine_tool(
            path=str(tmp_path),
            include_complexity=True,
            include_hotspots=True,
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_chronicle_tool(self, mcp_server, tmp_path):
        """Test the tenets_chronicle tool (may return error if not git repo)."""
        chronicle_tool = mcp_server._mcp.tools["tenets_chronicle"]
        result = await chronicle_tool(
            path=str(tmp_path),
            since="1 week",
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_momentum_tool(self, mcp_server, tmp_path):
        """Test the tenets_momentum tool (may return error if not git repo)."""
        momentum_tool = mcp_server._mcp.tools["tenets_momentum"]
        result = await momentum_tool(
            path=str(tmp_path),
            since="last-month",
            team=False,
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_session_pin_file_action(self, mcp_server, tmp_path):
        """Test the tenets_session tool with pin_file action."""
        # Create session and file
        test_file = tmp_path / "pin_test.py"
        test_file.write_text("# test file")

        session_tool = mcp_server._mcp.tools["tenets_session"]
        await session_tool(action="create", name="pin-session")

        result = await session_tool(
            action="pin_file",
            name="pin-session",
            file_path=str(test_file),
        )

        assert isinstance(result, dict)
        assert result["action"] == "pin_file"
        assert "session" in result
        assert result["session"] == "pin-session"

    @pytest.mark.asyncio
    async def test_session_pin_folder_action(self, mcp_server, tmp_path):
        """Test the tenets_session tool with pin_folder action."""
        # Create session and folder with files
        folder = tmp_path / "src"
        folder.mkdir()
        (folder / "a.py").write_text("# a")
        (folder / "b.py").write_text("# b")

        session_tool = mcp_server._mcp.tools["tenets_session"]
        await session_tool(action="create", name="folder-pin-session")

        result = await session_tool(
            action="pin_folder",
            name="folder-pin-session",
            folder_path=str(folder),
            patterns=["*.py"],
        )

        assert isinstance(result, dict)
        assert result["action"] == "pin_folder"
        assert "pinned_count" in result


class TestMCPResources:
    """Tests for MCP resource implementations."""

    @pytest.mark.asyncio
    async def test_sessions_list_resource(self, mcp_server):
        """Test the sessions list resource."""
        # Create a session first using consolidated tool
        session_tool = mcp_server._mcp.tools["tenets_session"]
        await session_tool(action="create", name="resource-test-session")

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
        # Create a session using consolidated tool
        session_tool = mcp_server._mcp.tools["tenets_session"]
        await session_tool(action="create", name="state-session", description="State test")

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

    @pytest.mark.asyncio
    async def test_session_state_resource_not_found(self, mcp_server):
        """Test session state resource with nonexistent session."""
        resource_func = mcp_server._mcp.resources["tenets://sessions/{name}/state"]
        result = await resource_func(name="nonexistent-session")

        assert isinstance(result, str)
        data = json.loads(result)
        assert "error" in data


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

    def test_build_context_for_task_without_focus(self, mcp_server):
        """Test build_context_for_task without focus areas."""
        prompt_func = mcp_server._mcp.prompts["build_context_for_task"]
        result = prompt_func(task="add logging")

        assert isinstance(result, str)
        assert "add logging" in result

    def test_code_review_prompt(self, mcp_server):
        """Test the code_review prompt."""
        prompt_func = mcp_server._mcp.prompts["code_review"]
        result = prompt_func(
            scope="recent",
            focus="security",
        )

        assert isinstance(result, str)
        assert "security" in result

    def test_code_review_scopes(self, mcp_server):
        """Test code_review with different scopes."""
        prompt_func = mcp_server._mcp.prompts["code_review"]

        for scope in ["recent", "file", "module"]:
            result = prompt_func(scope=scope)
            assert isinstance(result, str)

    def test_understand_codebase_prompt(self, mcp_server):
        """Test the understand_codebase prompt."""
        prompt_func = mcp_server._mcp.prompts["understand_codebase"]
        result = prompt_func(
            depth="overview",
            area="authentication",
        )

        assert isinstance(result, str)
        assert "authentication" in result

    def test_understand_codebase_depths(self, mcp_server):
        """Test understand_codebase with different depths."""
        prompt_func = mcp_server._mcp.prompts["understand_codebase"]

        for depth in ["overview", "detailed"]:
            result = prompt_func(depth=depth)
            assert isinstance(result, str)
            assert depth in result


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

    def test_run_custom_host_port(self, mcp_server):
        """Test running with custom host and port."""
        mcp_server.run(transport="sse", host="0.0.0.0", port=9000)
        assert mcp_server._mcp.last_run["host"] == "0.0.0.0"
        assert mcp_server._mcp.last_run["port"] == 9000


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

    def test_create_server_default_name(self, mock_mcp_module, tmp_path):
        """Test create_server with default name."""
        from tenets.config import TenetsConfig
        from tenets.mcp.server import create_server

        config = TenetsConfig()
        config.cache.directory = tmp_path / "cache"
        config.cache.directory.mkdir(parents=True, exist_ok=True)

        server = create_server(config=config)
        assert server.name == "tenets"


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

    def test_check_mcp_available_caches_result(self, mock_mcp_module):
        """Test that _check_mcp_available caches its result."""
        import tenets.mcp.server as server_module

        server_module._mcp_available = None
        result1 = server_module._check_mcp_available()
        result2 = server_module._check_mcp_available()

        assert result1 == result2 == True


class TestMCPToolDocstrings:
    """Tests to verify tool docstrings are present and informative."""

    def test_distill_has_docstring(self, mcp_server):
        """Test that tenets_distill tool has a proper docstring."""
        tool = mcp_server._mcp.tools["tenets_distill"]
        assert tool.__doc__ is not None
        # v0.9.3+ uses "Use when:" pattern
        assert "Use when" in tool.__doc__
        assert "context" in tool.__doc__.lower()

    def test_rank_files_has_docstring(self, mcp_server):
        """Test that tenets_rank_files tool has a proper docstring."""
        tool = mcp_server._mcp.tools["tenets_rank_files"]
        assert tool.__doc__ is not None
        assert "relevant" in tool.__doc__.lower()
        assert "Use when" in tool.__doc__

    def test_all_tools_have_docstrings(self, mcp_server):
        """Test that all tools have docstrings."""
        for name, tool in mcp_server._mcp.tools.items():
            assert tool.__doc__ is not None, f"Tool {name} missing docstring"
            assert len(tool.__doc__) > 50, f"Tool {name} has too short docstring"

    def test_docstrings_have_args_section(self, mcp_server):
        """Test that tool docstrings include Args or Inputs documentation."""
        key_tools = ["tenets_distill", "tenets_rank_files"]
        for name in key_tools:
            tool = mcp_server._mcp.tools[name]
            # v0.9.3 uses "Inputs:" pattern
            assert (
                "Inputs:" in tool.__doc__ or "Args:" in tool.__doc__
            ), f"Tool {name} missing Inputs/Args section"

    def test_docstrings_have_returns_section(self, mcp_server):
        """Test that tool docstrings include Returns documentation."""
        key_tools = ["tenets_distill", "tenets_rank_files", "tenets_tenet"]
        for name in key_tools:
            tool = mcp_server._mcp.tools[name]
            # Returns can be in docstring or as Return:
            assert "return" in tool.__doc__.lower(), f"Tool {name} missing Returns section"

    def test_docstrings_have_use_this_tool_trigger(self, mcp_server):
        """Test that all tools have 'Use when' trigger phrases (v0.9.3+ pattern)."""
        for name, tool in mcp_server._mcp.tools.items():
            assert (
                "Use when" in tool.__doc__ or "USE THIS TOOL" in tool.__doc__
            ), f"Tool {name} missing trigger phrase"


class TestMCPSessionPersistence:
    """Tests for session persistence across operations."""

    @pytest.mark.asyncio
    async def test_session_persists_across_operations(self, mcp_server, tmp_path):
        """Test that session data persists correctly."""
        session_tool = mcp_server._mcp.tools["tenets_session"]

        # Create session
        result = await session_tool(
            action="create", name="persist-test", description="Persistence test"
        )
        session_id = result["id"]

        # Verify it appears in list
        sessions = await session_tool(action="list")
        session_names = [s["name"] for s in sessions["sessions"]]
        assert "persist-test" in session_names

    @pytest.mark.asyncio
    async def test_pinned_files_persist(self, mcp_server, tmp_path):
        """Test that pinned files are remembered."""
        # Create a test file
        test_file = tmp_path / "pinned.py"
        test_file.write_text("# Important file")

        session_tool = mcp_server._mcp.tools["tenets_session"]

        # Create session
        await session_tool(action="create", name="pin-test")

        # Pin file
        result = await session_tool(action="pin_file", name="pin-test", file_path=str(test_file))
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self, mcp_server):
        """Test that multiple sessions are isolated from each other."""
        session_tool = mcp_server._mcp.tools["tenets_session"]

        # Create two sessions
        await session_tool(action="create", name="session-a", description="Session A")
        await session_tool(action="create", name="session-b", description="Session B")

        # Verify both exist
        sessions = await session_tool(action="list")
        session_names = [s["name"] for s in sessions["sessions"]]
        assert "session-a" in session_names
        assert "session-b" in session_names

    @pytest.mark.asyncio
    async def test_session_metadata_preserved(self, mcp_server):
        """Test that session metadata is preserved."""
        session_tool = mcp_server._mcp.tools["tenets_session"]
        await session_tool(action="create", name="metadata-test", description="Test description")

        resource_func = mcp_server._mcp.resources["tenets://sessions/{name}/state"]
        result = await resource_func(name="metadata-test")
        data = json.loads(result)

        assert data["metadata"].get("description") == "Test description"


class TestMCPTenetOperations:
    """Tests for tenet CRUD operations."""

    @pytest.mark.asyncio
    async def test_tenet_priority_levels(self, mcp_server):
        """Test that all priority levels work."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]

        for priority in ["low", "medium", "high", "critical"]:
            result = await tenet_tool(
                action="add",
                content=f"Test tenet with {priority} priority",
                priority=priority,
            )
            assert "id" in result

    @pytest.mark.asyncio
    async def test_tenet_categories(self, mcp_server):
        """Test that categories are stored correctly."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]

        result = await tenet_tool(
            action="add",
            content="Always use HTTPS",
            priority="high",
            category="security",
        )

        assert result["category"] == "security"

    @pytest.mark.asyncio
    async def test_tenet_instill(self, mcp_server):
        """Test the tenet instill functionality."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]

        # Add a pending tenet
        await tenet_tool(action="add", content="Pending tenet for instill test")

        # Instill it
        result = await tenet_tool(action="instill")

        assert isinstance(result, dict)
        assert result["action"] == "instill"

    @pytest.mark.asyncio
    async def test_tenet_with_session(self, mcp_server):
        """Test adding a tenet to a specific session."""
        session_tool = mcp_server._mcp.tools["tenets_session"]
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]

        # Create session
        await session_tool(action="create", name="tenet-session")

        # Add tenet to session
        result = await tenet_tool(
            action="add",
            content="Session-specific tenet",
            priority="medium",
            session="tenet-session",
        )

        assert "id" in result

    @pytest.mark.asyncio
    async def test_tenet_list_pending_only(self, mcp_server):
        """Test listing only pending tenets."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]

        # Add tenet
        await tenet_tool(action="add", content="Test pending tenet")

        # List with pending_only
        result = await tenet_tool(action="list", pending_only=True)

        assert isinstance(result, dict)
        assert result["action"] == "list"
        assert "tenets" in result


class TestMCPErrorHandling:
    """Tests for error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_session_pin_nonexistent_file(self, mcp_server):
        """Test pinning a nonexistent file."""
        session_tool = mcp_server._mcp.tools["tenets_session"]
        await session_tool(action="create", name="error-test-session")

        result = await session_tool(
            action="pin_file",
            name="error-test-session",
            file_path="/nonexistent/file.py",
        )

        # Should return failure indicator
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_session_pin_folder_empty(self, mcp_server, tmp_path):
        """Test pinning an empty folder."""
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        session_tool = mcp_server._mcp.tools["tenets_session"]
        await session_tool(action="create", name="empty-folder-session")

        result = await session_tool(
            action="pin_folder",
            name="empty-folder-session",
            folder_path=str(empty_folder),
        )

        assert isinstance(result, dict)
        assert result["pinned_count"] == 0

    @pytest.mark.asyncio
    async def test_system_instruction_positions(self, mcp_server):
        """Test all system instruction positions."""
        instruction_tool = mcp_server._mcp.tools["tenets_system_instruction"]

        for position in ["top", "after_header", "before_content"]:
            result = await instruction_tool(
                instruction="Test instruction",
                position=position,
            )
            assert result["success"] is True
            assert result["position"] == position

    @pytest.mark.asyncio
    async def test_session_missing_required_params(self, mcp_server):
        """Test consolidated session tool with missing required params."""
        session_tool = mcp_server._mcp.tools["tenets_session"]

        # Create without name should return error
        result = await session_tool(action="create")
        assert "error" in result

        # pin_file without name or file_path should return error
        result = await session_tool(action="pin_file")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_tenet_missing_required_params(self, mcp_server):
        """Test consolidated tenet tool with missing required params."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]

        # Add without content should return error
        result = await tenet_tool(action="add")
        assert "error" in result


class TestMCPIntegration:
    """Integration tests for MCP tool workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow_session_to_distill(self, mcp_server_with_mock_tenets, tmp_path):
        """Test a complete workflow: create session, pin files, distill."""
        # Create test files
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def util(): pass")

        session_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_session"]

        # Create session
        session_result = await session_tool(action="create", name="workflow-test")
        assert session_result["name"] == "workflow-test"

        # Pin a file
        pin_result = await session_tool(
            action="pin_file",
            name="workflow-test",
            file_path=str(tmp_path / "main.py"),
        )
        assert pin_result["success"] is True

        # Distill with session
        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        distill_result = await distill_tool(
            prompt="find main function",
            path=str(tmp_path),
            session="workflow-test",
            max_tokens=1000,
        )
        assert isinstance(distill_result, dict)
        assert "context" in distill_result

    @pytest.mark.asyncio
    async def test_tenet_workflow(self, mcp_server):
        """Test tenet workflow: add, list, instill."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]

        # Add tenet
        add_result = await tenet_tool(
            action="add",
            content="Always write tests",
            priority="high",
            category="quality",
        )
        tenet_id = add_result["id"]
        assert tenet_id is not None

        # List tenets
        list_result = await tenet_tool(action="list")
        assert isinstance(list_result, dict)

        # Instill
        instill_result = await tenet_tool(action="instill")
        assert isinstance(instill_result, dict)

    @pytest.mark.asyncio
    async def test_rank_then_distill_workflow(self, mcp_server_with_mock_tenets, tmp_path):
        """Test workflow: rank files first, then distill with specific files."""
        # Create test files
        (tmp_path / "auth.py").write_text("def authenticate(): pass")
        (tmp_path / "main.py").write_text("def main(): pass")

        # Rank first
        rank_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_rank_files"]
        rank_result = await rank_tool(
            prompt="authentication",
            path=str(tmp_path),
            top_n=5,
        )
        assert len(rank_result["files"]) > 0

        # Then distill
        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        distill_result = await distill_tool(
            prompt="authentication",
            path=str(tmp_path),
            max_tokens=1000,
        )
        assert "context" in distill_result


class TestMCPResourceContent:
    """Tests for MCP resource content format."""

    @pytest.mark.asyncio
    async def test_sessions_resource_json_format(self, mcp_server):
        """Test that sessions resource returns valid JSON."""
        # Create a session using consolidated tool
        session_tool = mcp_server._mcp.tools["tenets_session"]
        await session_tool(action="create", name="json-test")

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

    @pytest.mark.asyncio
    async def test_tenets_resource_json_format(self, mcp_server):
        """Test that tenets resource returns valid JSON."""
        # Add a tenet using consolidated tool
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]
        await tenet_tool(action="add", content="Resource format test")

        resource_func = mcp_server._mcp.resources["tenets://tenets/list"]
        result = await resource_func()

        # Should be valid JSON
        json.loads(result)


class TestMCPEdgeCases:
    """Edge case tests for MCP functionality."""

    @pytest.mark.asyncio
    async def test_empty_prompt_distill(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill with minimal prompt."""
        (tmp_path / "test.py").write_text("# test")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        result = await distill_tool(
            prompt="x",
            path=str(tmp_path),
            max_tokens=100,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill with very long prompt."""
        (tmp_path / "test.py").write_text("# test")

        long_prompt = "find " + "authentication " * 100

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        result = await distill_tool(
            prompt=long_prompt,
            path=str(tmp_path),
            max_tokens=100,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_unicode_in_tenet(self, mcp_server):
        """Test tenet with unicode characters."""
        tenet_tool = mcp_server._mcp.tools["tenets_tenet"]
        result = await tenet_tool(
            action="add",
            content="Always use UTF-8 encoding 🔒 für Sicherheit",
            priority="medium",
        )
        assert "id" in result

    @pytest.mark.asyncio
    async def test_special_chars_in_session_name(self, mcp_server):
        """Test session with special characters in name."""
        session_tool = mcp_server._mcp.tools["tenets_session"]
        # Use only safe characters
        result = await session_tool(action="create", name="test-session_123")
        assert result["name"] == "test-session_123"

    @pytest.mark.asyncio
    async def test_zero_max_tokens(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill with zero max_tokens (should use default)."""
        (tmp_path / "test.py").write_text("# test")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        # Most implementations should treat 0 as "use default"
        result = await distill_tool(
            prompt="test",
            path=str(tmp_path),
            max_tokens=0,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_include_tests_flag(self, mcp_server_with_mock_tenets, tmp_path):
        """Test tenets_distill with include_tests flag."""
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "test_main.py").write_text("def test_main(): pass")

        distill_tool = mcp_server_with_mock_tenets._mcp.tools["tenets_distill"]
        result = await distill_tool(
            prompt="test main",
            path=str(tmp_path),
            include_tests=True,
            max_tokens=1000,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_same_server(self, mcp_server):
        """Test multiple sequential tool calls on same server instance."""
        session_tool = mcp_server._mcp.tools["tenets_session"]

        # Create multiple sessions
        for i in range(5):
            result = await session_tool(action="create", name=f"multi-test-{i}")
            assert result["name"] == f"multi-test-{i}"

        # Verify all created
        sessions = await session_tool(action="list")
        names = [s["name"] for s in sessions["sessions"]]
        for i in range(5):
            assert f"multi-test-{i}" in names
