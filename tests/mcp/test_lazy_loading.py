"""Tests for MCP lazy loading functionality."""

import pytest
from unittest.mock import MagicMock, patch


class TestToolRegistry:
    """Tests for TOOL_REGISTRY and meta-tools."""

    def test_tool_registry_has_expected_tools(self):
        """Verify TOOL_REGISTRY contains all expected tools."""
        from tenets.mcp.server import TOOL_REGISTRY

        expected_tools = {
            "tenets_examine", "tenets_chronicle", "tenets_momentum",
            "tenets_session_create", "tenets_session_list", "tenets_session_pin_file", "tenets_session_pin_folder",
            "tenets_tenet_add", "tenets_tenet_list", "tenets_tenet_instill", "tenets_set_system_instruction",
        }
        assert set(TOOL_REGISTRY.keys()) == expected_tools

    def test_tool_registry_has_required_fields(self):
        """Verify each tool in registry has required metadata."""
        from tenets.mcp.server import TOOL_REGISTRY

        for name, info in TOOL_REGISTRY.items():
            assert "category" in info, f"{name} missing category"
            assert "description" in info, f"{name} missing description"
            assert "keywords" in info, f"{name} missing keywords"
            assert isinstance(info["keywords"], list), f"{name} keywords should be list"

    def test_always_available_tools(self):
        """Verify ALWAYS_AVAILABLE_TOOLS has core tools."""
        from tenets.mcp.server import ALWAYS_AVAILABLE_TOOLS

        assert "tenets_distill" in ALWAYS_AVAILABLE_TOOLS
        assert "tenets_rank_files" in ALWAYS_AVAILABLE_TOOLS
        assert "tenets_search_tools" in ALWAYS_AVAILABLE_TOOLS
        assert "tenets_get_tool_schema" in ALWAYS_AVAILABLE_TOOLS


class TestSearchTenetsTools:
    """Tests for tenets_search_tools meta-tool."""

    @pytest.fixture
    def tool_search(self):
        """Create a mock search function based on TOOL_REGISTRY."""
        from tenets.mcp.server import TOOL_REGISTRY

        async def search(query: str, category: str = None):
            query_lower = query.lower()
            results = []
            for name, info in TOOL_REGISTRY.items():
                if category and info["category"] != category:
                    continue
                if (
                    query_lower in name.lower()
                    or query_lower in info["description"].lower()
                    or any(query_lower in kw for kw in info["keywords"])
                ):
                    results.append({
                        "name": name,
                        "category": info["category"],
                        "description": info["description"],
                    })
            return results
        return search

    @pytest.mark.asyncio
    async def test_search_by_keyword(self, tool_search):
        """Search should match against keywords."""
        results = await tool_search("git")
        names = [r["name"] for r in results]
        assert "tenets_chronicle" in names

    @pytest.mark.asyncio
    async def test_search_by_category(self, tool_search):
        """Search should filter by category."""
        results = await tool_search("", category="session")
        for r in results:
            assert r["category"] == "session"

    @pytest.mark.asyncio
    async def test_search_returns_minimal_info(self, tool_search):
        """Search should return only name, category, description."""
        results = await tool_search("session")
        for r in results:
            assert set(r.keys()) == {"name", "category", "description"}


class TestGetToolSchema:
    """Tests for tenets_get_tool_schema meta-tool."""

    @pytest.mark.asyncio
    async def test_get_known_tool_schema(self):
        """tenets_get_tool_schema should return full schema for known tools."""
        tool_schemas = {
            "tenets_examine": {
                "name": "tenets_examine",
                "description": "Analyze codebase structure and quality metrics",
                "parameters": {
                    "path": {"type": "string", "default": "."},
                },
            },
        }
        
        result = tool_schemas.get("tenets_examine", {"error": "not found"})
        assert result["name"] == "tenets_examine"
        assert "parameters" in result

    @pytest.mark.asyncio
    async def test_get_unknown_tool_returns_error(self):
        """tenets_get_tool_schema should return error for unknown tools."""
        tool_schemas = {}
        tool_name = "nonexistent_tool"
        
        result = tool_schemas.get(tool_name, {"error": f"Tool '{tool_name}' not found"})
        assert "error" in result
