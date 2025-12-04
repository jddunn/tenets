"""Tenets MCP Server implementation.

This module provides the core MCP server that exposes tenets functionality
to AI coding assistants via the Model Context Protocol.

The server supports multiple transports:
- stdio: Local process communication (default, for IDE integration)
- sse: Server-Sent Events (for web-based clients)
- http: Streamable HTTP (for remote deployment)

All tools delegate to the existing tenets core library, ensuring consistent
behavior between CLI, Python API, and MCP interfaces.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from tenets import Tenets
    from tenets.config import TenetsConfig

# Lazy imports to avoid loading MCP dependencies unless needed
_mcp_available = None


def _check_mcp_available() -> bool:
    """Check if MCP dependencies are available."""
    global _mcp_available
    if _mcp_available is None:
        try:
            import mcp  # noqa: F401

            _mcp_available = True
        except ImportError:
            _mcp_available = False
    return _mcp_available


class TenetsMCP:
    """Tenets MCP Server.

    Wraps the tenets core library and exposes functionality via MCP protocol.
    This class manages the FastMCP server instance and handles lifecycle.

    Attributes:
        name: Server name for MCP identification.
        tenets: Underlying Tenets instance for actual functionality.
        config: Configuration for the MCP server.

    Example:
        >>> from tenets.mcp import TenetsMCP
        >>> server = TenetsMCP()
        >>> server.run(transport="stdio")
    """

    def __init__(
        self,
        name: str = "tenets",
        config: Optional[TenetsConfig] = None,
        project_path: Optional[Path] = None,
    ):
        """Initialize the MCP server.

        Args:
            name: Server name shown to MCP clients.
            config: Optional TenetsConfig. If not provided, uses defaults.
            project_path: Optional project root path. Defaults to cwd.
        """
        if not _check_mcp_available():
            raise ImportError(
                "MCP dependencies not installed. " "Install with: pip install tenets[mcp]"
            )

        self.name = name
        self._project_path = project_path or Path.cwd()
        self._config = config
        self._tenets: Optional[Tenets] = None
        self._mcp = None
        self._setup_server()

    @property
    def tenets(self) -> Tenets:
        """Lazy-load the Tenets instance."""
        if self._tenets is None:
            from tenets import Tenets

            self._tenets = Tenets(config=self._config)
        return self._tenets

    def _setup_server(self) -> None:
        """Configure the FastMCP server with tools, resources, and prompts."""
        from mcp.server.fastmcp import FastMCP

        self._mcp = FastMCP(self.name)
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def _register_tools(self) -> None:
        """Register all MCP tools."""
        mcp = self._mcp

        # === Context Tools ===

        @mcp.tool()
        async def distill(
            prompt: str,
            path: str = ".",
            mode: Literal["fast", "balanced", "thorough"] = "balanced",
            max_tokens: int = 100000,
            format: Literal["markdown", "xml", "json"] = "markdown",
            include_tests: bool = False,
            session: Optional[str] = None,
            include_patterns: Optional[list[str]] = None,
            exclude_patterns: Optional[list[str]] = None,
        ) -> dict[str, Any]:
            """Build optimized code context for a task or question.

            This is the primary tool for gathering relevant code. It finds,
            ranks, and aggregates files into token-optimized context ready
            for AI consumption.

            Use this when you need to:
            - Understand how a feature is implemented
            - Find code related to a bug or task
            - Gather context before making changes

            Args:
                prompt: What you're working on. Be specific for better results.
                    Examples: "implement OAuth2 login", "fix the payment bug",
                    "understand the caching layer"
                path: Directory to search. Use "." for current project.
                mode: Speed vs accuracy tradeoff:
                    - "fast": Quick keyword matching (~1s)
                    - "balanced": BM25 + structure analysis (~3s, recommended)
                    - "thorough": ML embeddings + deep analysis (~10s)
                max_tokens: Token budget. Default 100k works for most models.
                format: Output structure:
                    - "markdown": Human-readable with headers
                    - "xml": Claude-optimized with tags
                    - "json": Structured for programmatic use
                include_tests: Set True when debugging test failures.
                session: Link to a session for persistent pinned files.
                include_patterns: Only include matching files (e.g., ["*.py"]).
                exclude_patterns: Skip matching files (e.g., ["*.log", "*.min.js"]).

            Returns:
                {
                    "context": "# File: src/auth.py\\n...",
                    "token_count": 45000,
                    "files": ["src/auth.py", "src/user.py"],
                    "files_summarized": ["src/utils.py"],
                    "metadata": {"mode": "balanced", "total_scanned": 150}
                }
            """
            result = self.tenets.distill(
                prompt=prompt,
                files=path,
                mode=mode,
                max_tokens=max_tokens,
                format=format,
                include_tests=include_tests,
                session_name=session,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            )
            return result.to_dict()

        @mcp.tool()
        async def rank_files(
            prompt: str,
            path: str = ".",
            mode: Literal["fast", "balanced", "thorough", "ml"] = "balanced",
            top_n: int = 20,
            include_tests: bool = False,
            explain: bool = False,
        ) -> dict[str, Any]:
            """Preview which files are most relevant without fetching content.

            Faster than distill (~500ms). Use this to:
            - Quickly check if the right files would be found
            - Understand file relevance before full context retrieval
            - Debug ranking when distill results seem off

            Args:
                prompt: What you're looking for. Same as distill prompt.
                path: Directory to search.
                mode: Ranking algorithm (same as distill).
                top_n: How many files to return. Default 20 is usually enough.
                include_tests: Include test files in results.
                explain: Add breakdown of why each file ranked where it did.
                    Useful for debugging relevance issues.

            Returns:
                {
                    "files": [
                        {"path": "src/auth.py", "score": 0.85, "factors": {...}},
                        {"path": "src/user.py", "score": 0.72}
                    ],
                    "total_scanned": 150,
                    "mode": "balanced"
                }
            """
            result = self.tenets.rank_files(
                prompt=prompt,
                paths=path,
                mode=mode,
                include_tests=include_tests,
                explain=explain,
            )
            files_data = []
            for f in result.files[:top_n]:
                file_info = {
                    "path": str(f.path) if hasattr(f, "path") else str(f),
                    "score": getattr(f, "relevance_score", 0.0),
                }
                if explain and hasattr(f, "ranking_factors"):
                    file_info["factors"] = f.ranking_factors
                files_data.append(file_info)

            return {
                "files": files_data,
                "total_scanned": result.total_scanned,
                "mode": result.mode,
            }

        # === Analysis Tools ===

        @mcp.tool()
        async def examine(
            path: str = ".",
            include_complexity: bool = True,
            include_hotspots: bool = True,
        ) -> dict[str, Any]:
            """Examine codebase structure and quality metrics.

            Analyzes the codebase to identify structure, complexity patterns,
            and maintenance hotspots.

            Args:
                path: Root path to examine.
                include_complexity: Include complexity metrics.
                include_hotspots: Identify maintenance hotspots.

            Returns:
                Dictionary with examination results including file counts,
                language distribution, complexity metrics, and hotspots.
            """
            result = self.tenets.examine(
                path=path,
                deep=include_complexity,
            )
            return result if isinstance(result, dict) else {"result": str(result)}

        @mcp.tool()
        async def chronicle(
            path: str = ".",
            since: str = "1 week",
            author: Optional[str] = None,
        ) -> dict[str, Any]:
            """Analyze git history and development patterns.

            Returns commit activity, file churn, contributor patterns,
            and temporal development insights.

            Args:
                path: Repository path to analyze.
                since: Time period (e.g., "1 week", "3 days", "last month").
                author: Optional author filter.

            Returns:
                Dictionary with git history analysis.
            """
            result = self.tenets.track_changes(
                path=path,
                since=since,
                author=author,
            )
            return result if isinstance(result, dict) else {"result": str(result)}

        @mcp.tool()
        async def momentum(
            path: str = ".",
            since: str = "last-month",
            team: bool = False,
        ) -> dict[str, Any]:
            """Track development velocity and team momentum.

            Provides sprint velocity metrics, contribution patterns,
            and development trend analysis.

            Args:
                path: Repository path to analyze.
                since: Time period for analysis.
                team: Whether to show team-wide statistics.

            Returns:
                Dictionary with momentum metrics.
            """
            result = self.tenets.momentum(
                path=path,
                since=since,
                team=team,
            )
            return result if isinstance(result, dict) else {"result": str(result)}

        # === Session Tools ===

        @mcp.tool()
        async def session_create(
            name: str,
            description: Optional[str] = None,
        ) -> dict[str, Any]:
            """Create a new development session for stateful context building.

            Sessions allow pinning files, tracking context history, and
            maintaining state across multiple distill operations.

            Args:
                name: Unique session name.
                description: Optional session description.

            Returns:
                Dictionary with session information.
            """
            from tenets.storage.session_db import SessionDB

            db = SessionDB(self.tenets.config)
            metadata = {"description": description} if description else {}
            session = db.create_session(name, metadata=metadata)
            return {
                "id": session.id,
                "name": session.name,
                "created_at": session.created_at.isoformat(),
            }

        @mcp.tool()
        async def session_list() -> dict[str, Any]:
            """List all development sessions.

            Returns:
                Dictionary with list of sessions and their metadata.
            """
            from tenets.storage.session_db import SessionDB

            db = SessionDB(self.tenets.config)
            sessions = db.list_sessions()
            return {
                "sessions": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "created_at": s.created_at.isoformat(),
                        "metadata": s.metadata,
                    }
                    for s in sessions
                ]
            }

        @mcp.tool()
        async def session_pin_file(
            session: str,
            file_path: str,
        ) -> dict[str, Any]:
            """Pin a file to a session for guaranteed inclusion in future distills.

            Pinned files are always included in context generation for the
            session, regardless of relevance ranking.

            Args:
                session: Session name.
                file_path: Path to file to pin.

            Returns:
                Dictionary indicating success.
            """
            success = self.tenets.add_file_to_session(file_path, session=session)
            return {"success": success, "file": file_path, "session": session}

        @mcp.tool()
        async def session_pin_folder(
            session: str,
            folder_path: str,
            patterns: Optional[list[str]] = None,
        ) -> dict[str, Any]:
            """Pin all files in a folder to a session.

            Args:
                session: Session name.
                folder_path: Path to folder.
                patterns: Optional file patterns to include (e.g., ["*.py"]).

            Returns:
                Dictionary with count of pinned files.
            """
            count = self.tenets.add_folder_to_session(
                folder_path,
                session=session,
                include_patterns=patterns,
            )
            return {"pinned_count": count, "folder": folder_path, "session": session}

        # === Tenet Tools ===

        @mcp.tool()
        async def tenet_add(
            content: str,
            priority: Literal["low", "medium", "high", "critical"] = "medium",
            category: Optional[str] = None,
            session: Optional[str] = None,
        ) -> dict[str, Any]:
            """Add a guiding principle that will be injected into all context.

            Tenets combat "context drift" in long conversations by repeatedly
            injecting key principles. Use them for:
            - Coding standards: "Always use type hints in Python"
            - Security rules: "Never log sensitive data"
            - Architecture: "All API calls go through the gateway"
            - Style: "Use descriptive variable names, no abbreviations"

            Args:
                content: The principle. Keep it concise and actionable.
                    Good: "Validate all user input before database queries"
                    Bad: "Be careful with security"
                priority: How often to inject:
                    - "critical": Every context (security rules)
                    - "high": Most contexts (architecture)
                    - "medium": Regular contexts (style, default)
                    - "low": Occasional reminder
                category: Group related tenets (security, style, architecture).
                session: Bind to specific session, or global if None.

            Returns:
                {"id": "abc123", "content": "...", "priority": "high", "category": "security"}
            """
            tenet = self.tenets.add_tenet(
                content=content,
                priority=priority,
                category=category,
                session=session,
            )
            return {
                "id": tenet.id,
                "content": tenet.content,
                "priority": (
                    tenet.priority.value
                    if hasattr(tenet.priority, "value")
                    else str(tenet.priority)
                ),
                "category": (
                    tenet.category.value
                    if tenet.category and hasattr(tenet.category, "value")
                    else str(tenet.category) if tenet.category else None
                ),
            }

        @mcp.tool()
        async def tenet_list(
            session: Optional[str] = None,
            pending_only: bool = False,
        ) -> dict[str, Any]:
            """List all tenets with optional filtering.

            Args:
                session: Optional session filter.
                pending_only: Only show pending (not yet instilled) tenets.

            Returns:
                Dictionary with list of tenets.
            """
            tenets = self.tenets.list_tenets(
                session=session,
                pending_only=pending_only,
            )
            return {"tenets": tenets}

        @mcp.tool()
        async def tenet_instill(
            session: Optional[str] = None,
            force: bool = False,
        ) -> dict[str, Any]:
            """Instill pending tenets, marking them active for injection.

            Args:
                session: Optional session to instill tenets for.
                force: Re-instill even already instilled tenets.

            Returns:
                Dictionary with instillation results.
            """
            result = self.tenets.instill_tenets(session=session, force=force)
            return result if isinstance(result, dict) else {"result": str(result)}

        @mcp.tool()
        async def set_system_instruction(
            instruction: str,
            position: Literal["top", "after_header", "before_content"] = "top",
        ) -> dict[str, Any]:
            """Set a system instruction for AI interactions.

            System instructions are injected at the specified position in
            all generated context.

            Args:
                instruction: The system instruction text.
                position: Where to inject the instruction.

            Returns:
                Dictionary confirming the instruction was set.
            """
            self.tenets.set_system_instruction(
                instruction=instruction,
                enable=True,
                position=position,
            )
            return {
                "success": True,
                "instruction_length": len(instruction),
                "position": position,
            }

    def _register_resources(self) -> None:
        """Register all MCP resources."""
        mcp = self._mcp

        @mcp.resource("tenets://sessions/list")
        async def get_sessions_list() -> str:
            """List of all development sessions."""
            from tenets.storage.session_db import SessionDB
            import json

            db = SessionDB(self.tenets.config)
            sessions = db.list_sessions()
            return json.dumps(
                [
                    {
                        "name": s.name,
                        "created_at": s.created_at.isoformat(),
                        "metadata": s.metadata,
                    }
                    for s in sessions
                ],
                indent=2,
            )

        @mcp.resource("tenets://sessions/{name}/state")
        async def get_session_state(name: str) -> str:
            """Current state of a specific session."""
            from tenets.storage.session_db import SessionDB
            import json

            db = SessionDB(self.tenets.config)
            session = db.get_session(name)
            if not session:
                return json.dumps({"error": f"Session '{name}' not found"})
            return json.dumps(
                {
                    "name": session.name,
                    "created_at": session.created_at.isoformat(),
                    "metadata": session.metadata,
                },
                indent=2,
            )

        @mcp.resource("tenets://tenets/list")
        async def get_tenets_list() -> str:
            """List of all guiding principles (tenets)."""
            import json

            tenets = self.tenets.list_tenets()
            return json.dumps(tenets, indent=2, default=str)

        @mcp.resource("tenets://config/current")
        async def get_current_config() -> str:
            """Current tenets configuration (read-only)."""
            import json

            config_dict = self.tenets.config.to_dict()
            # Remove sensitive data
            if "llm" in config_dict and "api_keys" in config_dict["llm"]:
                config_dict["llm"]["api_keys"] = {k: "***" for k in config_dict["llm"]["api_keys"]}
            return json.dumps(config_dict, indent=2, default=str)

    def _register_prompts(self) -> None:
        """Register all MCP prompt templates."""
        mcp = self._mcp

        @mcp.prompt()
        def build_context_for_task(
            task: str,
            focus_areas: Optional[str] = None,
        ) -> str:
            """Build optimal context for a development task.

            Analyzes the task description and generates comprehensive context
            with relevant code files and guiding principles.

            Args:
                task: Description of the development task.
                focus_areas: Optional comma-separated focus areas.
            """
            prompt_parts = [
                f"I need to work on: {task}",
                "",
                "Please use the tenets `distill` tool to build relevant context.",
            ]
            if focus_areas:
                prompt_parts.append(f"Focus on these areas: {focus_areas}")
            return "\n".join(prompt_parts)

        @mcp.prompt()
        def code_review_context(
            scope: Literal["recent", "file", "module"] = "recent",
            focus: Optional[str] = None,
        ) -> str:
            """Prepare context for code review.

            Args:
                scope: Review scope - recent changes, specific file, or module.
                focus: Optional focus area (security, performance, etc.).
            """
            parts = ["Prepare context for a code review."]
            if scope == "recent":
                parts.append("Focus on recent changes (use chronicle tool first).")
            parts.append("Use distill to get relevant code context.")
            if focus:
                parts.append(f"Pay special attention to: {focus}")
            return "\n".join(parts)

        @mcp.prompt()
        def understand_codebase(
            depth: Literal["overview", "detailed"] = "overview",
            area: Optional[str] = None,
        ) -> str:
            """Generate codebase understanding context.

            Args:
                depth: Analysis depth - overview or detailed.
                area: Optional specific area to focus on.
            """
            parts = [f"Help me understand this codebase ({depth} level)."]
            if area:
                parts.append(f"Specifically, I want to understand: {area}")
            parts.extend(
                [
                    "",
                    "Steps:",
                    "1. Use `examine` to see codebase structure",
                    "2. Use `distill` with an understanding prompt",
                    "3. Identify key architectural patterns",
                ]
            )
            return "\n".join(parts)

    def run(
        self,
        transport: Literal["stdio", "sse", "http"] = "stdio",
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        """Run the MCP server with the specified transport.

        Args:
            transport: Transport type - stdio (local), sse, or http (remote).
            host: Host for network transports (sse, http).
            port: Port for network transports (sse, http).
        """
        if transport == "stdio":
            self._mcp.run(transport="stdio")
        elif transport == "sse":
            self._mcp.run(transport="sse", host=host, port=port)
        elif transport == "http":
            self._mcp.run(transport="streamable-http", host=host, port=port)
        else:
            raise ValueError(f"Unknown transport: {transport}")


def create_server(
    name: str = "tenets",
    config: Optional[TenetsConfig] = None,
) -> TenetsMCP:
    """Create a new Tenets MCP server instance.

    Factory function for creating MCP servers. This is the recommended way
    to instantiate the server for programmatic use.

    Args:
        name: Server name shown to MCP clients.
        config: Optional TenetsConfig for customization.

    Returns:
        Configured TenetsMCP instance ready to run.

    Example:
        >>> from tenets.mcp import create_server
        >>> server = create_server()
        >>> server.run(transport="stdio")
    """
    return TenetsMCP(name=name, config=config)


def main() -> None:
    """CLI entry point for tenets-mcp server.

    Parses command-line arguments and starts the MCP server with the
    specified transport configuration.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="tenets-mcp",
        description="Tenets MCP Server - Intelligent code context for AI assistants",
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for network transports (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Port for network transports (default: 8080)",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        from tenets import __version__

        print(f"tenets-mcp v{__version__}")
        sys.exit(0)

    try:
        server = create_server()
        server.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Install MCP dependencies with: pip install tenets[mcp]", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
