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
import logging
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from tenets import Tenets
    from tenets.config import TenetsConfig

# Lazy imports to avoid loading MCP dependencies unless needed
_mcp_available = None

# Singleton instance for MCP server (preserves warm state across invocations)
_mcp_instance: Optional["TenetsMCP"] = None
_mcp_instance_lock = threading.Lock()


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
        self._warmed = False
        self._setup_server()

    @classmethod
    def get_instance(
        cls,
        name: str = "tenets",
        config: Optional["TenetsConfig"] = None,
        project_path: Optional[Path] = None,
    ) -> "TenetsMCP":
        """Get or create a singleton MCP server instance.

        Using a singleton preserves warm state (loaded analyzers, cached results)
        across multiple tool invocations, significantly improving response times.

        Args:
            name: Server name shown to MCP clients.
            config: Optional TenetsConfig. Only used on first creation.
            project_path: Optional project root path. Only used on first creation.

        Returns:
            The singleton TenetsMCP instance.
        """
        global _mcp_instance

        with _mcp_instance_lock:
            if _mcp_instance is None:
                _mcp_instance = cls(name=name, config=config, project_path=project_path)
            return _mcp_instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        Primarily for testing. Clears the cached instance so the next
        call to get_instance() creates a fresh server.
        """
        global _mcp_instance

        with _mcp_instance_lock:
            _mcp_instance = None

    def warm_components(self) -> None:
        """Pre-warm critical components for faster tool responses.

        Triggers lazy loading of expensive components (analyzers, rankers)
        before the first tool invocation, reducing initial response latency.

        This is called automatically on server start when using run().
        """
        if self._warmed:
            return

        logger = logging.getLogger(__name__)
        logger.debug("Pre-warming MCP server components...")

        try:
            # Trigger lazy loading of the Tenets instance
            tenets = self.tenets

            # Warm the distiller's components
            if hasattr(tenets, "distiller"):
                distiller = tenets.distiller
                # Access analyzer to trigger initialization
                _ = distiller.analyzer
                # Access ranker to trigger initialization
                _ = distiller.ranker
                logger.debug("Distiller components warmed")

            # Pre-warm token encoding cache
            from tenets.utils.tokens import _get_cached_encoding

            _get_cached_encoding(None)  # Default encoding
            _get_cached_encoding("gpt-4o")  # Common model
            logger.debug("Token encoding cache warmed")

            self._warmed = True
            logger.debug("MCP server components pre-warmed successfully")

        except Exception as e:
            logger.warning(f"Failed to pre-warm components: {e}")
            # Don't fail startup if warming fails

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
        """Register all MCP tools.

        Tools are prefixed with 'tenets_' to distinguish them from other MCPs
        and improve LLM discoverability. Descriptions follow a pattern:
        1. WHAT it does (one line)
        2. WHEN to use it (explicit trigger phrases)
        3. WHY it's necessary (prevent LLM from skipping)
        4. Encouragement to explore further after using the tool
        """
        mcp = self._mcp

        # === Context Tools ===

        @mcp.tool()
        async def tenets_distill(
            prompt: str,
            path: str = ".",
            mode: Literal["fast", "balanced", "thorough"] = "balanced",
            max_tokens: int = 100000,
            format: Literal["markdown", "xml", "json", "html"] = "markdown",
            include_tests: bool = False,
            include_git: bool = True,
            session: Optional[str] = None,
            include_patterns: Optional[list[str]] = None,
            exclude_patterns: Optional[list[str]] = None,
            timeout: Optional[int] = 120,
        ) -> dict[str, Any]:
            """Find and retrieve the most relevant code for a task using semantic ranking.

            Use when: user asks "how does X work", "find code for", "where is Y implemented", "understand feature",
            "gather context for bug/task", or needs codebase exploration before coding.
            Do not use when: user already knows exact file paths or wants general project info (use tenets_examine).

            Inputs:
            - prompt (required): Specific task description. Good: "OAuth2 JWT auth flow", "payment error handling".
              Bad: "auth stuff", "the bug".
            - path: Directory to search (default "." for current project).
            - mode: "fast" (~1s keyword), "balanced" (~3s BM25+structure, recommended), "thorough" (~10s ML embeddings).
            - max_tokens: Context budget (default 100k).
            - format: "markdown" (default), "xml" (Claude-optimized), "json", "html".
            - include_tests: True to include test files (default False).
            - include_patterns/exclude_patterns: File filter globs (e.g., ["*.py"], ["*.min.js"]).
            - session: Session name to include pinned files.
            - timeout: Max seconds (default 120, <=0 disables).

            Returns: {context: str, token_count: int, files: [str], metadata: {mode, timing, session}}.
            The files list shows what was included—read specific files for deeper understanding.

            Common errors:
            - Timeout → reduce max_tokens or use "fast" mode.
            - No files found → check path exists and patterns allow target files.
            - Too many files → use exclude_patterns or more specific prompt.
            """
            result = self.tenets.distill(
                prompt=prompt,
                files=path,
                mode=mode,
                max_tokens=max_tokens,
                format=format,
                include_git=include_git,
                include_tests=include_tests,
                session_name=session,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                timeout=timeout,
            )
            return result.to_dict()

        @mcp.tool()
        async def tenets_rank_files(
            prompt: str,
            path: str = ".",
            mode: Literal["fast", "balanced", "thorough", "ml"] = "balanced",
            top_n: int = 20,
            include_tests: bool = False,
            exclude_tests: bool = False,
            include_patterns: Optional[list[str]] = None,
            exclude_patterns: Optional[list[str]] = None,
            explain: bool = False,
        ) -> dict[str, Any]:
            """Identify the most relevant files for a task without fetching content (fast file discovery).

            Use when: user asks "which files to check", "what files handle X", "scout codebase", "plan changes",
            "quick file overview", or wants to verify search area before full context fetch.
            Do not use when: user needs actual code content (use tenets_distill) or general stats (use tenets_examine).

            Inputs:
            - prompt (required): Task description, same format as tenets_distill.
            - path: Directory to search (default ".").
            - mode: "fast" (keyword), "balanced" (BM25+structure, recommended), "thorough" (deep), "ml" (embeddings).
            - top_n: Max files to return (default 20).
            - include_tests/exclude_tests: Control test file inclusion (default: exclude).
            - include_patterns/exclude_patterns: File filter globs (e.g., ["*.py"], ["*.min.js"]).
            - explain: True to show ranking factors breakdown (default False).

            Returns: {files: [{path: str, score: float, factors?: dict}], total_scanned: int, mode: str}.
            Faster than distill (~500ms vs ~3s). Recommended workflow: rank_files → read top files → distill if needed.

            Common errors:
            - Empty files array → check path exists and patterns match target files.
            - Low scores (<0.3) → prompt may be too vague or files don't match query.
            - explain=True for debugging relevance issues.
            """
            result = self.tenets.rank_files(
                prompt=prompt,
                paths=path,
                mode=mode,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                include_tests=include_tests,
                exclude_tests=exclude_tests,
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
        async def tenets_examine(
            path: str = ".",
            include_complexity: bool = True,
            include_hotspots: bool = True,
        ) -> dict[str, Any]:
            """Analyze codebase structure, complexity, and quality metrics from static analysis.

            Use when: user asks "what's in this repo", "project structure", "complex files", "hotspots",
            "tech debt", "language breakdown", "file counts", "how is this organized", or needs architecture overview.
            Do not use when: user wants specific code content (use tenets_distill) or file discovery (use tenets_rank_files).

            Inputs:
            - path: Root directory to examine (default ".").
            - include_complexity: Include cyclomatic complexity metrics (default True).
            - include_hotspots: Identify high-churn + high-complexity areas (default True).

            Returns: {file_counts: dict, languages: dict, complexity: dict, hotspots: [str]}.
            Use results to prioritize exploration—dive into hotspots or complex areas with tenets_distill/tenets_rank_files.

            Common errors:
            - Empty results → verify path is a valid codebase directory.
            - Missing git data → hotspots require git history; disable include_hotspots if not a git repo.
            """
            result = self.tenets.examine(
                path=path,
                deep=include_complexity,
            )
            return result if isinstance(result, dict) else {"result": str(result)}

        @mcp.tool()
        async def tenets_chronicle(
            path: str = ".",
            since: str = "1 week",
            author: Optional[str] = None,
        ) -> dict[str, Any]:
            """Analyze git history and recent development activity from commit data.

            Use when: user asks "what changed recently", "what's new", "recent commits", "who worked on X",
            "file history", "commit history", "review prep", "recent activity", "change patterns", "hot areas".
            Do not use when: user wants future plans (not in git) or non-git projects.

            Inputs:
            - path: Repository directory (default ".").
            - since: Time period like "1 week", "3 days", "1 month" (default "1 week").
            - author: Filter by author name or email pattern (optional).

            Returns: {commits: [dict], file_churn: dict, contributors: dict, temporal: dict}.
            High-churn files often indicate active development or problem areas—explore them with tenets_distill.

            Common errors:
            - Not a git repo → requires .git directory; returns empty if path is not a git repository.
            - No commits in period → try longer 'since' value like "1 month" or "6 months".
            """
            result = self.tenets.track_changes(
                path=path,
                since=since,
                author=author,
            )
            return result if isinstance(result, dict) else {"result": str(result)}

        @mcp.tool()
        async def tenets_momentum(
            path: str = ".",
            since: str = "1 week",
            team: bool = False,
        ) -> dict[str, Any]:
            """Track development velocity and contribution patterns over time from git metrics.

            Use when: user asks "how fast is development", "sprint progress", "throughput", "who's contributing",
            "team momentum", "bus factor", "development patterns", "activity trends", "slowdowns", "project health",
            "is this active", "maintenance status".
            Do not use when: user wants static analysis (use tenets_examine) or specific commits (use tenets_chronicle).

            Inputs:
            - path: Repository directory (default ".").
            - since: Time period like "1 week", "1 month", "3 months" (default "1 week").
            - team: True for team-wide stats, False for individual breakdown (default False).

            Returns: {velocity: dict, contributions: dict, trends: dict, health_score: float}.
            Use to assess project health and identify active vs stale areas.

            Common errors:
            - Not a git repo → requires .git directory.
            - Low commit count → try longer 'since' period for meaningful trend analysis.
            """
            result = self.tenets.momentum(
                path=path,
                since=since,
                team=team,
            )
            return result if isinstance(result, dict) else {"result": str(result)}

        # === Session Management (Consolidated) ===

        @mcp.tool()
        async def tenets_session(
            action: Literal["create", "list", "pin_file", "pin_folder"],
            name: Optional[str] = None,
            description: Optional[str] = None,
            file_path: Optional[str] = None,
            folder_path: Optional[str] = None,
            patterns: Optional[list[str]] = None,
        ) -> dict[str, Any]:
            """Manage development sessions for persistent context that survives across conversations.

            Use when: user says "start session", "create session", "resume work", "pin file", "track context",
            "remember this file", "always include", or needs to maintain focus across multiple conversations.
            Do not use when: user wants one-time context fetch (use tenets_distill directly).

            Actions (set via action parameter):
            - "create": Start new session. Required: name. Optional: description.
              Use when: "start session for X", "create session", "begin work on Y".
            - "list": Show all sessions. No parameters needed.
              Use when: "what sessions exist", "show sessions", "resume work" (to find session name).
            - "pin_file": Pin single file to session. Required: name, file_path.
              Use when: "always include this file", "pin config", "remember X.py".
            - "pin_folder": Pin all matching files in folder. Required: name, folder_path. Optional: patterns.
              Use when: "pin all tests", "remember entire module", "track folder".

            Pinned files are ALWAYS included in tenets_distill for that session regardless of ranking.

            Inputs: action (required), name, description, file_path, folder_path, patterns (see action descriptions).
            Returns: {action: str, ...action-specific fields}.

            Common errors:
            - Missing name → required for create/pin_file/pin_folder.
            - Missing file_path/folder_path → required for respective pin actions.
            - Session not found → use action="list" to see existing sessions.
            """
            from tenets.storage.session_db import SessionDB

            if action == "create":
                if not name:
                    return {"error": "name is required for create action"}
                db = SessionDB(self.tenets.config)
                metadata = {"description": description} if description else {}
                session = db.create_session(name, metadata=metadata)
                return {
                    "action": "create",
                    "id": session.id,
                    "name": session.name,
                    "created_at": session.created_at.isoformat(),
                }

            elif action == "list":
                db = SessionDB(self.tenets.config)
                sessions = db.list_sessions()
                return {
                    "action": "list",
                    "sessions": [
                        {
                            "id": s.id,
                            "name": s.name,
                            "created_at": s.created_at.isoformat(),
                            "metadata": s.metadata,
                        }
                        for s in sessions
                    ],
                }

            elif action == "pin_file":
                if not name or not file_path:
                    return {"error": "name and file_path are required for pin_file action"}
                success = self.tenets.add_file_to_session(file_path, session=name)
                return {
                    "action": "pin_file",
                    "success": success,
                    "file": file_path,
                    "session": name,
                }

            elif action == "pin_folder":
                if not name or not folder_path:
                    return {"error": "name and folder_path are required for pin_folder action"}
                count = self.tenets.add_folder_to_session(
                    folder_path,
                    session=name,
                    include_patterns=patterns,
                )
                return {
                    "action": "pin_folder",
                    "pinned_count": count,
                    "folder": folder_path,
                    "session": name,
                    "patterns": patterns,
                }

            else:
                return {"error": f"Unknown action: {action}"}

        # === Tenet Management (Consolidated) ===

        @mcp.tool()
        async def tenets_tenet(
            action: Literal["add", "list", "instill"],
            content: Optional[str] = None,
            priority: Literal["low", "medium", "high", "critical"] = "medium",
            category: Optional[str] = None,
            session: Optional[str] = None,
            pending_only: bool = False,
            force: bool = False,
        ) -> dict[str, Any]:
            """Manage guiding principles that get auto-injected into all generated context to prevent drift.

            Use when: user says "always use X", "remember rule", "enforce standard", "coding guideline",
            "never do Y", "architectural decision", "security rule", "maintain consistency", "follow pattern".
            Do not use when: user wants one-time instruction (use tenets_system_instruction).

            Actions (set via action parameter):
            - "add": Create principle. Required: content. Optional: priority, category, session.
              Use when: "always use type hints", "validate input", "no magic numbers".
              Good content: "Validate all user input before DB queries" (specific, actionable).
              Bad content: "Be careful with security" (vague).
            - "list": Show existing tenets. Optional: session, pending_only.
              Use when: "what are my rules", "show tenets", "check guidelines".
            - "instill": Activate pending tenets. Optional: session, force.
              Use when: "apply tenets", "activate rules" (usually automatic, rarely needed manually).

            Priority levels (for add): "critical" (every context), "high" (most contexts), "medium" (default),
            "low" (occasional). Higher priority = more frequent injection to combat context drift.

            Inputs: action (required), content, priority, category, session, pending_only, force.
            Returns: {action: str, ...action-specific fields}.

            Common errors:
            - Missing content → required for "add" action.
            - Tenets not appearing → run action="instill" to activate pending tenets.
            """
            if action == "add":
                if not content:
                    return {"error": "content is required for add action"}
                tenet = self.tenets.add_tenet(
                    content=content,
                    priority=priority,
                    category=category,
                    session=session,
                )
                return {
                    "action": "add",
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

            elif action == "list":
                tenets_list = self.tenets.list_tenets(
                    session=session,
                    pending_only=pending_only,
                )
                return {"action": "list", "tenets": tenets_list}

            elif action == "instill":
                result = self.tenets.instill_tenets(session=session, force=force)
                if isinstance(result, dict):
                    result["action"] = "instill"
                    return result
                return {"action": "instill", "result": str(result)}

            else:
                return {"error": f"Unknown action: {action}"}

        # === System Instruction ===

        @mcp.tool()
        async def tenets_system_instruction(
            instruction: str,
            position: Literal["top", "after_header", "before_content"] = "top",
        ) -> dict[str, Any]:
            """Set a one-time system instruction injected into all generated context.

            Use when: user says "add instruction", "always explain reasoning", "be concise", "focus on security",
            "use TypeScript conventions", "this is a Django project", or needs persistent behavioral guidance.
            Do not use when: user wants repeating principles (use tenets_tenet for rules that auto-inject).

            Inputs:
            - instruction (required): System instruction text. Be clear and specific.
            - position: Injection location—"top" (default, highest visibility), "after_header", "before_content".

            Returns: {success: bool, instruction_length: int, position: str}.
            Instruction appears at specified position in every tenets_distill output for consistent behavior.

            Common errors:
            - Instruction too vague → be specific (good: "Use TypeScript strict mode", bad: "use TS").
            - Wrong tool → for repeating rules use tenets_tenet; this is for one-time project-wide instructions.
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
        """Register all MCP resources.

        Resources provide read-only access to tenets data without requiring
        tool calls. LLMs can access these directly for quick lookups.
        """
        mcp = self._mcp

        # === Session Resources ===

        @mcp.resource("tenets://sessions/list")
        async def get_sessions_list() -> str:
            """List of all development sessions."""
            import json

            from tenets.storage.session_db import SessionDB

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
            import json

            from tenets.storage.session_db import SessionDB

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

        # === Tenet Resources ===

        @mcp.resource("tenets://tenets/list")
        async def get_tenets_list() -> str:
            """List of all guiding principles (tenets)."""
            import json

            tenets = self.tenets.list_tenets()
            return json.dumps(tenets, indent=2, default=str)

        # === Configuration Resources ===

        @mcp.resource("tenets://config/current")
        async def get_current_config() -> str:
            """Current tenets configuration (read-only)."""
            import json

            config_dict = self.tenets.config.to_dict()
            # Remove sensitive data
            if "llm" in config_dict and "api_keys" in config_dict["llm"]:
                config_dict["llm"]["api_keys"] = {k: "***" for k in config_dict["llm"]["api_keys"]}
            return json.dumps(config_dict, indent=2, default=str)

        # === Ranking Factors Resource ===

        @mcp.resource("tenets://ranking/factors")
        async def get_ranking_factors() -> str:
            """Explains how files are ranked for relevance.

            Use this to understand why certain files appear in results
            and how to improve your queries for better ranking.
            """
            import json

            factors = {
                "description": "How tenets ranks files for relevance to your query",
                "modes": {
                    "fast": {
                        "description": "Keyword-based matching (~1s)",
                        "factors": [
                            "Exact keyword matches in file content",
                            "Keyword matches in file path/name",
                            "Word frequency (TF-IDF)",
                        ],
                        "best_for": "Simple queries, known keywords",
                    },
                    "balanced": {
                        "description": "BM25 + structural analysis (~3s)",
                        "factors": [
                            "BM25 text relevance score",
                            "Code structure (classes, functions)",
                            "Import/dependency relationships",
                            "File recency (recent changes boost)",
                            "Path relevance (matches query terms)",
                        ],
                        "best_for": "Most queries (recommended default)",
                    },
                    "thorough": {
                        "description": "Deep semantic analysis (~10s)",
                        "factors": [
                            "All balanced mode factors",
                            "AST-based code understanding",
                            "Cross-file relationship mapping",
                            "Complexity-weighted scoring",
                        ],
                        "best_for": "Complex queries, architectural questions",
                    },
                    "ml": {
                        "description": "ML embedding similarity",
                        "factors": [
                            "Semantic embedding similarity",
                            "Contextual understanding",
                            "Concept matching (not just keywords)",
                        ],
                        "best_for": "Conceptual queries, 'how does X work'",
                    },
                },
                "tips": [
                    "Use specific terms over generic ones",
                    "Include function/class names if known",
                    "Describe the behavior, not just keywords",
                    "Use 'balanced' mode for most queries",
                    "Use 'ml' mode for conceptual questions",
                ],
            }
            return json.dumps(factors, indent=2)

        # === Active Session Resource ===

        @mcp.resource("tenets://sessions/active")
        async def get_active_session() -> str:
            """Returns the currently active session, if any.

            Active session is determined by most recent activity
            or explicit activation.
            """
            import json

            from tenets.storage.session_db import SessionDB

            db = SessionDB(self.tenets.config)
            sessions = db.list_sessions()

            if not sessions:
                return json.dumps(
                    {
                        "active": False,
                        "message": "No sessions exist. Create one with tenets_session(action='create').",
                    }
                )

            # Most recent session is considered active
            most_recent = max(sessions, key=lambda s: s.created_at)
            return json.dumps(
                {
                    "active": True,
                    "session": {
                        "name": most_recent.name,
                        "created_at": most_recent.created_at.isoformat(),
                        "metadata": most_recent.metadata,
                    },
                },
                indent=2,
            )

        # === Hotspots Resource ===

        @mcp.resource("tenets://analysis/hotspots")
        async def get_hotspots() -> str:
            """Pre-computed complexity hotspots in the codebase.

            Hotspots are files with high complexity and/or high churn,
            indicating areas that may need attention or refactoring.
            """
            import json

            try:
                # Get examination results which include hotspots
                result = self.tenets.examine(path=".", deep=True)
                if isinstance(result, dict):
                    hotspots = result.get("hotspots", [])
                    complexity = result.get("complexity", {})

                    return json.dumps(
                        {
                            "description": "Files with high complexity or frequent changes",
                            "hotspots": hotspots[:20] if hotspots else [],
                            "high_complexity_files": (
                                [
                                    {"path": k, "complexity": v}
                                    for k, v in sorted(
                                        complexity.items(),
                                        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                                        reverse=True,
                                    )[:10]
                                ]
                                if complexity
                                else []
                            ),
                            "recommendations": [
                                "Consider refactoring files with complexity > 20",
                                "High-churn + high-complexity = maintenance risk",
                                "Add tests for hotspot files first",
                            ],
                        },
                        indent=2,
                    )
                return json.dumps({"error": "Could not analyze hotspots"})
            except Exception as e:
                return json.dumps({"error": str(e)})

        # === Codebase Summary Resource ===

        @mcp.resource("tenets://analysis/summary")
        async def get_codebase_summary() -> str:
            """Quick summary of the codebase structure and metrics.

            Provides an overview without requiring a full examine call.
            """
            import json

            try:
                result = self.tenets.examine(path=".", deep=False)
                if isinstance(result, dict):
                    return json.dumps(
                        {
                            "description": "Codebase overview",
                            "languages": result.get("languages", {}),
                            "file_count": result.get("file_count", 0),
                            "total_lines": result.get("total_lines", 0),
                            "structure": result.get("structure", {}),
                        },
                        indent=2,
                    )
                return json.dumps({"error": "Could not analyze codebase"})
            except Exception as e:
                return json.dumps({"error": str(e)})

    def _register_prompts(self) -> None:
        """Register all MCP prompt templates.

        Prompts provide pre-built workflows that guide LLMs through common
        development tasks using the tenets tools effectively.
        """
        mcp = self._mcp

        # === Context Building Prompts ===

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
                "Please use the tenets_distill tool to build relevant context.",
            ]
            if focus_areas:
                prompt_parts.append(f"Focus on these areas: {focus_areas}")
            return "\n".join(prompt_parts)

        # === Refactoring Workflow Prompt ===

        @mcp.prompt()
        def refactoring_guide(
            target: str,
            goal: Optional[str] = None,
            safety_level: Literal["conservative", "moderate", "aggressive"] = "moderate",
        ) -> str:
            """Guide for safe code refactoring with proper context gathering.

            Provides a step-by-step workflow for refactoring code safely,
            ensuring all dependencies and usages are understood before changes.

            Args:
                target: What to refactor (function, class, module, or file path).
                goal: Optional refactoring goal (e.g., "improve readability").
                safety_level: How aggressive to be with changes.
            """
            parts = [
                f"# Refactoring Guide: {target}",
                "",
                "## Step 1: Understand Current State",
                f"Use `tenets_distill` with prompt: 'How does {target} work and what depends on it'",
                "",
                "## Step 2: Map Dependencies",
                f"Use `tenets_rank_files` with prompt: 'files that import or call {target}'",
                "Then READ the top results to understand usage patterns.",
                "",
                "## Step 3: Identify Test Coverage",
                f"Use `tenets_distill` with include_tests=True, prompt: 'tests for {target}'",
                "",
                "## Step 4: Plan Changes",
            ]

            if goal:
                parts.append(f"Goal: {goal}")

            safety_guidance = {
                "conservative": "Make minimal changes. Preserve all existing behavior.",
                "moderate": "Balance improvements with safety. Add deprecation warnings.",
                "aggressive": "Prioritize clean design. May require broader updates.",
            }
            parts.append(f"Safety level: {safety_level} - {safety_guidance[safety_level]}")

            parts.extend(
                [
                    "",
                    "## Step 5: Execute Refactoring",
                    "- Make changes incrementally",
                    "- Run tests after each change",
                    "- Update callers if needed",
                    "",
                    "## Step 6: Verify",
                    "- Run full test suite",
                    "- Check for type errors",
                    "- Review changed files",
                ]
            )

            return "\n".join(parts)

        # === Bug Investigation Prompt ===

        @mcp.prompt()
        def bug_investigation(
            symptom: str,
            location_hint: Optional[str] = None,
            include_history: bool = True,
        ) -> str:
            """Systematic workflow for investigating and fixing bugs.

            Guides through gathering context, understanding the bug,
            finding root cause, and implementing a fix.

            Args:
                symptom: Description of the bug or error message.
                location_hint: Optional hint about where the bug might be.
                include_history: Whether to check git history for related changes.
            """
            parts = [
                f"# Bug Investigation: {symptom}",
                "",
                "## Step 1: Gather Context",
            ]

            if location_hint:
                parts.append(
                    f"Use `tenets_distill` with prompt: '{symptom}' focused on {location_hint}"
                )
            else:
                parts.append(f"Use `tenets_distill` with prompt: 'code related to: {symptom}'")

            parts.extend(
                [
                    "",
                    "## Step 2: Find Related Code",
                    f"Use `tenets_rank_files` with prompt: '{symptom}'",
                    "READ the top 3-5 files to understand the code flow.",
                    "",
                    "## Step 3: Identify Error Handling",
                    "Look for:",
                    "- try/except blocks in the area",
                    "- Error messages that match the symptom",
                    "- Logging statements",
                    "",
                ]
            )

            if include_history:
                parts.extend(
                    [
                        "## Step 4: Check Recent Changes",
                        "Use `tenets_chronicle` with since='2 weeks' to see recent changes.",
                        "Recent changes to affected files may have introduced the bug.",
                        "",
                    ]
                )

            parts.extend(
                [
                    "## Step 5: Form Hypothesis",
                    "Based on context gathered:",
                    "1. What is the expected behavior?",
                    "2. What is the actual behavior?",
                    "3. What code path leads to the bug?",
                    "",
                    "## Step 6: Fix and Verify",
                    "- Implement fix",
                    "- Add test case that reproduces the bug",
                    "- Verify fix resolves the issue",
                    "- Check for similar patterns elsewhere",
                ]
            )

            return "\n".join(parts)

        # === Code Review Prompt ===

        @mcp.prompt()
        def code_review(
            scope: Literal["recent", "file", "pr", "module"] = "recent",
            focus: Optional[str] = None,
            since: str = "1 week",
        ) -> str:
            """Comprehensive code review workflow with context.

            Prepares context and checklist for thorough code review,
            including checking for common issues and patterns.

            Args:
                scope: Review scope - recent changes, specific file, PR, or module.
                focus: Optional focus area (security, performance, style, etc.).
                since: Time period for recent changes (default: 1 week).
            """
            parts = [
                f"# Code Review ({scope} scope)",
                "",
            ]

            if scope == "recent":
                parts.extend(
                    [
                        "## Step 1: Identify Changes",
                        f"Use `tenets_chronicle` with since='{since}' to see recent commits.",
                        "",
                        "## Step 2: Gather Context",
                        "For each changed file, use `tenets_distill` to understand:",
                        "- What the code does",
                        "- Why changes were made",
                        "- Impact on other parts of the codebase",
                    ]
                )
            elif scope == "file":
                parts.extend(
                    [
                        "## Step 1: Understand the File",
                        "Use `tenets_distill` with the file path to get context.",
                        "",
                        "## Step 2: Check Dependencies",
                        "Use `tenets_rank_files` to find related files.",
                    ]
                )
            elif scope == "pr":
                parts.extend(
                    [
                        "## Step 1: Review PR Changes",
                        "List all files changed in the PR.",
                        "",
                        "## Step 2: Understand Context",
                        "For each changed file, use `tenets_distill` to understand the change.",
                    ]
                )
            else:  # module
                parts.extend(
                    [
                        "## Step 1: Examine Module Structure",
                        "Use `tenets_examine` to see module structure and complexity.",
                        "",
                        "## Step 2: Understand Module",
                        "Use `tenets_distill` with the module path.",
                    ]
                )

            parts.extend(
                [
                    "",
                    "## Review Checklist",
                    "",
                    "### Correctness",
                    "- [ ] Logic is correct",
                    "- [ ] Edge cases handled",
                    "- [ ] Error handling appropriate",
                    "",
                    "### Security",
                    "- [ ] Input validated",
                    "- [ ] No hardcoded secrets",
                    "- [ ] SQL injection safe",
                    "- [ ] XSS prevented",
                    "",
                    "### Performance",
                    "- [ ] No unnecessary loops",
                    "- [ ] Database queries optimized",
                    "- [ ] Memory usage reasonable",
                    "",
                    "### Maintainability",
                    "- [ ] Code is readable",
                    "- [ ] Functions single-purpose",
                    "- [ ] Naming is clear",
                    "- [ ] Tests included",
                ]
            )

            if focus:
                parts.extend(
                    [
                        "",
                        f"## Special Focus: {focus}",
                        f"Pay extra attention to {focus}-related issues.",
                    ]
                )

            return "\n".join(parts)

        # === Onboarding Prompt ===

        @mcp.prompt()
        def onboarding(
            role: Literal["developer", "reviewer", "maintainer"] = "developer",
            focus_area: Optional[str] = None,
        ) -> str:
            """New developer onboarding workflow for understanding a codebase.

            Provides a structured approach to learning a new codebase,
            from high-level overview to specific implementation details.

            Args:
                role: The person's role - affects what to focus on.
                focus_area: Optional specific area to focus on first.
            """
            parts = [
                "# Codebase Onboarding",
                "",
                "## Step 1: High-Level Overview",
                "Use `tenets_examine` to see:",
                "- Languages used",
                "- File counts and structure",
                "- Complexity hotspots",
                "",
                "## Step 2: Architecture",
                "Use `tenets_distill` with prompt: 'main architecture and entry points'",
                "Identify:",
                "- Main modules/packages",
                "- Entry points",
                "- Core abstractions",
                "",
                "## Step 3: Recent Activity",
                "Use `tenets_chronicle` with since='1 month' to see:",
                "- Active areas of development",
                "- Key contributors",
                "- Recent focus areas",
                "",
            ]

            if focus_area:
                parts.extend(
                    [
                        f"## Step 4: Focus Area - {focus_area}",
                        f"Use `tenets_distill` with prompt: 'how {focus_area} works'",
                        "Then READ the returned files to understand the implementation.",
                        "",
                    ]
                )

            role_guidance = {
                "developer": [
                    "## Developer Focus",
                    "- Understand coding conventions",
                    "- Find example implementations",
                    "- Identify testing patterns",
                    "- Learn the build/deploy process",
                ],
                "reviewer": [
                    "## Reviewer Focus",
                    "- Understand quality standards",
                    "- Identify high-risk areas",
                    "- Learn the review checklist",
                    "- Know the security requirements",
                ],
                "maintainer": [
                    "## Maintainer Focus",
                    "- Identify technical debt",
                    "- Understand deployment pipeline",
                    "- Know the monitoring/alerting",
                    "- Map external dependencies",
                ],
            }

            parts.extend(role_guidance.get(role, []))

            parts.extend(
                [
                    "",
                    "## Key Questions to Answer",
                    "1. How do I run the project locally?",
                    "2. How do I run tests?",
                    "3. What's the deployment process?",
                    "4. Where do I find documentation?",
                    "5. Who do I ask for help?",
                ]
            )

            return "\n".join(parts)

        # === Understand Codebase (Updated) ===

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
                    "1. Use `tenets_examine` to see codebase structure",
                    "2. Use `tenets_distill` with an understanding prompt",
                    "3. Identify key architectural patterns",
                ]
            )
            return "\n".join(parts)

    def run(
        self,
        transport: Literal["stdio", "sse", "http"] = "stdio",
        host: str = "127.0.0.1",
        port: int = 8080,
        warm: bool = True,
    ) -> None:
        """Run the MCP server with the specified transport.

        Args:
            transport: Transport type - stdio (local), sse, or http (remote).
            host: Host for network transports (sse, http).
            port: Port for network transports (sse, http).
            warm: Whether to pre-warm components before starting (default True).
        """
        # Pre-warm components for faster first response
        if warm:
            self.warm_components()

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
    use_singleton: bool = True,
) -> TenetsMCP:
    """Create or get a Tenets MCP server instance.

    Factory function for creating MCP servers. This is the recommended way
    to instantiate the server for programmatic use.

    By default, uses a singleton pattern to preserve warm state across
    invocations, significantly improving response times for repeated calls.

    Args:
        name: Server name shown to MCP clients.
        config: Optional TenetsConfig for customization.
        use_singleton: If True (default), returns a shared singleton instance.
            Set to False to create a fresh instance each time.

    Returns:
        Configured TenetsMCP instance ready to run.

    Example:
        >>> from tenets.mcp import create_server
        >>> server = create_server()
        >>> server.run(transport="stdio")
    """
    if use_singleton:
        return TenetsMCP.get_instance(name=name, config=config)
    return TenetsMCP(name=name, config=config)


def _configure_mcp_logging(transport: str) -> None:
    """Configure logging for MCP server to avoid polluting stdout.

    MCP stdio transport uses stdout for JSON-RPC messages.
    Any non-JSON output (like colored logs) breaks the protocol.

    Args:
        transport: The transport type (stdio, sse, http)
    """
    import logging
    import os

    # Disable colored output - MCP expects clean JSON on stdout
    os.environ["NO_COLOR"] = "1"
    os.environ["FORCE_COLOR"] = "0"

    # For stdio transport, redirect all logging to stderr
    if transport == "stdio":
        # Configure root logger to use stderr with no colors
        logging.basicConfig(
            level=logging.WARNING,  # Only warnings and errors
            format="%(levelname)s: %(message)s",
            stream=sys.stderr,
            force=True,  # Override any existing configuration
        )

        # Suppress verbose loggers
        for logger_name in ["tenets", "mcp", "httpx", "httpcore", "asyncio"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


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

        print(f"tenets-mcp v{__version__}", file=sys.stderr)
        sys.exit(0)

    # Configure logging BEFORE importing anything else
    _configure_mcp_logging(args.transport)

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
        print("\nServer stopped.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
