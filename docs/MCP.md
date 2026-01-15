---
title: Tenets MCP Server - Free Open Source for Cursor, Claude, Windsurf
description: Tenets MCP server for AI coding assistants. Free open source Python library and CLI. Intelligent code context + automatic guiding principles. 100% local, works with Cursor, Claude Desktop, Windsurf.
---

# MCP Server

Tenets is a **free, open source, 100% local** MCP server that solves two critical problems for AI coding:

1. **Intelligent Code Context** — NLP-powered ranking finds and aggregates the most relevant code automatically
2. **Automatic Guiding Principles** — Your tenets (coding standards, rules) are injected into every prompt, preventing context drift

Integrates natively with Cursor, Claude Desktop, Windsurf, and custom AI agents.

<div class="grid cards" markdown>

-   :material-cursor-default-outline: **[Cursor Setup](cursor-mcp.md)**

    Step-by-step guide for Cursor IDE

-   :material-chat-outline: **[Claude Desktop Setup](claude-mcp.md)**

    Configure Claude Desktop with Tenets

-   :material-microsoft-visual-studio-code: **[VSCode Extension](https://marketplace.visualstudio.com/items?itemName=ManicAgency.tenets-mcp-server)**

    Install from marketplace • [Setup guide](vscode-setup.md)

-   :material-rocket-launch-outline: **[Quick Start](quickstart.md)**

    Get started in 30 seconds

-   :material-help-circle-outline: **[FAQ](faq.md)**

    Common questions answered

</div>

**Learn more:** [Model Context Protocol Explained](blog/mcp-explained.md) · [Why Context Matters](blog/why-context-matters.md)

## Installation

```bash
pip install tenets[mcp]
```

## Quick Start

### Start the Server

```bash
# Default: stdio transport for local IDE integration
tenets-mcp

# SSE transport for web clients
tenets-mcp --transport sse --port 8080

# HTTP transport for remote deployment
tenets-mcp --transport http --port 8080
```

### Verify Installation

```bash
tenets-mcp --version
```

### Python 3.14 compatibility

- Use tenets >= 0.7.0 on Python 3.14 to avoid the prior import recursion in `tenets.core`.
- No extra flags needed; stdio/SSE/HTTP transports all work on 3.14+.

## IDE Configuration

### Claude Code (CLI / VS Code Extension)

**Recommended:** Use the CLI command to automatically configure:

```bash
claude mcp add tenets -s user -- tenets-mcp
```

Or manually add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "tenets": {
      "type": "stdio",
      "command": "tenets-mcp",
      "args": []
    }
  }
}
```

If you installed tenets in a virtual environment, use the full path:

```json
{
  "mcpServers": {
    "tenets": {
      "type": "stdio",
      "command": "/path/to/venv/bin/tenets-mcp",
      "args": []
    }
  }
}
```

### Claude Desktop (macOS App)

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp",
      "args": []
    }
  }
}
```

If you installed tenets in a virtual environment, use the full path:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "/path/to/venv/bin/tenets-mcp",
      "args": []
    }
  }
}
```

### Cursor

Add to your Cursor MCP settings (Settings → MCP Servers):

```json
{
  "tenets": {
    "command": "tenets-mcp",
    "args": []
  }
}
```

### Windsurf / Codeium

Configure the MCP server in Windsurf settings:

```json
{
  "mcp": {
    "servers": {
      "tenets": {
        "command": "tenets-mcp"
      }
    }
  }
}
```

### VS Code with MCP Extension

If using an MCP extension for VS Code:

```json
{
  "mcp.servers": {
    "tenets": {
      "command": "tenets-mcp",
      "transport": "stdio"
    }
  }
}
```

## Available Tools

The MCP server exposes 10 tools to AI assistants (consolidated from 13 in v0.8.0):

### Meta-Tools (Tool Discovery)

| Tool | Description |
|------|-------------|
| `tenets_search_tools` | Search available tools by keyword or category. Returns [{name, category, description}]. Use to discover specialized tools beyond distill/rank. |
| `tenets_get_tool_schema` | Get full input schema for a specific tool. Returns parameters, types, and descriptions. Call after search to get args for a tool. |

> **Token savings**: Meta-tools enable ~80% reduction in initial context (from ~15k to ~3k tokens) by loading full tool schemas on-demand rather than upfront.

### Context Tools

| Tool | Description |
|------|-------------|
| `tenets_distill` | Find and retrieve the most relevant code using semantic ranking. Returns context, files, and metadata. Use when exploring codebase or gathering context for tasks. |
| `tenets_rank_files` | Identify most relevant files without fetching content (fast file discovery). Returns scored file list. ~500ms vs ~3s for distill. |

Default distill timeout is 120s; pass the `timeout` argument (seconds, `0` to disable) to override.

### Analysis Tools

| Tool | Description |
|------|-------------|
| `tenets_examine` | Analyze codebase structure, complexity, and quality metrics from static analysis. Returns file counts, languages, complexity, hotspots. |
| `tenets_chronicle` | Analyze git history and recent development activity. Returns commits, file churn, contributors, temporal insights. Default: last 1 week. |
| `tenets_momentum` | Track development velocity and contribution patterns over time. Returns velocity, contributions, trends, health score. Default: last 1 week. |

### Session Tool (consolidated)

| Tool | Description |
|------|-------------|
| `tenets_session` | Manage development sessions for persistent context across conversations. Actions: `create`, `list`, `pin_file`, `pin_folder`. Pinned files always included in distill. |

### Tenet Tools (consolidated)

| Tool | Description |
|------|-------------|
| `tenets_tenet` | Manage guiding principles that auto-inject into all generated context to prevent drift. Actions: `add`, `list`, `instill`. Priorities: critical/high/medium/low. |
| `tenets_system_instruction` | Set one-time system instruction injected into all generated context. Positions: top/after_header/before_content. For persistent behavioral guidance. |

## Available Resources

Resources are read-only data the AI can access:

| Resource URI | Description |
|--------------|-------------|
| `tenets://sessions/list` | All development sessions |
| `tenets://sessions/{name}/state` | Specific session state |
| `tenets://sessions/active` | Currently active session |
| `tenets://tenets/list` | All guiding principles |
| `tenets://config/current` | Current configuration |
| `tenets://ranking/factors` | Ranking factor explanations |
| `tenets://analysis/hotspots` | Complexity hotspots |
| `tenets://analysis/summary` | Codebase summary |

## Available Prompts

Prompts are reusable templates for common tasks:

| Prompt | Description |
|--------|-------------|
| `build_context_for_task` | Build context for a development task |
| `understand_codebase` | Generate codebase understanding |
| `refactoring_guide` | Step-by-step refactoring workflow |
| `bug_investigation` | Systematic bug investigation workflow |
| `code_review` | Comprehensive code review checklist |
| `onboarding` | New developer onboarding workflow |

## Tool-to-Prompt Matrix

Common development tasks and which tools to use:

| Task | Tool(s) | Example Prompt |
|------|---------|----------------|
| Find relevant code | `tenets_distill` | "Find code related to payment processing" |
| Quick file scan | `tenets_rank_files` | "Which files are most relevant to auth?" |
| Code review | `tenets_distill` + `tenets_chronicle` | "Review recent changes to the API" |
| Onboarding | `tenets_examine` + `tenets_distill` | "Help me understand this codebase" |
| Feature planning | `tenets_distill` + `tenets_session(action='create')` | "I'm building a new feature for X" |
| Refactoring | `tenets_distill` + `tenets_rank_files` | "Find all usages of deprecated function" |
| Bug investigation | `tenets_chronicle` + `tenets_distill` | "Find changes that could cause issue X" |
| Track velocity | `tenets_momentum` | "Show development activity this week" |

## Example Usage

Once configured, ask your AI assistant:

> "Use tenets to find relevant files for implementing user authentication"

The AI will call the `tenets_distill` tool and return ranked, optimized context.

> "Create a session called 'auth-feature' and pin the auth folder"

The AI will use `tenets_session` with `action='create'` then `action='pin_folder'`.

> "Add a tenet: Always validate user input before processing"

The AI will use `tenets_tenet` with `action='add'` to create a guiding principle.

## Tool Responses (Examples)

### tenets_distill
```json
{
  "context": "# File: src/auth.py\n...",
  "token_count": 45000,
  "files": ["src/auth.py", "src/user.py"],
  "files_summarized": ["src/utils.py"],
  "metadata": {"mode": "balanced", "total_scanned": 150}
}
```

### tenets_rank_files
```json
{
  "files": [
    {"path": "src/auth.py", "score": 0.85, "factors": {"keyword_match": 0.9, "bm25_score": 0.8}},
    {"path": "src/user.py", "score": 0.72, "factors": {"keyword_match": 0.7, "path_relevance": 0.75}}
  ],
  "total_scanned": 150,
  "mode": "balanced"
}
```

### tenets_session (action='create')
```json
{
  "action": "create",
  "id": "sess_abc123",
  "name": "auth-feature",
  "created_at": "2025-12-04T10:00:00"
}
```

### tenets_tenet (action='add')
```json
{
  "action": "add",
  "id": "tenet_xyz789",
  "content": "Always validate user input",
  "priority": "high",
  "category": "security"
}
```

## Error Semantics

Tenets returns clear errors that AI agents can act on:

- Path errors (missing or inaccessible):
  - Distill/Rank will raise an error or return an empty result — agents should retry with a valid path.
- Invalid parameters (mode/format):
  - MCP validation prevents tool invocation; agents should correct parameters based on schema.
- Session operations:
  - Pinning a nonexistent file returns a structured failure response; agents should prompt to confirm the path.

When in doubt, agents should:
1) Validate inputs against the tool schema, and 2) Ask the user to clarify scope (path, include/exclude patterns, session name).

## Remote Deployment

For team or cloud deployment, use HTTP transport:

```bash
# Start server on all interfaces
tenets-mcp --transport http --host 0.0.0.0 --port 8080
```

Configure clients to connect:

```json
{
  "mcpServers": {
    "tenets": {
      "url": "http://your-server:8080/mcp"
    }
  }
}
```

## Programmatic Usage

```python
from tenets.mcp import create_server

# Create and run server
server = create_server()
server.run(transport="stdio")

# Or with custom config
from tenets.config import TenetsConfig
config = TenetsConfig(max_tokens=150000)
server = create_server(config=config)
server.run(transport="http", port=8080)
```

## Configuration

MCP settings can be added to `.tenets.yml`:

```yaml
mcp:
  enabled: true
  
  # Tool availability
  tools:
    distill: true
    rank_files: true
    examine: true
    chronicle: true
    momentum: true
    session: true
    tenet: true
```

## Troubleshooting

### Server not starting

1. Verify MCP dependencies: `pip install tenets[mcp]`
2. Check Python version: requires 3.9+
3. Try verbose mode: `tenets-mcp --verbose`

### IDE not connecting

1. Restart the IDE after configuration changes
2. Check the server is running: `tenets-mcp --version`
3. Verify the command path is correct in IDE settings
4. Check IDE logs for MCP connection errors

### Tools not appearing

1. Ensure MCP server is running
2. Refresh MCP connection in IDE
3. Check server logs for initialization errors

## MCP Discovery & Marketplaces

### How AI Assistants Find MCP Servers

MCP servers are discovered through configuration files. Each AI assistant has its own configuration format:

| Assistant | Config Location | Format |
|-----------|-----------------|--------|
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) | JSON |
| Cursor | Settings → MCP Servers | JSON |
| VS Code + MCP Extension | `.vscode/settings.json` or workspace settings | JSON |

### Publishing to MCP Directories

To make tenets discoverable by other developers:

#### 1. Official MCP Server Registry

Submit to the [MCP Servers Directory](https://github.com/modelcontextprotocol/servers):

```bash
# Fork and clone the servers repository
git clone https://github.com/modelcontextprotocol/servers.git

# Add your server to the README
# Submit a pull request
```

#### 2. Awesome MCP Lists

Submit to community-curated lists:
- [awesome-mcp](https://github.com/punkpeye/awesome-mcp) - Community curated
- [mcp-directory](https://mcphub.io) - Searchable directory

#### 3. Package Managers

Tenets is available via pip, making it easy to install:

```bash
pip install tenets[mcp]
```

### Local Development Setup

For developers working on tenets locally:

```bash
# Clone the repository
git clone https://github.com/jddunn/tenets.git
cd tenets

# Install in development mode with MCP support
pip install -e ".[mcp,dev]"

# Run the MCP server
tenets-mcp
```

### Registering with IDEs Automatically

Some IDEs support automatic MCP server discovery through package metadata.

#### pyproject.toml Entry Point

Tenets registers an entry point that MCP-aware tools can discover:

```toml
[project.scripts]
tenets-mcp = "tenets.mcp.server:main"
```

#### Manual Registration

If automatic discovery isn't available, add to your IDE config:

**Claude Desktop** (macOS):
```bash
# Open config
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

**Cursor**:
1. Open Settings (⌘ + ,)
2. Navigate to MCP Servers
3. Add new server with command `tenets-mcp`

### Verifying Installation

After configuration, verify the server is recognized:

1. **Restart your IDE** after config changes
2. **Check server status** in IDE's MCP panel
3. **Test a tool call**: Ask "Use tenets to rank files for authentication"

### Distributing to Teams

For team deployment, create a shared configuration:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp",
      "env": {
        "TENETS_MAX_TOKENS": "150000",
        "TENETS_RANKING_ALGORITHM": "balanced"
      }
    }
  }
}
```

Share this configuration file with your team for consistent setup.

## Enterprise Features

The following features are available with enterprise support:

- **Privacy Redaction**: Automatically strip API keys, credentials, and PII
- **Anonymization**: Remove author information for compliance
- **Custom ML Models**: Fine-tuned models for your codebase
- **SSO/SAML**: Enterprise authentication
- **Audit Logging**: Compliance tracking
- **Air-gapped Deployment**: Offline operation support

[Contact us](mailto:team@tenets.dev) for enterprise inquiries.

---

## Related Resources

- **[Cursor MCP Setup Guide](cursor-mcp.md)** — Detailed Cursor configuration
- **[Claude Desktop MCP Guide](claude-mcp.md)** — Configure Claude Desktop
- **[Quick Start](quickstart.md)** — Get started in 30 seconds
- **[CLI Reference](CLI.md)** — Command-line interface documentation
- **[FAQ](faq.md)** — Frequently asked questions
- **[Model Context Protocol Explained](blog/mcp-explained.md)** — Technical deep dive into MCP
- **[Why Context Matters](blog/why-context-matters.md)** — The importance of intelligent context

