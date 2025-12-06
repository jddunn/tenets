# MCP Server

Tenets includes a Model Context Protocol (MCP) server that integrates with AI coding assistants like Cursor, Claude Desktop, Windsurf, and custom AI agents.

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

### Claude Desktop

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

The MCP server exposes these tools to AI assistants:

### Context Tools

| Tool | Description |
|------|-------------|
| `distill` | Build ranked, token-optimized context from codebase (markdown/xml/json/html) |
| `rank_files` | Rank files by relevance without full content (supports include/exclude patterns, test filters) |

### Analysis Tools

| Tool | Description |
|------|-------------|
| `examine` | Analyze codebase structure and complexity |
| `chronicle` | Analyze git history and patterns |
| `momentum` | Track development velocity |

### Session Tools

| Tool | Description |
|------|-------------|
| `session_create` | Create a development session |
| `session_list` | List all sessions |
| `session_pin_file` | Pin a file to a session |
| `session_pin_folder` | Pin a folder to a session |

### Tenet Tools

| Tool | Description |
|------|-------------|
| `tenet_add` | Add a guiding principle |
| `tenet_list` | List all tenets |
| `tenet_instill` | Activate pending tenets |
| `set_system_instruction` | Set system-level instruction |

## Available Resources

Resources are read-only data the AI can access:

| Resource URI | Description |
|--------------|-------------|
| `tenets://sessions/list` | All development sessions |
| `tenets://sessions/{name}/state` | Specific session state |
| `tenets://tenets/list` | All guiding principles |
| `tenets://config/current` | Current configuration |

## Available Prompts

Prompts are reusable templates for common tasks:

| Prompt | Description |
|--------|-------------|
| `build_context_for_task` | Build context for a development task |
| `code_review_context` | Prepare context for code review |
| `understand_codebase` | Generate codebase understanding |

## Tool-to-Prompt Matrix

Common development tasks and which tools to use:

| Task | Tool(s) | Example Prompt |
|------|---------|----------------|
| Find relevant code | `distill` | "Find code related to payment processing" |
| Quick file scan | `rank_files` | "Which files are most relevant to auth?" |
| Code review | `distill` + `chronicle` | "Review recent changes to the API" |
| Onboarding | `examine` + `distill` | "Help me understand this codebase" |
| Feature planning | `distill` + `session_create` | "I'm building a new feature for X" |
| Refactoring | `distill` + `rank_files` | "Find all usages of deprecated function" |
| Bug investigation | `chronicle` + `distill` | "Find changes that could cause issue X" |
| Track velocity | `momentum` | "Show development activity this week" |

## Example Usage

Once configured, ask your AI assistant:

> "Use tenets to find relevant files for implementing user authentication"

The AI will call the `distill` tool and return ranked, optimized context.

> "Create a session called 'auth-feature' and pin the auth folder"

The AI will use `session_create` and `session_pin_folder`.

> "Add a tenet: Always validate user input before processing"

The AI will use `tenet_add` to create a guiding principle.

## Tool Responses (Examples)

### distill
```json
{
  "context": "# File: src/auth.py\n...",
  "token_count": 45000,
  "files": ["src/auth.py", "src/user.py"],
  "files_summarized": ["src/utils.py"],
  "metadata": {"mode": "balanced", "total_scanned": 150}
}
```

### rank_files
```json
{
  "files": [
    {"path": "src/auth.py", "score": 0.85},
    {"path": "src/user.py", "score": 0.72}
  ],
  "total_scanned": 150,
  "mode": "balanced"
}
```

### session_create
```json
{
  "id": "sess_abc123",
  "name": "auth-feature",
  "created_at": "2025-12-04T10:00:00"
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

