---
title: "Model Context Protocol: Technical Deep Dive"
description: How MCP works and how Tenets implements it. JSON-RPC transport, tool schemas, intelligent code context for LLMs. Free open source Python MCP server.
author: Johnny Dunn
date: 2024-10-22
tags:
  - tenets
  - mcp
  - model-context-protocol
  - python
  - llm
  - ai-coding
  - json-rpc
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "Model Context Protocol: Technical Deep Dive",
  "description": "How MCP works and how Tenets implements it. JSON-RPC transport, tool schemas, intelligent code context for LLMs.",
  "author": {
    "@type": "Person",
    "name": "Johnny Dunn",
    "url": "https://jddunn.github.io"
  },
  "publisher": {
    "@type": "Organization",
    "name": "manic.agency",
    "logo": {
      "@type": "ImageObject",
      "url": "https://tenets.dev/logos/tenets_dark_icon_transparent_cropped.png"
    }
  },
  "datePublished": "2024-10-22",
  "dateModified": "2024-12-07",
  "mainEntityOfPage": "https://tenets.dev/blog/mcp-explained/",
  "image": "https://tenets.dev/assets/og-image.png",
  "keywords": ["MCP", "Model Context Protocol", "JSON-RPC", "AI coding", "Tenets", "Python"]
}
</script>

# Model Context Protocol: Technical Deep Dive

**Author:** Johnny Dunn | **Date:** October 22, 2024

---

## What is MCP?

**Model Context Protocol (MCP)** is a JSON-RPC 2.0 based protocol developed by Anthropic for AI applications to interact with external tools and data sources. It standardizes how LLM-powered applications discover, invoke, and receive results from external capabilities.

Tenets implements MCP to provide AI coding assistants with two capabilities: **intelligent code context** (finding relevant files automatically) and **automatic guiding principles injection** (your coding standards in every prompt).

The protocol defines three primitives:

| Primitive | Direction | Purpose |
|-----------|-----------|---------|
| **Tools** | AI → Server | Functions the AI can invoke |
| **Resources** | AI ← Server | Data endpoints the AI can read |
| **Prompts** | AI ← Server | Pre-built interaction templates |

---

## Protocol Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      MCP Host (Cursor, Claude Desktop)       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │   LLM Engine   │  │   MCP Client   │  │   UI Layer     │  │
│  └───────┬────────┘  └───────┬────────┘  └────────────────┘  │
│          │                   │                               │
│          │ Tool calls        │ JSON-RPC 2.0                  │
│          └───────────────────┤                               │
└──────────────────────────────┼───────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │     Transport       │
                    │  stdio | SSE | HTTP │
                    └──────────┬──────────┘
                               │
┌──────────────────────────────┼───────────────────────────────┐
│                      MCP Server (Tenets)                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │  Tool Handler  │  │ Resource Mgr   │  │ Prompt Registry│  │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘  │
│          │                   │                   │           │
│  ┌───────┴───────────────────┴───────────────────┴────────┐  │
│  │                    Tenets Core                         │  │
│  │   Distiller │ Ranker │ Analyzer │ Session │ Git        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Transport Mechanisms

MCP supports multiple transports:

**stdio** (default): Server runs as subprocess, communicates via stdin/stdout
```json
{"jsonrpc": "2.0", "method": "tools/call", "params": {...}, "id": 1}
```

**SSE (Server-Sent Events)**: HTTP-based streaming for web clients
```bash
tenets-mcp --sse --port 8080
```

**HTTP**: REST-like interface for integration scenarios
```bash
tenets-mcp --http --port 8080
```

---

## JSON-RPC Message Flow

### Tool Discovery

When the MCP host connects, it discovers available tools:

```json
// Request: List available tools
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}

// Response: Tool schemas
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "distill",
        "description": "Build optimized code context for a task",
        "inputSchema": {
          "type": "object",
          "properties": {
            "prompt": {
              "type": "string",
              "description": "Task or question to find context for"
            },
            "path": {
              "type": "string",
              "default": "."
            },
            "mode": {
              "type": "string",
              "enum": ["fast", "balanced", "thorough"],
              "default": "balanced"
            },
            "max_tokens": {
              "type": "integer",
              "default": 100000
            }
          },
          "required": ["prompt"]
        }
      }
    ]
  },
  "id": 1
}
```

### Tool Invocation

The AI constructs a tool call based on the schema:

```json
// Request: Call distill tool
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "distill",
    "arguments": {
      "prompt": "implement user authentication with JWT",
      "path": "/home/user/project",
      "mode": "balanced",
      "max_tokens": 80000
    }
  },
  "id": 2
}

// Response: Context result
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "# Context for: implement user authentication with JWT\n\n## File: src/auth/jwt.py (relevance: 0.92)\n```python\nimport jwt\nfrom datetime import datetime, timedelta\n..."
      }
    ],
    "isError": false
  },
  "id": 2
}
```

---

## Tenets MCP Server Implementation

### Tool Definitions

Tenets exposes its core capabilities as MCP tools:

```python
# tenets/mcp/server.py

@mcp.tool()
async def distill(
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
) -> dict[str, Any]:
    """
    Build optimized code context for a task or question.
    
    Uses multi-factor NLP ranking (BM25, keyword matching, import 
    centrality, git signals) to find and aggregate relevant code
    within your token budget.
    
    Args:
        prompt: What you're working on. Be specific for better results.
        path: Directory to search. Use "." for current project.
        mode: Speed vs accuracy tradeoff.
        max_tokens: Token budget for context.
        format: Output structure (markdown recommended for LLMs).
        include_tests: Set True when debugging test failures.
        session: Link to a session for pinned files.
    
    Returns:
        Dictionary with context, token_count, files, and metadata.
    """
    result = tenets_instance.distill(
        prompt=prompt,
        path=Path(path),
        mode=mode,
        max_tokens=max_tokens,
        format=format,
        include_tests=include_tests,
        include_git=include_git,
        session=session,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    return result.to_dict()
```

### Full Tool Surface

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `distill` | Build ranked code context | prompt, mode, max_tokens |
| `rank_files` | Preview file relevance scores | prompt, top_n, explain |
| `examine` | Analyze codebase structure | path, include_complexity |
| `chronicle` | Analyze git history | since, author |
| `momentum` | Track development velocity | since, team |
| `session_create` | Create development session | name, description |
| `session_pin_file` | Pin file to session | session, file_path |
| `session_pin_folder` | Pin folder to session | session, folder_path, patterns |
| `tenet_add` | Add guiding principle | content, priority, category |
| `tenet_list` | List active tenets | session, pending_only |
| `tenet_instill` | Activate pending tenets | session, force |
| `set_system_instruction` | Set AI instruction | instruction, position |

### Resource Definitions

Resources expose read-only data:

```python
@mcp.resource("tenets://sessions/list")
async def list_sessions() -> str:
    """List all active development sessions."""
    sessions = tenets_instance.list_sessions()
    return json.dumps(sessions, indent=2)

@mcp.resource("tenets://sessions/{name}/state")  
async def get_session_state(name: str) -> str:
    """Get detailed state for a specific session."""
    state = tenets_instance.get_session(name)
    return json.dumps(state.to_dict(), indent=2)

@mcp.resource("tenets://config/current")
async def get_config() -> str:
    """Get current Tenets configuration."""
    return json.dumps(tenets_instance.config.to_dict(), indent=2)
```

---

## How the AI Uses MCP

When you ask Claude or Cursor: *"Find the authentication code in my project"*

### Step 1: Intent Recognition
The AI recognizes this requires external tool access.

### Step 2: Tool Selection
The AI examines available tools and selects `distill`:

```
Available tools: distill, rank_files, examine, chronicle...
Selected: distill (builds code context for queries)
```

### Step 3: Parameter Construction
The AI constructs the call based on the schema:

```json
{
  "name": "distill",
  "arguments": {
    "prompt": "authentication code",
    "mode": "balanced"
  }
}
```

### Step 4: Result Integration
The MCP response becomes part of the AI's context:

```markdown
# Context for: authentication code

## File: src/auth/service.py (relevance: 0.91)
```python
class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.hasher = Argon2Hasher()
    
    async def authenticate(self, email: str, password: str) -> AuthResult:
        user = await self.user_repo.get_by_email(email)
        if not user or not self.hasher.verify(password, user.password_hash):
            raise AuthenticationError("Invalid credentials")
        return AuthResult(user=user, token=self._generate_token(user))
```

## File: src/auth/middleware.py (relevance: 0.87)
...
```

The AI now has **relevant, ranked code** to work with.

---

## Configuration Examples

### Cursor (`~/.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp",
      "args": [],
      "env": {
        "TENETS_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Claude Desktop

**macOS** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tenets": {
      "command": "/usr/local/bin/tenets-mcp"
    }
  }
}
```

**Windows** (`%APPDATA%\Claude\claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "tenets": {
      "command": "C:\\Python311\\Scripts\\tenets-mcp.exe"
    }
  }
}
```

### Multiple Project Configuration

```json
{
  "mcpServers": {
    "tenets-frontend": {
      "command": "tenets-mcp",
      "args": ["--path", "/projects/frontend"]
    },
    "tenets-backend": {
      "command": "tenets-mcp", 
      "args": ["--path", "/projects/backend"]
    }
  }
}
```

---

## Why Tenets + MCP

Most MCP servers provide **raw access**—file reading, command execution, API calls. Tenets provides **intelligent access**:

| Raw MCP Server | Tenets MCP Server |
|----------------|-------------------|
| Read file by path | Find relevant files automatically |
| List directory | Rank files by query relevance |
| Return full content | Optimize for token budget |
| Stateless | Session persistence + pinned files |
| No guidance | Automatic tenets injection |

The two core features:

1. **Intelligent code context**: NLP-powered ranking (BM25, import centrality, git signals) finds exactly what the LLM needs
2. **Automatic tenets injection**: Your guiding principles (coding standards, architecture rules) are injected into every prompt, preventing context drift in long conversations

Both run 100% locally—no API calls, no data leaving your machine.

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Tool discovery | <10ms | Cached after first call |
| `distill` (fast) | 500ms-1s | Keyword + path matching |
| `distill` (balanced) | 2-4s | Full NLP pipeline |
| `distill` (thorough) | 8-15s | ML embeddings |
| `rank_files` | 200ms-2s | Preview without content |
| `examine` | 1-3s | Structure analysis |

All processing runs **100% locally**—no API calls, no data leaving your machine.

---

## Learn More

- [MCP Specification](https://spec.modelcontextprotocol.io)
- [Tenets MCP Documentation](../MCP.md)
- [Architecture: MCP Integration](../architecture/mcp-integration-plan.md)
- [Architecture: Ranking System](../architecture/ranking-system.md)

---

## Related Posts

- [Why Context Is Everything in AI Coding](why-context-matters.md)
- [Setting Up Tenets with Cursor and Claude](cursor-claude-setup.md)

---

<div style="text-align: center; padding: 2rem; margin-top: 2rem; background: rgba(245, 158, 11, 0.05); border-radius: 12px;">
  <p>Built by <a href="https://manic.agency" target="_blank">manic.agency</a></p>
  <a href="https://manic.agency/contact" style="color: #f59e0b;">Need custom AI tooling? →</a>
</div>
