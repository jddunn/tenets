---
title: MCP Server Submission
description: Templates and guides for submitting Tenets to MCP directories and registries.
---

# MCP Server Submission Guide

This page contains templates for submitting Tenets to various MCP directories and registries.

---

## 1. Official MCP Registry (Primary)

**Submit at:** https://registry.modelcontextprotocol.io/

The MCP Registry is the official directory maintained by the MCP steering group.

### Submission Information

| Field | Value |
|-------|-------|
| **Name** | tenets |
| **Display Name** | Tenets |
| **Description** | Intelligent code context for AI coding assistants. Multi-factor ranking finds relevant code for any task. |
| **Repository** | https://github.com/jddunn/tenets |
| **Documentation** | https://tenets.dev/MCP/ |
| **PyPI** | https://pypi.org/project/tenets/ |
| **Install** | `pip install tenets[mcp]` |
| **Run** | `tenets-mcp` |
| **Author** | Johnny Dunn |
| **Organization** | manic.agency |
| **License** | MIT |
| **Categories** | Developer Tools, Code Analysis, IDE Integration |

### Features to Highlight

- **13 MCP Tools**: distill, rank_files, examine, chronicle, momentum, sessions, tenets
- **4 MCP Resources**: Session state, tenet lists, configuration
- **3 MCP Prompts**: Task context, code review, codebase understanding
- **Transports**: stdio (default), SSE, HTTP
- **IDE Support**: Cursor, Claude Desktop, Windsurf, VS Code

---

## 2. GitHub modelcontextprotocol/servers

**Repository:** https://github.com/modelcontextprotocol/servers

> Note: This README is being deprecated in favor of the MCP Registry, but submissions may still be accepted.

### PR Template

**Title:** `Add tenets - intelligent code context server`

**Description:**
```markdown
## Summary
Adding tenets to the third-party servers list.

## Server Details
- **Name:** tenets
- **Repository:** https://github.com/jddunn/tenets
- **Description:** MCP server for intelligent code context. Multi-factor ranking (BM25, git signals, import centrality) finds relevant code for AI coding tasks.
- **Install:** `pip install tenets[mcp]`
- **Run:** `tenets-mcp`

## Features
- 13 tools for code analysis, ranking, and session management
- 100% local processing - code never leaves your machine
- Works with Cursor, Claude Desktop, Windsurf

## Checklist
- [x] Server is publicly available
- [x] Documentation is complete
- [x] License is MIT (open source)
```

### README Entry (if accepted)
```markdown
- <img height="12" width="12" src="https://tenets.dev/favicon/favicon-32x32.png" alt="Tenets Logo" /> **[Tenets](https://github.com/jddunn/tenets)** - Intelligent code context for AI assistants. Multi-factor ranking finds relevant code for any task. Works with Cursor, Claude, Windsurf.
```

---

## 3. Community Awesome Lists

### awesome-mcp-servers (wong2)

**Repository:** https://github.com/wong2/awesome-mcp-servers

**PR Entry:**
```markdown
### Developer Tools

- [tenets](https://github.com/jddunn/tenets) - Intelligent code context for AI coding assistants. Multi-factor ranking finds relevant code. `pip install tenets[mcp]`
```

### awesome-mcp-servers (punkpeye)

**Repository:** https://github.com/punkpeye/awesome-mcp-servers

**PR Entry:**
```markdown
| [tenets](https://github.com/jddunn/tenets) | Intelligent code context for AI assistants | `pip install tenets[mcp]` | MIT |
```

---

## 4. Other Platforms

### mcp.run
- **URL:** https://mcp.run
- Submit via their web form

### Smithery
- **URL:** https://smithery.ai
- Create account and publish directly

### Glama
- **URL:** https://glama.ai/mcp/servers
- Submit via their directory

---

## Quick Config Snippets

### Claude Desktop
```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

### Cursor
```json
{
  "tenets": {
    "command": "tenets-mcp"
  }
}
```

### With Custom Path
```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp",
      "args": ["--path", "/path/to/your/project"]
    }
  }
}
```

---

## Social Announcement Template

### Twitter/X
```
🚀 Tenets v0.4.0 - MCP Server for AI Coding

Finally, proper context for your AI assistant.

✅ Works with Cursor, Claude Desktop, Windsurf
✅ Multi-factor ranking (BM25, git signals, imports)
✅ 100% local - your code never leaves your machine

pip install tenets[mcp]

https://tenets.dev
```

### Hacker News
```
Show HN: Tenets – MCP server that gives AI assistants intelligent code context

Tenets is an open-source MCP server that automatically finds, ranks, and aggregates relevant code for AI coding assistants.

Features:
- Multi-factor ranking (BM25, TF-IDF, git signals, import centrality)
- Session management with pinned files
- Guiding principles (tenets) to prevent AI drift
- 100% local - no data leaves your machine

Works with Cursor, Claude Desktop, Windsurf, and any MCP client.

Install: pip install tenets[mcp]
GitHub: https://github.com/jddunn/tenets
Docs: https://tenets.dev
```

---

## Contact

- **Email:** team@manic.agency
- **GitHub:** [@jddunn](https://github.com/jddunn)
- **Agency:** [manic.agency](https://manic.agency)

