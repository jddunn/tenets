---
title: "Model Context Protocol Explained"
description: What is MCP, why does it matter, and how Tenets fits into the ecosystem. A developer's guide to the open standard for AI tool integration.
author: Johnny Dunn
date: 2024-12-04
tags:
  - mcp
  - model-context-protocol
  - ai-tools
  - integration
---

# Model Context Protocol Explained

**Author:** Johnny Dunn | **Date:** December 4, 2024

---

## Table of Contents

- [What is MCP?](#what-is-mcp)
- [The Problem MCP Solves](#the-problem-mcp-solves)
- [MCP Architecture](#mcp-architecture)
- [MCP Primitives](#mcp-primitives)
- [How Tenets Uses MCP](#how-tenets-uses-mcp)
- [Getting Started](#getting-started)
- [The Future of MCP](#the-future-of-mcp)

---

## What is MCP?

**Model Context Protocol (MCP)** is an open standard developed by Anthropic that allows AI applications to interact with external tools, data sources, and services through a unified interface.

Think of it like USB for AI tools—a standard way for AI assistants to "plug in" to external capabilities.

Before MCP:
```
AI Assistant → Custom integration → Tool A
AI Assistant → Different integration → Tool B
AI Assistant → Yet another integration → Tool C
```

After MCP:
```
AI Assistant → MCP → Tool A, Tool B, Tool C
```

---

## The Problem MCP Solves

AI assistants face a fundamental limitation: they can only work with what's in their context window. They can't:

- Read your files
- Run commands
- Query databases
- Access APIs

**Integrations** solve this, but every integration requires custom code on both sides. This doesn't scale.

MCP provides a **standard interface** so:

1. Tool developers build MCP servers once
2. AI application developers support MCP once
3. Everything connects automatically

### Real-World Analogy

Remember when every device had a different charger? Then USB came along.

MCP is USB for AI tools.

---

## MCP Architecture

```
┌─────────────────┐     ┌─────────────┐     ┌─────────────┐
│  AI Application │────▶│ MCP Client  │────▶│ MCP Server  │
│  (Claude, etc.) │     │  (in app)   │     │  (Tenets)   │
└─────────────────┘     └─────────────┘     └─────────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │  Transport  │
                        │ stdio/http  │
                        └─────────────┘
```

### Components

**MCP Host**: The AI application (Claude Desktop, Cursor, etc.)

**MCP Client**: Built into the host, communicates with servers

**MCP Server**: Provides tools, resources, and prompts (like Tenets)

**Transport**: Communication method (stdio for local, HTTP for remote)

---

## MCP Primitives

MCP defines three primitives:

### 1. Tools

**Functions the AI can call.** Like REST endpoints but for AI.

```json
{
  "name": "distill",
  "description": "Build optimized code context for a task",
  "inputSchema": {
    "type": "object",
    "properties": {
      "prompt": {"type": "string"},
      "mode": {"type": "string", "enum": ["fast", "balanced", "thorough"]}
    }
  }
}
```

The AI sees this schema and knows how to call the tool.

### 2. Resources

**Data endpoints the AI can read.** Like GET endpoints.

```
tenets://sessions/list
tenets://config/current
tenets://tenets/list
```

Resources are read-only and provide state/data to the AI.

### 3. Prompts

**Templates for common workflows.** Pre-built interaction patterns.

```json
{
  "name": "build_context_for_task",
  "description": "Build context for a coding task",
  "arguments": [
    {"name": "task_description", "required": true}
  ]
}
```

---

## How Tenets Uses MCP

Tenets implements an MCP server that exposes:

### Tools (Actions)

| Tool | Purpose |
|------|---------|
| `distill` | Build optimized code context |
| `rank_files` | Preview file relevance scores |
| `examine` | Analyze codebase structure |
| `chronicle` | Analyze git history |
| `session_create` | Create development sessions |
| `session_pin_file` | Pin files to sessions |
| `tenet_add` | Add guiding principles |

### Resources (Data)

| Resource | Returns |
|----------|---------|
| `tenets://sessions/list` | All active sessions |
| `tenets://sessions/{name}/state` | Session details |
| `tenets://tenets/list` | Active tenets |
| `tenets://config/current` | Current configuration |

### Prompts (Templates)

| Prompt | Use Case |
|--------|----------|
| `build_context_for_task` | Standard coding tasks |
| `code_review_context` | Reviewing code changes |
| `understand_codebase` | Onboarding/exploration |

---

## Getting Started

### 1. Install Tenets

```bash
pip install tenets[mcp]
```

### 2. Test the Server

```bash
tenets-mcp --version
```

### 3. Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

### 4. Restart Claude Desktop

### 5. Use It

In Claude, try:

> "Use tenets to find code related to user authentication"

Claude will call the Tenets MCP server automatically.

---

## The Future of MCP

MCP is rapidly becoming the standard for AI tool integration:

### Current Adoption

- **Claude Desktop** — Full MCP support
- **Cursor** — Native MCP integration
- **Continue** — MCP-compatible
- **Windsurf** — Adding MCP support

### Ecosystem Growth

The [MCP Servers Directory](https://github.com/modelcontextprotocol/servers) is growing rapidly with servers for:

- Databases (PostgreSQL, MongoDB)
- Development tools (GitHub, GitLab)
- Productivity (Slack, Notion)
- Infrastructure (AWS, Kubernetes)

### What This Means

Building an MCP server (like Tenets) means automatic compatibility with the entire ecosystem. One integration, everywhere.

---

## Learn More

- [Official MCP Documentation](https://modelcontextprotocol.io)
- [MCP Servers Directory](https://github.com/modelcontextprotocol/servers)
- [Tenets MCP Documentation](../MCP.md)
- [Tenets Tutorial](../tutorial.md)

---

## Related Posts

- [Why Context Is Everything in AI Coding](why-context-matters.md)
- [Setting Up Tenets with Cursor and Claude](cursor-claude-setup.md)

---

<div style="text-align: center; padding: 2rem; margin-top: 2rem; background: rgba(245, 158, 11, 0.05); border-radius: 12px;">
  <p>Built by <a href="https://manic.agency" target="_blank">manic.agency</a></p>
  <a href="https://manic.agency/contact" style="color: #f59e0b;">Need custom AI tooling? →</a>
</div>

