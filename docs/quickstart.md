---
title: Quick Start - Install Free MCP Server in 30 Seconds
description: Get started with Tenets MCP server for AI coding assistants. Free, open source, 100% local. Works with Cursor, Claude Desktop, Windsurf.
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "HowTo",
  "name": "How to Install Tenets MCP Server for AI Coding",
  "description": "Install and configure Tenets MCP server to give AI coding assistants like Cursor and Claude intelligent code context. Free and open source.",
  "totalTime": "PT2M",
  "tool": [
    {
      "@type": "HowToTool",
      "name": "Python 3.9+"
    },
    {
      "@type": "HowToTool",
      "name": "pip package manager"
    }
  ],
  "step": [
    {
      "@type": "HowToStep",
      "position": 1,
      "name": "Install Tenets with MCP support",
      "text": "Run: pip install tenets[mcp]",
      "url": "https://tenets.dev/quickstart/#mcp-first-recommended"
    },
    {
      "@type": "HowToStep",
      "position": 2,
      "name": "Add MCP configuration to your IDE",
      "text": "Add {\"mcpServers\": {\"tenets\": {\"command\": \"tenets-mcp\"}}} to your IDE's MCP config file (Cursor: ~/.cursor/mcp.json, Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json)",
      "url": "https://tenets.dev/quickstart/#mcp-first-recommended"
    },
    {
      "@type": "HowToStep",
      "position": 3,
      "name": "Restart your IDE and use Tenets",
      "text": "Restart your IDE and ask the AI: 'Use tenets to find the auth code'",
      "url": "https://tenets.dev/quickstart/#mcp-first-recommended"
    }
  ]
}
</script>

# Quickstart

Tenets gives your AI two superpowers: **intelligent code context** (finds the right files automatically) and **automatic tenets injection** (your guiding principles in every prompt).

## MCP-first (recommended)

```bash
pip install tenets[mcp]
tenets-mcp
```

Add to your IDE:

- **Cursor** (`~/.cursor/mcp.json`)
  ```json
  { "mcpServers": { "tenets": { "command": "tenets-mcp" } } }
  ```
- **Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`)
  ```json
  { "mcpServers": { "tenets": { "command": "tenets-mcp" } } }
  ```
- **Windsurf / VS Code MCP**
  ```json
  { "tenets": { "command": "tenets-mcp" } }
  ```

Ask your AI: *"Use tenets to find the auth code and pin src/auth."*

---

## The Two Core Features

### 1. Intelligent Code Context
Your AI asks tenets to find relevant code. Tenets uses NLP (BM25, import centrality, git signals) to rank and aggregate the best files within your token budget.

### 2. Automatic Tenets Injection
Define guiding principles once—they're automatically injected into every prompt. This prevents context drift in long conversations and ensures consistent coding standards.

---

## Real-world flow: System instruction + Tenets + Sessions

```bash
# Create a working session
tenets session create auth-refresh

# Add guiding principles (tenets)
tenets tenet add "Prefer small, safe diffs" --priority high --category style
tenets tenet add "Always validate user input" --priority critical --category security

# Apply tenets for this session
tenets instill --session auth-refresh

# Set a global system instruction
tenets system-instruction set "You are a senior engineer. Add tests and document trade-offs." --enable

# Build context with transformations for token efficiency
tenets distill "add OAuth2 refresh tokens" --session auth-refresh --remove-comments --condense

# Pin files as you learn what matters
tenets instill --session auth-refresh --add-file src/auth/service.py --add-folder src/auth/routes
tenets instill --session auth-refresh --list-pinned
```

See also: CLI > System Instruction Commands, Tenet Commands, and Instill.
# Quick Start

Get productive with Tenets in under 60 seconds.

## 1. Install

```bash
pip install tenets
```

## 2. Generate Context (CLI)

```bash
tenets distill "add optimistic locking to order updates"
```

Copy straight to your clipboard:

```bash
tenets distill "refactor payment flow" --copy
```

Or enable auto-copy in `tenets.toml`:

```toml
[output]
copy_on_distill = true
```

## 3. Refine

Pin or force-include critical files:

```bash
# Build context for investigation
tenets distill "investigate cache stampede"

# Pin files are managed through instill command for sessions
tenets instill --add-file cache/*.py --add-file config/settings.py
```

Exclude noise:

```bash
tenets distill "debug webhook" --exclude "**/migrations/**,**/tests/**"
```

## 4. Python API

```python
from tenets import Tenets

tenets = Tenets()
result = tenets.distill(
    prompt="implement bulk import",
    max_tokens=80_000,
)
print(result.token_count, "tokens")
# Copy is done via CLI flag --copy or config setting
```

## 5. Sessions (Iterate)

```python
tenets = Tenets()
# Sessions are managed through distill parameters
first = tenets.distill("trace 500 errors in checkout", session_name="checkout-fixes")
second = tenets.distill("add instrumentation around payment retries", session_name="checkout-fixes")
```

## 6. Visualization & Insight

```bash
# Complexity & hotspots
tenets examine . --show-details --hotspots

# Dependency graph (Interactive HTML)
tenets viz deps --format html --output deps.html
```

## 7. Next

* See full CLI options: [CLI Reference](CLI.md)
* Tune ranking & tokens: [Configuration](CONFIG.md)
* Dive deeper: [Architecture](ARCHITECTURE.md)
