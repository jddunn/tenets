---
title: Tenets MCP Server for Cursor IDE - Free Setup Guide
description: Set up Tenets MCP server in Cursor IDE. Free, open source AI code context. NLP-powered ranking gives your AI the right files automatically.
---

# Tenets MCP Server for Cursor IDE

**Free, open source MCP server** that gives Cursor's AI the intelligent code context it needs. NLP-powered ranking finds relevant files automatically.

## Why Use Tenets with Cursor?

| Without Tenets | With Tenets |
|----------------|-------------|
| AI guesses which files to read | AI gets ranked, relevant files |
| Manual @-mentioning of files | Automatic context building |
| Context window filled with noise | Token-optimized, high-signal context |
| Each prompt starts fresh | Persistent sessions with pinned files |

## Quick Setup (2 minutes)

### Step 1: Install

```bash
pip install tenets[mcp]
```

### Step 2: Configure Cursor

Create or edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

### Step 3: Restart Cursor

Quit and reopen Cursor for the MCP server to connect.

### Step 4: Use It

In Cursor's AI chat, try:

> "Use tenets to find the authentication code"

Cursor will call the Tenets MCP server and get ranked, relevant files.

## What Tenets Provides to Cursor

### Tools

| Tool | What It Does |
|------|--------------|
| `distill` | Build optimized context for any task |
| `rank_files` | Preview file relevance scores |
| `examine` | Analyze codebase structure |
| `session_create` | Create persistent sessions |
| `session_pin_file` | Pin critical files |
| `tenet_add` | Add coding guidelines |

### Example Interactions

**Find code:**
> "Use tenets to find the payment processing code"

**Build context for a task:**
> "Use tenets distill to get context for implementing OAuth2"

**Pin important files:**
> "Use tenets to pin src/auth/ to session auth-feature"

**Add guidelines:**
> "Use tenets to add a tenet: Always validate user input"

## Advanced Configuration

### Full Path (Recommended)

If Cursor can't find `tenets-mcp`, use the absolute path:

```bash
# Find the path
which tenets-mcp
```

```json
{
  "mcpServers": {
    "tenets": {
      "command": "/usr/local/bin/tenets-mcp"
    }
  }
}
```

### Multiple Projects

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

### Debug Mode

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp",
      "env": {
        "TENETS_LOG_LEVEL": "DEBUG"
      }
    }
  }
}
```

## Troubleshooting

### "Command not found"

Use absolute path to `tenets-mcp`. Find it with `which tenets-mcp`.

### "Tools not showing"

1. Restart Cursor completely (not just reload)
2. Check JSON syntax in config file
3. Verify config file location is correct

### "Server timeout"

For large codebases, add exclusions to `.tenets.yml`:

```yaml
scanner:
  exclude:
    - node_modules/
    - dist/
    - build/
```

## Why Tenets?

- **Free & open source** — MIT license, no cost
- **100% local** — Your code never leaves your machine
- **NLP-powered** — BM25, TF-IDF, import graph analysis
- **Fast** — Analyzes thousands of files in seconds

## Next Steps

- [Full MCP Documentation](MCP.md)
- [CLI Reference](CLI.md)
- [Configuration Guide](CONFIG.md)
- [FAQ](faq.md)

---

<div style="text-align: center; padding: 2rem; margin: 2rem 0; background: rgba(245, 158, 11, 0.05); border-radius: 12px;">
  <p><strong>Tenets is 100% free and open source.</strong></p>
  <p>MIT License · <a href="https://github.com/jddunn/tenets">GitHub</a> · <a href="https://pypi.org/project/tenets/">PyPI</a></p>
</div>

