---
title: MCP Server for Claude Desktop - Free Setup Guide
description: Configure Tenets MCP server for Claude Desktop. Free, open source AI code context. Give Claude intelligent access to your codebase.
---

# MCP Server for Claude Desktop

**Free, open source MCP server** that gives Claude Desktop intelligent code context. NLP-powered ranking finds the most relevant files for any task.

## Why Use Tenets with Claude?

| Without Tenets | With Tenets |
|----------------|-------------|
| Manually copy-paste code | Claude calls Tenets directly |
| Random file selection | NLP-ranked relevance |
| Context window waste | Token-optimized output |
| No persistence | Sessions with pinned files |

## Quick Setup (2 minutes)

### Step 1: Install

```bash
pip install tenets[mcp]
```

### Step 2: Configure Claude Desktop

**macOS:** Edit `~/Library/Application Support/Claude/claude_desktop_config.json`

**Windows:** Edit `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

### Step 3: Restart Claude Desktop

Quit and reopen Claude Desktop.

### Step 4: Verify

Ask Claude:

> "What MCP tools do you have available?"

Claude should list `distill`, `rank_files`, `examine`, `session_create`, etc.

## What Tenets Provides to Claude

### Available Tools

| Tool | Purpose |
|------|---------|
| `distill` | Build optimized code context |
| `rank_files` | Preview file relevance |
| `examine` | Analyze codebase structure |
| `chronicle` | Git history analysis |
| `momentum` | Development velocity |
| `session_create` | Persistent sessions |
| `session_pin_file` | Pin files to session |
| `tenet_add` | Add coding guidelines |

### Example Prompts

**Build context for a task:**
> "Use tenets to find code related to user authentication in /path/to/project"

**Rank files without content:**
> "Use tenets rank_files to show the top 10 files for 'payment processing'"

**Create a working session:**
> "Use tenets to create a session called 'auth-refactor' and pin src/auth/"

**Add guidelines:**
> "Use tenets to add a critical tenet: Never log sensitive user data"

## Configuration Options

### Full Path (Recommended)

```bash
# Find your path
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

### With Working Directory

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp",
      "args": ["--path", "/Users/you/projects/myapp"]
    }
  }
}
```

### Debug Logging

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

### Claude says "I don't have access to tenets tools"

1. Check config file location is correct
2. Verify JSON syntax (use a JSON validator)
3. Fully restart Claude Desktop
4. Use absolute path to `tenets-mcp`

### "Command not found"

Find the binary location:

```bash
which tenets-mcp
# /usr/local/bin/tenets-mcp
```

Use that full path in config.

### Tools timeout on large projects

Add exclusions to `.tenets.yml` in your project:

```yaml
scanner:
  exclude:
    - node_modules/
    - .git/
    - dist/
    - "*.min.js"
```

## Why Tenets?

- **Free forever** — MIT license, open source
- **100% local** — Code never leaves your machine
- **NLP-powered** — BM25, TF-IDF, import centrality
- **Fast** — Thousands of files in seconds

## Next Steps

- [Full MCP Documentation](MCP.md)
- [Tutorial](tutorial.md)
- [CLI Reference](CLI.md)
- [FAQ](faq.md)

---

<div style="text-align: center; padding: 2rem; margin: 2rem 0; background: rgba(245, 158, 11, 0.05); border-radius: 12px;">
  <p><strong>Tenets is 100% free and open source.</strong></p>
  <p>MIT License · <a href="https://github.com/jddunn/tenets">GitHub</a> · <a href="https://pypi.org/project/tenets/">PyPI</a></p>
</div>

