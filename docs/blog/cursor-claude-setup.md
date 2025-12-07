---
title: "Setting Up Tenets MCP Server: Complete Guide"
description: Technical walkthrough for integrating Tenets MCP server with Cursor, Claude Desktop, VS Code, and other MCP-compatible tools.
author: Johnny Dunn
date: 2024-12-04
tags:
  - cursor
  - claude
  - mcp
  - setup
  - configuration
---

# Setting Up Tenets MCP Server: Complete Guide

**Author:** Johnny Dunn | **Date:** December 4, 2024

---

## Overview

Tenets is a **local MCP server** that provides NLP-powered code context to AI coding assistants. This guide covers installation, configuration, and verification for all major MCP hosts.

**What you'll set up:**
- Tenets MCP server binary
- IDE/tool configuration
- Verification that tools are available

**Time required:** 5-10 minutes

---

## Prerequisites

### Python Environment

Tenets requires Python 3.9+:

```bash
python3 --version
# Python 3.11.x or higher recommended
```

### pip Package Manager

```bash
pip --version
# pip 23.x or higher
```

### MCP-Compatible Host

One of:
- **Cursor** (native MCP support)
- **Claude Desktop** (macOS, Windows)
- **Windsurf** (MCP extension)
- **VS Code** with Continue or MCP extension

---

## Installation

### Standard Installation

```bash
pip install tenets[mcp]
```

This installs:
- `tenets` CLI and Python library
- `tenets-mcp` server binary
- MCP protocol dependencies (`mcp`, `httpx`)

### Verify Installation

```bash
tenets-mcp --version
# tenets-mcp v0.7.x

which tenets-mcp
# /usr/local/bin/tenets-mcp (or your Python bin path)
```

**Save this path**—you'll need it for configuration.

### Optional: ML Features

For semantic embeddings (slower but more accurate):

```bash
pip install tenets[ml]
```

Adds `sentence-transformers` for embedding-based ranking.

---

## Cursor Configuration

Cursor has native MCP support. Configuration lives in `~/.cursor/mcp.json`.

### Step 1: Create/Edit Config

```bash
# Create directory if needed
mkdir -p ~/.cursor

# Edit config
nano ~/.cursor/mcp.json
```

### Step 2: Add Tenets Server

**Minimal configuration:**

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

**With full path (recommended for reliability):**

```json
{
  "mcpServers": {
    "tenets": {
      "command": "/usr/local/bin/tenets-mcp"
    }
  }
}
```

**With environment variables:**

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp",
      "env": {
        "TENETS_LOG_LEVEL": "DEBUG",
        "TENETS_CACHE_DIR": "/tmp/tenets-cache"
      }
    }
  }
}
```

### Step 3: Restart Cursor

Fully quit and reopen Cursor (not just reload window).

### Step 4: Verify

In Cursor's AI chat:

```
What MCP tools do you have available?
```

You should see `distill`, `rank_files`, `examine`, etc.

---

## Claude Desktop Configuration

### macOS

**Config path:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```bash
# Create directory
mkdir -p ~/Library/Application\ Support/Claude

# Create/edit config
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Configuration:**

```json
{
  "mcpServers": {
    "tenets": {
      "command": "/usr/local/bin/tenets-mcp"
    }
  }
}
```

### Windows

**Config path:** `%APPDATA%\Claude\claude_desktop_config.json`

```powershell
# Open config location
explorer %APPDATA%\Claude
```

**Configuration:**

```json
{
  "mcpServers": {
    "tenets": {
      "command": "C:\\Users\\YourName\\AppData\\Local\\Programs\\Python\\Python311\\Scripts\\tenets-mcp.exe"
    }
  }
}
```

Find your path:
```powershell
where tenets-mcp
```

### Restart Claude Desktop

Quit and reopen the application.

---

## Windsurf Configuration

Windsurf uses VS Code-style settings with MCP extension.

### Step 1: Open Settings

`Cmd/Ctrl + ,` → Search "MCP"

### Step 2: Edit JSON

```json
{
  "mcp.servers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

### Step 3: Restart

Reload Windsurf window.

---

## VS Code with Continue

Continue extension provides MCP support for VS Code.

### Step 1: Install Continue

Install from VS Code marketplace.

### Step 2: Configure MCP

Open Continue settings and add:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

---

## Advanced Configuration

### Project-Specific Servers

Run different Tenets instances for different projects:

```json
{
  "mcpServers": {
    "tenets-frontend": {
      "command": "tenets-mcp",
      "args": ["--path", "/home/user/projects/frontend"]
    },
    "tenets-backend": {
      "command": "tenets-mcp",
      "args": ["--path", "/home/user/projects/backend"]
    },
    "tenets-infra": {
      "command": "tenets-mcp",
      "args": ["--path", "/home/user/projects/infrastructure"]
    }
  }
}
```

### Debug Mode

Enable verbose logging:

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

### Custom Config File

Point to a specific `.tenets.yml`:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp",
      "env": {
        "TENETS_CONFIG": "/path/to/.tenets.yml"
      }
    }
  }
}
```

---

## Verification

### Test 1: Tool Discovery

Ask your AI:

```
What tools are available from tenets?
```

**Expected response:**
- `distill` — Build optimized code context
- `rank_files` — Preview file relevance scores
- `examine` — Analyze codebase structure
- `session_create`, `session_pin_file`, etc.

### Test 2: Basic Distill

```
Use tenets to find code related to user authentication in /path/to/project
```

**Expected:** The AI calls `distill` and returns ranked code context.

### Test 3: Examine Codebase

```
Use tenets examine on /path/to/project
```

**Expected:** Structure analysis with file counts, languages, complexity.

### Test 4: Session Creation

```
Create a tenets session called "auth-feature"
```

**Expected:** Confirmation of session creation.

---

## Troubleshooting

### "Command not found"

**Cause:** Shell can't find `tenets-mcp` binary.

**Fix:** Use absolute path:

```bash
# Find the path
which tenets-mcp
# /home/user/.local/bin/tenets-mcp

# Use in config
{
  "mcpServers": {
    "tenets": {
      "command": "/home/user/.local/bin/tenets-mcp"
    }
  }
}
```

### "Server not responding"

**Cause:** Server crashes on startup.

**Debug:**

```bash
# Run manually to see errors
tenets-mcp

# Check for Python errors
python -c "import tenets; print(tenets.__version__)"
```

**Common fixes:**
```bash
# Upgrade to latest
pip install tenets[mcp] --upgrade

# Reinstall
pip uninstall tenets && pip install tenets[mcp]
```

### "Tools not appearing"

**Cause:** Config not loaded or JSON syntax error.

**Debug:**

1. Validate JSON syntax: https://jsonlint.com
2. Check config file path is correct
3. Fully restart application (not just reload)

### "Permission denied"

**Cause:** Binary not executable.

**Fix:**
```bash
chmod +x $(which tenets-mcp)
```

### Server Hangs on Large Codebases

**Cause:** Scanning too many files.

**Fix:** Add excludes to `.tenets.yml`:

```yaml
scanner:
  exclude:
    - node_modules/
    - .git/
    - dist/
    - build/
    - "*.min.js"
```

---

## Server Logs

### Enable Debug Logging

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

### Manual Server Testing

```bash
# Run server and send test request
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | tenets-mcp
```

### Log File Location

Logs write to stderr by default. Capture them:

```bash
tenets-mcp 2>&1 | tee /tmp/tenets-mcp.log
```

---

## Performance Tuning

### For Large Codebases (>10k files)

```yaml
# .tenets.yml
scanner:
  max_files: 5000  # Limit scan
  exclude:
    - "*.generated.*"
    - vendor/
    - third_party/

ranking:
  algorithm: fast  # Use faster mode by default
```

### For Slow Machines

```yaml
context:
  max_tokens: 50000  # Reduce token budget

ranking:
  use_ml: false  # Disable ML features
```

---

## Next Steps

Now that Tenets is configured:

1. **Try distill**: Ask your AI to find code for a specific task
2. **Create sessions**: Pin files for ongoing work
3. **Add tenets**: Define guiding principles for consistency
4. **Explore modes**: Test `fast` vs `balanced` vs `thorough`

---

## Resources

- [MCP Documentation](../MCP.md) — Full tool reference
- [CLI Reference](../CLI.md) — Command-line usage
- [Configuration Guide](../configuration.md) — All config options
- [Architecture](../architecture.md) — How Tenets works

---

## Related Posts

- [Why Context Is Everything in AI Coding](why-context-matters.md)
- [Model Context Protocol Explained](mcp-explained.md)

---

<div style="text-align: center; padding: 2rem; margin-top: 2rem; background: rgba(245, 158, 11, 0.05); border-radius: 12px;">
  <p>Built by <a href="https://manic.agency" target="_blank">manic.agency</a></p>
  <a href="https://manic.agency/contact" style="color: #f59e0b;">Need custom AI tooling? →</a>
</div>
