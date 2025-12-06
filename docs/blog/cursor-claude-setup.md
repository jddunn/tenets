---
title: "Setting Up Tenets with Cursor and Claude"
description: Step-by-step guide to integrating Tenets MCP server with Cursor IDE and Claude Desktop for AI-powered coding.
author: Johnny Dunn
date: 2024-12-04
tags:
  - cursor
  - claude
  - setup
  - tutorial
---

# Setting Up Tenets with Cursor and Claude

**Author:** Johnny Dunn | **Date:** December 4, 2024

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Claude Desktop Setup](#claude-desktop-setup)
- [Cursor Setup](#cursor-setup)
- [VS Code Setup](#vs-code-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.9+** installed
- **pip** package manager
- One of:
    - Claude Desktop (macOS, Windows)
    - Cursor IDE
    - VS Code with MCP extension

Check your Python version:

```bash
python3 --version
# Python 3.11.4 or higher
```

---

## Installation

Tenets is an MCP server for AI coding assistants. Install with MCP support:

```bash
pip install tenets[mcp]
```

Verify the installation:

```bash
tenets-mcp --version
```

You should see:

```
tenets-mcp v0.7.1
```

### Finding the Executable Path

You'll need the full path to `tenets-mcp` for configuration:

```bash
which tenets-mcp
# /usr/local/bin/tenets-mcp
# or
# /Users/yourname/.local/bin/tenets-mcp
```

Note this path for the next steps.

---

## Claude Desktop Setup

### macOS

1. **Open the config file:**

```bash
open ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

If it doesn't exist, create it:

```bash
mkdir -p ~/Library/Application\ Support/Claude
touch ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

2. **Add Tenets configuration:**

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

**Or with full path (recommended):**

```json
{
  "mcpServers": {
    "tenets": {
      "command": "/usr/local/bin/tenets-mcp"
    }
  }
}
```

3. **Restart Claude Desktop**

Quit and reopen Claude Desktop for changes to take effect.

### Windows

1. **Open the config file:**

```
%APPDATA%\Claude\claude_desktop_config.json
```

2. **Add configuration:**

```json
{
  "mcpServers": {
    "tenets": {
      "command": "C:\\Users\\YourName\\AppData\\Local\\Programs\\Python\\Python311\\Scripts\\tenets-mcp.exe"
    }
  }
}
```

3. **Restart Claude Desktop**

---

## Cursor Setup

### Method 1: Settings UI

1. Open Cursor
2. Go to **Settings** (⌘/Ctrl + ,)
3. Search for "MCP"
4. Click "Edit in settings.json"
5. Add:

```json
{
  "mcp.servers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

### Method 2: Edit JSON Directly

1. Open Command Palette (⌘/Ctrl + Shift + P)
2. Type "Open Settings (JSON)"
3. Add the MCP configuration above

### Restart Cursor

Restart Cursor for changes to take effect.

---

## VS Code Setup

VS Code requires an MCP extension. Install **Continue** or another MCP-compatible extension.

### With Continue

1. Install Continue extension
2. Open Continue settings
3. Add Tenets as an MCP server:

```json
{
  "models": [...],
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

---

## Verification

### Test in Claude Desktop

Open Claude and type:

> "What tools do you have available from tenets?"

Claude should respond with a list of available tools like `distill`, `examine`, `rank_files`, etc.

### Test Functionality

Try:

> "Use tenets to examine the structure of /path/to/your/project"

Claude should call the `examine` tool and return codebase analysis.

### Test in Cursor

In Cursor's AI chat, try:

> "Use tenets to find code related to authentication"

---

## Troubleshooting

### "Command not found"

**Problem:** The MCP server can't find `tenets-mcp`.

**Solution:** Use the full path:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "/full/path/to/tenets-mcp"
    }
  }
}
```

Find the path with:

```bash
which tenets-mcp
```

### "MCP server not responding"

**Problem:** Server starts but doesn't respond.

**Solutions:**

1. Check the server manually:
   ```bash
   tenets-mcp
   ```
   
2. Look for Python errors in output

3. Verify dependencies:
   ```bash
   pip install tenets[mcp] --upgrade
   ```

### "Tools not appearing"

**Problem:** Claude/Cursor doesn't show Tenets tools.

**Solutions:**

1. Fully restart the application (not just reload)
2. Check config JSON syntax (use a JSON validator)
3. Verify config file location

### "Permission denied"

**Problem:** Can't execute `tenets-mcp`.

**Solution:**

```bash
chmod +x $(which tenets-mcp)
```

### Check Server Logs

Run the server with debug output:

```bash
tenets-mcp 2>&1 | tee mcp-debug.log
```

---

## Advanced Configuration

### Working Directory

Set a default project directory:

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

### Multiple Projects

You can configure multiple Tenets servers for different projects:

```json
{
  "mcpServers": {
    "tenets-frontend": {
      "command": "tenets-mcp",
      "args": ["--path", "/path/to/frontend"]
    },
    "tenets-backend": {
      "command": "tenets-mcp",
      "args": ["--path", "/path/to/backend"]
    }
  }
}
```

### Environment Variables

Pass environment variables:

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

---

## Next Steps

- [Complete Tutorial](../tutorial.md) — Learn all Tenets features
- [MCP Documentation](../MCP.md) — Deep dive into MCP capabilities
- [CLI Reference](../CLI.md) — Use Tenets from the command line

---

## Related Posts

- [Why Context Is Everything in AI Coding](why-context-matters.md)
- [Model Context Protocol Explained](mcp-explained.md)

---

<div style="text-align: center; padding: 2rem; margin-top: 2rem; background: rgba(245, 158, 11, 0.05); border-radius: 12px;">
  <p>Built by <a href="https://manic.agency" target="_blank">manic.agency</a></p>
  <a href="https://manic.agency/contact" style="color: #f59e0b;">Need custom AI tooling? →</a>
</div>

