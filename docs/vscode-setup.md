# VSCode Setup Guide

This guide covers multiple ways to use Tenets with Visual Studio Code.

## Option 1: VSCode Extension (Recommended)

The easiest way to use Tenets with VSCode is through the official extension.

### Installation

1. **Install Tenets with MCP support:**
   ```bash
   pipx install tenets[mcp]
   ```

2. **Install the VSCode extension:**
   - **[Install from VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=ManicAgency.tenets-mcp-server)** (recommended)
   - Or search "Tenets MCP Server" in VSCode Extensions panel
   - Or install via command line: `code --install-extension ManicAgency.tenets-mcp-server`

3. **Reload VSCode** (the extension auto-starts the server on launch)

### Features

- ✅ **Auto-start**: Server starts automatically when VSCode launches
- ✅ **Status indicator**: Shows server status in status bar
- ✅ **Log viewer**: View server logs in Output panel
- ✅ **Commands**: Start, stop, restart server via Command Palette

### Commands

Access via Command Palette (Cmd+Shift+P / Ctrl+Shift+P):

- **Tenets: Start MCP Server** - Manually start the server
- **Tenets: Stop MCP Server** - Stop the running server
- **Tenets: Restart MCP Server** - Restart the server
- **Tenets: View Logs** - Open the Tenets MCP output channel

### Configuration

Open VSCode Settings and search for "Tenets":

```json
{
  "tenets.mcpPath": "",           // Path to tenets-mcp (leave empty for auto-detect)
  "tenets.autoStart": true        // Auto-start server on VSCode launch
}
```

### Troubleshooting

#### Server not starting

1. **Verify installation:**
   ```bash
   which tenets-mcp
   # Should output: /Users/YOUR_USER/.local/bin/tenets-mcp (or similar)
   ```

2. **Check if `[mcp]` extra is installed:**
   ```bash
   python -c "import mcp; print('MCP installed')"
   # If ImportError, run: pipx install tenets[mcp]
   ```

3. **View logs:**
   - Click status bar item (shows "Tenets: Active" or "Tenets: Inactive")
   - Or run "Tenets: View Logs" command
   - Check for error messages

4. **Manual path configuration:**
   - If auto-detection fails, find your tenets-mcp path:
     ```bash
     which tenets-mcp
     ```
   - Set `tenets.mcpPath` in VSCode settings to the full path

## Option 2: Manual MCP Configuration

If you prefer not to use the extension, you can configure Tenets as an MCP server manually (though this won't work with Claude Code extension v2.0.65 as it doesn't support custom MCP servers).

### For Cursor IDE

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

### For Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

Or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)

## Option 3: Claude Code CLI

Use Tenets with Claude Code from the terminal:

1. **Configure CLI** (`~/.config/claude/settings.json`):
   ```json
   {
     "mcpServers": {
       "tenets": {
         "command": "/Users/YOUR_USER/.local/bin/tenets-mcp"
       }
     }
   }
   ```

2. **Run Claude Code:**
   ```bash
   claude
   ```

3. **Use Tenets tools:**
   ```
   Use the distill tool to find relevant code for authentication
   ```

## Verification

After installation, verify Tenets is working:

1. **Check server status:**
   - VSCode extension: Look for status bar item
   - CLI: The server starts automatically when Claude Code connects

2. **Test a tool:**
   - In your AI assistant, ask:
     ```
     Use the rank_files tool to show me the most important files for authentication
     ```

3. **View available tools:**
   - Ask your AI:
     ```
     What Tenets tools are available?
     ```

Expected tools:
- `distill` - Build ranked, token-optimized context
- `rank_files` - Fast file relevance ranking
- `examine` - Code structure analysis
- `chronicle` - Git history analysis
- `momentum` - Development velocity tracking
- `session_create`, `session_list`, `session_pin_file`, `session_pin_folder`
- `tenet_add`, `tenet_list`, `tenet_instill`

## Common Issues

### "Tenets MCP Server not found"

**Cause:** `tenets-mcp` executable is not on PATH

**Solution:**
```bash
# Find it
which tenets-mcp

# If not found, install with [mcp]
pipx install tenets[mcp]

# Verify
tenets-mcp --version
```

### "ModuleNotFoundError: No module named 'mcp'"

**Cause:** Installed `tenets` without `[mcp]` extra

**Solution:**
```bash
# Reinstall with MCP support
pipx uninstall tenets
pipx install tenets[mcp]
```

### Extension installed but server not running

**Cause:** Auto-start might be disabled or tenets-mcp not found

**Solution:**
1. Check settings: `tenets.autoStart` should be `true`
2. Manually start: Run "Tenets: Start MCP Server" command
3. View logs to see error messages
4. Set `tenets.mcpPath` if auto-detection fails

### Server keeps restarting

**Cause:** The tenets-mcp process is crashing

**Solution:**
1. View logs ("Tenets: View Logs")
2. Check for Python errors
3. Verify `[mcp]` extra is installed correctly
4. Try running `tenets-mcp` manually in terminal to see errors

## Performance Tips

1. **Session pinning** - Pin frequently used files to avoid re-ranking:
   ```
   Use session_pin_file to pin src/auth.py
   ```

2. **Tenet injection** - Set up guiding principles once:
   ```
   Use tenet_add to add "Always use TypeScript strict mode"
   ```

3. **Distill modes:**
   - `fast` - Quick BM25 ranking (default)
   - `balanced` - BM25 + TF-IDF + git signals
   - `thorough` - All ranking factors + import centrality

## Next Steps

- **Read the MCP documentation:** [docs/MCP.md](MCP.md)
- **Explore CLI usage:** [docs/CLI.md](CLI.md)
- **Configure tenets:** [docs/CONFIG.md](CONFIG.md)
- **Learn about architecture:** [docs/architecture/](architecture/)

## Reporting Issues

If you encounter problems:

1. Check logs ("Tenets: View Logs" in VSCode)
2. Run `tenets-mcp` manually in terminal to see errors
3. Verify installation: `pip list | grep tenets`
4. Report issues: https://github.com/jddunn/tenets/issues

Include in your report:
- VSCode version
- Extension version
- Python version
- Output from `which tenets-mcp`
- Relevant logs from Output panel
