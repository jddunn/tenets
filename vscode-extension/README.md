# Tenets MCP Server for VSCode

This extension automatically manages the Tenets MCP (Model Context Protocol) server, enabling intelligent code context for AI assistants like Claude Code.

## Features

- 🚀 **Auto-start**: Automatically starts the Tenets MCP server when VSCode launches
- 📊 **Status indicator**: Shows server status in the status bar
- 📝 **Log viewer**: View server logs directly in VSCode Output panel
- ⚙️ **Manual control**: Start, stop, and restart the server with commands

## Requirements

You must have Tenets installed with MCP support:

```bash
pipx install tenets[mcp]
```

Or with pip:

```bash
pip install tenets[mcp]
```

## Installation

1. Install the extension from the VSCode marketplace (or install the `.vsix` file)
2. Make sure `tenets[mcp]` is installed (see Requirements above)
3. The extension will auto-start the MCP server on VSCode launch

## Usage

### Commands

Access these commands via the Command Palette (Cmd+Shift+P / Ctrl+Shift+P):

- **Tenets: Start MCP Server** - Manually start the server
- **Tenets: Stop MCP Server** - Stop the running server
- **Tenets: Restart MCP Server** - Restart the server
- **Tenets: View Logs** - Open the Tenets MCP output channel

### Status Bar

Click the status bar item (showing "Tenets: Active" or "Tenets: Inactive") to view logs.

### Configuration

Configure the extension in VSCode settings:

- **tenets.mcpPath**: Custom path to `tenets-mcp` executable (leave empty for auto-detect)
- **tenets.autoStart**: Automatically start server on VSCode launch (default: true)

## How It Works

This extension:
1. Finds the `tenets-mcp` executable on your system
2. Spawns it as a subprocess
3. Captures stdout/stderr to the Output panel
4. Shows server status in the status bar
5. Provides commands to control the server lifecycle

The Tenets MCP server provides intelligent code context ranking and distillation for AI assistants through the Model Context Protocol.

## Troubleshooting

### Server not starting

1. **Check installation**: Run `which tenets-mcp` in terminal to verify it's installed
2. **Install Tenets**: If not found, install with `pipx install tenets[mcp]`
3. **View logs**: Use "Tenets: View Logs" command to see error messages
4. **Manual path**: Set `tenets.mcpPath` in settings if auto-detection fails

### Finding tenets-mcp path

```bash
# With pipx (recommended)
which tenets-mcp
# Usually: ~/.local/bin/tenets-mcp

# Or find it manually
find ~ -name tenets-mcp 2>/dev/null
```

### Common Issues

- **"Tenets MCP Server not found"**: Install `tenets[mcp]` with pipx or pip
- **Server keeps restarting**: Check the logs for error messages
- **No logs appearing**: The server might not be outputting to stdout - this is normal for stdio transport

## About Tenets

Tenets is an intelligent code context system that helps AI assistants understand your codebase by:
- Ranking files by relevance to your query
- Building token-optimized context
- Analyzing code structure and complexity
- Tracking development momentum
- Managing sessions and guiding principles

Learn more: https://github.com/jddunn/tenets

## License

MIT - See the Tenets repository for details
