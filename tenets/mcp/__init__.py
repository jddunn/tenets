"""Tenets MCP Server - Model Context Protocol integration.

This module provides an MCP server that exposes tenets functionality to AI
coding assistants like Cursor, Claude Desktop, Windsurf, and custom agents.

The MCP server wraps the existing tenets core library, providing:
- Tools: Actions AI can invoke (distill, rank, examine, etc.)
- Resources: Data AI can read (context history, session state, analysis)
- Prompts: Reusable interaction templates

Usage:
    # Start the MCP server (stdio transport for local IDE integration)
    $ tenets-mcp

    # Or with specific transport
    $ tenets-mcp --transport stdio     # Local (default)
    $ tenets-mcp --transport sse       # Server-Sent Events
    $ tenets-mcp --transport http      # Streamable HTTP

    # Programmatic usage
    >>> from tenets.mcp import create_server
    >>> server = create_server()
    >>> server.run(transport="stdio")

Configuration:
    MCP settings can be configured in .tenets.yml:

    ```yaml
    mcp:
      enabled: true
      transports:
        stdio: true
        sse: false
        http: false
    ```

Example IDE Configuration (Claude Desktop):
    ```json
    {
      "mcpServers": {
        "tenets": {
          "command": "tenets-mcp",
          "args": []
        }
      }
    }
    ```
"""

from tenets.mcp.server import TenetsMCP, create_server, main

__all__ = [
    "TenetsMCP",
    "create_server",
    "main",
]

