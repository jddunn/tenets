---
title: Frequently Asked Questions
description: Common questions about Tenets MCP server, installation, configuration, and usage with AI coding assistants like Cursor and Claude.
---

# Frequently Asked Questions

## General

### What is Tenets?

Tenets is an intelligent code context platform that automatically finds, ranks, and aggregates relevant code for AI coding assistants. It works as a **CLI tool**, **Python library**, and **MCP server**.

### What is MCP?

[Model Context Protocol (MCP)](https://modelcontextprotocol.io) is an open standard that allows AI assistants to interact with external tools and data sources. Tenets implements an MCP server so AI assistants like Claude and Cursor can directly access its functionality.

### Is Tenets free?

Yes! Tenets is **100% free and open source** under the MIT license. There are no usage limits, API costs, or premium tiers for the core functionality.

### Does my code leave my machine?

**No.** All processing happens locally. Tenets never sends your code to external servers. This is a core design principle.

---

## Installation

### How do I install Tenets?

```bash
# Basic installation
pip install tenets

# With MCP server support
pip install tenets[mcp]

# Everything included
pip install tenets[all]
```

### What Python versions are supported?

Tenets supports **Python 3.9 through 3.13**. We recommend Python 3.11 for best performance.

### Do I need GPU for ML features?

No. ML features (semantic search, embeddings) work on CPU. GPU acceleration is optional and automatic if available.

---

## MCP Server

### How do I start the MCP server?

```bash
# Install with MCP support
pip install tenets[mcp]

# Start the server
tenets-mcp
```

### How do I configure Claude Desktop?

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

See the full [MCP documentation](MCP.md) for more configuration options.

### How do I configure Cursor?

In Cursor Settings → MCP Servers, add:

```json
{
  "tenets": {
    "command": "tenets-mcp"
  }
}
```

### What tools does the MCP server provide?

| Tool | Purpose |
|------|---------|
| `distill` | Build optimized code context for a task |
| `rank_files` | Preview file relevance without fetching content |
| `examine` | Analyze codebase structure and complexity |
| `chronicle` | Analyze git history and patterns |
| `momentum` | Track development velocity |
| `session_create` | Create stateful development sessions |
| `tenet_add` | Add guiding principles for consistency |

---

## Usage

### How does ranking work?

Tenets uses multi-factor ranking:

1. **BM25 scoring** — Text relevance to your prompt
2. **Keyword extraction** — Important terms from your query
3. **Import centrality** — Files that many others depend on
4. **Git signals** — Recently modified, frequently changed
5. **Path relevance** — Filename/path matches query terms
6. **Complexity metrics** — Weighted by code significance

### What's the difference between modes?

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| `fast` | ~1s | Good | Quick exploration |
| `balanced` | ~3s | Better | Most tasks (default) |
| `thorough` | ~10s | Best | Complex refactoring |

### How do I include/exclude files?

```bash
# Include only Python files
tenets distill "find auth" --include "*.py"

# Exclude tests and generated files
tenets distill "find auth" --exclude "test_*" --exclude "*.min.js"
```

### What's a session?

Sessions let you:
- **Pin files** for guaranteed inclusion in context
- **Track history** across multiple prompts
- **Maintain state** for long-running tasks

```bash
tenets session create auth-feature
tenets session pin auth/ --session auth-feature
tenets distill "implement OAuth" --session auth-feature
```

### What are tenets (guiding principles)?

Tenets are rules injected into every context to prevent "drift" in AI responses:

```bash
tenets tenet add "Always validate user input before database queries" --priority high
tenets tenet add "Use type hints in all Python functions" --category style
```

---

## Troubleshooting

### "MCP dependencies not installed"

Run: `pip install tenets[mcp]`

### "No files found"

- Check your path is correct
- Ensure files aren't in `.gitignore`
- Try `--include-tests` if looking for test files

### "Token limit exceeded"

- Reduce `--max-tokens`
- Use `--mode fast` for less content
- Add `--exclude` patterns for large generated files

### Server not connecting to IDE

1. Verify installation: `tenets-mcp --version`
2. Check config path is correct
3. Restart the IDE after config changes
4. Try absolute path to `tenets-mcp` binary

---

## Contributing

### How can I contribute?

- **Report bugs**: [GitHub Issues](https://github.com/jddunn/tenets/issues)
- **Submit PRs**: Fork, branch, and open a pull request
- **Improve docs**: Documentation PRs are always welcome
- **Share feedback**: [Discord](https://discord.gg/DzNgXdYm) or [email](mailto:team@tenets.dev)

### Where's the roadmap?

See our [Architecture Roadmap](architecture/roadmap.md) for planned features.

---

## Still have questions?

- **Discord**: [Join our community](https://discord.gg/DzNgXdYm)
- **Email**: [team@tenets.dev](mailto:team@tenets.dev)
- **GitHub**: [Open an issue](https://github.com/jddunn/tenets/issues)

<div style="text-align: center; padding: 2rem; margin: 2rem 0; background: rgba(245, 158, 11, 0.05); border-radius: 12px;">
  <p style="margin-bottom: 1rem;">Need enterprise support or custom development?</p>
  <a href="https://manic.agency/contact" target="_blank" rel="noopener" style="color: #f59e0b; font-weight: 600;">Contact manic.agency →</a>
</div>

