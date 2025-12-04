---
title: "Tutorial: From Zero to AI-Powered Coding"
description: Complete step-by-step tutorial for using Tenets with AI coding assistants. Learn to build context, manage sessions, and supercharge your workflow.
---

# Tutorial: From Zero to AI-Powered Coding

This tutorial walks you through using Tenets to supercharge your AI coding workflow. By the end, you'll be able to:

- Build optimal context for any coding task
- Use sessions to maintain state across prompts  
- Add tenets to keep AI responses consistent
- Integrate with Claude, Cursor, or any MCP client

**Time required**: ~15 minutes

---

## Prerequisites

- Python 3.9+
- A codebase to work with (or use any open-source project)
- Optional: Claude Desktop or Cursor IDE

---

## Step 1: Installation

### Basic Install (CLI + Python API)

```bash
pip install tenets
```

### With MCP Server (for AI Assistants)

```bash
pip install tenets[mcp]
```

Verify installation:

```bash
tenets --version
# tenets v0.4.0

tenets-mcp --version  
# tenets-mcp v0.4.0
```

---

## Step 2: Your First Distill

Navigate to any codebase and run:

```bash
cd /path/to/your/project

tenets distill "how does authentication work"
```

**What happens:**

1. Tenets scans all files in the directory
2. Ranks them by relevance to "authentication"
3. Aggregates the most relevant code
4. Outputs token-optimized context

**Example output:**

```markdown
# Context for: how does authentication work

## Relevant Files (5 of 127 scanned)

### src/auth/login.py (relevance: 0.92)
```python
def authenticate_user(username: str, password: str) -> User:
    """Authenticate user with username and password."""
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return user
    raise AuthenticationError("Invalid credentials")
```

### src/auth/middleware.py (relevance: 0.87)
...
```

---

## Step 3: Understanding Ranking Modes

Tenets offers three ranking modes:

### Fast Mode (~1 second)

```bash
tenets distill "find payment code" --mode fast
```

Uses keyword matching and path analysis. Best for quick exploration.

### Balanced Mode (~3 seconds, default)

```bash
tenets distill "find payment code" --mode balanced
```

Adds BM25 scoring, structure analysis, and git signals. Best for most tasks.

### Thorough Mode (~10 seconds)

```bash
tenets distill "refactor the database layer" --mode thorough
```

Adds ML embeddings and deep semantic analysis. Best for complex refactoring.

---

## Step 4: Working with Sessions

Sessions let you maintain state across multiple prompts.

### Create a Session

```bash
tenets session create auth-refactor
```

### Pin Important Files

```bash
# Pin a specific file
tenets session pin src/auth/login.py --session auth-refactor

# Pin an entire folder
tenets session pin src/auth/ --session auth-refactor
```

### Use the Session

```bash
tenets distill "add OAuth2 support" --session auth-refactor
```

Pinned files are **always included** in context, regardless of ranking.

### List Sessions

```bash
tenets session list
```

---

## Step 5: Adding Tenets (Guiding Principles)

Tenets prevent AI "drift" by injecting consistent rules into every context.

### Add Security Rules

```bash
tenets tenet add "Always validate and sanitize user input" --priority critical --category security
```

### Add Code Style Rules

```bash
tenets tenet add "Use type hints for all function parameters and returns" --priority high --category style
```

### Add Architecture Rules

```bash
tenets tenet add "All database access must go through the repository layer" --priority high --category architecture
```

### View Your Tenets

```bash
tenets tenet list
```

### Activate Tenets

```bash
tenets instill
```

Now every `distill` command will include your tenets at the top of the context.

---

## Step 6: Output Formats

### Markdown (default)

```bash
tenets distill "find auth" --format markdown
```

Human-readable with headers and code blocks.

### XML (Claude-optimized)

```bash
tenets distill "find auth" --format xml
```

```xml
<context>
  <file path="src/auth/login.py" relevance="0.92">
    <content>def authenticate_user(...)...</content>
  </file>
</context>
```

### JSON (programmatic)

```bash
tenets distill "find auth" --format json
```

```json
{
  "files": [
    {"path": "src/auth/login.py", "relevance": 0.92, "content": "..."}
  ],
  "token_count": 4500,
  "mode": "balanced"
}
```

---

## Step 7: MCP Integration

### Configure Claude Desktop

1. Open Claude Desktop settings
2. Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

3. Restart Claude Desktop

### Configure Cursor

1. Open Settings → MCP Servers
2. Add configuration:

```json
{
  "tenets": {
    "command": "tenets-mcp"
  }
}
```

3. Restart Cursor

### Using with AI

Now you can ask Claude or Cursor:

> "Use tenets to find code related to user authentication"

The AI will call the `distill` tool and receive optimized context automatically.

---

## Step 8: Advanced Patterns

### Filter by File Type

```bash
# Only Python files
tenets distill "find utilities" --include "*.py"

# Exclude tests
tenets distill "find utilities" --exclude "test_*"
```

### Include Test Files

```bash
tenets distill "debug the login test" --include-tests
```

### Set Token Budget

```bash
# Small context for quick questions
tenets distill "what does X do" --max-tokens 10000

# Large context for complex refactoring
tenets distill "refactor authentication" --max-tokens 150000
```

### Copy to Clipboard

```bash
tenets distill "find auth" --copy
```

### Save to File

```bash
tenets distill "find auth" --output context.md
```

---

## Step 9: Examine Your Codebase

Get insights about your codebase structure:

```bash
tenets examine
```

**Output includes:**

- File count by language
- Complexity hotspots
- Code health score
- Ownership analysis (from git)

### Focus on Complexity

```bash
tenets examine --hotspots
```

### Check Code Ownership

```bash
tenets examine --ownership
```

---

## Step 10: Track Development Velocity

See what's changing in your codebase:

```bash
tenets chronicle --since "1 week ago"
```

### Check Momentum

```bash
tenets momentum
```

Shows activity trends, hot files, and velocity metrics.

---

## Putting It All Together

Here's a complete workflow for a new feature:

```bash
# 1. Create a session
tenets session create new-feature

# 2. Add guiding principles
tenets tenet add "Follow existing patterns in the codebase" --priority high
tenets tenet add "Add tests for all new functions" --priority high

# 3. Pin relevant code
tenets session pin src/features/ --session new-feature

# 4. Build context for your AI
tenets distill "implement user notifications feature" \
  --session new-feature \
  --mode balanced \
  --max-tokens 50000 \
  --copy

# 5. Paste into Claude/Cursor and start coding!
```

---

## Next Steps

- **[MCP Documentation](MCP.md)** — Deep dive into MCP server features
- **[CLI Reference](CLI.md)** — Complete command documentation
- **[Configuration](CONFIG.md)** — Customize Tenets for your workflow
- **[Architecture](ARCHITECTURE.md)** — Understand how Tenets works

---

## Get Help

- **Discord**: [Join our community](https://discord.gg/DzNgXdYm)
- **GitHub**: [Report issues](https://github.com/jddunn/tenets/issues)
- **Email**: [team@tenets.dev](mailto:team@tenets.dev)

<div style="text-align: center; padding: 2rem; margin: 2rem 0; background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%); border-radius: 16px;">
  <h3>Built by manic.agency</h3>
  <p style="margin-bottom: 1rem;">Need custom AI tooling for your team?</p>
  <a href="https://manic.agency/contact" target="_blank" rel="noopener" style="display: inline-block; background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); color: #1a2332; padding: 0.75rem 2rem; border-radius: 8px; text-decoration: none; font-weight: 600;">Let's Talk →</a>
</div>

