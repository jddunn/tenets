# Tenets CLI Reference

**tenets** - Context that feeds your prompts. A command-line tool for intelligent code aggregation, analysis, and visualization.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Commands](#core-commands)
  - [make-context](#make-context)
  - [analyze](#analyze)
  - [track-changes](#track-changes)
  - [velocity](#velocity)
- [Visualization Commands](#visualization-commands)
  - [viz deps](#viz-deps)
  - [viz complexity](#viz-complexity)
  - [viz coupling](#viz-coupling)
  - [viz contributors](#viz-contributors)
- [Session Commands](#session-commands)
- [Configuration](#configuration)
- [Common Use Cases](#common-use-cases)
- [Examples](#examples)

## Installation

```bash
# Basic install (core features only)
pip install tenets

# With visualization support
pip install tenets[viz]

# With ML-powered ranking
pip install tenets[ml]

# Everything
pip install tenets[all]
```

## Quick Start

```bash
# Generate context for AI pair programming
tenets make-context "implement OAuth2" ./src

# Analyze your codebase
tenets analyze

# Track recent changes
tenets track-changes --since yesterday

# Visualize dependencies (ASCII)
tenets viz deps --format ascii
```

## Core Commands

### make-context

Generate optimized context for LLMs from your codebase.

```bash
tenets make-context <prompt> [path] [options]
```

**Arguments:**
- `prompt`: Your query or task description (can be text or URL)
- `path`: Directory or files to analyze (default: current directory)

**Options:**
- `--format, -f`: Output format: `markdown` (default), `xml`, `json`
- `--model, -m`: Target LLM model (e.g., `gpt-4o`, `claude-3-opus`)
- `--output, -o`: Save to file instead of stdout
- `--max-tokens`: Maximum tokens for context
- `--mode`: Analysis mode: `fast`, `balanced` (default), `thorough`
- `--no-git`: Disable git context inclusion
- `--include, -i`: Include file patterns (e.g., `"*.py,*.js"`)
- `--exclude, -e`: Exclude file patterns (e.g., `"test_*,*.backup"`)
- `--session, -s`: Use session for stateful context
- `--estimate-cost`: Show token usage and cost estimate
- `--verbose, -v`: Show detailed analysis info

**Examples:**

```bash
# Basic usage - finds all relevant files for implementing OAuth2
tenets make-context "implement OAuth2 authentication"

# From a GitHub issue
tenets make-context https://github.com/org/repo/issues/123

# Target specific model with cost estimation
tenets make-context "add caching layer" --model gpt-4o --estimate-cost

# Filter by file types
tenets make-context "review API endpoints" --include "*.py,*.yaml" --exclude "test_*"

# Save context to file
tenets make-context "debug login issue" --output context.md

# Use thorough analysis for complex tasks
tenets make-context "refactor authentication system" --mode thorough

# Session-based context (maintains state)
tenets make-context "build payment system" --session payment-feature
```

### analyze

Analyze codebase structure, complexity, and patterns.

```bash
tenets analyze [path] [options]
```

**Options:**
- `--deep, -d`: Perform deep analysis with AST parsing
- `--output, -o`: Save results to file
- `--metrics`: Show detailed code metrics
- `--complexity`: Show complexity analysis
- `--ownership`: Show code ownership (requires git)
- `--hotspots`: Show frequently changed files
- `--format, -f`: Output format: `table` (default), `json`, `yaml`
- `--no-git`: Disable git analysis

**Examples:**

```bash
# Basic analysis with summary table
tenets analyze

# Deep analysis with metrics
tenets analyze --deep --metrics

# Show complexity hotspots
tenets analyze --complexity --hotspots

# Export full analysis as JSON
tenets analyze --output analysis.json --format json

# Analyze specific directory
tenets analyze ./src --ownership
```

**Output Example (Table Format):**
```
Codebase Analysis
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric          ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total Files     │ 156      │
│ Total Lines     │ 24,531   │
│ Languages       │ Python,  │
│                 │ JavaScript│
│ Avg Complexity  │ 4.32     │
│ Git Branch      │ main     │
│ Contributors    │ 8        │
└─────────────────┴──────────┘
```

### track-changes

Track code changes over time using git history.

```bash
tenets track-changes [options]
```

**Options:**
- `--since, -s`: Time period (default: `"last-week"`)
  - Examples: `"yesterday"`, `"last-month"`, `"2024-01-01"`, `"3 weeks"`
- `--path, -p`: Repository path (default: current directory)
- `--pattern`: Filter files by pattern
- `--summary`: Show summary statistics

**Examples:**

```bash
# Changes in the last week
tenets track-changes

# Changes since yesterday
tenets track-changes --since yesterday

# Changes to Python files in the last month
tenets track-changes --since "last-month" --pattern "*.py"

# Changes with author summary
tenets track-changes --since "2 weeks" --summary
```

**Output Example:**
```
📊 Change Summary
════════════════
Total commits: 47
Active authors: 5

Changed Files
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ File                   ┃ Commits ┃ Lines Changed ┃ Last Modified  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ src/auth/oauth.py      │ 12      │ 324          │ 2024-03-15     │
│ src/models/user.py     │ 8       │ 156          │ 2024-03-14     │
│ tests/test_auth.py     │ 6       │ 89           │ 2024-03-15     │
└────────────────────────┴─────────┴──────────────┴────────────────┘
```

### velocity

Track development velocity and team productivity metrics.

```bash
tenets velocity [options]
```

**Options:**
- `--path, -p`: Repository path (default: current directory)
- `--since, -s`: Time period (default: `"last-month"`)
- `--team`: Show team-wide statistics
- `--author, -a`: Show stats for specific author

**Examples:**

```bash
# Personal velocity for last month
tenets velocity

# Team velocity for the quarter
tenets velocity --team --since "3 months"

# Individual contributor stats
tenets velocity --author "alice@example.com"
```

**Output Example:**
```
Weekly Velocity
══════════════
W10 ████████████████████ 45 commits
W11 ██████████████ 32 commits
W12 ████████████████████████ 52 commits
W13 ███████████ 28 commits

Team Statistics
Total contributors: 8

Top Contributors:
  Alice Smith: 127 commits, 3,421 lines added
  Bob Johnson: 89 commits, 2,156 lines added
```

## Visualization Commands

All visualization commands support ASCII output for terminal display, with optional graphical formats.

### viz deps

Visualize code dependencies and architecture.

```bash
tenets viz deps [path] [options]
```

**Options:**
- `--output, -o`: Save to file
- `--format, -f`: Format: `ascii` (default for terminal), `svg`, `png`, `html`
- `--cluster-by`: Group by: `directory`
- `--max-nodes`: Maximum nodes to display (default: 100)

**Examples:**

```bash
# ASCII dependency tree in terminal
tenets viz deps

# Save as SVG
tenets viz deps --output dependencies.svg

# Cluster by directory structure
tenets viz deps --cluster-by directory

# Limit to core files
tenets viz deps --max-nodes 50
```

**ASCII Output Example:**
```
Dependency Tree
══════════════════════════════════════════════════

├── main.py
│   ├── auth/handler.py
│   │   ├── auth/oauth.py
│   │   │   └── utils/crypto.py
│   │   └── models/user.py
│   │       └── db/base.py
│   └── api/routes.py
│       ├── api/endpoints.py
│       └── middleware/cors.py
└── config.py
```

### viz complexity

Visualize code complexity metrics.

```bash
tenets viz complexity [path] [options]
```

**Options:**
- `--output, -o`: Save to file
- `--format, -f`: Format: `ascii`, `png`, `html`
- `--metric, -m`: Metric type: `cyclomatic` (default), `cognitive`
- `--threshold`: Highlight files above threshold
- `--hotspots`: Focus on complexity hotspots

**Examples:**

```bash
# ASCII bar chart of complexity
tenets viz complexity

# Show only high-complexity files
tenets viz complexity --threshold 10 --hotspots

# Save as image
tenets viz complexity --output complexity.png
```

**ASCII Output Example:**
```
Complexity Analysis (cyclomatic)
══════════════════════════════════════════════════

auth/oauth.py                 ● ████████████████████████████ 28
models/user.py               ◐ ██████████████ 15
api/endpoints.py             ◐ ████████████ 12
utils/validators.py          ● ██████ 8
config/settings.py           ● ████ 5

Legend: ● Low  ◐ Medium  ◑ High  ○ Very High
```

### viz coupling

Visualize files that frequently change together.

```bash
tenets viz coupling [path] [options]
```

**Options:**
- `--output, -o`: Save to file
- `--format, -f`: Format: `ascii`, `html`
- `--min-coupling`: Minimum coupling count (default: 2)

**Examples:**

```bash
# Show file coupling matrix
tenets viz coupling

# Only strong couplings
tenets viz coupling --min-coupling 5

# Interactive HTML matrix
tenets viz coupling --output coupling.html
```

**ASCII Output Example:**
```
File Coupling Matrix
══════════════════════════════════════════════════

                    auth.py  user.py  api.py  test.py
auth.py               -        8       3       12
user.py               8        -       5       10
api.py                3        5       -       7
test_auth.py         12       10      7        -
```

### viz contributors

Visualize contributor activity and code ownership.

```bash
tenets viz contributors [path] [options]
```

**Options:**
- `--output, -o`: Save to file
- `--format, -f`: Format: `ascii`, `png`
- `--active`: Show only currently active contributors

**Examples:**

```bash
# Contributor stats
tenets viz contributors

# Active contributors only
tenets viz contributors --active
```

## Session Commands

Tenets can persist session state across distill runs. When a configuration is loaded, sessions are stored in a local SQLite database under the cache directory (see Storage below). Use `--session <name>` with commands like `distill` to build iterative context.

- `--session` behavior: When provided, Tenets maintains a named session that accumulates context history and preferences. With configuration present, this state is stored in `${CACHE_DIR}/tenets.db` and restored in subsequent runs.

### session start

Create a new analysis session.

```bash
tenets session start <name> [options]
```

**Options:**
- `--path, -p`: Project path (default: current directory)

**Example:**
```bash
tenets session start payment-integration
```

### session list

List all sessions.

```bash
tenets session list
```

### session show

Show specific files in session context.

```bash
tenets session show <session-name> <files...>
```

**Example:**
```bash
tenets session show payment-integration src/payment.py src/stripe.py
```

### session find

Search for patterns in session files.

```bash
tenets session find <session-name> <pattern> [options]
```

**Options:**
- `--context, -c`: Lines of context around matches
- `--lang, -l`: Filter by language

## Session Management

```
# Create or ensure session exists
tenets session create feature-x

# List sessions
tenets session list

# Show details
tenets session show feature-x

# Attach a generated context artifact
tenets session add feature-x context_result context.json

# Reset (delete & recreate) a session with all context
tenets session reset feature-x

# Delete session but keep context artifacts
tenets session delete feature-x --keep-context
```

## Cache Management

```
# Show cache stats (path, file count, size)
tenets config cache-stats

# Cleanup old/oversized entries respecting TTL
tenets config cleanup-cache

# Clear ALL caches (analysis + general) – destructive
tenets config clear-cache --yes
```

Git data is used strictly for ranking relevance unless explicitly requested via commands like `chronicle` or `viz contributors`; it is not embedded in `distill` output.

## Configuration

### config set

Set configuration values.

```bash
tenets config set <key> <value>
```

**Examples:**
```bash
# Set default ranking algorithm
tenets config set ranking.algorithm balanced

# Set maximum file size
tenets config set scanner.max_file_size 10000000

# Enable ML features
tenets config set nlp.use_embeddings true
```

### config show

Show configuration.

```bash
tenets config show [options]
```

**Options:**
- `--key, -k`: Show specific key

**Examples:**
```bash
# Show all config
tenets config show

# Show model costs
tenets config show --key costs

# Show specific setting
tenets config show --key ranking.algorithm
```

## Storage

Writable data is stored in a user/project cache directory:

- Default: `${HOME}/.tenets/cache` (Windows: `%USERPROFILE%\\.tenets\\cache`)
- Main DB: `${CACHE_DIR}/tenets.db` (sessions and future state)
- Analysis cache: `${CACHE_DIR}/analysis/analysis.db`

Override via `.tenets.yml`:

```yaml
cache:
  directory: /path/to/custom/cache
```

Or environment:

```bash
TENETS_CACHE_DIRECTORY=/path/to/custom/cache
```

Note on cost estimation: When `--estimate-cost` is used with `make-context`, Tenets estimates costs using model limits and the built-in pricing table from `SUPPORTED_MODELS`.

## Common Use Cases

### 1. AI Pair Programming

Generate context for ChatGPT/Claude when working on features:

```bash
# Initial context for new feature
tenets make-context "implement user authentication with JWT" > auth_context.md

# Paste auth_context.md into ChatGPT, then iterate:
tenets make-context "add password reset functionality" --session auth-feature

# AI needs to see specific files?
tenets session show auth-feature src/auth/password_reset.py
```

### 2. Code Review Preparation

Understand what changed and why:

```bash
# See what changed in the sprint
tenets track-changes --since "2 weeks" --summary

# Get context for reviewing a PR
tenets make-context "review payment processing changes" --since "feature-branch"

# Check complexity of changed files
tenets analyze --complexity --hotspots
```

### 3. Onboarding to New Codebase

Quickly understand project structure:

```bash
# Get project overview
tenets analyze --metrics

# Visualize architecture
tenets viz deps --format ascii

# Find the most complex areas
tenets viz complexity --hotspots

# See who knows what
tenets viz contributors
```

### 4. Debugging Production Issues

Find relevant code for debugging:

```bash
# Get all context related to the error
tenets make-context "users getting 500 error on checkout" --mode thorough

# Include recent changes
tenets make-context "debug payment timeout" --since "last-deploy"

# Find all error handlers
tenets session find debug-session "error|exception" --context 10
```

### 5. Technical Debt Assessment

Identify areas needing refactoring:

```bash
# Find complex files
tenets analyze --complexity --threshold 15

# Find tightly coupled code
tenets viz coupling --min-coupling 5

# Track velocity trends
tenets velocity --team --since "6 months"
```

### 6. Architecture Documentation

Generate architecture insights:

```bash
# Export dependency graph
tenets viz deps --output architecture.svg --cluster-by directory

# Generate comprehensive analysis
tenets analyze --deep --output analysis.json --format json

# Create context for documentation
tenets make-context "document API architecture" ./src/api
```

## Examples

### Complete Workflow Example

```bash
# 1. Start a new feature
tenets session start oauth-integration

# 2. Get initial context
tenets make-context "implement OAuth2 with Google and GitHub" \
  --session oauth-integration \
  --include "*.py,*.yaml" \
  --exclude "test_*" \
  --model gpt-4o \
  --estimate-cost > oauth_context.md

# 3. Paste into ChatGPT, start coding...

# 4. AI needs more specific context
tenets session show oauth-integration src/auth/providers.py

# 5. Check your progress
tenets track-changes --since "today"

# 6. Visualize what you built
tenets viz deps src/auth --format ascii

# 7. Check complexity
tenets analyze src/auth --complexity

# 8. Prepare for review
tenets make-context "OAuth implementation ready for review" \
  --session oauth-integration \
  --since "feature/oauth"
```

### Configuration File Example

Create `.tenets.yml` in your project:

```yaml
# .tenets.yml
context:
  ranking: balanced
  max_tokens: 100000
  include_git: true

scanner:
  respect_gitignore: true
  max_file_size: 5000000
  
ignore:
  - "*.generated.*"
  - "vendor/"
  - "build/"
  
output:
  format: markdown
  summarize_long_files: true
```

## Tips and Tricks

1. **Start with fast mode** for quick exploration, use thorough for complex tasks
2. **Use sessions** for multi-step features to maintain context
3. **ASCII visualizations** are great for README files and documentation
4. **Combine commands** - analyze first, then make-context with insights
5. **Git integration** works automatically - no setup needed
6. **Include/exclude patterns** support standard glob syntax
7. **Cost estimation** helps budget API usage before sending to LLMs

## Environment Variables

- `TENETS_CONFIG_PATH`: Custom config file location
- `TENETS_LOG_LEVEL`: Set log level (DEBUG, INFO, WARNING, ERROR)
- `TENETS_CACHE_DIR`: Custom cache directory
- `TENETS_NO_COLOR`: Disable colored output

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Invalid arguments
- `3`: File not found
- `4`: Git repository required but not found

---

For more information, visit [https://github.com/jddunn/tenets](https://github.com/jddunn/tenets)

### Verbosity & Output Controls

Control log verbosity globally:

```bash
# Default (warnings and above only)
TENETS_LOG_LEVEL=WARNING tenets distill "add caching layer"

# Verbose
tenets --verbose distill "add caching layer"

# Quiet / errors only
tenets --quiet distill "add caching layer"
# or
tenets --silent distill "add caching layer"
```

The `distill` command includes a Suggestions section when no files are included, with tips to adjust relevance thresholds, token budget, and include patterns.