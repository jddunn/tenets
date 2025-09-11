# Tenets CLI Reference

**tenets** - Context that feeds your prompts. A command-line tool for intelligent code aggregation, analysis, and visualization.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Commands](#core-commands)
  - [distill](#distill)
  - [instill](#instill)
  - [rank](#rank)
  - [examine](#examine)
  - [chronicle](#chronicle)
  - [momentum](#momentum)
  - [tenet](#tenet)
- [Visualization Commands](#visualization-commands)
  - [viz deps](#viz-deps)
  - [viz complexity](#viz-complexity)
  - [viz coupling](#viz-coupling)
  - [viz contributors](#viz-contributors)
- [Session Commands](#session-commands)
- [Tenet Commands](#tenet-commands)
- [Instill Command](#instill-command)
- [System Instruction Commands](#system-instruction-commands)
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
tenets distill "implement OAuth2" ./src

# Analyze your codebase
tenets examine

# Track recent changes
tenets chronicle --since yesterday

# Visualize dependencies (ASCII by default)
tenets viz deps
```

## Core Commands

### distill

Generate optimized context for LLMs from your codebase.

```bash
tenets distill [path] <prompt> [options]
```


**Arguments:**

- **path**: Directory or files to analyze (optional, defaults to current directory)
- **prompt**: Your query or task description (can be text or URL, required)

**Options:**

- `--format`, `-f`: Output format: markdown (default), xml, json
- `--model`, `-m`: Target LLM model (e.g., gpt-4o, claude-3-opus)
- `--output`, `-o`: Save to file instead of stdout
- `--max-tokens`: Maximum tokens for context
- `--mode`: Analysis mode: fast, balanced (default), thorough
- `--no-git`: Disable git context inclusion
- `--use-stopwords`: Enable stopword filtering for keyword analysis
- `--include`, `-i`: Include file patterns (e.g., ".py,.js")
- `--exclude`, `-e`: Exclude file patterns (e.g., "test_,.backup")
- `--session`, `-s`: Use a named session for stateful context
- `--estimate-cost`: Show token usage and cost estimate
- `--verbose`, `-v`: Show detailed analysis info
- `--full`: Include full content for all ranked files (no summarization) until token budget reached
- `--condense`: Condense whitespace (collapse large blank runs, trim trailing spaces) before token counting
- `--remove-comments`: Strip comments (heuristic, language-aware) before token counting
- `--copy`: Copy distilled context directly to clipboard (or set output.copy_on_distill: true in config)

**Examples:**

```bash
# Basic usage - path is optional, defaults to current directory
tenets distill . "implement OAuth2 authentication"  # Explicit current directory
tenets distill ./src "implement OAuth2"             # Specific directory
tenets distill "implement OAuth2 authentication"    # Path defaults to current directory

# From a GitHub issue (path optional)
tenets distill https://github.com/org/repo/issues/123

# Target specific model with cost estimation
tenets distill "add caching layer" --model gpt-4o --estimate-cost

# Filter by file types (scans current directory by default)
tenets distill "review API endpoints" --include "*.py,*.yaml" --exclude "test_*"

# Save context to file
tenets distill "debug login issue" --output context.md

# Use thorough analysis for complex tasks
tenets distill "refactor authentication system" --mode thorough

# Session-based context (maintains state)
tenets distill "build payment system" --session payment-feature

# Full mode (force raw content inclusion)
tenets distill "inspect performance code" --full --max-tokens 60000

# Reduce token usage by stripping comments & whitespace
tenets distill "understand API surface" --remove-comments --condense --stats
```

#### Content Transformations

You can optionally transform file content prior to aggregation/token counting:

| Flag | Effect | Safety |
|------|--------|--------|
| `--full` | Disables summarization; includes raw file content until budget is hit | Budget only |
| `--remove-comments` | Removes line & block comments (language-aware heuristics) | Aborts if >60% of non-empty lines would vanish |
| `--condense` | Collapses 3+ blank lines to 1, trims trailing spaces, ensures final newline | Lossless for code logic |

Transformations are applied in this order: comment stripping -> whitespace condensation. Statistics (e.g. removed comment lines) are tracked internally and may be surfaced in future `--stats` expansions.

#### Pinned Files

Pin critical files so they're always considered first in subsequent distill runs for the same session:

```bash
# Pin individual files
tenets instill --session refactor-auth --add-file src/auth/service.py --add-file src/auth/models.py

# Pin all files in a folder (respects .gitignore)
tenets instill --session refactor-auth --add-folder src/auth

# List pinned files
tenets instill --session refactor-auth --list-pinned

# Generate context (pinned files prioritized)
tenets distill "add JWT refresh tokens" --session refactor-auth --remove-comments
```

Pinned files are stored in the session metadata (SQLite) and reloaded automatically—no extra flags needed when distilling.

#### Ranking presets and thresholds

- Presets (selected via `--mode` or config `ranking.algorithm`):
  - `fast` – keyword + path signals (broad, quick)
  - `balanced` (default) – multi-factor (keywords, path, imports, git, complexity)
  - `thorough` – deeper analysis (heavier)

- Threshold (config `ranking.threshold`) controls inclusion. Lower = include more files.
  - Typical ranges:
    - fast: 0.05–0.10
    - balanced: 0.10–0.20
    - thorough: 0.10–0.20

Configure in `.tenets.yml` (repo root):

```yaml
ranking:
  algorithm: fast      # fast | balanced | thorough
  threshold: 0.05      # 0.0–1.0
```

One-off overrides (environment, Git Bash):

```bash
TENETS_RANKING_THRESHOLD=0.05 TENETS_RANKING_ALGORITHM=fast \
  tenets distill "implement OAuth2" --include "*.py,*.md" --max-tokens 50000

# Copy output to clipboard directly
tenets distill "implement OAuth2" --copy

# Enable automatic copying in config
output:
  copy_on_distill: true
```

Inspect current config:

```bash
tenets config show --key ranking
```

See also: docs/CONFIG.md for full configuration details.



### rank

Show ranked files by relevance without their content.

```bash
tenets rank <prompt> [path] [options]
```

**Arguments:**

- **prompt**: Your query or task to rank files against (required)
- **path**: Directory or files to analyze (optional, defaults to current directory)

**Options:**

- `--format`, `-f`: Output format: markdown (default), json, xml, html, tree
- `--output`, `-o`: Save to file instead of stdout
- `--mode`, `-m`: Ranking mode: fast, balanced (default), thorough
- `--top`, `-t`: Show only top N files
- `--min-score`: Minimum relevance score (0.0-1.0)
- `--max-files`: Maximum number of files to show
- `--tree`: Show results as directory tree
- `--scores/--no-scores`: Show/hide relevance scores (default: show)
- `--factors`: Show ranking factor breakdown
- `--path-style`: Path display: relative (default), absolute, name
- `--include`, `-i`: Include file patterns (e.g., "*.py,*.js")
- `--exclude`, `-e`: Exclude file patterns (e.g., "test_*,*.backup")
- `--include-tests`: Include test files
- `--exclude-tests`: Explicitly exclude test files
- `--no-git`: Disable git signals in ranking
- `--session`, `-s`: Use session for stateful ranking
- `--stats`: Show ranking statistics
- `--verbose`, `-v`: Show detailed debug information
- `--copy`: Copy file list to clipboard (also enabled automatically if config.output.copy_on_rank is true)

**Examples:**

```bash
# Basic usage - prompt first, path optional (defaults to current directory)
tenets rank "implement OAuth2" --top 10         # Scans current directory by default
tenets rank "implement OAuth2" . --top 10       # Explicit current directory
tenets rank "implement OAuth2" ./src --top 10   # Specific directory

# Show files above a relevance threshold
tenets rank "fix authentication bug" . --min-score 0.3

# Tree view with ranking factors (path defaults to current dir if omitted)
tenets rank "add caching layer" --tree --factors

# Export ranking as JSON for automation
tenets rank "review API endpoints" --format json -o ranked_files.json

# Quick file list to clipboard (no scores)
tenets rank "database queries" --top 20 --copy --no-scores

# Show only Python files with detailed factors
tenets rank "refactor models" --include "*.py" --factors --stats

# HTML report with interactive tree view
tenets rank "security audit" --format html -o security_files.html --tree
```

**Use Cases:**

1. **Understanding Context**: See which files would be included in a `distill` command without generating the full context
2. **File Discovery**: Find relevant files for manual inspection
3. **Automation**: Export ranked file lists for feeding into other tools or scripts
4. **Code Review**: Identify files most relevant to a particular feature or bug
5. **Impact Analysis**: See which files are most connected to a specific query

**Output Formats:**

- **Markdown**: Numbered list sorted by relevance with scores and optional factors
- **Tree**: Directory tree structure sorted by relevance (directories ordered by their highest-scoring file)
- **JSON**: Structured data with paths, scores, ranks, and factors (preserves relevance order)
- **XML**: Structured XML for integration with other tools
- **HTML**: Interactive web page with relevance-sorted display

The ranking uses the same intelligent multi-factor analysis as `distill`:
- Semantic similarity (ML-based when available)
- Keyword matching
- BM25/TF-IDF statistical relevance
- Import/dependency centrality
- Path relevance
- Git signals (recent changes, frequency)

### examine

Analyze codebase structure, complexity, and patterns.

```bash
tenets examine [path] [options]
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
tenets examine

# Deep analysis with metrics
tenets examine --deep --metrics

# Show complexity hotspots
tenets examine --complexity --hotspots

# Export full analysis as JSON
tenets examine --output analysis.json --format json

# Generate HTML examination report
tenets examine --format html --output examination_report.html

# Generate detailed HTML report with all analyses
tenets examine --ownership --hotspots --show-details --format html -o report.html

# Analyze specific directory with ownership tracking
tenets examine ./src --ownership

# Generate multiple format reports
tenets examine --format json -o analysis.json
tenets examine --format html -o analysis.html
tenets examine --format markdown -o analysis.md
```

**Coverage Reports:**

```bash
# Run tests with coverage and generate HTML report
pytest --cov=tenets --cov-report=html

# View HTML coverage report (opens htmlcov/index.html)
python -m webbrowser htmlcov/index.html

# Run tests with multiple coverage formats
pytest --cov=tenets --cov-report=html --cov-report=xml --cov-report=term

# Run specific test module with coverage
pytest tests/cli/commands/test_examine.py --cov=tenets.cli.commands.examine --cov-report=html
```

**Output Example (Table Format):**
```
Codebase Analysis
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric          ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Total Files     │ 156       │
│ Total Lines     │ 24,531    │
│ Languages       │ Python,   │
│                 │ JavaScript│
│ Avg Complexity  │ 4.32      │
│ Git Branch      │ main      │
│ Contributors    │ 8         │
└─────────────────┴───────────┘
```

### chronicle

Track code changes over time using git history.

```bash
tenets chronicle [options]
```

**Options:**
- `--since, -s`: Time period (e.g., "yesterday", "last-month", "2024-01-01")
- `--path, -p`: Repository path (default: current directory)
- `--author, -a`: Filter by author
- `--limit, -n`: Maximum commits to display

**Examples:**

```bash
# Changes in the last week
tenets chronicle --since "last-week"

# Changes since yesterday
tenets chronicle --since yesterday

# Filter by author
tenets chronicle --author "alice@example.com"
```

### momentum

Track development velocity and team productivity metrics.

```bash
tenets momentum [options]
```

**Options:**
- `--path, -p`: Repository path (default: current directory)
- `--since, -s`: Time period (default: "last-month")
- `--team`: Show team-wide statistics
- `--author, -a`: Show stats for specific author

**Examples:**

```bash
# Personal velocity for last month
tenets momentum

# Team velocity for the quarter
tenets momentum --team --since "3 months"

# Individual contributor stats
tenets momentum --author "alice@example.com"
```

### instill

Apply tenets to your current context by injecting them into prompts and outputs.

```bash
tenets instill [context] [options]
```

**Options:**
- `--session, -s`: Session name for tracking
- `--frequency`: Injection frequency: `always`, `periodic`, `adaptive`
- `--priority`: Minimum tenet priority: `low`, `medium`, `high`, `critical`
- `--max-tokens`: Maximum tokens to add
- `--format`: Output format

**Examples:**

```bash
# Apply all pending tenets
tenets instill "Current code context"

# Apply tenets for specific session
tenets instill --session feature-x

# Adaptive injection based on complexity
tenets instill --frequency adaptive
```

### tenet

Manage project tenets - rules and guidelines for your codebase.

```bash
tenets tenet [subcommand] [options]
```

**Subcommands:**
- `add`: Add a new tenet
- `list`: List all tenets
- `remove`: Remove a tenet
- `show`: Show tenet details
- `export`: Export tenets
- `import`: Import tenets

**Examples:**

```bash
# Add a new tenet
tenets tenet add "Always use type hints"

# List all tenets
tenets tenet list

# Remove a tenet
tenets tenet remove <tenet-id>
```

## Visualization Commands

All visualization commands support ASCII output for terminal display, with optional graphical formats.

### viz deps

Visualize code dependencies and architecture with intelligent project detection.

```bash
tenets viz deps [path] [options]
```

**Options:**
- `--output, -o`: Save to file (e.g., architecture.svg)
- `--format, -f`: Output format: `ascii`, `svg`, `png`, `html`, `json`, `dot`
- `--level, -l`: Dependency level: `file` (default), `module`, `package`
- `--cluster-by`: Group nodes by: `directory`, `module`, `package`
- `--max-nodes`: Maximum nodes to display
- `--include, -i`: Include file patterns (e.g., "*.py")
- `--exclude, -e`: Exclude file patterns (e.g., "*test*")
- `--layout`: Graph layout: `hierarchical`, `circular`, `shell`, `kamada`

**Features:**
- **Auto-detection**: Automatically detects project type (Python, Node.js, Java, Go, etc.)
- **Smart aggregation**: Three levels of dependency views (file, module, package)
- **Interactive HTML**: D3.js or Plotly-based interactive visualizations
- **Pure Python**: All visualization libraries installable via `pip install tenets[viz]`

**Examples:**

```bash
# Auto-detect project type and show dependencies
tenets viz deps

# Generate interactive HTML visualization
tenets viz deps --format html --output deps.html

# Module-level dependencies as SVG
tenets viz deps --level module --format svg --output modules.svg

# Package architecture with clustering
tenets viz deps --level package --cluster-by package --output packages.png

# Circular layout for better visibility
tenets viz deps --layout circular --format svg --output circular.svg

# Limit to top 50 nodes for large projects
tenets viz deps --max-nodes 50 --format png --output top50.png

# Export to Graphviz DOT format
tenets viz deps --format dot --output graph.dot

# Filter specific files
tenets viz deps src/ --include "*.py" --exclude "*test*"
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

- Only one session is considered active at a time. Resuming a session will mark all others inactive.
- If a session NAME is omitted for `resume` or `exit`, Tenets operates on the currently active session.

### session create

Create a new analysis session.

```bash
tenets session create <name>
```

**Example:**
```bash
tenets session create payment-integration
```

### session start

Alias of `session create`.

```bash
tenets session start <name>
```

### session resume

Mark an existing session as active.

```bash
# Resume the active session (if one exists)
tenets session resume

# Or specify by name
tenets session resume <name>
```

### session exit

Mark a session as inactive.

```bash
# Exit the current active session
tenets session exit

# Or exit a specific session by name
tenets session exit <name>
```

### session list

List all sessions.

```bash
tenets session list
```

The output includes an Active column ("yes" indicates the current session).

### session delete

Delete a specific session.

```bash
tenets session delete <name> [--keep-context]
```

Options:
- `--keep-context`: Keep stored context artifacts (default: false)

### session reset

Reset (delete and recreate) a session, purging its context.

```bash
tenets session reset <name>
```

### session clear

Delete ALL sessions at once. Useful for clearing cache and starting fresh.

```bash
tenets session clear [--keep-context]
```

Options:
- `--keep-context`: Keep stored artifacts (default: false, deletes everything)

**Example:**
```bash
# Clear all sessions and their data
tenets session clear

# Clear sessions but preserve context files
tenets session clear --keep-context
```

### session show

Show details for a specific session.

```bash
tenets session show <name>
```

### session add

Attach arbitrary content to a session.

```bash
tenets session add <name> <kind> <file>
```

Arguments:
- `name`: Session name
- `kind`: Content type tag (e.g., note, context_result)
- `file`: File to attach

Notes:
- Creating or resetting a session marks it active.
- Only one session is active at a time (resuming one deactivates others).
- Session data is stored in SQLite under `~/.tenets/cache/sessions.db`

## Tenet Commands

Create and manage guiding principles (“tenets”) that can be injected into context.

### tenet add

Add a new tenet.

```bash
tenets tenet add "Always use type hints" --priority high --category style
tenets tenet add "Validate all user inputs" --priority critical --category security
tenets tenet add "Use async/await for I/O" --session feature-x
```

Options:
- `--priority, -p`: low | medium | high | critical (default: medium)
- `--category, -c`: Freeform tag (e.g., architecture, security, style, performance, testing)
- `--session, -s`: Bind tenet to a session

### tenet list

List tenets with filters.

```bash
tenets tenet list
tenets tenet list --pending
tenets tenet list --session oauth --category security --verbose
```

Options:
- `--pending`: Only pending
- `--instilled`: Only instilled
- `--session, -s`: Filter by session
- `--category, -c`: Filter by category
- `--verbose, -v`: Show full content and metadata

### tenet remove

Remove a tenet by ID (partial ID accepted).

```bash
tenets tenet remove abc123
tenets tenet remove abc123 --force
```

### tenet show

Show details for a tenet.

```bash
tenets tenet show abc123
```

### tenet export / import

Export/import tenets.

```bash
# Export to stdout or file
tenets tenet export
tenets tenet export --format json --session oauth -o team-tenets.json

# Import from file (optionally into a session)
tenets tenet import team-tenets.yml
tenets tenet import standards.json --session feature-x
```

## Instill Command

Apply tenets to the current context with smart strategies (periodic/adaptive/manual).

```bash
tenets instill [options]
```

Common options:
- `--session, -s`: Use a named session for history and pinned files
- `--force`: Force instillation regardless of frequency
- `--max-tenets`: Cap number of tenets applied

Examples:

```bash
# Apply pending tenets for a session
tenets instill --session refactor-auth

# Force all tenets once
tenets instill --force
```

## System Instruction Commands

Manage the system instruction (system prompt) that can be auto-injected at the start of a session’s first distill (or every output if no session is used).

### system-instruction set

Set/update the system instruction and options.

```bash
tenets system-instruction set "You are a helpful coding assistant" \
  --enable \
  --position top \
  --format markdown

# From file
tenets system-instruction set --file prompts/system.md --enable
```

Options:
- `--file, -f`: Read instruction from file
- `--enable/--disable`: Enable or disable auto-injection
- `--position`: Placement: `top`, `after_header`, `before_content`
- `--format`: Format of injected block: `markdown`, `xml`, `comment`, `plain`
- `--save/--no-save`: Persist to config

### system-instruction show

Display current configuration and instruction.

```bash
tenets system-instruction show
tenets system-instruction show --raw
```

Options:
- `--raw`: Print raw instruction only

### system-instruction clear

Clear and disable the system instruction.

```bash
tenets system-instruction clear
tenets system-instruction clear --yes
```

Options:
- `--yes, -y`: Skip confirmation

### system-instruction test

Preview how injection would modify content.

```bash
tenets system-instruction test
tenets system-instruction test --session my-session
```

Options:
- `--session`: Test with a session to respect once-per-session behavior

### system-instruction export

Export the instruction to a file.

```bash
tenets system-instruction export prompts/system.md
```

### system-instruction validate

Validate the instruction for basic issues and optional token estimates.

```bash
tenets system-instruction validate
tenets system-instruction validate --tokens --max-tokens 800
```

Options:
- `--tokens`: Show a rough token estimate
- `--max-tokens`: Threshold for warnings/errors

### system-instruction edit

Edit the instruction in your editor and save changes back to config.

```bash
tenets system-instruction edit
tenets system-instruction edit --editor code
```

### session show

Show session details.

```bash
tenets session show <name>
```

### session add

Attach an artifact (stored as text) to a session.

```bash
tenets session add <name> <kind> <file>
```

Examples of `kind`: `note`, `context_result`, `summary`

### session reset

Reset (delete and recreate) a session and purge its context.

```bash
tenets session reset <name>
```

### session delete

Delete a session. Optionally keep stored artifacts.

```bash
tenets session delete <name> [--keep-context]
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

Note on cost estimation: When `--estimate-cost` is used with `distill`, Tenets estimates costs using model limits and the built-in pricing table from `SUPPORTED_MODELS`.

## Common Use Cases

### 1. AI Pair Programming

Generate context for ChatGPT/Claude when working on features:

```bash
# Initial context for new feature
tenets distill "implement user authentication with JWT" > auth_context.md

# Paste auth_context.md into ChatGPT, then iterate:
tenets distill "add password reset functionality" --session auth-feature

# AI needs to see session info?
tenets session show auth-feature
```

### 2. Code Review Preparation

Understand what changed and why:

```bash
# See what changed in the sprint
tenets chronicle --since "2 weeks" --summary

# Get context for reviewing a PR
tenets distill "review payment processing changes"

# Check complexity of changed files
tenets examine --complexity --hotspots
```

### 3. Onboarding to New Codebase

Quickly understand project structure:

```bash
# Get project overview
tenets examine --metrics

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
tenets distill "users getting 500 error on checkout" --mode thorough

# Include recent changes summary
tenets chronicle --since "last-deploy"

# Search for patterns within a session by iterating with prompts
tenets distill "find error handlers" --session debug-session
```

### 5. Technical Debt Assessment

Identify areas needing refactoring:

```bash
# Find complex files
tenets examine --complexity --threshold 15

# Find tightly coupled code
tenets viz coupling --min-coupling 5

# Track velocity trends
tenets momentum --team --since "6 months"
```

### 6. Architecture Documentation

Generate architecture insights:

```bash
# Export dependency graph
tenets viz deps --output architecture.svg --cluster-by directory

# Generate comprehensive analysis
tenets examine --deep --output analysis.json --format json

# Create context for documentation
tenets distill "document API architecture" ./src/api
```

## Examples

### Complete Workflow Example

```bash
# 1. Start a new feature
tenets session create oauth-integration

# 2. Get initial context
tenets distill "implement OAuth2 with Google and GitHub" \
  --session oauth-integration \
  --include "*.py,*.yaml" \
  --exclude "test_*" \
  --model gpt-4o \
  --estimate-cost > oauth_context.md

# 3. Paste into ChatGPT, start coding...

# 4. AI needs more specific context
# (Show session details)
tenets session show oauth-integration

# 5. Check your progress
tenets chronicle --since "today"

# 6. Visualize what you built
tenets viz deps src/auth --format ascii

# 7. Check complexity
tenets examine src/auth --complexity

# 8. Prepare for review
tenets distill "OAuth implementation ready for review" \
  --session oauth-integration
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
4. **Combine commands** - examine first, then distill with insights
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
