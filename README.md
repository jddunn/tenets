# **tenets**

<a href="https://tenets.dev"><img src="./docs/logos/tenets_dark_icon_transparent.png" alt="tenets logo" width="140" /></a>

**context that feeds your prompts.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/tenets.svg)](https://pypi.org/project/tenets/)
[![CI](https://github.com/jddunn/tenets/actions/workflows/ci.yml/badge.svg)](https://github.com/jddunn/tenets/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jddunn/tenets/branch/main/graph/badge.svg)](https://codecov.io/gh/jddunn/tenets)
[![Coverage Status](https://coveralls.io/repos/github/jddunn/tenets/badge.svg?branch=main)](https://coveralls.io/github/jddunn/tenets?branch=main)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://docs.tenets.dev)

**tenets** automatically finds and builds the most relevant context from your codebase. Instead of manually copying files or searching for documentation, tenets intelligently aggregates exactly what you need - whether you're debugging, building features, or chatting with an AI assistant.

## What is tenets?

Think of it as intelligent context aggregation. You give it a prompt or query, and it:

- 🔍 **Finds** all relevant files automatically 
- 🎯 **Ranks** them by importance using multiple factors
- 📦 **Aggregates** them within your token budget
- 📋 **Formats** perfectly for any use case

Plus powerful development intelligence:

- 📊 **Visualize** dependencies and architecture
- 📈 **Track** velocity and code evolution  
- 🔥 **Identify** hotspots and technical debt
- 👥 **Understand** team patterns and expertise

```bash
# Instead of manually finding and copying files...
tenets distill "implement OAuth2" ./src

# It automatically finds auth.py, user.py, config.yaml,
# related tests, dependency files — everything you actually need.
# (Git activity is used for relevance only; not shown in the output.)
```

## Why tenets?

### The Problem

When working with large codebases, you're constantly:

- Manually searching for relevant files
- Copy-pasting code into ChatGPT/Claude
- Missing important context and dependencies
- Repeating the same searches over and over
- Paying for AI to process irrelevant code

### The Solution

tenets uses intelligent algorithms to solve this:

```bash
# Old way: Manual search and copy
$ grep -r "payment" . 
$ cat payment.py api.py models.py  # Did I miss anything?
# Copy... paste... hope for the best

# With tenets: Automatic context building
$ tenets distill "fix payment processing bug"
# Finds: payment.py, API endpoints, models, config files, tests, error handlers
# (recent changes inform ranking) — all ranked by relevance
```

## Key Features

### 🎯 Intelligent Context Distillation

Like repomix on steroids - smart filters, automatic relevance ranking, and configurable aggregation:

```bash
# Distill by file types (include only Python & JS; exclude tests)
tenets distill "review API" --include "*.py,*.js" --exclude "test_*" --stats

# Smart aggregation with larger budget
tenets distill "understand auth flow" --max-tokens 50000 --mode balanced --stats

# Session-based for iterative work
# 1) Create a named session
tenets session create "new-feature"
# 2) Build broad context
tenets distill "design database schema" --session new-feature --stats
# 3) Narrow follow-up
tenets distill "add user model" --session new-feature --stats
```

### 🧭 Guiding Principles (Tenets)

Add persistent instructions that guide AI interactions:

```bash
# Add guiding principles (tenets)
tenets tenet add "Always use type hints in Python"
tenets tenet add "Follow RESTful conventions"
tenets tenet add "Include error handling"
```

### 📊 Code Intelligence & Visualization

Understand your codebase at a glance:

```bash
# Dependency graphs
tenets viz deps . --output architecture.svg

# Complexity analysis
tenets viz complexity . --hotspots

# Git integration - automatic, no setup (used for relevance, not shown in output)
tenets chronicle --since "last week"
tenets viz contributors --active
```

### 🚀 Developer Productivity

Track velocity, identify bottlenecks, measure progress:

```bash
# Sprint velocity
tenets momentum --team --since "sprint-start"

# Code ownership
tenets examine . --ownership

# Technical debt trends
tenets examine . --complexity-trend
```

### 🔧 Flexible Configuration

Works instantly with smart defaults, fully configurable when needed:

```yaml
# .tenets.yml (optional)
context:
  ranking: balanced  # fast, balanced, thorough
  include_git: true  # Use git signals for relevance (not shown in output)
  max_tokens: 100000

ignore:
  - vendor/
  - "*.generated.*"
  
output:
  format: markdown  # markdown, json, xml
```

## Installation

### Quick Install (pip)

```bash
# Core features only - lightweight, no ML dependencies
pip install tenets

# Add specific features
pip install tenets[light]  # Adds numpy, scikit-learn for TF-IDF ranking
pip install tenets[viz]    # Adds visualization capabilities
pip install tenets[ml]     # Adds deep learning models (large dependencies)

# Everything
pip install tenets[all]
```

### Install with Poetry (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/jddunn/tenets.git
cd tenets

# Install with poetry
poetry install           # Core only
poetry install -E light  # With TF-IDF ranking
poetry install -E viz    # With visualization
poetry install -E ml     # With deep learning
poetry install -E all    # Everything

# Activate shell
poetry shell
```

### Feature Sets Explained

| Feature Set | Includes | Use When |
|-------------|----------|----------|
| **core** (default) | Basic file scanning, keyword matching, git integration | You want fast, lightweight context building |
| **light** | + numpy, scikit-learn, YAKE keyword extraction, TF-IDF ranking | You want better ranking without heavy ML dependencies |
| **viz** | + matplotlib, networkx, dependency graphs, complexity charts | You want to visualize your codebase |
| **ml** | + PyTorch, transformers, semantic search, embeddings | You want state-of-the-art ranking (slower, 2GB+ dependencies) |
| **web** | + FastAPI, web UI (coming soon) | You want to run tenets as a service |
| **all** | Everything above | You want all features |

### Troubleshooting Installation

### Local Development / Editable Install

If you want to hack on Tenets locally and immediately use the `tenets` CLI with your changes, install it in editable mode.

```bash
# (Windows) create & activate venv
python -m venv .venv
.venv\\Scripts\\activate

# (macOS/Linux)
python -m venv .venv
source .venv/bin/activate

# Upgrade tooling
pip install --upgrade pip

# Editable install (core + dev tooling)
pip install -e ".[dev]"

# Or everything (all optional feature groups + dev)
pip install -e ".[all,dev]"

# Verify CLI
tenets --help
```

### Building Distribution Artifacts (sdist / wheel)

You usually only need this when testing packaging or publishing. The project uses PEP 517/518 with `hatchling`.

```bash
# Ensure build deps installed (included in dev extras)
pip install build

# Build (creates dist/*.whl and dist/*.tar.gz)
python -m build

# (Optional) install the freshly built wheel
pip install --force-reinstall dist/tenets-*.whl
```

After an editable install you don't need to rebuild for Python code changes— they are picked up immediately. Rebuild only when validating the packaging metadata or non-Python asset inclusion.

#### Using Poetry (Alternative)
If you prefer Poetry instead of pip + venv:
```bash
poetry install -E all -E dev   # or omit extras for core only
poetry run tenets --help
```
To add a new extra later:
```bash
poetry add --optional SOME_PKG
```

#### Makefile Shortcuts
Common tasks are wrapped in the Makefile:
```bash
make dev      # editable install with all + dev extras
make install  # core editable install
make test     # run full test suite with coverage
make build    # build sdist + wheel
```


**numpy/scipy installation fails on Python 3.9?**

```bash
# Use compatible versions
pip install "numpy<2.0" "scikit-learn<1.4"
# Or upgrade to Python 3.10+
```

**Import errors after installation?**

```bash
# Ensure you have the right extras
pip install tenets[light]  # For NLP features
pip install tenets[ml]     # For deep learning
```

**Poetry dependency conflicts?**

```bash
# Clear cache and reinstall
rm poetry.lock
poetry cache clear pypi --all
poetry install -E light
```

## Quick Start

### Basic Context Building

```bash
# For debugging - finds error handling, logs, and recent changes (used for ranking)
tenets distill "users getting 401 errors" --stats

# For new features - finds related code, patterns, examples
tenets distill "add caching layer" --stats

# For code review - includes recent changes in ranking, tests, dependencies
tenets distill "review payment refactor" --since yesterday --stats
```

### Working with AI Assistants

```bash
# Generate context for ChatGPT/Claude
tenets distill "explain authentication flow" > context.md
# Paste context.md into your AI chat

# Add guiding principles
tenets tenet add "Always validate user input"
tenets tenet add "Use async/await for I/O operations"
# Apply to future contexts
tenets instill

# Interactive session for back-and-forth
tenets session create "debug-memory-leak"
# AI: "Show me the memory allocation code"
tenets distill "malloc|alloc" --include "*.c" --session debug-memory-leak
# AI: "I need to see the cleanup functions"
tenets distill "free|cleanup" --include "*.c" --session debug-memory-leak
```

### Exploration & Analysis

```bash
# Instant project overview
tenets examine .

# What's been happening?
tenets chronicle --since "1 month" --summary

# Visualize your architecture
tenets viz deps . --cluster-by directory

# Find complex areas
tenets examine . --complexity --threshold 10
```

## How It Works

1. **Scans** your codebase respecting .gitignore
2. **Analyzes** code structure, imports, and dependencies
3. **Ranks** files using multi-factor scoring:
   - Keyword matching (TF-IDF with `light` extra)
   - Import relationships
   - Git activity and recency (used for relevance; not shown in output)
   - Code complexity
   - Path relevance
   - Semantic similarity (with `ml` extra)
4. **Aggregates** intelligently within token limits
5. **Formats** for optimal consumption

*All processing is local. Your code never leaves your machine.*

## Storage and Cache

- Main SQLite database lives at `${CACHE_DIR}/tenets.db` where `CACHE_DIR` is `config.cache.directory`.
  - Defaults: `~/.tenets/cache` (Linux/macOS), `%USERPROFILE%\.tenets\cache` (Windows)
- Sub-directories for caches (future/optional), e.g. `analysis/`, `general/` under the same cache root
- SQLite PRAGMAs applied by default for safety and performance: `journal_mode=WAL`, `synchronous=NORMAL`, in-memory temp store, and larger page cache. This keeps writes safe and works well even when tenets is installed via pip because the DB resides in a writable user cache directory.
- Override location via either environment or config file:
  - Env: `TENETS_CACHE_DIRECTORY=/path/to/cache`
  - `.tenets.yml`:
    ```yaml
    cache:
      directory: /path/to/cache
    ```
- Sessions: when you pass `--session <name>` and a configuration is loaded, session state is persisted to the SQLite database so your context history can survive CLI runs.

## Advanced Features

### Multi-Factor Ranking

Choose the algorithm that fits your needs:

- **fast** - Quick keyword matching (always available)
- **balanced** - Keywords + structure + git (requires `light` extra)
- **thorough** - Deep analysis with AST + ML (requires `ml` extra)

### Smart Summarization

When files exceed token budgets, tenets intelligently preserves:

- Function/class signatures
- Import statements
- Complex logic blocks
- Recent changes
- Documentation

### Session Management

Maintain context across multiple interactions:

```python
session = tenets.create_session("feature-x")
session.distill("design API")         # Full context
session.distill("implement auth")     # Builds on previous
session.show_files(["api/auth.py"])   # Specific files
session.ignore_files(["old/*.py"])    # Refine relevance
```

### Guiding Principles (Tenets)

Maintain consistent coding principles across AI interactions:

```bash
# Add principles that should guide the AI
tenets tenet add "Use dependency injection"
tenets tenet add "Write tests for all new functions"
tenets tenet add "Follow PEP 8 strictly"

# List current tenets
tenets tenet list

# Apply them to your context
tenets instill

# Use them in a distillation
tenets distill "implement OAuth2 with Google" --stats
```

## What Makes tenets Different

| Feature | Other Tools | tenets |
|---------|-------------|---------|
| **File Selection** | Manual or basic search | Automatic multi-factor ranking |
| **Context Building** | Simple concatenation | Intelligent aggregation |
| **Token Management** | Hit limits or waste tokens | Smart budgeting & summaries |
| **Git Integration** | Afterthought or none | First-class, automatic |
| **Visualization** | Separate tools | Built-in graphs & analysis |
| **Setup Required** | Config files everywhere | Zero config, just works |
| **Persistent Instructions** | None | Tenets system for consistency |

## Real-World Use Cases

### 🐛 Debugging Production Issues

```bash
tenets distill "users can't login after deploy" --since "last-deploy"
# Automatically includes: auth code, config changes, deployment files,
# related error handlers (recent changes inform ranking)
```

### 🏗️ Building New Features

```bash
tenets distill "add PDF export" --examples
# Finds: existing export code, similar features, file I/O patterns,
# library usage examples, relevant tests
```

### 📚 Code Understanding

```bash
tenets distill "how does the payment system work?" --visualize
# Includes: payment files, dependency graph, database models,
# API endpoints, configuration, with visual architecture diagram
```

### 🤖 AI Pair Programming

```bash
# Create a session
tenets session create "implement-oauth"

# Add coding principles
tenets tenet add "Use existing auth patterns"
tenets tenet add "Include comprehensive error handling"
# Apply them
tenets instill

# Initial context
tenets distill "implement OAuth2 with Google" > context.md

# As you work, update context
# (Optionally attach artifacts to the session)
# tenets session add implement-oauth context_result context.md

tenets distill "add refresh token handling" --session implement-oauth > context_update.md
```

## Command Reference

### Primary Commands

```bash
# Distill context from codebase
tenets distill <prompt> [path] [options]

# Examine codebase structure
tenets examine [path] [options]

# Chronicle git history
tenets chronicle [options]

# Track development momentum
tenets momentum [options]

# Instill guiding principles
tenets instill [options]
```

### Tenet Management

```bash
# Add a guiding principle
tenets tenet add <principle> [--priority high]

# List all tenets
tenets tenet list [--pending | --instilled]

# Remove a tenet
tenets tenet remove <id>

# Export/import tenets
tenets tenet export > my-tenets.yml
tenets tenet import my-tenets.yml
```

### Session Commands

```bash
# Create a new session
tenets session create <name>

# Start a session (alias of create)
tenets session start <name>

# Resume the current active session (or specify a name)
tenets session resume [<name>]

# Exit the current active session (or specify a name)
tenets session exit [<name>]

# Show session details
tenets session show <name>

# List sessions (shows an Active column)
tenets session list

# Attach an artifact to a session (stored as text)
# kind examples: note, context_result, summary
tenets session add <name> <kind> <file>

# Reset (delete & recreate) a session and purge its context
tenets session reset <name>

# Delete a session (optionally keep context)
# Add --keep-context to retain stored artifacts
tenets session delete <name> [--keep-context]

# Clear ALL sessions (optionally keep context)
tenets session clear [--keep-context]
```

### Visualization

```bash
# Visualize dependencies
tenets viz deps [path] [options]

# Show complexity heatmap
tenets viz complexity [path] [options]

# Contributor activity
tenets viz contributors [path] [options]
```

## 📚 Documentation

Core docs in `docs/`:

- [CLI Reference](docs/CLI.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Deep Dive](docs/DEVELOPMENT.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Testing Guide](docs/TESTING.md)

See `docs/CONFIG.md` for configuration details.

Full documentation is available at https://docs.tenets.dev.

### Building Documentation Locally

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material pymdown-extensions

# Serve locally with live reload
make docs
# Or directly:
mkdocs serve

# Build static site
make docs-build
# Or directly:
mkdocs build

# Deploy to GitHub Pages (requires permissions)
make docs-deploy
# Or directly:
mkdocs gh-deploy
```

## Configuration (quick start)

- Create a starter file: `tenets config init` (writes `.tenets.yml` at project root)
- Lower threshold to include more files:

```yaml
ranking:
  algorithm: fast
  threshold: 0.05
```

- Or set environment variables for one run:

```bash
TENETS_RANKING_THRESHOLD=0.05 tenets distill "implement OAuth2" .
```

See the full guide: `docs/CONFIG.md`.

## Contributing

We love contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Development sponsored by [manic.agency](https://manic.agency)*

## Storage & Persistence

Tenets stores all writable state in a user/project cache directory. This avoids writing into the installed package location (which may be read‑only under pip installs).

- Default cache directory: `${HOME}/.tenets/cache` (Windows: `%USERPROFILE%\.tenets\cache`)
- Main SQLite database: `${CACHE_DIR}/tenets.db`
- Subdirectories:
  - `${CACHE_DIR}/analysis/` – analysis cache (SQLite)
  - `${CACHE_DIR}/general/` – general cache (SQLite)

You can change the cache location via config or environment variables:

- Config file `.tenets.yml`:
  ```yaml
  cache:
    directory: /path/to/custom/cache
  ```
- Environment variable: `TENETS_CACHE_DIRECTORY=/path/to/custom/cache`

Sessions are persisted to the main SQLite database by default when a `TenetsConfig` is provided. The `SessionManager` uses an in‑memory mirror for speed and will write session metadata and context snapshots to `${CACHE_DIR}/tenets.db`.

No project code leaves your machine; all processing and storage are local.

## Supported languages and formats

The analyzer includes specialized parsers for many languages and formats. Files are routed based on extensions; HTML analyzer covers HTML and Vue SFCs. JSX/TSX are handled by the JavaScript analyzer.

### Languages with dedicated analyzers

| Language | Analyzer | Extensions |
|---|---|---|
| Python | PythonAnalyzer | .py, .pyw, .pyi |
| JavaScript/TypeScript | JavaScriptAnalyzer | .js, .jsx, .ts, .tsx, .mjs, .cjs |
| HTML + Vue SFC | HTMLAnalyzer | .html, .htm, .xhtml, .vue |
| CSS/SCSS/Sass/Less | CSSAnalyzer | .css, .scss, .sass, .less, .styl, .stylus, .pcss, .postcss |
| Go | GoAnalyzer | .go |
| Java | JavaAnalyzer | .java |
| C/C++ | CppAnalyzer | .c, .cc, .cpp, .cxx, .c++, .h, .hh, .hpp, .hxx, .h++ |
| Ruby | RubyAnalyzer | .rb, .rake, .gemspec, .ru |
| PHP | PhpAnalyzer | .php, .phtml, .inc, .php3, .php4, .php5, .phps |
| Rust | RustAnalyzer | .rs |
| Dart (Flutter aware) | DartAnalyzer | .dart |
| Kotlin | KotlinAnalyzer | .kt, .kts |
| Scala | ScalaAnalyzer | .scala, .sc |
| Swift | SwiftAnalyzer | .swift |
| C# | CSharpAnalyzer | .cs, .csx |
| GDScript (Godot) | GDScriptAnalyzer | .gd, .tres, .tscn |

### Configuration, docs, and structured text (GenericAnalyzer)

These formats are analyzed with GenericAnalyzer. YAML files include heuristics for common ecosystems.

| Category | Examples | Extensions / Filenames | Notes |
|---|---|---|---|
| YAML (Compose/K8s/etc.) | docker-compose.yml, deployment.yaml, chart.yaml, kustomization.yaml, .github/workflows/*.yml | .yaml, .yml | Detects Docker Compose (services, images), Kubernetes (apiVersion/kind, images, refs), hints Helm, Kustomize, GitHub Actions |
| TOML | pyproject.toml, Cargo.toml | .toml | Extracts top-level keys |
| INI/CFG/CONF | app.ini, settings.cfg, nginx.conf, my.cnf | .ini, .cfg, .conf, .cnf | Sections and keys parsed |
| Properties | application.properties | .properties, .props | Key/value parsing |
| ENV files | .env, .env.local, .env.production | .env, .env.* | Routed to Generic by filename |
| JSON/XML | package.json, config.json, pom.xml | .json, .xml | JSON deps/keys detected |
| Markdown | README.md, docs/*.mdx | .md, .markdown, .mdx, .mdown, .mkd, .mkdn, .mdwn | Sections and headings extracted |
| SQL | schema.sql, queries.sql | .sql | Basic metrics only |
| Lock/HashiCorp | yarn.lock, *.tf, *.tfvars, *.hcl | .lock, .tf, .tfvars, .hcl | Routed to Generic |
| Shell scripts | build.sh, hooks/*.bash | .sh, .bash, .zsh, .fish | Routed to Generic |
| Special files | Dockerfile, Makefile, CMakeLists.txt, .gitignore, .editorconfig, .npmrc, .yarnrc, .nvmrc | by name | Routed to Generic by special-name handling |

Notes
- JSX/TSX are owned by JavaScriptAnalyzer; HTMLAnalyzer focuses on HTML and Vue SFCs.
- YAML heuristics set structure.framework (e.g., docker-compose, kubernetes) and populate modules with services/resources.
- Generic analyzer extracts imports/references from config where possible (images, depends_on, ConfigMaps/Secrets, etc.).