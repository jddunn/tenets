# API Reference

Welcome to the Tenets API Reference. This documentation is automatically generated from the source code docstrings.

## Core Modules

### Main Interface

#### [Tenets](tenets/)
Main API interface for all Tenets functionality
```python
from tenets import Tenets
```

#### [Config](config/)
Configuration management and settings
```python
from tenets.config import TenetsConfig
```

### Core Components

#### [Distiller](distiller/)
Context extraction and distillation
```python
from tenets.core.distiller import Distiller
```

#### [Analyzer](analyzer/)
Code analysis and metrics
```python
from tenets.core.analysis import Analyzer
```

#### [Ranker](ranker/)
Relevance ranking algorithms
```python
from tenets.core.ranking import Ranker
```

#### [Session](session/)
Stateful session management
```python
from tenets.core.session import Session
```

## Quick Start Guide

### Installation

```bash
pip install tenets
```

### Basic Usage

Initialize Tenets with default configuration and generate context:

```python
from tenets import Tenets

# Initialize with defaults
t = Tenets()

# Generate context for your prompt
result = t.make_context("implement caching layer")
print(result.content)
```

### Advanced Configuration

Customize Tenets behavior with detailed configuration options:

```python
from tenets import Tenets, TenetsConfig

# Create custom configuration
config = TenetsConfig(
    max_tokens=150_000,
    ranking_algorithm="thorough",
    output_format="markdown",
    include_tests=True,
    include_docs=True
)

# Initialize with custom config
t = Tenets(config=config)

# Generate context with custom settings
result = t.make_context(
    prompt="refactor authentication module",
    max_files=50
)
```

### Session Management

Use sessions for iterative development and context building:

```python
from tenets import Tenets

t = Tenets()

# Create a named session for feature development
session = t.create_session("feature-cache-implementation")

# Build context iteratively
initial_context = session.make_context("setup caching infrastructure")
print(f"Initial files: {len(initial_context.files)}")

# Add more context based on previous results
refined_context = session.make_context("add Redis support")
print(f"Refined files: {len(refined_context.files)}")

# Session maintains state between calls
final_context = session.make_context("implement cache invalidation")
```

### Working with Tenets

Manage project-specific principles and constraints:

```python
from tenets import Tenets

t = Tenets()

# Add project tenets
t.add_tenet("Use TypeScript strict mode")
t.add_tenet("Follow REST API conventions")
t.add_tenet("Maintain 90% test coverage")

# Tenets are included in context generation
result = t.make_context("create new API endpoint")
# Result will consider your project tenets
```

### Git Integration

Leverage Git history for better context:

```python
from tenets import Tenets, TenetsConfig

config = TenetsConfig(
    use_git_history=True,
    git_history_depth=100,
    include_commit_messages=True
)

t = Tenets(config=config)

# Context will include relevant Git history
result = t.make_context("fix recent regression in user service")
```

## Module Organization

```
tenets/
├── __init__.py              # Main Tenets class and exports
├── config.py                # Configuration management
├── core/
│   ├── distiller/           # Context distillation engine
│   │   ├── __init__.py
│   │   ├── extractor.py    # Content extraction
│   │   └── processor.py    # Processing pipeline
│   ├── analysis/            # Code analysis tools
│   │   ├── __init__.py
│   │   ├── analyzer.py     # Main analyzer
│   │   ├── metrics.py      # Code metrics
│   │   └── patterns.py     # Pattern detection
│   ├── ranking/             # Relevance ranking system
│   │   ├── __init__.py
│   │   ├── ranker.py       # Ranking algorithms
│   │   └── scorers.py      # Scoring functions
│   ├── session/             # Session management
│   │   ├── __init__.py
│   │   ├── manager.py      # Session lifecycle
│   │   └── state.py        # State persistence
│   ├── instiller/           # Tenet management
│   │   ├── __init__.py
│   │   └── tenets.py       # Tenet operations
│   └── git/                 # Git integration
│       ├── __init__.py
│       ├── history.py       # History analysis
│       └── diff.py          # Diff processing
├── models/                  # Data models
│   ├── __init__.py
│   ├── context.py           # Context models
│   ├── file.py              # File representations
│   └── result.py            # Result structures
├── storage/                 # Caching and persistence
│   ├── __init__.py
│   ├── cache.py             # Caching layer
│   └── persist.py           # Persistence layer
└── utils/                   # Utility functions
    ├── __init__.py
    ├── validators.py        # Input validation
    ├── formatters.py        # Output formatting
    └── helpers.py           # Helper functions
```

## Configuration Options

### TenetsConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | int | 100,000 | Maximum tokens in generated context |
| `ranking_algorithm` | str | "balanced" | Algorithm for ranking relevance ("fast", "balanced", "thorough") |
| `output_format` | str | "markdown" | Output format ("markdown", "json", "xml") |
| `include_tests` | bool | False | Include test files in context |
| `include_docs` | bool | True | Include documentation files |
| `use_git_history` | bool | False | Analyze Git history for context |
| `git_history_depth` | int | 50 | Number of commits to analyze |
| `cache_enabled` | bool | True | Enable context caching |
| `cache_ttl` | int | 3600 | Cache time-to-live in seconds |

## API Methods

### Tenets Class

#### `__init__(config: Optional[TenetsConfig] = None)`
Initialize a new Tenets instance.

#### `make_context(prompt: str, **kwargs) -> ContextResult`
Generate context based on the provided prompt.

#### `create_session(name: str) -> Session`
Create a new named session for iterative context building.

#### `add_tenet(tenet: str) -> None`
Add a project-specific tenet or principle.

#### `remove_tenet(tenet: str) -> None`
Remove a previously added tenet.

#### `list_tenets() -> List[str]`
Get all current project tenets.

#### `clear_cache() -> None`
Clear the context cache.

### Session Class

#### `make_context(prompt: str, **kwargs) -> ContextResult`
Generate context within the session scope.

#### `get_history() -> List[ContextResult]`
Retrieve session history.

#### `clear() -> None`
Clear session state.

#### `save(path: str) -> None`
Save session to disk.

#### `load(path: str) -> Session`
Load session from disk.

## Error Handling

```python
from tenets import Tenets, TenetsError

try:
    t = Tenets()
    result = t.make_context("implement feature")
except TenetsError as e:
    print(f"Error generating context: {e}")
    # Handle error appropriately
```

## Best Practices

1. **Use Sessions for Complex Features**: When working on multi-step features, use sessions to maintain context between iterations.

2. **Configure Token Limits**: Adjust `max_tokens` based on your LLM's context window to avoid truncation.

3. **Leverage Git History**: Enable Git integration for bug fixes and refactoring tasks to include relevant historical context.

4. **Cache for Performance**: Keep caching enabled for faster subsequent context generation.

5. **Add Project Tenets**: Define project-specific principles to ensure generated context aligns with your team's standards.

## Examples

### Example 1: Bug Fix Context

```python
from tenets import Tenets, TenetsConfig

# Configure for bug fixing
config = TenetsConfig(
    use_git_history=True,
    git_history_depth=200,
    ranking_algorithm="thorough"
)

t = Tenets(config=config)
result = t.make_context("fix null pointer exception in UserService.authenticate")
```

### Example 2: Feature Development

```python
from tenets import Tenets

t = Tenets()
session = t.create_session("payment-integration")

# Add relevant tenets
t.add_tenet("Follow PCI compliance standards")
t.add_tenet("Use Stripe API v3")

# Generate comprehensive context
context = session.make_context(
    "implement payment processing with Stripe",
    include_tests=True,
    max_files=100
)
```

### Example 3: Refactoring

```python
from tenets import Tenets, TenetsConfig

config = TenetsConfig(
    include_tests=True,
    include_docs=True,
    ranking_algorithm="balanced"
)

t = Tenets(config=config)
result = t.make_context("refactor database layer to use repository pattern")
```

## Support

For issues, feature requests, or questions:
- GitHub Issues: [github.com/yourusername/tenets/issues](https://github.com/yourusername/tenets/issues)
- Documentation: [tenets.readthedocs.io](https://tenets.readthedocs.io)
- Email: support@tenets.dev

## License

MIT License - see LICENSE file for details.