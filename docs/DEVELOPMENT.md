# Development Guide

This guide covers everything you need to contribute to **tenets**.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Documentation](#documentation)
- [Debugging](#debugging)
- [Performance Profiling](#performance-profiling)
- [Contributing Guidelines](#contributing-guidelines)

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Make (optional but recommended)
- Docker (optional, for container testing)

### Initial Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tenets.git
   cd tenets
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   make dev  # Or: pip install -e ".[all,dev,test,docs]"
   ```

4. **Set up pre-commit hooks**:
   ```bash
   pre-commit install --install-hooks
   pre-commit install --hook-type commit-msg
   pre-commit install --hook-type push
   ```

5. **Verify installation**:
   ```bash
   tenets --version
   make test-fast  # Run quick tests
   ```

## Project Structure

```
tenets/
├── .github/workflows/    # CI/CD pipelines
│   ├── ci.yml           # Continuous integration
│   └── release.yml      # Release automation
├── assets/logos/        # Brand assets
├── docs/                # Documentation source
│   ├── api.md          # API reference
│   ├── architecture.md # System design
│   ├── cli.md          # CLI documentation
│   └── deep-dive.md    # Advanced topics
├── tenets/             # Main package
│   ├── __init__.py     # Package initialization
│   ├── cli/            # CLI implementation
│   │   └── main.py     # CLI entry point
│   ├── core/           # Core functionality
│   │   ├── analyzer.py # Code analysis
│   │   ├── nlp.py      # NLP processing
│   │   └── prompt_parser.py
│   ├── models/         # Data models
│   │   ├── analysis.py
│   │   ├── context.py
│   │   └── llm.py
│   └── utils/          # Utilities
│       └── file_scanner.py
├── tests/              # Test suite
├── pyproject.toml      # Project configuration
├── Makefile           # Development tasks
└── README.md          # Main documentation
```

## Development Workflow

### 1. Creating a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### 2. Making Changes

Follow the coding standards:
- Write clean, readable code
- Add comprehensive docstrings (Google style)
- Include type hints for all functions
- Write tests for new functionality

### 3. Committing Changes

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Interactive commit
make commit  # or: cz commit

# Manual commit (must follow format)
git commit -m "feat(analyzer): add support for Rust AST parsing"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `chore`: Maintenance tasks

### 4. Running Tests

```bash
# Run all tests
make test

# Run fast tests only
make test-fast

# Run specific test file
pytest tests/test_analyzer.py

# Run with coverage
pytest --cov=tenets --cov-report=html
```

### 5. Code Quality Checks

```bash
# Run all checks
make lint

# Auto-format code
make format

# Individual tools
black .
isort .
ruff check .
mypy tenets --strict
bandit -r tenets
```

### 6. Pushing Changes

```bash
# Pre-commit hooks will run automatically
git push origin feature/your-feature-name
```

### 7. Creating a Pull Request

1. Go to GitHub and create a PR
2. Fill out the PR template
3. Ensure all CI checks pass
4. Request review from maintainers

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_analyzer.py
│   ├── test_nlp.py
│   └── test_scanner.py
├── integration/       # Integration tests
│   ├── test_cli.py
│   └── test_workflow.py
├── fixtures/         # Test data
│   └── sample_repo/
└── conftest.py      # Pytest configuration
```

### Writing Tests

```python
"""Test module for analyzer functionality."""

import pytest
from tenets.core.analysis import CodeAnalyzer


class TestCodeAnalyzer:
    """Test suite for CodeAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CodeAnalyzer()
    
    def test_analyze_python_file(self, analyzer, tmp_path):
        """Test Python file analysis."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'")
        
        # Analyze
        result = analyzer.analyze_file(test_file)
        
        # Assertions
        assert result.language == "python"
        assert len(result.functions) == 1
        assert result.functions[0]["name"] == "hello"
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run tests requiring git
pytest -m requires_git
```

## Code Quality

### Style Guide

We follow PEP 8 with these modifications:
- Line length: 100 characters
- Use Black for formatting
- Use Google-style docstrings

### Type Hints

All functions must have type hints:

```python
from typing import List, Optional, Dict, Any


def analyze_files(
    paths: List[Path],
    deep: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze multiple files in parallel.
    
    Args:
        paths: List of file paths to analyze
        deep: Whether to perform deep analysis
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary containing analysis results
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_relevance(
    file: FileAnalysis,
    prompt: PromptContext,
    algorithm: str = "balanced"
) -> float:
    """
    Calculate relevance score for a file.
    
    Uses multi-factor scoring to determine how relevant a file is
    to the given prompt context.
    
    Args:
        file: Analyzed file data
        prompt: Parsed prompt context
        algorithm: Ranking algorithm to use
        
    Returns:
        Relevance score between 0.0 and 1.0
        
    Raises:
        ValueError: If algorithm is not recognized
        
    Example:
        >>> relevance = calculate_relevance(file, prompt, "thorough")
        >>> print(f"Relevance: {relevance:.2f}")
        0.85
    """
    ...
```

## Documentation

### Building Documentation

```bash
# Build docs
make docs

# Serve locally
make serve-docs
# Visit http://localhost:8000
```

### Writing Documentation

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Written in Markdown in `docs/`
3. **Examples**: Include code examples in docstrings

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep it up-to-date with code changes

## Debugging

### Debug Mode

```bash
# Enable debug logging
export TENETS_DEBUG=true
tenets make-context "test" . --verbose

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Using VS Code

`.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug tenets CLI",
            "type": "python",
            "request": "launch",
            "module": "tenets.cli.main",
            "args": ["make-context", "test query", "."],
            "console": "integratedTerminal"
        }
    ]
}
```

### Common Issues

1. **Import errors**: Ensure you've installed in development mode (`pip install -e .`)
2. **Type errors**: Run `mypy` to catch type issues
3. **Test failures**: Check if you need to install optional dependencies

## Performance Profiling

### CPU Profiling

```bash
# Profile a command
python -m cProfile -o profile.stats tenets analyze .

# View results
python -m pstats profile.stats
> sort cumtime
> stats 20
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    pass
```

### Benchmarking

```python
import pytest

@pytest.mark.benchmark
def test_performance(benchmark):
    result = benchmark(function_to_test, arg1, arg2)
    assert result == expected
```

## Contributing Guidelines

### Before You Start

1. Check existing issues and PRs
2. Open an issue to discuss large changes
3. Read the architecture documentation

### Code Review Checklist

- [ ] Tests pass locally
- [ ] Code is formatted (black, isort)
- [ ] Type hints are present
- [ ] Docstrings are complete
- [ ] Tests cover new functionality
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No security issues (bandit)

### Getting Help

- Open an issue for bugs
- Start a discussion for features
- Join our Discord (coming soon)
- Email: team@tenets.dev

## Release Process

Releases are automated but here's the manual process:

1. **Ensure main is stable**:
   ```bash
   git checkout main
   git pull origin main
   make test
   ```

2. **Create release**:
   ```bash
   make release  # Interactive
   # Or manually:
   cz bump
   git push && git push --tags
   ```

3. **Monitor CI/CD**:
   - Check GitHub Actions
   - Verify PyPI deployment
   - Check documentation update

## Advanced Topics

### Adding a New Language Analyzer

1. Create analyzer in `tenets/core/analysis/`:
   ```python
   class RustAnalyzer(LanguageAnalyzer):
       language_name = "rust"
       
       def extract_imports(self, content: str) -> List[Import]:
           # Implementation
           ...
   ```

2. Register in `analysis/analyzer.py`:
   ```python
   analyzers['.rs'] = RustAnalyzer()
   ```

3. Add tests in `tests/unit/test_rust_analyzer.py`

### Creating Custom Ranking Algorithms

1. Implement algorithm:
   ```python
   class SecurityRanking:
       def score_file(self, file, prompt):
           # Custom scoring logic
           ...
   ```

2. Register algorithm:
   ```python
   @register_algorithm("security")
   class SecurityRanking:
       ...
   ```

3. Document usage in `docs/api.md`

---

Happy coding! 🚀 Remember: context is everything.