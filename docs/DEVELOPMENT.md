# Development Guide

This guide provides instructions for setting up your development environment, running tests, and contributing to the Tenets project.

## 1. Initial Setup

### Prerequisites
- Python 3.9+
- Git
- An activated Python virtual environment (e.g., `venv`, `conda`).

### Fork and Clone
1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/tenets.git
   cd tenets
   ```

### Install Dependencies
Install the project in "editable" mode along with all development dependencies. This allows you to modify the source code and have the changes immediately reflected.

```bash
pip install -e ".[all,dev]"
```
This command installs everything needed for development, including core dependencies, optional features (`all`), and development tools (`dev`).

### Set up Pre-Commit Hooks
This project uses `pre-commit` to automatically run linters and formatters before each commit.

```bash
pre-commit install

### Alternative Installs

If you only need core + dev tooling (faster):
```bash
pip install -e ".[dev]"
```
If you need a minimal footprint for quick iteration (no optional extras):
```bash
pip install -e .
```

### Verifying the CLI
```bash
tenets --version
tenets --help | head
```

If the command is not found, ensure your virtualenv is activated and that the `scripts` (Windows) or `bin` (Unix) directory is on PATH.

## 1.1 Building Distribution Artifacts (Optional)

You typically do NOT need to build wheels / sdists for day‑to‑day development; the editable install auto-reflects code edits. Build only when testing packaging or release steps.

```bash
python -m build               # creates dist/*.whl and dist/*.tar.gz
pip install --force-reinstall dist/tenets-*.whl  # sanity check install
```

To inspect what went into the wheel:
```bash
unzip -l dist/tenets-*.whl | grep analysis/implementations | head
```

## 1.2 Clean Environment Tasks

```bash
pip cache purge        # optional: clear wheel cache
find . -name "__pycache__" -exec rm -rf {} +
rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
```

## 1.3 Using Poetry Instead of pip (Optional)

Poetry can manage the virtual environment and extras if you prefer:
```bash
poetry install -E all -E dev   # full feature + dev toolchain
poetry run pytest              # run tests
poetry run tenets --help       # invoke CLI
```
Update dependencies:
```bash
poetry update
```
Add a new optional dependency (example):
```bash
poetry add --optional rich
```
```

## 2. Running Tests

The test suite uses `pytest`. We have a comprehensive configuration in `pytest.ini` that handles most settings automatically.

### Running All Tests
To run the entire test suite:
```bash
pytest
```

### Running Tests with Coverage
To generate a test coverage report:
```bash
pytest --cov
```
This command is configured in `pytest.ini` to:
- Measure coverage for the `tenets` package.
- Generate reports in the terminal, as XML (`coverage.xml`), and as a detailed HTML report (`htmlcov/`).
- Fail the build if coverage drops below 70%.

To view the interactive HTML report:
```bash
# On macOS
open htmlcov/index.html

# On Windows
start htmlcov/index.html

# On Linux
xdg-open htmlcov/index.html
```

## 3. Required / Optional Secrets

Configure these in GitHub: Settings → Secrets and variables → Actions.

| Secret | Required? | Purpose | Notes |
|--------|-----------|---------|-------|
| `PYPI_API_TOKEN` | Yes* | Upload package in `release.yml` | *If using PyPI Trusted Publishing you can omit and approve first publication manually. Keep token while bootstrapping. |
| `CODECOV_TOKEN` | Yes (private repo) / No (public) | Coverage uploads in CI | Public repos sometimes auto-detect; set to be explicit. |
| `DOCKER_USERNAME` | Optional | Auth for Docker image push (if enabled) | Only needed if/when container publishing is turned on. |
| `DOCKER_TOKEN` | Optional | Password / token for Docker Hub | Pair with username. |
| `GH_PAT` | No | Only for advanced workflows (e.g. cross‑repo automation) | Not needed for standard release pipeline. |

Additional environment driven configs (rarely needed):
| Variable | Effect |
|----------|-------|
| `TENETS_CACHE_DIRECTORY` | Override default cache directory |
| `TENETS_DEBUG` | Enables verbose debug logging when `true` |

Security tips:
- Grant least privilege (PyPI token scoped to project if possible)
- Rotate any credentials annually or on role changes
- Prefer Trusted Publishing over long‑lived API tokens once stable

## 4. Code Style and Linting

We use `ruff` for linting and formatting. The pre-commit hook runs it automatically, but you can also run it manually:

```bash
# Check for linting errors
ruff check .

# Automatically fix linting errors
ruff check . --fix

# Format the code
ruff format .
```

## 5. Building Documentation

The documentation is built using `mkdocs`.

```bash
# Serve the documentation locally
mkdocs serve

# Build the static site
mkdocs build
```
The site will be available at `http://127.0.0.1:8000`.

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

## Release & Versioning

Releases are automated. Merging conventional commits into `main` (from PRs) is all you normally do.

### Branch Model
| Branch | Purpose |
|--------|---------|
| `dev` (or feature branches) | Integration / iterative work |
| `main` | Always releasable; auto-versioned on merge |

### Workflows (high level)
1. PR merged into `main`.
2. `version-bump.yml` runs:
    - Collects commits since last tag
    - Determines next version:
      - Major: commit body contains `BREAKING CHANGE:` or type suffixed with `!`
      - Minor: at least one `feat:` or `perf:` commit (performance treated as minor to signal impact)
      - Patch: any `fix`, `refactor`, `chore` (unless a higher bump already chosen)
      - Skip: only docs / test / style commits (no release)
    - Updates `pyproject.toml`
    - Appends a section to `CHANGELOG.md` grouping commits (Features / Performance / Fixes / Refactoring / Chore)
    - Commits with message `chore(release): vX.Y.Z` and creates annotated tag `vX.Y.Z`
3. Tag push triggers `release.yml`:
    - Builds wheel + sdist
    - Publishes to PyPI (token or Trusted Publishing)
    - (Optional) Builds & publishes Docker image (future enablement)
    - Deploys docs (if configured) / updates site
4. `release-drafter` (config) ensures GitHub Release notes reflect categorized changes (either via draft or final publish depending on config state).

You do NOT run `cz bump` manually during normal flow; the workflow handles versioning.

### Conventional Commit Expectations
Use clear scopes where possible:
```
feat(ranking): add semantic similarity signal
fix(cli): prevent crash on empty directory
perf(analyzer): cache parsed ASTs
refactor(config): simplify loading logic
docs: update quickstart for --copy flag
```

Edge cases:
- Multiple commit types: highest precedence decides (major > minor > patch)
- Mixed docs + fix: still releases (fix wins)
- Only docs/test/style: skipped; no tag produced

### First Release (Bootstrap)
If no existing tag:
1. Merge initial feature set into `main`
2. Push a commit with `feat: initial release` (or similar)
3. Workflow sets version to `0.1.0` (or bump logic starting point defined in workflow)

If you need a different starting version (e.g. `0.3.0`): create an annotated tag manually once, then subsequent merges resume automation.

### Manual / Emergency Release
Only when automation is blocked:
```bash
git checkout main && git pull
cz bump --increment PATCH  # or MINOR / MAJOR
git push && git push --tags
```
Monitor `release.yml`. After resolution, revert to automated flow.

### Verifying a Release
After automation completes:
```bash
pip install --no-cache-dir tenets==<new_version>
tenets --version
```
Smoke test a core command:
```bash
tenets distill "smoke" . --max-tokens 2000 --mode fast --stats || true
```

### Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No new tag after merge | Only docs/test/style commits | Land a non-skipped commit (e.g. fix) |
| Wrong bump size | Commit type misclassified | Amend / add corrective commit (e.g. feat) |
| PyPI publish failed | Missing / invalid `PYPI_API_TOKEN` or Trusted Publishing not approved yet | Add token or approve in PyPI UI |
| Changelog missing section | Commit type not in allowed list | Ensure conventional type used |
| Duplicate release notes | Manual tag + automated tag | Avoid manual tagging except emergencies |

### Philosophy
Keep `main` always shippable. Small, frequent releases reduce risk and keep context fresh for users.

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