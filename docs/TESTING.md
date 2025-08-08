# Testing

## Quick Start

```bash
# One-liner (editable install + test deps + coverage helpers)
pip install -e '.[test]' pytest pytest-cov

# Run all tests (quiet)
pytest -q

# Run with coverage + fail if below threshold (adjust as policy evolves)
pytest --cov=tenets --cov-report=term-missing --cov-fail-under=80

# Generate XML (CI) + HTML
pytest --cov=tenets --cov-report=xml --cov-report=html

# Open HTML (macOS/Linux)
open htmlcov/index.html || xdg-open htmlcov/index.html || true

# Specific test file / test
pytest tests/core/analysis/test_analyzer.py::test_basic_python_analysis -q

# Pattern match
pytest -k analyzer -q

# Parallel (if pytest-xdist installed)
pytest -n auto
```

Optional feature extras (install before running related tests):
```bash
pip install -e '.[light]'   # TF-IDF / YAKE ranking tests
pip install -e '.[viz]'     # Visualization tests
pip install -e '.[ml]'      # Embedding / semantic tests (heavy)
```

## Test Structure

```text
tests/
├── conftest.py              # Shared fixtures
├── test_config.py           # Config tests
├── test_tenets.py           # Main module tests
├── core/
│   ├── analysis/           # Code analysis tests
│   ├── distiller/          # Context distillation tests
│   ├── git/                # Git integration tests
│   ├── prompt/             # Prompt parsing tests
│   ├── ranker/             # File ranking tests
│   ├── session/            # Session management tests
│   └── summarizer/         # Summarization tests
├── storage/
│   ├── test_cache.py       # Caching system tests
│   ├── test_session_db.py  # Session persistence tests
│   └── test_sqlite.py      # SQLite utilities tests
└── utils/
    ├── test_scanner.py     # File scanning tests
    ├── test_tokens.py      # Token counting tests
    └── test_logger.py      # Logging tests
```

## Running Tests

### By Category

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Tests requiring git
pytest -m requires_git

# Tests requiring ML dependencies
pytest -m requires_ml
```

### Coverage Reports

```bash
# Terminal report
pytest --cov=tenets --cov-report=term-missing

# Enforce minimum (CI/local gate)
pytest --cov=tenets --cov-report=term-missing --cov-fail-under=80

# HTML report
pytest --cov=tenets --cov-report=html

# XML for CI services (Codecov)
pytest --cov=tenets --cov-report=xml
```

### Debug Mode

```bash
# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Verbose output
pytest -vv
```

## Writing Tests

### Basic Test

```python
def test_feature(config, analyzer):
    """Test feature description."""
    result = analyzer.analyze_file(Path("test.py"))
    assert result.language == "python"
```

### Using Fixtures

```python
@pytest.fixture
def temp_project(tmp_path):
    """Create temporary project structure."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("print('hello')")
    return tmp_path

def test_with_project(temp_project):
    files = list(temp_project.glob("**/*.py"))
    assert len(files) == 1
```

### Mocking

```python
from unittest.mock import Mock, patch

def test_with_mock():
    with patch('tenets.utils.tokens.count_tokens') as mock_count:
        mock_count.return_value = 100
        # test code
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("test.py", "python"),
    ("test.js", "javascript"),
    ("test.go", "go"),
])
def test_language_detection(analyzer, input, expected):
    assert analyzer._detect_language(Path(input)) == expected
```

## Test Markers

Add to test functions:

```python
@pytest.mark.slow
def test_heavy_operation():
    pass

@pytest.mark.requires_git
def test_git_features():
    pass

@pytest.mark.skipif(not HAS_TIKTOKEN, reason="tiktoken not installed")
def test_token_counting():
    pass
```

## CI Integration

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pytest --cov=tenets --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: tests
      name: tests
      entry: pytest
      language: system
      pass_filenames: false
      always_run: true
```

## Release Test Checklist

Before tagging a release:

```bash
# 1. Clean environment
rm -rf .venv dist build *.egg-info && python -m venv .venv && source .venv/bin/activate

# 2. Install with all needed extras for full test surface
pip install -e '.[all,test]' pytest pytest-cov

# 3. Lint / type (if tools configured)
# ruff check .
# mypy tenets

# 4. Run tests with coverage gate
pytest --cov=tenets --cov-report=term-missing --cov-fail-under=80

# 5. Spot-check critical CLI commands
for cmd in \
  "distill 'smoke test' --stats" \
  "instill 'example tenet'" \
  "session create release-smoke" \
  "config cache-stats"; do
  echo "tenets $cmd"; tenets $cmd || exit 1; done

# 6. Build sdist/wheel
python -m build

# 7. Install built artifact in fresh venv & re-smoke
python -m venv verify && source verify/bin/activate && pip install dist/*.whl && tenets version
```

Minimal CHANGELOG update + version bump in `tenets/__init__.py` must precede tagging.

## Performance Testing

```bash
# Benchmark tests
pytest tests/performance/ --benchmark-only

# Profile slow tests
pytest --durations=10
```

## Troubleshooting

### Common Issues

**Import errors**: Ensure package is installed with test extras:
```bash
pip install -e ".[test]"
```

**Slow tests**: Use parallel execution:
```bash
pytest -n auto
```

**Flaky tests**: Re-run failures:
```bash
pytest --reruns 3
```

**Memory issues**: Run tests in chunks:
```bash
pytest tests/core/
pytest tests/storage/
pytest tests/utils/
```

## Coverage Goals

- **Overall**: >80%
- **Core logic**: >90%
- **Error paths**: >70%
- **Utils**: >85%

Check current coverage:
```bash
pytest --cov=tenets --cov-report=term-missing | grep TOTAL
```