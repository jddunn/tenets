# Contributing to Tenets

Thanks for your interest in improving Tenets! Contributions of all kinds are welcome: bug reports, docs, tests, features, performance improvements, refactors, and feedback.

## Quick Start (TL;DR)

```bash
# Fork / clone
 git clone https://github.com/jddunn/tenets.git
 cd tenets

# Create a virtual environment (or use pyenv / conda)
 python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install core + dev extras
 pip install -e .[dev]
 # (or: make dev)

# Run tests
 pytest -q

# Lint & type check
 ruff check .
 mypy tenets

# Format
 black .

# Run a sample command
 tenets distill "hello world" . --stats
```

## Project Philosophy

Tenets is:
- Local-first, privacy-preserving
- Fast with graceful scalability (analyze only as deep as necessary)
- Extensible without forcing heavyweight ML (opt-in extras)
- Transparent in ranking decisions (explanations where reasonable)

## Issue Tracking

Before filing:
1. Search existing issues (open + closed)
2. For questions / ideas, consider starting a GitHub Discussion (if enabled) or Discord
3. Provide reproduction steps and environment info (OS, Python version, extras installed)

Good bug report template:
```
### Description
Clear, concise description of the problem.

### Reproduction
Commands or code snippet that reproduces the issue.

### Expected vs Actual
What you expected / what happened.

### Environment
OS / Python / tenets version / installed extras.
```

## Branch & Commit Conventions

- Create feature branches off `dev` (default contribution branch)
- Keep PRs narrowly scoped when possible
- Conventional Commit prefixes (enforced via commitizen config):
  - feat: new user-facing feature
  - fix: bug fix
  - refactor: code change without feature/bug semantics
  - perf: performance improvement
  - docs: docs only changes
  - test: add or improve tests
  - chore: tooling / infra / build

Example:
```
feat(ranking): add parallel TF-IDF corpus prepass
```

Use `cz commit` if you have commitizen installed.

## Code Style & Tooling

| Tool | Purpose | Command |
|------|---------|---------|
| black | Formatting | `black .` |
| ruff | Linting (multi-plugin) | `ruff check .` |
| mypy | Static typing | `mypy tenets` |
| pytest | Tests + coverage | `pytest -q` |
| coverage | HTML / XML reports | `pytest --cov` |
| commitizen | Conventional versioning | `cz bump` |

Pre-commit hooks (optional):
```bash
pip install pre-commit
pre-commit install
```

## Tests

Guidelines:
- Place tests under `tests/` mirroring module paths
- Use `pytest` fixtures; prefer explicit data over deep mocks
- Mark slow tests with `@pytest.mark.slow`
- Keep unit tests fast (<300ms ideally)
- Add at least one failing test before a bug fix

Run selectively:
```bash
pytest tests/core/analysis -k python_analyzer
pytest -m "not slow"
```

## Type Hints

- New/modified public functions must be fully typed
- Avoid `Any` unless absolutely necessary; justify in a comment
- mypy config is strict—fix or silence with narrow `# type: ignore[...]`

## Documentation

User docs live in `docs/` (MkDocs Material). For changes affecting users:
- Update `README.md`
- Update or create relevant page under `docs/`
- Add examples (`quickstart.md`) if CLI/API behavior changes
- Link new pages in `mkdocs.yml`

Serve docs locally:
```bash
mkdocs serve
```

## Adding a Language Analyzer

1. Create `<language>_analyzer.py` under `tenets/core/analysis/implementations/`
2. Subclass `LanguageAnalyzer`
3. Implement `match(path)` and `analyze(content)`
4. Add tests under `tests/core/analysis/implementations/`
5. Update `supported-languages.md`

## Ranking Extensions

- Register custom rankers via provided registration API (see `tenets/core/ranking/ranker.py`)
- Provide deterministic output; avoid network calls in ranking stage
- Document new algorithm flags in `config.md`

## Performance Considerations

- Avoid O(n^2) scans over file lists when possible
- Cache expensive analysis (see existing caching layer)
- Add benchmarks if adding heavy operations (future / optional)

## Security / Privacy

- Never exfiltrate code or send network requests without explicit user config
- Keep default extras minimal

## Release Process (Maintainers)

1. Ensure `dev` is green (CI + coverage)
2. Bump version: `cz bump` (updates `pyproject.toml`, tag, CHANGELOG)
3. Build: `make build` (or `python -m build`)
4. Publish: `twine upload dist/*`
5. Merge `dev` -> `master` and push tags

## Code of Conduct

This project follows the [Code of Conduct](code_of_conduct.md). By participating you agree to uphold it.

## License

By contributing you agree your contributions are licensed under the MIT License.

---
Questions? Open an issue or reach out via Discord.
