# Examples

## Basic Analysis Examples

### Analyzing a Python Project

```bash
# Basic examination
tenets examine my_python_project/

# With specific focus
tenets examine my_python_project/ --language python --depth 3

# Output to JSON
tenets examine my_python_project/ --format json > analysis.json
```

### Analyzing a JavaScript/TypeScript Project

```bash
# Examine with TypeScript support
tenets examine frontend/ --language typescript

# Exclude node_modules
tenets examine frontend/ --exclude "**/node_modules/**"
```

## Chronicle Examples

### Team Contribution Analysis

```bash
# Last month's team activity
tenets chronicle --days 30 --format table

# Specific developer's contributions
tenets chronicle --author "Jane Doe" --days 90

# Focus on feature branch
tenets chronicle --branch feature/new-ui --days 14
```

### Release Analysis

```bash
# Changes between releases
tenets chronicle --from v1.0.0 --to v2.0.0

# Recent hotfixes
tenets chronicle --pattern "**/hotfix/**" --days 7
```

## Distill Examples

### Project Insights

```bash
# Generate comprehensive insights
tenets distill --comprehensive

# Quick summary
tenets distill --quick

# Export for reporting
tenets distill --format markdown > insights.md
```

## Visualization Examples

### Architecture Visualization

```bash
# Interactive HTML graph
tenets viz --output architecture.html

# Include all relationships
tenets viz --include-all --output full-graph.html

# Focus on core modules
tenets viz --filter "core/**" --output core-modules.html
```

## Momentum Tracking

### Development Velocity

```bash
# Weekly momentum report
tenets momentum --period week

# Monthly trends
tenets momentum --period month --format chart

# Team momentum
tenets momentum --team --days 30
```

## Advanced Combinations

### Pre-Release Audit

```bash
# Full pre-release analysis
tenets examine --comprehensive > examine-report.txt
tenets chronicle --days 30 > chronicle-report.txt
tenets distill --format json > insights.json
tenets viz --output release-architecture.html
```

### Technical Debt Assessment

```bash
# Identify complex areas
tenets examine --metric complexity --threshold high

# Find stale code
tenets chronicle --stale --days 180

# Ownership gaps
tenets examine --ownership --unowned
```

### Team Performance Review

```bash
# Individual contributions
for author in "Alice" "Bob" "Charlie"; do
  tenets chronicle --author "$author" --days 90 > "$author-report.txt"
done

# Team visualization
tenets viz --team --output team-collaboration.html
```

## Integration Examples

### GitHub Actions

```yaml
name: Code Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Tenets
        run: pip install tenets
      - name: Run Analysis
        run: |
          tenets examine --format json > examine.json
          tenets chronicle --days 7 --format json > chronicle.json
      - name: Upload Results
        uses: actions/upload-artifact@v2
        with:
          name: analysis-results
          path: |
            examine.json
            chronicle.json
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tenets-complexity
        name: Check Code Complexity
        entry: tenets examine --metric complexity --fail-on high
        language: system
        pass_filenames: false
```

## Next Steps

- Review [Best Practices](best_practices.md) for optimal usage
- See [CLI Reference](cli.md) for all available options
- Check [Configuration](config.md) for customization options
