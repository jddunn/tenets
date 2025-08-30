# Best Practices

## Repository Setup

### Clean Working Directory
Always run Tenets on a clean working directory for accurate results:
```bash
git status  # Ensure no uncommitted changes
tenets examine
```

### Gitignore Configuration
Ensure your `.gitignore` is properly configured to exclude:
- Build artifacts
- Node modules
- Virtual environments
- Temporary files

## Command Usage

### Examine Command
- Use `--format` for different output formats
- Filter by language with `--language`
- Focus on specific paths for targeted analysis

### Chronicle Command
- Use time ranges appropriate to your project's activity
- Filter by author for team member contributions
- Combine with `--pattern` for specific file analysis

### Distill Command
- Run after significant development milestones
- Use to generate weekly or monthly insights
- Combine with chronicle for historical context

## Performance Optimization

### Large Repositories
For repositories with many files:
```bash
# Focus on specific directories
tenets examine src/ --depth 3

# Exclude certain patterns
tenets examine --exclude "**/test/**"
```

### Memory Management
- Use `--batch-size` for large analyses
- Enable streaming output with `--stream`

## Integration

### CI/CD Pipeline
Add Tenets to your CI pipeline:
```yaml
- name: Code Analysis
  run: |
    pip install tenets
    tenets examine --format json > analysis.json
```

### Pre-commit Hooks
Use Tenets in pre-commit hooks:
```yaml
repos:
  - repo: local
    hooks:
      - id: tenets-check
        name: Tenets Analysis
        entry: tenets examine --quick
        language: system
        pass_filenames: false
```

## Team Collaboration

### Sharing Reports
- Generate HTML reports for stakeholder review
- Export JSON for further processing
- Use visualizations for architecture discussions

### Code Reviews
Use Tenets output to:
- Identify complex areas needing review
- Track ownership changes
- Monitor technical debt

## Next Steps

- See [Examples](examples.md) for real-world scenarios
- Review [CLI Reference](CLI.md) for all options
- Check [Configuration](CONFIG.md) for customization