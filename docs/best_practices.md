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

### Prompt Writing Tips

**Learn one to learn both**: Rank and distill commands are designed to be interchangeable, with the only difference being file content versus just filepaths.

**Be brutally curt**: Intentions and concepts are mapped (specifically in programming context), so the more you list the more matches/crossover you'll get. For example, "fix, analyze, and debug this issue" is worse than just "fix".

**Don't use many synonyms**: Sometimes in LLM interactions you mention multiple variations of a term, hoping to expand its context and connections. This is a static tool; more words in the prompt = lower processing speed and more crossover. Let tenets decide relationships, not you.

**Planning**: Identify key functions, lines, variables before making changes helps greatly. If you specify method names in the prompt, it will search for and aggregate all files that call it and smartly summarize the file's context around mentioned matches.

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

### Choosing the Right Mode

Tenets offers three ranking modes that balance speed vs. thoroughness:

**Fast Mode** (100% baseline speed)
- Best for: Quick exploration, CI/CD pipelines, large codebases (1000+ files)
- Use when: You need results quickly or are working programmatically
- Trade-off: Less accurate matching, may miss nuanced connections

**Balanced Mode** (150% - 1.5x slower, default)
- Best for: Day-to-day development, bug fixing, feature building
- Use when: You need good accuracy without waiting too long
- Trade-off: Only 50% slower than fast mode with significantly better results

**Thorough mode** (400% - 4x slower)
- Best for: Major refactoring, architectural analysis, semantic search
- Use when: You need to find semantically similar code, not just keyword matches
- Trade-off: Requires ML dependencies, takes longer but finds deeper connections

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
- Review [CLI Reference](cli.md) for all options
- Check [Configuration](config.md) for customization
