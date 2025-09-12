# Versioning Strategy

This project uses automated semantic versioning based on conventional commits.

## Automatic Version Bumps

Version bumps are triggered automatically in two ways:

1. **On Pull Request Merge** - When a PR is merged to master/main branch
2. **On Direct Push** - When feat/fix commits are pushed directly to master/main branch

## Conventional Commit Types

- `feat:` - New feature (triggers minor version bump: 0.1.0 → 0.2.0)
- `fix:` - Bug fix (triggers patch version bump: 0.1.0 → 0.1.1)
- `perf:` - Performance improvement (triggers patch version bump)
- `refactor:` - Code refactoring (triggers patch version bump)
- `BREAKING CHANGE` - Breaking change (triggers major version bump: 0.1.0 → 1.0.0)

## Non-Versioned Commits

The following commit types do not trigger version bumps:
- `docs:` - Documentation only changes
- `chore:` - Maintenance tasks
- `ci:` - CI/CD changes
- `style:` - Code style changes
- `test:` - Test changes

## Release Process

1. Commit with appropriate conventional commit message
2. GitHub Actions automatically:
   - Determines version bump type
   - Updates version in pyproject.toml and __init__.py
   - Updates CHANGELOG.md
   - Creates git tag
   - Triggers release workflow
3. Release workflow automatically:
   - Builds Python package
   - Creates GitHub release
   - Publishes to PyPI

## Manual Version Bump

If needed, you can manually trigger a version bump using the manual-version-bump.yml workflow from the Actions tab.

## Skip Version Bump

To skip automatic version bump:
- Add `[skip ci]` to commit message
- Or use non-versioned commit types (docs, chore, etc.)