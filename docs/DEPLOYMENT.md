# Deployment Guide

This guide outlines the process for releasing new versions of Tenets to PyPI and deploying documentation.

## Release Process (Automated)

Standard path: merge conventional commits into `main`; automation versions & publishes.

### How It Works
1. Merge PR → `version-bump.yml` runs
2. Determines bump size (major / minor / patch / skip) from commit messages
3. Updates `pyproject.toml` + appends grouped section to `CHANGELOG.md`
4. Commits `chore(release): vX.Y.Z` and tags `vX.Y.Z`
5. Tag triggers `release.yml`: build, publish to PyPI, (future) Docker, docs deploy
6. Release notes composed from changelog / draft config

### Bump Rules (Summary)
| Commit Types Seen | Result |
|-------------------|--------|
| BREAKING CHANGE / `!` | Major |
| feat / perf | Minor |
| fix / refactor / chore | Patch (unless higher trigger present) |
| Only docs / test / style | Skip |

### Manual Overrides (Rare)
If automation blocked (workflow infra outage):
```bash
git checkout main && git pull
cz bump --increment PATCH  # or MINOR / MAJOR
git push && git push --tags
```
Resume automation next merge.

### First Release Bootstrap (v0.1.0)

**For the initial v0.1.0 release, follow this manual process:**

1. **Update CHANGELOG.md** with v0.1.0 entries (on dev branch)
2. **Commit and push to dev:** `git commit -m "docs: update CHANGELOG for v0.1.0"`
3. **Merge dev → master**
4. **From master, create and push tag:**
   ```bash
   git checkout master && git pull
   git tag -a v0.1.0 -m "Release v0.1.0 - Initial public release"
   git push origin v0.1.0  # This triggers everything!
   ```
5. **The tag push automatically triggers:**
   - GitHub Release creation with artifacts
   - PyPI package publishing (if PYPI_API_TOKEN is set)
   - Documentation deployment to GitHub Pages

**After v0.1.0:** Automation takes over - commits trigger version bumps based on conventional commit messages.

### Verification Checklist
| Step | Command / Action |
|------|------------------|
| Install published wheel | `pip install --no-cache-dir tenets==X.Y.Z` |
| CLI version matches | `tenets --version` |
| Release notes present | Check GitHub Release page |
| Docs updated | Visit docs site / gh-pages commit |

### Troubleshooting
| Issue | Cause | Resolution |
|-------|-------|------------|
| No tag created | Only docs/test/style commits | Land a fix/feat/perf commit |
| Wrong bump size | Mis-typed commit message | Amend & force push before merge; or follow-up commit |
| PyPI publish failed | Missing PyPI token / trust approval pending | Add `PYPI_API_TOKEN` or approve trusted publisher |
| Duplicate releases | Manual tag + automated tag | Avoid manual tagging unless emergency |

## Documentation Deployment

Docs are (a) built in CI on PR for validation; (b) deployed on release tag push by `release.yml` (or dedicated docs deploy step on main). GitHub Pages serves from `gh-pages`.

## Required / Optional Secrets

| Secret | Required | Purpose | Notes |
|--------|----------|---------|-------|
| `PYPI_API_TOKEN` | Yes* | PyPI publish in `release.yml` | *Omit if using Trusted Publishing (approve first build). |
| `CODECOV_TOKEN` | Public: often no / Private: yes | Coverage uploads | Set to be explicit. |
| `GOOGLE_ANALYTICS_ID` | Optional | GA4 measurement ID for docs analytics | Used by MkDocs Material via `!ENV` in `mkdocs.yml` (e.g., `G-XXXXXXXXXX`). If unset/empty, analytics are disabled. |
| `DOCKER_USERNAME` / `DOCKER_TOKEN` | Optional | Future Docker image publishing | Not required yet. |
| `GH_PAT` | No | Cross-repo automation (not standard) | Avoid storing if unused. |

Environment (optional): `TENETS_DEBUG`, `TENETS_CACHE_DIRECTORY`.

### Google Analytics (optional)

MkDocs Material analytics are wired to an environment variable:

- In `mkdocs.yml`: `extra.analytics.property: !ENV [GOOGLE_ANALYTICS_ID, ""]`
- Provide a GA4 Measurement ID (format `G-XXXXXXXXXX`). If the variable is unset or empty, analytics are disabled automatically.

Local usage

```bash
# bash / Git Bash / WSL
export GOOGLE_ANALYTICS_ID=G-XXXXXXXXXX
mkdocs serve
```

```powershell
# PowerShell
$env:GOOGLE_ANALYTICS_ID = 'G-XXXXXXXXXX'
mkdocs serve
```

GitHub Actions (recommended)

```yaml
jobs:
   docs:
      runs-on: ubuntu-latest
      env:
         GOOGLE_ANALYTICS_ID: ${{ secrets.GOOGLE_ANALYTICS_ID }}
      steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
            with:
               python-version: '3.12'
         - run: pip install -e '.[docs]'
         - run: mkdocs build --clean
```

Store your GA4 Measurement ID as a repository secret named `GOOGLE_ANALYTICS_ID`. The docs build will inject it at build time; if not present, analytics are off.

# With specific features
pip install tenets[ml]  # ML features
pip install tenets[viz]  # Visualization
pip install tenets[all]  # Everything
```

### 2. Development Installation

```bash
# From source
git clone https://github.com/jddunn/tenets.git
cd tenets
pip install -e ".[dev]"
```

### 3. Docker Container

```bash
# Pull from Docker Hub
docker pull tenets/tenets:latest

# Run command
docker run --rm -v $(pwd):/workspace tenets/tenets make-context "query" .

# Interactive shell
docker run -it --rm -v $(pwd):/workspace tenets/tenets bash
```

### 4. Standalone Binary

Download from [GitHub Releases](https://github.com/jddunn/tenets/releases):

```bash
# Linux/macOS
curl -L https://github.com/jddunn/tenets/releases/latest/download/tenets-linux -o tenets
chmod +x tenets
./tenets --version

# Windows
# Download tenets-windows.exe from releases page
```

## PyPI Publishing

### First-Time Setup

1. **Create PyPI account**:
   - Register at [pypi.org](https://pypi.org)
   - Enable 2FA (required)

2. **Configure trusted publishing**:
   - Go to your project settings on PyPI
   - Add GitHub Actions as trusted publisher:
     - Owner: `jddunn`
     - Repository: `tenets`
     - Workflow: `release.yml`
     - Environment: `pypi`

### Manual Publishing (Emergency Only)

```bash
# Build distribution
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ tenets

# Upload to PyPI
twine upload dist/*
```

## Docker Deployment

### Building Images

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash tenets

# Set working directory
WORKDIR /app

# Install tenets
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER tenets

# Set entrypoint
ENTRYPOINT ["tenets"]
```

### Multi-Architecture Build

```bash
# Setup buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag tenets/tenets:latest \
  --tag tenets/tenets:v0.1.0 \
  --push .
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  tenets:
    image: tenets/tenets:latest
    volumes:
      - .:/workspace
      - ~/.tenets:/home/tenets/.tenets
    working_dir: /workspace
    environment:
      - TENETS_LOG_LEVEL=INFO
    command: make-context "implement feature" .
```

## Binary Distribution

### Building Binaries

```bash
# Install PyInstaller
pip install pyinstaller

# Build for current platform
pyinstaller \
  --onefile \
  --name tenets \
  --add-data "tenets:tenets" \
  --hidden-import tenets.core \
  --hidden-import tenets.models \
  --hidden-import tenets.utils \
  tenets/__main__.py

# Output in dist/tenets
```

### Cross-Platform Building

Use GitHub Actions for multi-platform builds:
- Linux: Ubuntu runner
- macOS: macOS runner
- Windows: Windows runner

### Code Signing (Optional)

```bash
# macOS
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: Your Name" \
  dist/tenets

# Windows (using signtool)
signtool sign /t http://timestamp.digicert.com dist/tenets.exe
```

## Documentation Deployment

### Building Documentation

```bash
# Install dependencies
pip install -e ".[docs]"

# Build docs
mkdocs build

# Test locally
mkdocs serve
```

### Versioned Documentation

```bash
# Deploy new version
mike deploy --push --update-aliases 0.1.0 latest

# Deploy development docs
mike deploy --push dev

# Set default version
mike set-default --push latest
```

### GitHub Pages Setup

1. Enable GitHub Pages in repository settings
2. Set source to `gh-pages` branch
3. Documentation auto-deploys on release

## Security Considerations

### Release Security

1. **Sign commits and tags**:
   ```bash
   git config --global commit.gpgsign true
   git config --global tag.gpgsign true
   ```

2. **Verify dependencies**:
   ```bash
   # Check for vulnerabilities
   safety check

   # Audit dependencies
   pip-audit
   ```

3. **Scan for secrets**:
   ```bash
   # Pre-release scan
   detect-secrets scan --all-files
   ```

### Deployment Security

1. **Use minimal base images**:
   ```dockerfile
   FROM python:3.11-slim  # Not full python image
   ```

2. **Run as non-root**:
   ```dockerfile
   USER nobody
   ```

3. **Scan images**:
   ```bash
   # Scan for vulnerabilities
   docker scan tenets/tenets:latest
   ```

## Monitoring & Maintenance

### Release Monitoring

1. **PyPI Statistics**:
   - Check download stats
   - Monitor for unusual activity

2. **GitHub Insights**:
   - Track clone/download metrics
   - Monitor issue trends

3. **Error Tracking**:
   - Set up Sentry (optional)
   - Monitor GitHub issues

### Maintenance Tasks

#### Weekly
- Review and triage issues
- Check for security advisories
- Update dependencies

#### Monthly
- Review performance metrics
- Update documentation
- Clean up old releases

#### Quarterly
- Major dependency updates
- Security audit
- Performance benchmarking

### Rollback Procedure

If a release has critical issues:

1. **Yank from PyPI** (last resort):
   ```bash
   # This prevents new installations
   # Existing installations continue to work
   twine yank tenets==0.1.0
   ```

2. **Create hotfix**:
   ```bash
   git checkout -b hotfix/critical-bug
   # Fix issue
   git commit -m "fix: critical bug in analyzer"
   cz bump --increment PATCH
   git push origin hotfix/critical-bug
   ```

3. **Fast-track release**:
   - Create PR with hotfix
   - Bypass normal review (emergency)
   - Merge and tag immediately

## Deployment Environments

### Development
```bash
pip install -e ".[dev]"
export TENETS_ENV=development
```

### Staging
```bash
pip install tenets==0.1.0rc1  # Release candidate
export TENETS_ENV=staging
```

### Production
```bash
pip install tenets==0.1.0
export TENETS_ENV=production
```

## Troubleshooting

### Common Issues

1. **PyPI upload fails**:
   - Check PyPI status
   - Verify credentials
   - Ensure version doesn't exist

2. **Docker build fails**:
   - Clear builder cache
   - Check Docker Hub limits
   - Verify multi-arch support

3. **Documentation not updating**:
   - Check GitHub Pages settings
   - Verify mike configuration
   - Clear browser cache

### Getting Help

- GitHub Issues for bugs
- Discussions for questions
- team@tenets.dev for security issues

---

Remember: Every release should make developers' lives easier. 🚀
