# Deployment Guide

This guide outlines the process for releasing new versions of Tenets to PyPI and deploying documentation.

## Release Process

We follow a manual, tag-based release process. Releases are not automatically created on every commit to `main`.

### 1. Pre-Release Checklist

Before creating a new release, ensure the following are complete:

- [ ] All relevant feature branches have been merged into `main`.
- [ ] The `main` branch has passed all CI checks (tests, linting, etc.).
- [ ] The `CHANGELOG.md` file has been updated with all notable changes for the new version.
- [ ] The version number in `pyproject.toml` has been incremented according to [Semantic Versioning](https://semver.org/).

### 2. Creating a Release

1.  **Create a Git Tag**: Create a new Git tag that matches the version in `pyproject.toml`.

    ```bash
    # Example for version 1.2.3
    git tag -a v1.2.3 -m "Release version 1.2.3"
    ```

2.  **Push the Tag**: Push the new tag to the `origin` remote.

    ```bash
    git push origin v1.2.3
    ```

### 3. Automated Release Workflow

Pushing a new tag triggers the `release.yml` GitHub Actions workflow, which automates the following steps:

1.  **Builds the Package**: Builds the source distribution (`sdist`) and wheel (`bdist_wheel`) for the project.
2.  **Publishes to PyPI**: Publishes the built package to the Python Package Index (PyPI) using the `PYPI_API_TOKEN` secret.
3.  **Creates a GitHub Release**: Creates a new release on GitHub, using the tag and release notes from the `CHANGELOG.md`.

## Documentation Deployment

The documentation is automatically deployed to GitHub Pages whenever a commit is pushed to the `main` branch. This is handled by the `ci.yml` workflow.

## Required Secrets

The following secrets must be configured in the GitHub repository settings under **Settings > Secrets and variables > Actions**:

-   **`PYPI_API_TOKEN`**: An API token from PyPI with permission to upload packages to the `tenets` project.
-   **`CODECOV_TOKEN`**: The repository upload token from Codecov, used to upload coverage reports.

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