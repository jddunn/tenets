# Deployment & Release Guide

This guide covers releasing **tenets** and deploying it in various environments.

## Table of Contents

- [Release Process](#release-process)
- [Deployment Options](#deployment-options)
- [PyPI Publishing](#pypi-publishing)
- [Docker Deployment](#docker-deployment)
- [Binary Distribution](#binary-distribution)
- [Documentation Deployment](#documentation-deployment)
- [Security Considerations](#security-considerations)
- [Monitoring & Maintenance](#monitoring--maintenance)

## Release Process

### Prerequisites

- Maintainer access to the repository
- PyPI account with project access
- Docker Hub account (optional)
- GPG key for signing (recommended)

### Automated Release (Recommended)

1. **Prepare the release**:
   ```bash
   # Ensure you're on main
   git checkout main
   git pull origin main
   
   # Run all tests
   make test
   
   # Check what will be released
   cz bump --dry-run
   ```

2. **Create the release**:
   ```bash
   # Interactive release
   make release
   
   # This will:
   # 1. Run tests
   # 2. Bump version (major/minor/patch)
   # 3. Update CHANGELOG.md
   # 4. Create git commit and tag
   # 5. Push to GitHub
   ```

3. **Monitor the release**:
   - Go to [GitHub Actions](https://github.com/jddunn/tenets/actions)
   - Watch the "Release" workflow
   - Verify all steps complete successfully

### Manual Release Process

1. **Update version**:
   ```bash
   # Choose version bump type
   cz bump --increment PATCH  # 0.1.0 -> 0.1.1
   cz bump --increment MINOR  # 0.1.0 -> 0.2.0
   cz bump --increment MAJOR  # 0.1.0 -> 1.0.0
   ```

2. **Review changes**:
   ```bash
   # Check CHANGELOG.md
   cat CHANGELOG.md
   
   # Verify version files
   grep version pyproject.toml
   grep __version__ tenets/__init__.py
   ```

3. **Create release**:
   ```bash
   # Push commits and tags
   git push origin main
   git push origin --tags
   ```

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md reflects changes
- [ ] Version numbers are consistent
- [ ] Security scan passes
- [ ] Performance benchmarks acceptable

## Deployment Options

### 1. PyPI Installation (Users)

```bash
# Basic installation
pip install tenets

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