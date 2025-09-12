# Security Policy

## Supported Versions

The project is pre-1.0; security fixes are applied to the latest released version. Older versions may not receive backports.

## Reporting a Vulnerability

Email: team@tenets.dev (or team@manic.agency if unreachable)

Please include:
- Description of the issue
- Steps to reproduce / proof-of-concept
- Potential impact / affected components
- Your environment (OS, Python, tenets version)

We aim to acknowledge within 3 business days and provide a remediation ETA after triage.

## Responsible Disclosure

Do not open public issues for exploitable vulnerabilities. Use the private email above. We will coordinate disclosure and credit (if desired) after a fix is released.

## Scope

Tenets runs locally. Primary concerns:
- Arbitrary code execution via file parsing
- Directory traversal / path injection
- Insecure temporary file handling
- Leakage of private repository data beyond intended output

Out of scope:
- Issues requiring malicious local user privilege escalation
- Vulnerabilities in optional third-party dependencies (report upstream)

## Security Best Practices (Users)

- Pin versions in production workflows
- Run latest patch release
- Review output before sharing externally
- Avoid running against untrusted repositories without isolation (use containers)

## Patching Process

1. Triage & reproduce
2. Develop fix in private branch
3. Add regression tests
4. Coordinate release (patch version bump)
5. Publish advisory in CHANGELOG / release notes

## Contact

team@tenets.dev // team@manic.agency