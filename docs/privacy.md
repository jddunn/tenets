---
title: Privacy Policy
hide:
  - navigation
  - toc
---

# Privacy Policy

**Last Updated: December 2024**

## Overview

Tenets is an open-source project committed to user privacy. This policy explains what data we collect on this documentation website (tenets.dev) and how the tenets software handles your code.

## The Tenets Software

**Your code never leaves your machine.**

The tenets CLI and library process everything locally:

- ✅ All code analysis runs on your computer
- ✅ No code is transmitted to external servers
- ✅ No API calls required for core functionality
- ✅ Optional ML features use local models
- ✅ Session data stored locally in SQLite

When you use `tenets distill`, `tenets rank`, or any other command, your code stays private.

### Optional External Connections

These features require explicit opt-in:

| Feature | What It Does | Data Sent |
|---------|--------------|-----------|
| **External Issue Fetching** | Fetches GitHub/JIRA issues | Issue URL only |
| **LLM Summarization** | Uses external AI for summaries | Selected code snippets (opt-in) |
| **Update Checks** | Checks for new versions | Version number only |

You can disable all network features in `.tenets.yml`:

```yaml
network:
  enabled: false
```

## This Website (tenets.dev)

### Analytics

We use analytics to understand how people use our documentation:

| Service | Purpose | Data Collected |
|---------|---------|----------------|
| **Google Analytics** | Page views, navigation patterns | Anonymized IP, pages viewed, browser info |
| **Microsoft Clarity** | Heatmaps, session recordings | Mouse movements, clicks, scroll depth |

**You can opt out at any time** using the cookie banner or by:
- Using browser Do Not Track
- Using an ad blocker
- Clearing cookies and rejecting on next visit

### Cookies

| Cookie | Purpose | Duration |
|--------|---------|----------|
| `cookieConsent` | Remembers your cookie choice | Persistent |
| `_ga`, `_gid` | Google Analytics | 2 years, 24 hours |
| `_clck`, `_clsk` | Microsoft Clarity | 1 year, 1 day |

### GDPR Rights (EU Visitors)

Under GDPR, you have the right to:

- **Access** your data
- **Rectify** incorrect data
- **Erase** your data ("right to be forgotten")
- **Restrict** processing
- **Data portability**
- **Object** to processing

To exercise these rights, contact: [privacy@tenets.dev](mailto:privacy@tenets.dev)

## Data Retention

- **Analytics data**: Automatically deleted after 14 months
- **No personal data collected** from software usage
- **Local caches**: Managed by you (default: 7 days)

## Children's Privacy

This website and software are not intended for children under 13. We do not knowingly collect data from children.

## Changes to This Policy

We may update this policy occasionally. Changes will be posted here with an updated date.

## Contact

For privacy questions:

- Email: [privacy@tenets.dev](mailto:privacy@tenets.dev)
- GitHub: [github.com/jddunn/tenets](https://github.com/jddunn/tenets)

---

*Tenets is open source software. You can inspect exactly what the software does in our [GitHub repository](https://github.com/jddunn/tenets).*

