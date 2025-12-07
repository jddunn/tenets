---
title: "Why Context Is Everything in AI Coding"
description: The difference between great AI-assisted code and hallucinated garbage comes down to context quality. Learn how to provide better context to LLMs.
author: Johnny Dunn
date: 2024-09-15
tags:
  - ai-coding
  - context
  - llm
  - best-practices
---

# Why Context Is Everything in AI Coding

**Author:** Johnny Dunn | **Date:** September 15, 2024

---

## Table of Contents

- [The Context Problem](#the-context-problem)
- [What Good Context Looks Like](#what-good-context-looks-like)
- [The Cost of Bad Context](#the-cost-of-bad-context)
- [How Tenets Solves This](#how-tenets-solves-this)
- [Practical Tips](#practical-tips)
- [Conclusion](#conclusion)

---

## The Context Problem

You've probably experienced this: you ask an AI assistant to help with code, and it returns something that:

- Uses libraries you don't have
- Ignores your existing patterns
- Invents APIs that don't exist
- Duplicates functionality you already wrote

This isn't because the AI is stupid. It's because **the AI doesn't know what you know**.

When you look at your codebase, you see:

- Years of decisions and conventions
- Dependencies and their versions
- Custom utilities and helpers
- Architectural patterns
- Team preferences

When the AI looks at your codebase (or doesn't), it sees... nothing. It fills in the blanks with training data—generic patterns that may have nothing to do with your project.

**Context is the difference between an AI that helps and an AI that wastes your time.**

---

## What Good Context Looks Like

Good context for AI coding includes:

### 1. Relevant Code

Not just the file you're working on, but:

- Files that import/export from the current file
- Similar implementations to reference
- Utilities and helpers you should reuse
- Types and interfaces to match

### 2. Conventions

- How do you name variables?
- What patterns do you follow?
- What frameworks and libraries are standard?
- What's the testing strategy?

### 3. Constraints

- What shouldn't the AI do?
- What security requirements exist?
- What performance considerations matter?

### 4. Structure

The AI needs to understand:

- Where does this code live in the architecture?
- What layers exist?
- How do components communicate?

---

## The Cost of Bad Context

Bad context isn't just annoying—it's expensive:

| Problem | Time Cost | Risk |
|---------|-----------|------|
| Wrong patterns | 15-30min to refactor | Inconsistent codebase |
| Missing utilities | 10-20min to rewrite | Code duplication |
| Wrong dependencies | 5-60min debugging | Broken builds |
| Hallucinated APIs | 20-45min to fix | Runtime errors |
| Security anti-patterns | Hours to days | Vulnerabilities |

A study by GitHub found that developers accept ~30% of Copilot suggestions. That means 70% need modification or rejection. Better context can significantly improve this ratio.

---

## How Tenets Solves This

Tenets automatically builds optimal context through:

### Multi-Factor Ranking

Instead of just keyword matching, Tenets uses:

1. **BM25 scoring** — Finds semantically relevant code
2. **Import analysis** — Includes dependencies
3. **Git signals** — Prioritizes recently-modified code
4. **Complexity weighting** — Focuses on significant code
5. **Path relevance** — Matches file/folder names

### Token Optimization

LLMs have context windows. Tenets:

- Ranks all files by relevance
- Fills context up to your token budget
- Prioritizes high-signal code
- Truncates intelligently

### Session State

For ongoing work, Tenets maintains:

- Pinned files (always included)
- Conversation history
- Guiding principles (tenets)

---

## Practical Tips

### Tip 1: Be Specific in Your Prompts

Instead of:
> "Fix the bug"

Try:
> "Fix the authentication bug where users can't log in with valid credentials"

More specific prompts = better ranking = better context.

### Tip 2: Pin Related Files

If you're working on authentication:

```bash
tenets session pin src/auth/ --session current-task
```

These files will always be in context.

### Tip 3: Add Tenets for Consistency

```bash
tenets tenet add "Use bcrypt for password hashing" --priority high
tenets tenet add "All API endpoints require authentication middleware"
```

These rules appear at the top of every context.

### Tip 4: Match Mode to Task

| Task | Mode |
|------|------|
| Quick question | `fast` |
| Normal development | `balanced` |
| Major refactoring | `thorough` |

### Tip 5: Review Before Sending

```bash
tenets distill "your task" --copy
```

Glance at the context before pasting. Sometimes you'll spot missing pieces.

---

## Conclusion

AI coding assistants are only as good as the context they receive. Random file selection and manual context building don't scale.

**Tenets automates optimal context**, letting you focus on the creative work while ensuring the AI has everything it needs.

---

*Ready to try it?* 

```bash
pip install tenets[mcp]
```

See the [Tutorial](../tutorial.md) for a complete walkthrough.

---

## Related Posts

- [Model Context Protocol Explained](mcp-explained.md)
- [Setting Up Tenets with Cursor and Claude](cursor-claude-setup.md)

---

<div style="text-align: center; padding: 2rem; margin-top: 2rem; background: rgba(245, 158, 11, 0.05); border-radius: 12px;">
  <p>Built by <a href="https://manic.agency" target="_blank">manic.agency</a></p>
  <a href="https://manic.agency/contact" style="color: #f59e0b;">Need custom AI tooling? →</a>
</div>

