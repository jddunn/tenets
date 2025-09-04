---
title: Integration & APIs
description: How to integrate Tenets with other tools and systems
---

# Integration & APIs

## Overview

Tenets provides multiple integration points for embedding its functionality into your development workflow, CI/CD pipelines, and custom tools.

## Python API

### Basic Integration

```python
from tenets import Tenets

# Initialize
tenets = Tenets()

# Basic usage
result = tenets.distill("implement authentication")
print(f"Generated {result.token_count} tokens")
print(result.content)
```

### Advanced Integration

```python
from tenets import Tenets, Config
from tenets.core.ranking import RelevanceRanker, RankingAlgorithm

# Custom configuration
config = Config(
    algorithm=RankingAlgorithm.THOROUGH,
    max_tokens=100000,
    include_tests=False
)

# Initialize with config
tenets = Tenets(config=config)

# Use sessions
tenets.session.create("feature-x")
tenets.session.pin_file("src/core/auth.py")

# Rank files
ranker = RelevanceRanker(algorithm="balanced")
files = tenets.discover_files("./src")
ranked = ranker.rank(files, "optimize database queries")

# Custom processing
for file in ranked[:10]:
    print(f"{file.path}: {file.relevance_score:.3f}")
```

## CLI Integration

### Shell Scripts

```bash
#!/bin/bash
# Generate context and copy to clipboard
tenets distill "fix bug in payment processing" --copy

# Export ranked files for processing
tenets rank "refactor auth" --format json | \
  jq -r '.files[].path' | \
  xargs pylint

# Create session and build context
tenets session create feature-123
tenets instill --session feature-123 --add-file src/main.py
tenets distill "implement feature" --session feature-123
```

### Makefiles

```makefile
# Makefile integration
.PHONY: context
context:
	@tenets distill "$(PROMPT)" --format markdown > context.md

.PHONY: analyze
analyze:
	@tenets examine . --complexity --threshold 10

.PHONY: deps
deps:
	@tenets viz deps --output architecture.svg
```

## CI/CD Integration

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
        run: pip install tenets[all]

      - name: Generate Context
        run: |
          tenets distill "${{ github.event.pull_request.title }}" \
            --format markdown > pr_context.md

      - name: Check Complexity
        run: |
          tenets examine . --complexity --threshold 15 \
            --format json > complexity.json

      - name: Upload Artifacts
        uses: actions/upload-artifact@v2
        with:
          name: analysis-results
          path: |
            pr_context.md
            complexity.json
```

### GitLab CI

```yaml
analyze:
  stage: test
  script:
    - pip install tenets
    - tenets examine . --complexity --threshold 10
    - tenets viz deps --format html -o deps.html
  artifacts:
    paths:
      - deps.html
    expire_in: 1 week
```

### Jenkins

```groovy
pipeline {
    agent any
    stages {
        stage('Analysis') {
            steps {
                sh 'pip install tenets'
                sh 'tenets distill "${CHANGE_TITLE}" --format html -o context.html'
                publishHTML([
                    reportDir: '.',
                    reportFiles: 'context.html',
                    reportName: 'Code Context'
                ])
            }
        }
    }
}
```

## IDE Integration

### VS Code Extension

```javascript
// Extension integration example
const { exec } = require('child_process');

function generateContext(prompt) {
    return new Promise((resolve, reject) => {
        exec(`tenets distill "${prompt}" --format json`,
            (error, stdout, stderr) => {
                if (error) reject(error);
                else resolve(JSON.parse(stdout));
            }
        );
    });
}

// Use in extension
vscode.commands.registerCommand('tenets.generateContext', async () => {
    const prompt = await vscode.window.showInputBox({
        prompt: 'Enter context prompt'
    });

    const result = await generateContext(prompt);
    // Process result...
});
```

### Vim Integration

```vim
" .vimrc configuration
function! TenetsContext(prompt)
    let output = system('tenets distill "' . a:prompt . '" --format markdown')
    new
    setlocal buftype=nofile
    call setline(1, split(output, '\n'))
endfunction

command! -nargs=1 Tenets call TenetsContext(<q-args>)
```

## API Endpoints (Future)

### REST API Design

```yaml
openapi: 3.0.0
info:
  title: Tenets API
  version: 1.0.0

paths:
  /distill:
    post:
      summary: Generate context
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                prompt:
                  type: string
                path:
                  type: string
                options:
                  type: object
      responses:
        200:
          description: Success
          content:
            application/json:
              schema:
                type: object
                properties:
                  content:
                    type: string
                  token_count:
                    type: integer
                  files:
                    type: array
```

## Plugin System

### Creating Plugins

```python
from tenets.plugins import Plugin, hook

class CustomPlugin(Plugin):
    """Example custom plugin."""

    @hook('pre_rank')
    def modify_ranking(self, files, context):
        """Modify files before ranking."""
        # Custom logic
        return files

    @hook('post_distill')
    def process_output(self, result):
        """Process distilled output."""
        # Custom processing
        return result

# Register plugin
from tenets import register_plugin
register_plugin(CustomPlugin())
```

### Available Hooks

- `pre_discover`: Before file discovery
- `post_discover`: After file discovery
- `pre_rank`: Before ranking
- `post_rank`: After ranking
- `pre_distill`: Before distillation
- `post_distill`: After distillation
- `pre_output`: Before output formatting
- `post_output`: After output formatting

## Integration Patterns

### 1. AI Assistant Integration

```python
# OpenAI GPT Integration
import openai
from tenets import Tenets

def get_ai_response(user_query):
    # Generate context
    t = Tenets()
    context = t.distill(user_query, max_tokens=4000)

    # Send to AI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": f"Context:\n{context.content}\n\nQuery: {user_query}"}
        ]
    )

    return response.choices[0].message.content
```

### 2. Documentation Generation

```python
from tenets import Tenets
from tenets.utils.markdown import MarkdownGenerator

def generate_docs():
    t = Tenets()

    # Analyze codebase
    result = t.distill("document public APIs",
                      include_patterns=["*.py"],
                      exclude_patterns=["test_*.py"])

    # Generate documentation
    gen = MarkdownGenerator()
    docs = gen.generate_api_docs(result)

    with open("API_DOCS.md", "w") as f:
        f.write(docs)
```

### 3. Code Review Automation

```python
def automated_review(pr_files):
    t = Tenets()

    issues = []
    for file in pr_files:
        # Check complexity
        complexity = t.examine(file, metric="complexity")
        if complexity > 15:
            issues.append(f"{file}: High complexity ({complexity})")

        # Check for patterns
        result = t.distill(f"find security issues in {file}")
        if "password" in result.content.lower():
            issues.append(f"{file}: Potential security issue")

    return issues
```

## Webhook Integration

### GitHub Webhooks

```python
from flask import Flask, request
from tenets import Tenets

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def github_webhook():
    payload = request.json

    if payload['action'] == 'opened':
        pr_title = payload['pull_request']['title']

        # Generate context for PR
        t = Tenets()
        context = t.distill(pr_title)

        # Post as comment
        post_github_comment(
            repo=payload['repository']['full_name'],
            pr=payload['number'],
            comment=f"## Context\n{context.content}"
        )

    return 'OK'
```

## Performance Considerations

### Batch Processing

```python
# Process multiple prompts efficiently
from tenets import Tenets

t = Tenets()
prompts = ["fix auth bug", "optimize database", "add caching"]

results = []
for prompt in prompts:
    result = t.distill(prompt, use_cache=True)
    results.append(result)
```

### Async Integration

```python
import asyncio
from tenets import AsyncTenets

async def process_requests(prompts):
    tenets = AsyncTenets()

    tasks = [
        tenets.distill(prompt)
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks)
    return results
```

## Error Handling

```python
from tenets import Tenets, TenetsError

try:
    t = Tenets()
    result = t.distill("analyze code")
except TenetsError as e:
    print(f"Error: {e}")
    # Handle specific errors
    if e.code == "TOKEN_LIMIT_EXCEEDED":
        # Retry with lower limit
        result = t.distill("analyze code", max_tokens=50000)
```
