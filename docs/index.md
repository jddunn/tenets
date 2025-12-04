---
template: home.html
title: Tenets - MCP Server for AI Coding Assistants
hide:
  - navigation
  - toc
---

<div class="victorian-hero">
  <div class="lantern-container">
    <div class="lantern-glow"></div>
    <img src="logos/tenets_light_icon_transparent.png" alt="Tenets" class="lantern-icon">
  </div>
  <h1 class="hero-title">
    <span class="typewriter" data-text="tenets"></span>
  </h1>
  <p class="hero-tagline">MCP Server for AI Coding Assistants</p>
  <p class="hero-subtitle">Give Cursor, Claude & Windsurf the context they need</p>

  <!-- Single install command - MCP focused -->
  <div class="hero-install">
    <code>pip install tenets[mcp]</code>
    <button class="copy-btn-inline" data-clipboard-text="pip install tenets[mcp]" aria-label="Copy">
      <svg viewBox="0 0 16 16" width="14" height="14"><path fill="currentColor" d="M13 0H6a2 2 0 0 0-2 2 2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2 2 2 0 0 0 2-2V2a2 2 0 0 0-2-2zm0 13V4a2 2 0 0 0-2-2H5a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1zM3 4a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4z"/></svg>
    </button>
  </div>

  <div class="hero-actions">
    <a href="MCP/" class="btn-primary">
      <span class="btn-icon">
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="#fff" stroke-width="2" fill="none">
          <circle cx="12" cy="12" r="4"/>
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4"/>
        </svg>
      </span>
      Setup MCP
    </a>
    <a href="https://github.com/jddunn/tenets" class="btn-secondary">
      <span class="btn-icon">
        <svg class="github-icon" viewBox="0 0 24 24">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
      </span>
      GitHub
    </a>
  </div>
</div>

<!-- MCP Setup Section - RIGHT AFTER HERO -->
<div class="mcp-setup-section" style="padding: 4rem 2rem; background: linear-gradient(180deg, rgba(245, 158, 11, 0.05) 0%, transparent 100%);">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>30-Second Setup</h2>
    <div class="ornament right"></div>
  </div>

  <div style="max-width: 900px; margin: 0 auto;">
    
    <!-- Step 1: Install -->
    <div style="display: flex; align-items: flex-start; gap: 1.5rem; margin-bottom: 2rem;">
      <div style="background: #f59e0b; color: #000; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">1</div>
      <div style="flex: 1;">
        <h3 style="margin: 0 0 0.5rem;">Install</h3>

```bash
pip install tenets[mcp]
```

</div>
    </div>

    <!-- Step 2: Configure -->
    <div style="display: flex; align-items: flex-start; gap: 1.5rem; margin-bottom: 2rem;">
      <div style="background: #f59e0b; color: #000; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">2</div>
      <div style="flex: 1;">
        <h3 style="margin: 0 0 0.5rem;">Add to your AI tool</h3>
        
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">

<div style="background: var(--md-code-bg-color); border-radius: 8px; overflow: hidden; border: 1px solid rgba(245, 158, 11, 0.2);">
<div style="padding: 0.75rem 1rem; background: rgba(245, 158, 11, 0.1); font-weight: 600; font-size: 0.9rem;">Cursor</div>
<div style="padding: 1rem;">

```json
{
  "tenets": {
    "command": "tenets-mcp"
  }
}
```

<small style="opacity: 0.7;">Settings → MCP Servers → Add</small>
</div>
</div>

<div style="background: var(--md-code-bg-color); border-radius: 8px; overflow: hidden; border: 1px solid rgba(245, 158, 11, 0.2);">
<div style="padding: 0.75rem 1rem; background: rgba(245, 158, 11, 0.1); font-weight: 600; font-size: 0.9rem;">Claude Desktop</div>
<div style="padding: 1rem;">

```json
{
  "mcpServers": {
    "tenets": {
      "command": "tenets-mcp"
    }
  }
}
```

<small style="opacity: 0.7;">~/Library/Application Support/Claude/claude_desktop_config.json</small>
</div>
</div>

</div>
      </div>
    </div>

    <!-- Step 3: Use -->
    <div style="display: flex; align-items: flex-start; gap: 1.5rem;">
      <div style="background: #f59e0b; color: #000; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;">3</div>
      <div style="flex: 1;">
        <h3 style="margin: 0 0 0.5rem;">Ask your AI</h3>
        <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 8px; border-left: 3px solid #f59e0b;">
          <em>"Use tenets to find the authentication code"</em><br><br>
          <em>"Ask tenets to build context for implementing payments"</em><br><br>
          <em>"Have tenets analyze the database models"</em>
        </div>
      </div>
    </div>

  </div>
</div>

<!-- What Your AI Gets -->
<div class="mcp-response-section" style="padding: 4rem 2rem;">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>What Your AI Gets</h2>
    <div class="ornament right"></div>
  </div>
  
  <p style="text-align: center; max-width: 600px; margin: 0 auto 3rem; opacity: 0.9;">
    Tenets finds, ranks, and delivers the most relevant code — so your AI understands your codebase.
  </p>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 2rem; max-width: 1100px; margin: 0 auto;">
    
    <!-- Request -->
    <div style="background: var(--md-code-bg-color); border-radius: 12px; overflow: hidden; border: 1px solid rgba(245, 158, 11, 0.2);">
      <div style="padding: 1rem 1.5rem; background: rgba(245, 158, 11, 0.1); border-bottom: 1px solid rgba(245, 158, 11, 0.1);">
        <strong>🔧 Your AI calls Tenets</strong>
      </div>
      <div style="padding: 1.5rem;">

```json
{
  "tool": "distill",
  "arguments": {
    "prompt": "implement Stripe webhooks",
    "mode": "balanced"
  }
}
```

</div>
    </div>

    <!-- Response -->
    <div style="background: var(--md-code-bg-color); border-radius: 12px; overflow: hidden; border: 1px solid rgba(245, 158, 11, 0.2);">
      <div style="padding: 1rem 1.5rem; background: rgba(245, 158, 11, 0.1); border-bottom: 1px solid rgba(245, 158, 11, 0.1);">
        <strong>📦 Tenets returns ranked context</strong>
      </div>
      <div style="padding: 1.5rem;">

```json
{
  "files_included": 8,
  "total_tokens": 32000,
  "top_files": [
    "src/payments/stripe.py",
    "src/payments/webhooks.py", 
    "src/models/order.py"
  ],
  "context": "# Payment System\n\n## stripe.py\n```python\nclass StripeClient:\n..."
}
```

</div>
    </div>
  </div>
</div>

<!-- Available MCP Tools -->
<div class="mcp-tools-section" style="padding: 4rem 2rem; background: linear-gradient(180deg, transparent 0%, rgba(245, 158, 11, 0.03) 100%);">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>13 MCP Tools</h2>
    <div class="ornament right"></div>
  </div>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; max-width: 1200px; margin: 2rem auto;">
    
    <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
      <code style="font-size: 1.1rem; color: #f59e0b;">distill</code>
      <p style="margin: 0.75rem 0 0; opacity: 0.85;">Build optimized code context for any task. The main tool your AI will use.</p>
    </div>

    <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
      <code style="font-size: 1.1rem; color: #f59e0b;">rank_files</code>
      <p style="margin: 0.75rem 0 0; opacity: 0.85;">Preview which files are most relevant without loading content.</p>
    </div>

    <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
      <code style="font-size: 1.1rem; color: #f59e0b;">examine</code>
      <p style="margin: 0.75rem 0 0; opacity: 0.85;">Analyze codebase structure, dependencies, and complexity.</p>
    </div>

    <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
      <code style="font-size: 1.1rem; color: #f59e0b;">session_create</code>
      <p style="margin: 0.75rem 0 0; opacity: 0.85;">Create persistent sessions that remember pinned files across requests.</p>
    </div>

    <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
      <code style="font-size: 1.1rem; color: #f59e0b;">session_pin_file</code>
      <p style="margin: 0.75rem 0 0; opacity: 0.85;">Pin critical files so they're always included in context.</p>
    </div>

    <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
      <code style="font-size: 1.1rem; color: #f59e0b;">tenet_add</code>
      <p style="margin: 0.75rem 0 0; opacity: 0.85;">Add coding guidelines that get injected into every context.</p>
    </div>

    <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
      <code style="font-size: 1.1rem; color: #f59e0b;">chronicle</code>
      <p style="margin: 0.75rem 0 0; opacity: 0.85;">Query git history — recent changes, active files, contributors.</p>
    </div>

    <div style="background: var(--md-code-bg-color); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.15);">
      <code style="font-size: 1.1rem; color: #f59e0b;">momentum</code>
      <p style="margin: 0.75rem 0 0; opacity: 0.85;">Track development velocity and sprint progress.</p>
    </div>

  </div>

  <p style="text-align: center; margin-top: 2rem;">
    <a href="MCP/#available-tools" style="color: #f59e0b; font-weight: 500;">See all 13 tools with examples →</a>
  </p>
</div>

<!-- How Ranking Works -->
<div class="ranking-section" style="padding: 4rem 2rem;">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>Intelligent Ranking</h2>
    <div class="ornament right"></div>
  </div>

  <p style="text-align: center; max-width: 700px; margin: 0 auto 3rem; opacity: 0.9;">
    Tenets uses 8+ signals to find the most relevant code for your task.
  </p>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; max-width: 1000px; margin: 0 auto;">
    <div style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
      <strong>BM25 Scoring</strong>
      <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Statistical text relevance</p>
    </div>
    <div style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔤</div>
      <strong>Keyword Match</strong>
      <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Direct term matching</p>
    </div>
    <div style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 2rem; margin-bottom: 0.5rem;">📁</div>
      <strong>Path Relevance</strong>
      <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Directory structure</p>
    </div>
    <div style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔗</div>
      <strong>Import Graph</strong>
      <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Dependency centrality</p>
    </div>
    <div style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 2rem; margin-bottom: 0.5rem;">📅</div>
      <strong>Git Recency</strong>
      <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Recent changes</p>
    </div>
    <div style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔥</div>
      <strong>Change Frequency</strong>
      <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Hot files</p>
    </div>
    <div style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧩</div>
      <strong>Complexity</strong>
      <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Code metrics</p>
    </div>
    <div style="text-align: center; padding: 1.5rem;">
      <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
      <strong>ML Embeddings</strong>
      <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Semantic similarity</p>
    </div>
  </div>
</div>

<!-- Why Tenets -->
<div class="why-section" style="padding: 4rem 2rem; background: linear-gradient(180deg, rgba(245, 158, 11, 0.03) 0%, transparent 100%);">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>Why Tenets?</h2>
    <div class="ornament right"></div>
  </div>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem; max-width: 1100px; margin: 2rem auto;">
    
    <div style="text-align: center; padding: 2rem;">
      <div style="font-size: 3rem; margin-bottom: 1rem;">🔒</div>
      <h3>100% Local</h3>
      <p style="opacity: 0.85;">All processing happens on your machine. Your code never leaves. Zero API calls to external services.</p>
    </div>

    <div style="text-align: center; padding: 2rem;">
      <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
      <h3>Fast</h3>
      <p style="opacity: 0.85;">Analyzes thousands of files in seconds. Intelligent caching makes repeat queries instant.</p>
    </div>

    <div style="text-align: center; padding: 2rem;">
      <div style="font-size: 3rem; margin-bottom: 1rem;">🎯</div>
      <h3>Accurate</h3>
      <p style="opacity: 0.85;">Multi-factor ranking beats naive keyword search. Your AI gets the right files, not just matching files.</p>
    </div>

  </div>
</div>

<!-- Also Works As -->
<div class="also-section" style="padding: 4rem 2rem;">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>Also Works As</h2>
    <div class="ornament right"></div>
  </div>

  <p style="text-align: center; max-width: 600px; margin: 0 auto 2rem; opacity: 0.9;">
    Not using MCP? Tenets also works as a CLI and Python library.
  </p>

  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 2rem; max-width: 900px; margin: 0 auto;">
    
    <div style="background: var(--md-code-bg-color); border-radius: 12px; overflow: hidden; border: 1px solid rgba(245, 158, 11, 0.2);">
      <div style="padding: 1rem 1.5rem; background: rgba(245, 158, 11, 0.1);">
        <strong>CLI</strong>
      </div>
      <div style="padding: 1.5rem;">

```bash
# Build context and copy to clipboard
tenets distill "implement OAuth" --copy

# See ranked files
tenets rank "fix auth bug" --top 10
```

</div>
    </div>

    <div style="background: var(--md-code-bg-color); border-radius: 12px; overflow: hidden; border: 1px solid rgba(245, 158, 11, 0.2);">
      <div style="padding: 1rem 1.5rem; background: rgba(245, 158, 11, 0.1);">
        <strong>Python</strong>
      </div>
      <div style="padding: 1.5rem;">

```python
from tenets import Tenets

t = Tenets()
result = t.distill("implement webhooks")
print(result.context)
```

</div>
    </div>
  </div>

  <p style="text-align: center; margin-top: 2rem;">
    <a href="CLI/" style="color: #f59e0b;">CLI Reference →</a> · <a href="api/" style="color: #f59e0b;">Python API →</a>
  </p>
</div>

<!-- Enterprise CTA Section -->
<div class="enterprise-section" style="padding: 4rem 2rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(251, 191, 36, 0.03) 100%); margin: 2rem 0;">
  <div style="max-width: 900px; margin: 0 auto; text-align: center;">
    <span style="background: linear-gradient(135deg, #f59e0b, #fbbf24); color: #1a1a1a; padding: 0.25rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">ENTERPRISE</span>
    <h2 style="margin: 1.5rem 0 1rem;">Need More?</h2>
    <p style="opacity: 0.9; margin-bottom: 2rem;">
      Tenets is 100% open source. For teams needing advanced features, we offer enterprise support.
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
      <div style="padding: 1.5rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔒</div>
        <strong>Privacy Redaction</strong>
        <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Auto-remove sensitive data</p>
      </div>
      <div style="padding: 1.5rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
        <strong>Audit Logging</strong>
        <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">Track AI interactions</p>
      </div>
      <div style="padding: 1.5rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔑</div>
        <strong>SSO Integration</strong>
        <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">SAML/OIDC for teams</p>
      </div>
      <div style="padding: 1.5rem; background: rgba(0,0,0,0.2); border-radius: 12px;">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🛡️</div>
        <strong>Air-Gapped</strong>
        <p style="font-size: 0.85rem; opacity: 0.8; margin: 0.5rem 0 0;">On-prem installation</p>
      </div>
    </div>
    <a href="https://manic.agency/contact" target="_blank" rel="noopener" style="display: inline-block; background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); color: #1a2332; padding: 0.875rem 2.5rem; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 1.1rem;">Contact manic.agency →</a>
  </div>
</div>

<!-- Final CTA -->
<div class="cta-section">
  <div class="cta-content neumorphic">
    <h2>Ready to supercharge your AI coding?</h2>
    <p>Install in 30 seconds. Works with Cursor, Claude, Windsurf, and any MCP client.</p>
    <div class="cta-actions">
      <a href="MCP/" class="btn-primary large">
        Setup Guide
      </a>
      <div class="install-command">
        <code>pip install tenets[mcp]</code>
        <button class="copy-btn" data-clipboard-text="pip install tenets[mcp]">
          <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none">
            <rect x="6" y="5" width="12" height="15" rx="2" ry="2" />
            <rect x="9" y="2" width="6" height="3" rx="1" ry="1" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</div>
