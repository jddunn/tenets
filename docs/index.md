---
template: home.html
title: Tenets - Context that feeds your prompts
hide:
  - navigation
  - toc
---

<div class="victorian-hero">
  <div class="lantern-container">
    <div class="lantern-glow"></div>
    <!-- Use the light icon with transparency for improved contrast -->
    <img src="logos/tenets_light_icon_transparent.png" alt="Tenets" class="lantern-icon">
  </div>
  <h1 class="hero-title">
    <span class="typewriter" data-text="tenets"></span>
  </h1>
  <p class="hero-tagline">A powerful CLI & Python library for context that feeds your prompts</p>

  <!-- Elegant minimal install copy at top right -->
  <div class="hero-quick-install">
    <code>pip install tenets</code>
    <button class="copy-btn-mini" data-clipboard-text="pip install tenets" aria-label="Copy">
      <svg viewBox="0 0 16 16" width="14" height="14">
        <path fill="currentColor" d="M13 0H6a2 2 0 0 0-2 2 2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2 2 2 0 0 0 2-2V2a2 2 0 0 0-2-2zm0 13V4a2 2 0 0 0-2-2H5a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1zM3 4a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4z"/>
      </svg>
    </button>
  </div>

  <!-- Installation command at the top -->
  <div class="hero-install">
    <code>pip install tenets</code>
    <button class="copy-btn minimal" data-clipboard-text="pip install tenets" aria-label="Copy command">
      <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
        <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
      </svg>
    </button>
  </div>
  
  <!-- Elegant feature list -->
  <div class="hero-features">
    <div class="feature-line">
      <span class="feature-icon">✦</span>
      <span>Command-line tool and Python API for intelligent code aggregation and analysis</span>
    </div>
    <div class="feature-line">
      <span class="feature-icon">✦</span>
      <span>Surface relevant files • Build optimal context • Track velocity</span>
    </div>
    <div class="feature-line">
      <span class="feature-icon">✦</span>
      <span>Examine code quality metrics • Evaluate and improve your codebase</span>
    </div>
    <div class="feature-line">
      <span class="feature-icon">✦</span>
      <span>All processing stays local — your code never leaves your machine</span>
    </div>
  </div>
  <div class="hero-actions">
    <a href="quickstart/" class="btn-primary">
      <span class="btn-icon">
        <!-- Rocket icon (upward arrow) as inline SVG -->
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="#fff" stroke-width="2" fill="none" aria-hidden="true">
          <path d="M12 2 L12 22" />
          <path d="M5 9 L12 2 L19 9" />
        </svg>
      </span>
      Quick Start
    </a>
    <a href="https://github.com/jddunn/tenets" class="btn-secondary">
      <span class="btn-icon">
        <svg class="github-icon" viewBox="0 0 24 24">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
      </span>
      View on GitHub
    </a>
  </div>

  <div class="terminal-showcase">
    <div class="terminal">
      <div class="terminal-header">
        <div class="terminal-buttons">
          <span class="terminal-button red"></span>
          <span class="terminal-button yellow"></span>
          <span class="terminal-button green"></span>
        </div>
        <div class="terminal-title">Terminal</div>
      </div>
      <div class="terminal-body">
        <pre><code class="language-bash"><span class="prompt">$</span> pip install tenets
<span class="prompt">$</span> tenets distill "add mistral api to summarizer"
<span class="output">✨ Finding relevant files...
📊 Ranking by importance...
📦 Aggregating context (251,146 tokens)
✅ Context ready for your LLM!</span></code></pre>
      </div>
    </div>
  </div>
</div>

<div class="features-section">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>Illuminating Features</h2>
    <div class="ornament right"></div>
  </div>

  <!-- Why Tenets Section -->
  <div class="why-tenets-section">
    <div class="section-header">
      <div class="ornament left"></div>
      <h2>Why&nbsp;Tenets?</h2>
      <div class="ornament right"></div>
    </div>
    <div class="why-tenets-grid">
      <div class="why-card neumorphic">
        <div class="why-icon">
          <!-- Lightbulb icon -->
          <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
            <circle cx="12" cy="10" r="6"/>
            <path d="M12 16 V20"/>
            <line x1="9" y1="20" x2="15" y2="20"/>
          </svg>
        </div>
        <h3>Context on Demand</h3>
        <p>Stop hunting for files. Tenets discovers, ranks and assembles your code for you—so you can focus on solving the problem.</p>
      </div>
      <div class="why-card neumorphic">
        <div class="why-icon">
          <!-- Magnifying glass icon -->
          <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
            <circle cx="11" cy="11" r="7"/>
            <line x1="16.65" y1="16.65" x2="21" y2="21"/>
          </svg>
        </div>
        <h3>Deeper Insight</h3>
        <p>Visualize dependencies, uncover complexity hotspots and track velocity trends. Know your codebase like never before.</p>
      </div>
      <div class="why-card neumorphic">
        <div class="why-icon">
          <!-- Lock icon -->
          <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
            <rect x="6" y="10" width="12" height="10" rx="2"/>
            <path d="M8 10 V7 C8 4.8 10 3 12 3 C14 3 16 4.8 16 7 V10"/>
            <circle cx="12" cy="15" r="2"/>
          </svg>
        </div>
        <h3>Local &amp; Private</h3>
        <p>Your source never leaves your machine. With zero external API calls, Tenets keeps your intellectual property safe.</p>
      </div>
      <div class="why-card neumorphic">
        <div class="why-icon">
          <!-- Gear icon -->
          <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
            <circle cx="12" cy="12" r="3"/>
            <path d="M12 2 V4"/>
            <path d="M12 20 V22"/>
            <path d="M4.93 4.93 L6.34 6.34"/>
            <path d="M17.66 17.66 L19.07 19.07"/>
            <path d="M2 12 H4"/>
            <path d="M20 12 H22"/>
            <path d="M4.93 19.07 L6.34 17.66"/>
            <path d="M17.66 6.34 L19.07 4.93"/>
          </svg>
        </div>
        <h3>Flexible &amp; Extensible</h3>
        <p>Dial the ranking algorithm, expand the token budget and add plugins when you need more. Tenets grows with you.</p>
      </div>
    </div>
  </div>

  <!-- Architecture Section -->
  <div class="architecture-section">
    <div class="section-header">
      <div class="ornament left"></div>
      <h2>Architecture at a Glance</h2>
      <div class="ornament right"></div>
    </div>
    <div class="architecture-wrapper">
      <svg class="architecture-svg" viewBox="0 0 900 320" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrow-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"></path>
          </marker>
        </defs>

        <!-- Boxes -->
        <rect x="20" y="120" width="140" height="70" rx="8" class="arch-box"></rect>
        <text x="90" y="155" class="arch-label" text-anchor="middle">Input</text>

        <rect x="200" y="120" width="140" height="70" rx="8" class="arch-box"></rect>
        <text x="270" y="155" class="arch-label" text-anchor="middle">Scanner</text>

        <rect x="380" y="120" width="140" height="70" rx="8" class="arch-box"></rect>
        <text x="450" y="155" class="arch-label" text-anchor="middle">Analyzer</text>

        <rect x="560" y="120" width="140" height="70" rx="8" class="arch-box"></rect>
        <text x="630" y="155" class="arch-label" text-anchor="middle">Ranker</text>

        <rect x="740" y="120" width="140" height="70" rx="8" class="arch-box"></rect>
        <text x="810" y="155" class="arch-label" text-anchor="middle">Aggregator</text>

        <!-- Arrows -->
        <line x1="160" y1="155" x2="200" y2="155" class="arch-arrow" marker-end="url(#arrow-head)"></line>
        <line x1="340" y1="155" x2="380" y2="155" class="arch-arrow" marker-end="url(#arrow-head)"></line>
        <line x1="520" y1="155" x2="560" y2="155" class="arch-arrow" marker-end="url(#arrow-head)"></line>
        <line x1="700" y1="155" x2="740" y2="155" class="arch-arrow" marker-end="url(#arrow-head)"></line>
      </svg>
    </div>
    <p class="architecture-note">Tenets flows your query through a pipeline of scanners, analyzers, rankers and aggregators, delivering context precisely tailored to your task.</p>
  </div>

  <div class="features-grid">
    <div class="feature-card neumorphic">
      <div class="feature-icon">
        <!-- Target icon -->
        <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
          <circle cx="12" cy="12" r="8"/>
          <circle cx="12" cy="12" r="3"/>
          <line x1="12" y1="4" x2="12" y2="7"/>
          <line x1="12" y1="20" x2="12" y2="17"/>
          <line x1="4" y1="12" x2="7" y2="12"/>
          <line x1="20" y1="12" x2="17" y2="12"/>
        </svg>
      </div>
      <h3>Intelligent Context</h3>
      <p>Multi-factor ranking finds exactly what you need. No more manual file hunting.</p>
    </div>

    <div class="feature-card neumorphic">
      <div class="feature-icon">
        <!-- Shield icon -->
        <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
          <path d="M12 2l8 4v5c0 5.5-3.5 10.7-8 12-4.5-1.3-8-6.5-8-12V6l8-4z"/>
        </svg>
      </div>
      <h3>100% Local</h3>
      <p>Your code never leaves your machine. Complete privacy, zero API calls.</p>
    </div>

    <div class="feature-card neumorphic">
      <div class="feature-icon">
        <!-- Lightning bolt icon -->
        <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
          <polyline points="13 2 3 14 11 14 11 22 21 10 13 10 13 2"/>
        </svg>
      </div>
      <h3>Lightning Fast</h3>
      <p>Analyzes thousands of files in seconds with intelligent caching.</p>
    </div>

    <div class="feature-card neumorphic">
      <div class="feature-icon">
        <!-- Compass star icon -->
        <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
          <circle cx="12" cy="12" r="9"/>
          <line x1="12" y1="4" x2="12" y2="20"/>
          <line x1="4" y1="12" x2="20" y2="12"/>
          <line x1="8" y1="8" x2="16" y2="16"/>
          <line x1="16" y1="8" x2="8" y2="16"/>
        </svg>
      </div>
      <h3>Guiding Principles</h3>
      <p>Add persistent instructions that maintain consistency across AI sessions.</p>
    </div>

    <div class="feature-card neumorphic">
      <div class="feature-icon">
        <!-- Bar chart icon -->
        <svg viewBox="0 0 24 24" width="36" height="36" stroke="none" fill="#f59e0b">
          <rect x="4" y="10" width="3" height="10" rx="1"></rect>
          <rect x="10" y="6" width="3" height="14" rx="1"></rect>
          <rect x="16" y="13" width="3" height="7" rx="1"></rect>
        </svg>
      </div>
      <h3>Code Intelligence</h3>
      <p>Visualize dependencies, track velocity, identify hotspots at a glance.</p>
    </div>

    <div class="feature-card neumorphic">
      <div class="feature-icon">
        <!-- Upward arrow / rocket icon -->
        <svg viewBox="0 0 24 24" width="36" height="36" stroke="#f59e0b" stroke-width="2" fill="none">
          <path d="M12 2 L12 22"/>
          <path d="M5 9 L12 2 L19 9"/>
        </svg>
      </div>
      <h3>Zero Config</h3>
      <p>Works instantly with smart defaults. Just install and start distilling.</p>
    </div>
  </div>
</div>

<div class="workflow-section">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>How It Works</h2>
    <div class="ornament right"></div>
  </div>

  <!-- Enhanced Pipeline Visualization -->
  <div class="pipeline-grid">
    <div class="pipeline-card">
      <div class="pipeline-header">
        <span class="pipeline-icon">📝</span>
        <h4>1. Input & Parsing</h4>
      </div>
      <ul class="pipeline-details">
        <li><strong>Natural language processing</strong> extracts intent, keywords, and entities</li>
        <li><strong>Temporal parsing</strong> identifies time-based queries ("last week", "since Monday")</li>
        <li><strong>External reference detection</strong> automatically fetches GitHub issues, JIRA tickets, Linear tasks</li>
        <li><strong>Intent detection</strong> classifies prompts (implement, debug, refactor, test, document)</li>
        <li><strong>Test inclusion logic</strong> automatically includes/excludes test files based on context</li>
      </ul>
    </div>

    <div class="pipeline-card">
      <div class="pipeline-header">
        <span class="pipeline-icon">🔍</span>
        <h4>2. Scan & Discover</h4>
      </div>
      <ul class="pipeline-details">
        <li><strong>Parallel file discovery</strong> respects .gitignore and custom filters</li>
        <li><strong>Language detection</strong> with specialized analyzers for 15+ languages</li>
        <li><strong>Incremental scanning</strong> caches unchanged files for speed</li>
        <li><strong>Pattern matching</strong> identifies file types and purposes</li>
      </ul>
    </div>

    <div class="pipeline-card">
      <div class="pipeline-header">
        <span class="pipeline-icon">🧠</span>
        <h4>3. Analyze & Extract</h4>
      </div>
      <ul class="pipeline-details">
        <li><strong>AST parsing</strong> extracts classes, functions, imports, symbols</li>
        <li><strong>Dependency graph building</strong> maps import relationships</li>
        <li><strong>Complexity metrics</strong> (cyclomatic, cognitive, Halstead)</li>
        <li><strong>Git history mining</strong> for recency and change frequency</li>
        <li><strong>Content tokenization</strong> for similarity calculations</li>
      </ul>
    </div>

    <div class="pipeline-card">
      <div class="pipeline-header">
        <span class="pipeline-icon">📊</span>
        <h4>4. Rank & Score</h4>
      </div>
      <ul class="pipeline-details">
        <li><strong>Multi-factor scoring</strong> combines 8+ ranking signals:
          <ul style="margin-top: 0.5em;">
            <li>Keyword matching with position weighting</li>
            <li>BM25/TF-IDF text similarity</li>
            <li>Semantic similarity (ML mode)</li>
            <li>Path and structure relevance</li>
            <li>Import centrality in dependency graph</li>
            <li>Git activity signals</li>
            <li>Code pattern matching</li>
            <li>Task-specific boosts</li>
          </ul>
        </li>
        <li><strong>Adaptive weighting</strong> based on detected intent</li>
        <li><strong>Three ranking modes:</strong>
          <ul style="margin-top: 0.5em;">
            <li><em>Fast:</em> Keyword & path matching (< 5s)</li>
            <li><em>Balanced:</em> BM25 + structure analysis (default)</li>
            <li><em>Thorough:</em> ML embeddings + patterns (4x slower)</li>
          </ul>
        </li>
      </ul>
    </div>

    <div class="pipeline-card">
      <div class="pipeline-header">
        <span class="pipeline-icon">🎯</span>
        <h4>5. Tenets Injection</h4>
      </div>
      <ul class="pipeline-details">
        <li><strong>Guiding principles</strong> maintain consistency across sessions</li>
        <li><strong>Context drift prevention</strong> in long AI conversations</li>
        <li><strong>Adaptive injection frequency</strong> based on interaction patterns</li>
        <li><strong>Session persistence</strong> for multi-turn workflows</li>
      </ul>
    </div>

    <div class="pipeline-card">
      <div class="pipeline-header">
        <span class="pipeline-icon">📦</span>
        <h4>6. Aggregate & Output</h4>
      </div>
      <ul class="pipeline-details">
        <li><strong>Token-aware budgeting</strong> for different LLM models</li>
        <li><strong>Intelligent summarization</strong> preserves critical code structure</li>
        <li><strong>Multiple output formats:</strong> Markdown, XML (Claude), HTML, JSON</li>
        <li><strong>Progressive enhancement</strong> - summaries only if needed</li>
        <li><strong>Pinned files</strong> guarantee critical files are always included</li>
      </ul>
    </div>
  </div>

  <style>
    .pipeline-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.5rem;
      margin: 2rem 0;
    }

    .pipeline-card {
      background: var(--md-code-bg-color);
      border: 1px solid var(--md-default-fg-color--lightest);
      border-radius: 8px;
      padding: 1.5rem;
    }

    .pipeline-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    .pipeline-icon {
      font-size: 1.5rem;
    }

    .pipeline-header h4 {
      margin: 0;
      color: var(--md-primary-fg-color);
    }

    .pipeline-details {
      margin: 0;
      padding-left: 1.25rem;
      font-size: 0.9rem;
      line-height: 1.6;
    }

    .pipeline-details li {
      margin: 0.5rem 0;
    }

    .pipeline-details ul {
      padding-left: 1rem;
      margin-top: 0.25rem;
    }

    .pipeline-details ul li {
      margin: 0.25rem 0;
      font-size: 0.85rem;
      opacity: 0.9;
    }
  </style>

  <div class="workflow-diagram">
    <div class="workflow-step">
      <div class="step-number">1</div>
      <h4>Parse</h4>
      <p>Understands intent & extracts context</p>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
      <div class="step-number">2</div>
      <h4>Scan</h4>
      <p>Discovers files in parallel</p>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
      <div class="step-number">3</div>
      <h4>Analyze</h4>
      <p>Extracts AST & metrics</p>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
      <div class="step-number">4</div>
      <h4>Rank</h4>
      <p>Multi-factor scoring</p>
    </div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-step">
      <div class="step-number">5</div>
      <h4>Output</h4>
      <p>Token-aware aggregation</p>
    </div>
  </div>
</div>

<div class="see-action">
  <div class="section-header">
    <div class="ornament left"></div>
    <h2>See it in action</h2>
    <div class="ornament right"></div>
  </div>

  <!-- Example 1: CLI vs Output -->
  <div class="see-grid">
    <div class="see-code">
      <div class="see-header">
        <div class="see-dots">
          <span class="see-dot red"></span>
          <span class="see-dot yellow"></span>
          <span class="see-dot green"></span>
        </div>
        <div class="see-title">CLI</div>
      </div>
      <div class="see-body">

```bash
$ tenets distill "implement OAuth2 authentication"
✨ Finding relevant files...
📊 Ranking by importance...
📦 Aggregating context (45,231 tokens)
✅ Context ready for your LLM!
```

      </div>
    </div>
  <div class="see-output" data-shot="context-building" data-caption="Context builder assembling the most relevant files">
      <div class="see-header">
        <div class="see-dots">
          <span class="see-dot red"></span>
          <span class="see-dot yellow"></span>
          <span class="see-dot green"></span>
        </div>
        <div class="see-title">Output</div>
      </div>
      <div class="see-body">
        <div class="screenshots-container">
          <div class="screenshot-item">
            <img src="assets/images/screenshots/context-building-1.png" alt="Context building - Analyzing files" loading="lazy" onerror="this.onerror=null; this.src='logos/tenets_dark_icon_transparent.png'" />
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Example 2: Rank Command vs Output -->
  <div class="see-grid">
    <div class="see-code">
      <div class="see-header">
        <div class="see-dots">
          <span class="see-dot red"></span>
          <span class="see-dot yellow"></span>
          <span class="see-dot green"></span>
        </div>
        <div class="see-title">Rank Files</div>
      </div>
      <div class="see-body">

```bash
$ tenets rank "fix authentication bug" --top 10 --factors
🔍 Scanning codebase...
📊 Ranking files by relevance...

1. src/auth/service.py - Score: 0.892
   - semantic_similarity: 85%
   - keyword_match: 92%
   - import_centrality: 78%
   
2. src/auth/middleware.py - Score: 0.834
   - semantic_similarity: 79%
   - keyword_match: 88%
   - import_centrality: 65%
```

      </div>
    </div>
    <div class="see-output" data-shot="ranking" data-caption="File ranking with relevance factors">
      <div class="see-header">
        <div class="see-dots">
          <span class="see-dot red"></span>
          <span class="see-dot yellow"></span>
          <span class="see-dot green"></span>
        </div>
        <div class="see-title">Tree View</div>
      </div>
      <div class="see-body">

```bash
$ tenets rank "add caching" --tree --scores
📁 Ranked Files
├── 📂 src/
│   ├── 📄 cache_manager.py [0.892]
│   ├── 📄 redis_client.py [0.834]
│   └── 📄 config.py [0.756]
├── 📂 src/api/
│   ├── 📄 endpoints.py [0.723]
│   └── 📄 middleware.py [0.689]
└── 📂 tests/
    └── 📄 test_cache.py [0.534]
```

      </div>
    </div>
  </div>

  <!-- Example 3: Python API vs Output -->
  <div class="see-grid">
    <div class="see-code">
      <div class="see-header">
        <div class="see-dots">
          <span class="see-dot red"></span>
          <span class="see-dot yellow"></span>
          <span class="see-dot green"></span>
        </div>
        <div class="see-title">Python</div>
      </div>
      <div class="see-body">

```python
from tenets import Tenets
t = Tenets()
result = t.distill(
    prompt="map request lifecycle"
)
print(result.context[:500])  # First 500 chars
```

      </div>
    </div>
  <div class="see-output" data-shot="python-summary" data-caption="Python API summary output">
      <div class="see-header">
        <div class="see-dots">
          <span class="see-dot red"></span>
          <span class="see-dot yellow"></span>
          <span class="see-dot green"></span>
        </div>
        <div class="see-title">Output</div>
      </div>
      <div class="see-body">
        <div class="shot-frame">
          <img src="assets/images/screenshots/code-analysis.png" alt="Code analysis" loading="lazy" onerror="this.onerror=null; this.src='logos/tenets_dark_icon_transparent.png'" />
        </div>
      </div>
    </div>
  </div>

  <!-- Example 3: Sessions vs Output -->
  <div class="see-grid">
    <div class="see-code">
      <div class="see-header">
        <div class="see-dots">
          <span class="see-dot red"></span>
          <span class="see-dot yellow"></span>
          <span class="see-dot green"></span>
        </div>
        <div class="see-title">Sessions</div>
      </div>
      <div class="see-body">

```python
# Sessions are managed through distill parameters
ctx = t.distill("design payment flow", session_name="checkout-flow")
# Pin files through pin_file method
t.pin_file("payment.py")
t.pin_file("stripe.py")
ctx = t.distill("add refund support", session_name="checkout-flow")
```

      </div>
    </div>
  <div class="see-output" data-shot="sessions-flow" data-caption="Sessions flow with iterative context">
      <div class="see-header">
        <div class="see-dots">
          <span class="see-dot red"></span>
          <span class="see-dot yellow"></span>
          <span class="see-dot green"></span>
        </div>
        <div class="see-title">Output</div>
      </div>
      <div class="see-body">
        <div class="shot-frame">
          <img src="assets/images/screenshots/sessions-1.png" alt="Sessions flow" loading="lazy" onerror="this.onerror=null; this.src='logos/tenets_dark_icon_transparent.png'" />
        </div>
      </div>
    </div>
  </div>
</div>

<div class="cta-section">
  <div class="cta-content neumorphic">
    <h2>Ready to illuminate your codebase?</h2>
    <p>Join thousands of developers building better with Tenets.</p>
    <div class="cta-actions">
      <a href="installation/" class="btn-primary large">
        Get Started Now
      </a>
      <div class="install-command">
        <code>pip install tenets</code>
        <button class="copy-btn" data-clipboard-text="pip install tenets">
          <!-- Clipboard icon as inline SVG -->
          <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" aria-hidden="true">
            <rect x="6" y="5" width="12" height="15" rx="2" ry="2" />
            <rect x="9" y="2" width="6" height="3" rx="1" ry="1" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</div>
