---
title: Architecture Overview
---

# Tenets Architecture Overview

## Quick Navigation

Choose your level of detail:

### **[System Overview](overview.md)**
High-level architecture, core principles, and system design philosophy.

### **[Core Systems](core-systems.md)**
Detailed breakdown of analysis engines, ranking systems, and processing pipelines.

### **[Data & Storage](data-storage.md)**
Session management, caching architecture, and persistence layers.

### **[Integration & APIs](integration.md)**
CLI architecture, Git integration, and extensibility systems.

### **[Performance & Deployment](performance.md)**
Performance architecture, scalability, and deployment strategies.

---

## What is Tenets?

Tenets is a sophisticated, **local-first code intelligence platform** that revolutionizes how developers interact with their codebases when working with AI assistants.

## Core Architecture Principles

### 1. **Local-First Processing**
All analysis, ranking, and context generation happens on the developer's machine. No code ever leaves the local environment.

### 2. **Progressive Enhancement**
The system provides value immediately with just Python installed, and scales up with optional dependencies.

### 3. **Intelligent Caching**
Every expensive operation is cached at multiple levels - memory, SQLite, disk, and specialized embedding caches.

### 4. **Configurable Intelligence**
Every aspect of ranking and analysis can be configured. Users can adjust factor weights, enable/disable features, and add custom functions.

### 5. **Streaming Architecture**
Uses streaming and incremental processing. Files are analyzed as discovered, rankings computed in parallel, results stream to users.

---

## 🗺️ High-Level System Flow

```mermaid
graph TB
    A[User Prompt] --> B[Prompt Parser]
    B --> C[Intent Detection]
    C --> D[File Discovery]
    D --> E[Code Analysis]
    E --> F[Relevance Ranking]
    F --> G[Context Assembly]
    G --> H[Output Generation]

    subgraph "Caching Layers"
        I[Memory Cache]
        J[SQLite DB]
        K[File System Cache]
    end

    E --> I
    F --> J
    G --> K

    style A fill:#e1f5fe
    style H fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#fce4ec
```

---

## 📚 Documentation Sections

| Section | What You'll Learn | Best For |
|---------|-------------------|----------|
| **[Overview](overview.md)** | System design, philosophy, key concepts | Product managers, architects |
| **[Core Systems](core-systems.md)** | Analysis engines, ML pipelines, ranking | Senior developers, integrators |
| **[Data & Storage](data-storage.md)** | Database design, caching, sessions | Backend developers, DBAs |
| **[Integration](integration.md)** | APIs, CLI, Git integration, plugins | DevOps, tool builders |
| **[Performance](performance.md)** | Scalability, optimization, deployment | Performance engineers, SREs |

---

## 🔍 Quick Architecture Facts

<div class="architecture-facts">
  <div class="fact-card">
    <h4>🏗️ Modular Design</h4>
    <p>20+ specialized modules working in harmony</p>
  </div>

  <div class="fact-card">
    <h4>⚡ Performance First</h4>
    <p>Multi-level caching, parallel processing, streaming</p>
  </div>

  <div class="fact-card">
    <h4>🔒 Privacy Focused</h4>
    <p>100% local processing, no data leaves your machine</p>
  </div>

  <div class="fact-card">
    <h4>🔧 Highly Configurable</h4>
    <p>Every ranking factor and feature can be tuned</p>
  </div>
</div>

---

## 🎯 Where to Start?

- **New to Tenets?** → [System Overview](overview.md)
- **Want technical details?** → [Core Systems](core-systems.md)
- **Building integrations?** → [Integration & APIs](integration.md)
- **Performance questions?** → [Performance & Deployment](performance.md)

<style>
.architecture-nav {
  background: var(--md-primary-fg-color--light);
  padding: 1rem;
  border-radius: 8px;
  margin: 1rem 0 2rem 0;
  text-align: center;
}

.arch-stats {
  display: flex;
  justify-content: center;
  gap: 2rem;
  flex-wrap: wrap;
}

.stat {
  font-weight: 600;
  color: var(--md-primary-fg-color);
}

.architecture-facts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin: 2rem 0;
}

.fact-card {
  background: var(--md-default-bg-color--light);
  padding: 1.5rem;
  border-radius: 8px;
  border: 1px solid var(--md-default-fg-color--lightest);
}

.fact-card h4 {
  margin: 0 0 0.5rem 0;
  color: var(--md-primary-fg-color);
}

.fact-card p {
  margin: 0;
  color: var(--md-default-fg-color--light);
}

.next-steps {
  background: var(--md-accent-fg-color--transparent);
  padding: 2rem;
  border-radius: 8px;
  margin: 2rem 0;
}

.next-steps h2 {
  margin-top: 0;
}

@media screen and (max-width: 768px) {
  .arch-stats {
    gap: 1rem;
  }

  .architecture-facts {
    grid-template-columns: 1fr;
  }
}
</style>
