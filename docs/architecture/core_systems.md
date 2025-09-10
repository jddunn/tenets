---
title: Core Systems
description: Deep dive into Tenets core systems, algorithms, and processing pipelines
---

# Core Systems Architecture

## Overview

Tenets is built around a sophisticated, multi-layered architecture that transforms raw code into intelligent, contextual insights. At its heart, the system uses advanced ranking algorithms, comprehensive code analysis, and intelligent caching to provide developers with exactly the right context at exactly the right time.

## Core Design Decisions

### Text Matching Philosophy

Tenets employs **three distinct matching strategies** optimized for different use cases:

| Mode | Matching Behavior | Design Rationale |
|------|------------------|------------------|
| **Fast** | Simple substring matching (no word boundaries) | Maximum speed, no corpus building |
| **Balanced** | Word boundaries + BM25 ranking + text processing | Accurate results for development |
| **Thorough** | Balanced features + ML embeddings + semantic search | Deep code understanding |

### Key Matching Principles

1. **Mode-Specific Matching Behavior**
   - **Fast**: Simple substring matching (e.g., "auth" matches "authentication", "oauth", "authorized")
   - **Balanced**: Word boundary enforcement (e.g., "auth" only matches standalone "auth", not "oauth")
   - **Thorough**: Semantic matching (e.g., "auth" matches "login", "authentication", "security")
   - **Rationale**: Trade-off between speed and precision based on use case

2. **No Typo Tolerance By Design**
   - ❌ "auht" does NOT match "auth" in any mode
   - **Rationale**: Professional development assumes correct spelling
   - **Exception**: Hyphen/space variations (e.g., "open-source" = "open source" = "opensource")

3. **Hierarchical Feature Inheritance**
   ```
   Fast Mode:
     - Simple substring matching (no word boundaries)
     - Case-insensitive
     - Path relevance
     - NO corpus building (key performance optimization)
   
   Balanced Mode (completely different from Fast):
     - Word boundary matching with regex
     - BM25 corpus building and scoring
     - CamelCase/snake_case splitting (authManager → auth, manager)
     - Common abbreviation expansion (config → configuration)
     - Plural/singular normalization
   
   Thorough Mode (extends Balanced, includes all its features plus):
     - Semantic similarity (auth → authentication, login)
     - ML-based embeddings (requires tenets[ml])
     - Both BM25 AND TF-IDF scoring
     - Context-aware matching
   ```

4. **Case Insensitivity Throughout**
   - All matching is case-insensitive
   - **Rationale**: Users may type queries in any case

5. **Import Preservation**
   - Import statements are intelligently truncated while preserving structure
   - Critical imports are always retained
   - **Rationale**: Maintains code context while optimizing token usage

## System Flow & Pipeline

The entire Tenets system follows a carefully orchestrated pipeline:

```mermaid
graph TB
    A[User Input] --> B[Prompt Parser]
    B --> C[Intent Detection]
    C --> D[File Discovery Engine]
    D --> E[Language-Specific Analysis]
    E --> F[Multi-Factor Ranking]
    F --> G[Token Optimization]
    G --> H[Context Assembly]
    H --> I[Output Formatting]

    subgraph "Analysis Pipeline"
        E1[Static Analysis]
        E2[Complexity Analysis]
        E3[Git History Analysis]
        E4[Dependency Analysis]
        E5[Pattern Recognition]
    end

    subgraph "Ranking Pipeline"
        F1[Keyword Matching]
        F2[BM25 Scoring]
        F3[Semantic Similarity]
        F4[Git Signals]
        F5[File Characteristics]
        F6[Project Structure]
    end

    E --> E1
    E --> E2
    E --> E3
    E --> E4
    E --> E5

    F --> F1
    F --> F2
    F --> F3
    F --> F4
    F --> F5
    F --> F6

    style A fill:#e1f5fe
    style I fill:#e8f5e8
    style E fill:#fff3e0
    style F fill:#fce4ec
```

## NLP/ML Pipeline Architecture

The NLP/ML pipeline powers Tenets' semantic understanding capabilities, enabling intelligent keyword extraction, similarity matching, and context-aware ranking.

### Pipeline Component Flow

```mermaid
graph TD
    subgraph "Input Processing"
        INPUT[Raw Text Input]
        PROMPT[User Prompt]
        CODE[Code Content]
    end

    subgraph "Tokenization Layer"
        CODE_TOK[Code Tokenizer<br/>camelCase, snake_case]
        TEXT_TOK[Text Tokenizer<br/>NLP processing]
    end

    subgraph "Keyword Extraction"
        YAKE_EXT[YAKE Extractor<br/>Statistical]
        TFIDF_EXT[TF-IDF Extractor<br/>Alternative option]
        FREQ_EXT[Frequency Extractor<br/>Fallback]
    end

    subgraph "Embedding Generation"
        LOCAL_EMB[Local Embeddings<br/>sentence-transformers]
        MODEL_SEL[Model Selection<br/>MiniLM, MPNet]
        FALLBACK[BM25 Fallback<br/>No ML required]
    end

    subgraph "Similarity Computing"
        COSINE[Cosine Similarity]
        BATCH[Batch Processing]
        CACHE[Result Caching]
    end

    INPUT --> CODE_TOK
    INPUT --> TEXT_TOK
    PROMPT --> TEXT_TOK
    CODE --> CODE_TOK

    CODE_TOK --> YAKE_EXT
    TEXT_TOK --> YAKE_EXT
    YAKE_EXT --> TFIDF_EXT
    TFIDF_EXT --> FREQ_EXT

    FREQ_EXT --> LOCAL_EMB
    LOCAL_EMB --> MODEL_SEL
    MODEL_SEL --> FALLBACK

    FALLBACK --> COSINE
    COSINE --> BATCH
    BATCH --> CACHE

    style INPUT fill:#e1f5fe
    style CACHE fill:#e8f5e8
```

### Tokenization Strategy

Tenets uses specialized tokenizers for different content types:

#### Code Tokenizer
```python
class CodeTokenizer:
    """Handles programming language tokens."""

    def tokenize(self, text: str) -> List[str]:
        # Split on camelCase: 'getUserName' → ['get', 'User', 'Name']
        # Split on snake_case: 'get_user_name' → ['get', 'user', 'name']
        # Split on kebab-case: 'get-user-name' → ['get', 'user', 'name']
        # Preserve special tokens: '__init__', 'UTF-8'
        # Handle operators: '++', '==', '!='
```

#### Stopword Management
- **Code Stopwords**: Minimal set (~30 words) - 'function', 'class', 'return'
- **Prompt Stopwords**: Aggressive filtering (~200+ words) - common English words
- **Intent Action Words**: Filters generic action words from keyword matching
  - Prevents words like "fix", "debug", "implement" from affecting file ranking
  - Preserves domain-specific terms for accurate matching
  - Configurable per intent type
- **Context-Aware**: Different stopword sets for different operations

### Embedding Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **all-MiniLM-L6-v2** | 90MB | ~180ms/file | Good | Default, semantic searches |
| **all-MiniLM-L12-v2** | 120MB | ~250ms/file | Better | Higher accuracy needs |
| **all-mpnet-base-v2** | 420MB | Moderate | Best | Thorough mode |
| **multi-qa-MiniLM** | 90MB | Fast | Specialized | Q&A optimized tasks |

The system automatically selects models based on:
- Available memory
- Task complexity
- User configuration
- Performance requirements

## File Discovery & Scanning System

The file discovery system efficiently traverses codebases of any size, applying intelligent filtering and parallel processing.

### Scanner Architecture

```mermaid
graph TD
    subgraph "Entry Points"
        ROOT[Project Root]
        PATHS[Specified Paths]
        PATTERNS[Include Patterns]
    end

    subgraph "Ignore System Hierarchy"
        CLI[CLI Arguments<br/>Highest Priority]
        TENETS[.tenetsignore]
        GIT[.gitignore]
        GLOBAL[Global Ignores<br/>Lowest Priority]
    end

    subgraph "Detection Systems"
        BINARY[Binary Detection]
        SIZE[Size Check<br/>Max 10MB]
        CONTENT[Content Analysis]
    end

    subgraph "Parallel Processing"
        QUEUE[Work Queue]
        WORKERS[Thread Pool]
        PROGRESS[Progress Tracking]
    end

    ROOT --> CLI
    PATHS --> CLI
    PATTERNS --> CLI

    CLI --> TENETS
    TENETS --> GIT
    GIT --> GLOBAL

    GLOBAL --> BINARY
    BINARY --> SIZE
    SIZE --> CONTENT

    CONTENT --> QUEUE
    QUEUE --> WORKERS
    WORKERS --> PROGRESS

    style ROOT fill:#e1f5fe
    style PROGRESS fill:#e8f5e8
```

### Binary Detection Strategy

```mermaid
flowchart TD
    FILE[Input File] --> EXT{Known Binary<br/>Extension?}
    EXT -->|Yes| BINARY[Mark as Binary]
    EXT -->|No| SIZE{Size > 10MB?}
    SIZE -->|Yes| SKIP[Skip File]
    SIZE -->|No| SAMPLE[Sample First 8KB]
    SAMPLE --> NULL{Contains<br/>Null Bytes?}
    NULL -->|Yes| BINARY
    NULL -->|No| RATIO[Calculate Text Ratio]
    RATIO --> THRESHOLD{>95%<br/>Printable?}
    THRESHOLD -->|Yes| TEXT[Mark as Text]
    THRESHOLD -->|No| BINARY
    TEXT --> ANALYZE[Ready for Analysis]
    BINARY --> IGNORE[Skip Analysis]
    SKIP --> IGNORE
```

## 1. Ranking System Deep Dive

The ranking system (`tenets.core.ranking`) is the intelligence core of Tenets, using multiple sophisticated algorithms to determine file relevance.

### Ranking Algorithms

#### Fast Algorithm (`FastRankingStrategy`)
- **Use Case**: Quick exploration, CI/CD, initial discovery
- **Relative Performance**: Baseline (100%)
- **Implementation**: 
  - Lightweight file analysis (8KB samples)
  - No corpus building (saves significant time)
  - Simple keyword matching with word boundaries
  - Deep analysis only on top 20 files
- **Factors**: Simple keyword matching, path analysis, file type relevance
- **Accuracy**: Good for obvious matches, may miss nuanced connections

```python
# Fast algorithm priorities:
1. Lightweight analysis for all files
2. Exact keyword matches in content samples
3. File extension relevance to prompt
4. Directory importance (src/ > tests/)
5. Deep analysis on top-ranked files only
```

#### Balanced Algorithm (`BalancedRankingStrategy`) - **Default**
- **Use Case**: Daily development, general context building
- **Relative Performance**: 150% (1.5x slower than fast)
- **Implementation**: 
  - Full AST analysis for all files
  - BM25 corpus building for accurate ranking
  - Word boundary matching for precision
  - Intelligent summarization for token optimization
- **Factors**: BM25 scoring, word boundaries, abbreviation expansion, structure analysis
- **Accuracy**: Excellent balance of speed and accuracy

```python
# Balanced algorithm combines:
- BM25 relevance scoring (35% weight)
- BM25 document ranking (30% weight)
- Git activity signals (15% weight)
- File structure analysis (15% weight)
```

#### Thorough Algorithm (`ThoroughRankingStrategy`)
- **Use Case**: Complex refactoring, architecture reviews, semantic search
- **Relative Performance**: 400% (4x slower than fast)
- **Implementation**: 
  - ML model loading (all-MiniLM-L6-v2, ~10s first run)
  - Builds both BM25 and TF-IDF corpus (~5s)
  - Semantic embeddings for all files
  - Comprehensive ranking with ML (~23s)
  - Pattern matching and dependency analysis
- **Factors**: Dual scoring algorithms, semantic similarity, dependency graphs, architectural patterns
- **Accuracy**: Best possible with ML-powered understanding

```python
# Thorough algorithm includes:
- ML embeddings (384-dim vectors)
- Semantic similarity via cosine distance
- Dual corpus (BM25 + TF-IDF)
- Programming pattern recognition
- Complex dependency analysis
- Cross-reference analysis
- Advanced git history mining
```

### Multi-Factor Ranking Architecture

The ranking system combines multiple signals to determine file relevance:

```mermaid
graph TD
    subgraph "Ranking Strategies"
        FAST[Fast Strategy<br/>0.5ms/file<br/>Word Boundaries]
        BALANCED[Balanced Strategy<br/>2.1ms/file<br/>BM25 + Compounds]
        THOROUGH[Thorough Strategy<br/>2.0ms/file<br/>Semantic + ML]
    end

    subgraph "Core Ranking Factors"
        SEM[Semantic Similarity<br/>25% weight]
        TEXT[Text Matching<br/>30% weight]
        STRUCT[Code Structure<br/>20% weight]
        GIT[Git Signals<br/>15% weight]
        FILE[File Characteristics<br/>10% weight]
    end

    subgraph "Scoring Engine"
        COMBINE[Weighted Combination]
        NORMALIZE[Score Normalization]
        FILTER[Threshold Filtering]
        RANK[Final Rankings]
    end

    FAST --> TEXT
    BALANCED --> STRUCT
    THOROUGH --> SEM

    SEM --> COMBINE
    TEXT --> COMBINE
    STRUCT --> COMBINE
    GIT --> COMBINE
    FILE --> COMBINE

    COMBINE --> NORMALIZE
    NORMALIZE --> FILTER
    FILTER --> RANK

    style FAST fill:#90caf9
    style BALANCED fill:#a5d6a7
    style THOROUGH fill:#ffab91
```

### Ranking Factors Explained

#### 1. **BM25 (Best Matching 25) - Primary Ranking Algorithm**
```python
class BM25Calculator:
    def score(self, query, document):
        """
        BM25 is the primary probabilistic ranking function used
        throughout Tenets for document relevance scoring.

        Key features:
        - Document length normalization (parameter b)
        - Term frequency saturation (parameter k1)
        - Significantly faster than alternatives
        - Industry standard in search engines
        """
```

**Use Case**: Primary ranking algorithm with 35% weight in balanced mode, handling varying document lengths and preventing term repetition over-weighting.

#### 2. **TF-IDF (Term Frequency-Inverse Document Frequency) - Optional**
```python
class TFIDFCalculator:
    def calculate_relevance(self, document, query_terms, corpus):
        """
        TF-IDF is available as an alternative ranking method
        for experimentation. Not recommended for production use.

        TF = (term frequency in doc) / (total terms in doc)
        IDF = log(total docs / docs containing term)
        TF-IDF = TF × IDF

        Note: BM25 provides superior performance and accuracy.
        """
```

**Status**: Available for experimentation but not recommended. Use BM25 for production workloads.

#### 3. **Git Activity Signals**
```python
class GitRankingFactor:
    def calculate_git_signals(self, file_path):
        return {
            'recent_commits': recent_commit_count,      # Files changed recently
            'commit_frequency': historical_changes,     # Frequently modified files
            'author_diversity': unique_contributors,    # Files many people work on
            'recency_score': days_since_last_change,   # How fresh is the file
            'blame_distribution': line_ownership,      # Code ownership patterns
        }
```

**Why This Matters**: Recently changed files are more likely relevant to current work. Files with many contributors often contain core logic.

#### 4. **Structural Analysis Factors**
```python
class StructuralRankingFactor:
    def analyze_structure(self, file_analysis):
        return {
            'complexity_score': cyclomatic_complexity,   # Code complexity
            'import_centrality': incoming_references,    # How many files import this
            'export_richness': outgoing_dependencies,    # What this file provides
            'directory_importance': path_significance,    # /src vs /tests importance
            'file_role': detected_file_type,            # Config, model, util, etc.
        }
```

### Ranking Strategy Selection

The system intelligently selects strategies based on context:

```python
def select_strategy(self, context: PromptContext, config: TenetsConfig) -> RankingStrategy:
    """
    Auto-select the best strategy based on:
    - Codebase size
    - Available dependencies
    - User preferences
    - Time constraints
    """
    if config.ranking.algorithm == "auto":
        if file_count > 10000:
            return FastRankingStrategy()
        elif has_ml_dependencies and context.complexity_level == "high":
            return ThoroughRankingStrategy()
        else:
            return BalancedRankingStrategy()
```

## 2. Analysis Engine Architecture

The analysis engine (`tenets.core.analysis`) provides deep code understanding through language-specific parsers and cross-language analysis.

### Language Analyzer System

#### Analyzer Architecture
```python
class LanguageAnalyzer(ABC):
    """Base class for all language-specific analyzers"""

    @abstractmethod
    def analyze_structure(self, content: str) -> CodeStructure:
        """Extract functions, classes, imports, etc."""

    @abstractmethod
    def calculate_complexity(self, content: str) -> ComplexityMetrics:
        """Calculate cyclomatic complexity, maintainability index"""

    @abstractmethod
    def extract_dependencies(self, content: str) -> List[ImportInfo]:
        """Find imports, requires, includes"""
```

#### Supported Languages & Features

| Language | Parser | Complexity | Dependencies | AST Analysis | Special Features |
|----------|--------|------------|--------------|--------------|------------------|
| **Python** | Full AST | ✅ CC, MI | ✅ imports, from | ✅ Full | Decorators, async/await |
| **JavaScript/TypeScript** | Full AST | ✅ CC, MI | ✅ import/require | ✅ Full | React components, Node.js |
| **Java** | Full AST | ✅ CC, MI | ✅ import, package | ✅ Full | Annotations, Spring |
| **C#** | Full AST | ✅ CC, MI | ✅ using, namespace | ✅ Full | Attributes, LINQ |
| **Go** | Full AST | ✅ CC, MI | ✅ import, package | ✅ Full | Goroutines, interfaces |
| **Rust** | Full AST | ✅ CC, MI | ✅ use, extern | ✅ Full | Traits, lifetimes |
| **C++** | Regex+ | ✅ Basic | ✅ #include | Partial | Headers, namespaces |
| **Ruby** | Full AST | ✅ CC, MI | ✅ require, gem | ✅ Full | Gems, Rails detection |
| **PHP** | Full AST | ✅ CC, MI | ✅ require, use | ✅ Full | Composer, namespaces |
| **Kotlin** | Full AST | ✅ CC, MI | ✅ import, package | ✅ Full | Coroutines, DSLs |

*CC = Cyclomatic Complexity, MI = Maintainability Index*

### Complexity Analysis Deep Dive

#### Cyclomatic Complexity
```python
def calculate_cyclomatic_complexity(self, node):
    """
    Measures the number of linearly independent paths through code.

    Decision points that increase complexity:
    - if/elif statements
    - for/while loops
    - try/catch blocks
    - switch/case statements
    - logical operators (&&, ||)
    - ternary operators
    """
    complexity = 1  # Base complexity
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity
```

**Complexity Levels:**
- 1-10: Simple, low risk
- 11-20: Moderate complexity
- 21-50: High complexity, refactor recommended
- 50+: Very high risk, immediate attention needed

#### Maintainability Index
```python
def calculate_maintainability_index(self, metrics):
    """
    MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)

    Where:
    - HV = Halstead Volume (operators + operands)
    - CC = Cyclomatic Complexity
    - LOC = Lines of Code

    Scale: 0-100 (higher is more maintainable)
    """
```

**MI Interpretation:**
- 85-100: Excellent maintainability
- 70-85: Good maintainability
- 50-70: Moderate maintainability
- 25-50: Below average, needs attention
- 0-25: Difficult to maintain

### Ranking Factor Details

| Factor Category | Weight | Components | Description |
|-----------------|--------|------------|-------------|
| **Semantic Similarity** | 25% | Embedding cosine similarity, contextual relevance | ML-based understanding of semantic meaning |
| **Text Matching** | 35% | Keyword matches (20%), BM25 (15%) | Direct term matching and probabilistic relevance |
| **Code Structure** | 20% | Import centrality (10%), path relevance (10%) | File importance in codebase architecture |
| **Git Signals** | 15% | Recency (5%), frequency (5%), authors (5%) | Version control activity indicators |
| **File Characteristics** | 10% | File type (5%), code patterns (5%) | Language and pattern-based relevance |

### Dependency Analysis

#### Import Graph Construction
```python
class DependencyAnalyzer:
    def build_dependency_graph(self, project_files):
        """
        Builds a directed graph of file dependencies:

        1. Extract all imports/includes from each file
        2. Resolve import paths to actual files
        3. Create edges between dependent files
        4. Calculate centrality metrics
        5. Identify circular dependencies
        """

        graph = nx.DiGraph()

        for file in project_files:
            imports = self.extract_imports(file)
            for imp in imports:
                target = self.resolve_import_path(imp, file)
                if target:
                    graph.add_edge(file.path, target.path)

        return self.calculate_centrality_metrics(graph)
```

#### Dependency Metrics

The system calculates various centrality metrics to understand file importance:

```mermaid
graph LR
    subgraph "Centrality Metrics"
        IN[In-Degree<br/>Files importing this]
        OUT[Out-Degree<br/>Files imported]
        BETWEEN[Betweenness<br/>Bridge importance]
        PAGE[PageRank<br/>Link analysis]
    end

    subgraph "Analysis"
        CRITICAL[Critical Files<br/>High in-degree]
        HUBS[Hub Files<br/>High out-degree]
        BRIDGES[Bridge Files<br/>High betweenness]
        CENTRAL[Central Files<br/>High PageRank]
    end

    IN --> CRITICAL
    OUT --> HUBS
    BETWEEN --> BRIDGES
    PAGE --> CENTRAL

    style CRITICAL fill:#ffcdd2
    style CENTRAL fill:#c5e1a5
```

- **In-Degree**: Number of files that import this file (dependency count)
- **Out-Degree**: Number of files this file imports (dependency scope)
- **Betweenness**: How often this file appears in shortest paths between other files
- **PageRank**: Importance based on the importance of files that import it
- **Clustering Coefficient**: How connected this file's dependencies are to each other

## 3. Summarization System

When files exceed token limits, Tenets uses intelligent summarization to preserve the most important content.

### Summarization Strategies

#### 1. **Structural Preservation**
```python
class StructuralSummarizer:
    def summarize(self, content, target_size):
        """
        Preserves code structure while reducing content:

        Priority Order:
        1. Function/class signatures (always keep)
        2. Public API definitions
        3. Complex logic blocks
        4. Important comments/docstrings
        5. Configuration/constants
        6. Simple variable assignments (truncate)
        7. Repetitive code (remove)
        """
```

#### 2. **Semantic Summarization** (ML mode)
```python
class SemanticSummarizer:
    def summarize(self, content, context, target_size):
        """
        Uses ML to understand semantic importance:

        1. Split content into logical segments
        2. Generate embeddings for each segment
        3. Calculate relevance to user prompt
        4. Rank segments by importance
        5. Select top segments that fit token budget
        6. Maintain code coherence
        """
```

#### 3. **Contextual Summarization**
```python
class ContextualSummarizer:
    def summarize_for_prompt(self, file_content, prompt_context):
        """
        Tailors summarization to specific prompt needs:

        - "debug bug": Keep error handling, logging
        - "add feature": Keep interfaces, extension points
        - "refactor": Keep complex logic, dependencies
        - "security": Keep authentication, validation
        """
```

## 4. Context Optimization & Token Management

Tenets intelligently manages token budgets to maximize relevant context while staying within model limits.

### Token Budget Allocation

```mermaid
graph TB
    subgraph "Token Budget"
        TOTAL[Total Token Limit]
        PROMPT[Prompt Reservation<br/>~2K tokens]
        RESPONSE[Response Buffer<br/>~4K tokens]
        CONTEXT[Available for Context]
    end

    subgraph "Optimization Strategy"
        RANK[Rank Files]
        SELECT[Select Top N]
        MEASURE[Measure Tokens]
        SUMMARIZE[Summarize if Needed]
    end

    subgraph "Content Priority"
        CRITICAL[Critical Files<br/>Always included]
        HIGH[High Relevance<br/>Full content]
        MEDIUM[Medium Relevance<br/>Summarized]
        LOW[Low Relevance<br/>Signatures only]
    end

    TOTAL --> PROMPT
    TOTAL --> RESPONSE
    TOTAL --> CONTEXT

    CONTEXT --> RANK
    RANK --> SELECT
    SELECT --> MEASURE
    MEASURE --> SUMMARIZE

    SUMMARIZE --> CRITICAL
    SUMMARIZE --> HIGH
    SUMMARIZE --> MEDIUM
    SUMMARIZE --> LOW

    style TOTAL fill:#e1f5fe
    style CRITICAL fill:#ffcdd2
```

## 5. Caching & Performance Architecture

Tenets uses sophisticated multi-level caching to provide instant responses after initial analysis.

### Cache Hierarchy

```mermaid
graph TB
    A[Request] --> B[Memory Cache]
    B -->|Miss| C[SQLite Session DB]
    C -->|Miss| D[File System Cache]
    D -->|Miss| E[Analysis Pipeline]

    E --> F[Store in File Cache]
    F --> G[Store in SQLite]
    G --> H[Store in Memory]
    H --> I[Return Results]

    subgraph "Cache Types"
        J[Hot Data - Memory]
        K[Structured Data - SQLite]
        L[Analysis Results - Disk]
        M[Embeddings - Specialized]
    end

    style B fill:#ffeb3b
    style C fill:#4caf50
    style D fill:#2196f3
    style E fill:#ff5722
```

### Cache Invalidation Strategy

```python
class CacheManager:
    def should_invalidate(self, file_path, cache_entry):
        """
        Smart cache invalidation based on:

        1. File modification time (mtime)
        2. Content hash changes (for accuracy)
        3. Git commit changes (for git-based signals)
        4. Configuration changes (for ranking params)
        5. Dependency changes (for import analysis)
        """

        reasons = []

        if file_path.stat().st_mtime > cache_entry.mtime:
            reasons.append("file_modified")

        if self.calculate_content_hash(file_path) != cache_entry.content_hash:
            reasons.append("content_changed")

        if self.get_git_commit_hash() != cache_entry.git_hash:
            reasons.append("git_changed")

        return len(reasons) > 0, reasons
```

### Prompt Parsing & Understanding

```mermaid
graph LR
    subgraph "Prompt Analysis"
        INPUT[User Prompt]
        INTENT[Intent Detection]
        KEYWORDS[Keyword Extraction]
        FILTER[Intent Filtering]
        CONTEXT[Context Building]
    end

    subgraph "Intent Types"
        DEBUG[Debug/Fix<br/>Error handling focus]
        FEATURE[Feature/Add<br/>New functionality]
        REFACTOR[Refactor/Optimize<br/>Code improvement]
        EXPLORE[Explore/Understand<br/>Learning focus]
    end

    subgraph "Extraction Methods"
        YAKE[YAKE Algorithm<br/>Statistical extraction]
        BM25[BM25<br/>Probabilistic relevance]
        NER[Named Entity<br/>Code entities]
    end

    INPUT --> INTENT
    INTENT --> KEYWORDS
    KEYWORDS --> FILTER
    FILTER --> CONTEXT

    INTENT --> DEBUG
    INTENT --> FEATURE
    INTENT --> REFACTOR
    INTENT --> EXPLORE

    KEYWORDS --> YAKE
    KEYWORDS --> BM25
    KEYWORDS --> NER
```

### Keyword Extraction and Matching

#### How Keywords Are Extracted

Tenets extracts both individual words and multi-word phrases from prompts:

- **Individual words**: `token`, `debug`, `issue`, `parser`
- **Multi-word phrases**: `debug tokenizing issue`, `authentication module`, `database connection`

#### Intent Action Word Filtering

Common action words are filtered from individual keywords but preserved in phrases:

**Example**: "fix the debug tokenizing issue"
1. **Extracted keywords**: `['fix', 'debug', 'issue', 'token', 'debug tokenizing issue']`
2. **After filtering**: `['token', 'debug tokenizing issue']`
3. **Why this works**:
   - `fix`, `debug`, `issue` are removed as standalone generic words
   - `debug tokenizing issue` is kept as a specific phrase
   - `token` is kept as a domain-specific term

#### How Keywords Match Files

Keywords match files using different strategies based on the mode:

**Fast Mode** (substring matching):
- `token` matches: `token`, `tokens`, `tokenizer`, `tokenization`, `tokenizing`
- Simple substring search: if keyword is contained in file content, it matches
- Example: searching for "token" will find all files containing that substring

**Balanced/Thorough Mode** (word boundary matching):
- More precise matching with word boundaries
- Still matches variations (plural forms, different word forms)
- Example: "token" matches "tokens" but not "atoken" or "tokenx"

**Multi-word Phrase Matching**:
- Phrases like "debug tokenizing issue" match if:
  - The exact phrase appears in the file, OR
  - Individual words from the phrase appear (partial matching)
- Higher scores for exact phrase matches

#### Real-World Example

Query: **"fix the tokenizing bug in the parser"**

1. **Keywords extracted**: 
   - Individual: `fix`, `bug`, `tokenizing`, `parser`
   - Phrases: `tokenizing bug`, `bug in the parser`

2. **After intent filtering**:
   - Filtered out: `fix`, `bug` (generic action words)
   - Kept: `tokenizing`, `parser`, `tokenizing bug`, `bug in the parser`

3. **Files that will match**:
   - Files containing "tokenizing", "tokenizer", "tokenization", etc.
   - Files containing "parser", "parsing", "parse", etc.
   - Files with the exact phrases get higher scores
   - `tokenizer.py`, `parser.py`, `parse_tokens.py` all rank high

This approach ensures:
- Generic action words don't pollute search results
- Domain-specific terms are preserved for accurate matching
- Multi-word phrases provide additional context
- Related word forms are still discovered

### Performance Optimizations

#### Parallel Processing
```python
def analyze_files_parallel(self, files, max_workers=None):
    """
    Parallel analysis with intelligent work distribution:

    - Small files: Batch process in single thread
    - Large files: Individual threads
    - I/O bound: Higher thread count
    - CPU bound: Thread count = CPU cores
    """

    with ThreadPoolExecutor(max_workers=self.optimal_worker_count()) as executor:
        future_to_file = {
            executor.submit(self.analyze_file, f): f
            for f in files
        }
```

#### Memory Management
```python
class MemoryManager:
    def manage_analysis_memory(self):
        """
        Prevents memory bloat during large project analysis:

        1. Stream file processing (don't load all at once)
        2. Release analysis objects after ranking
        3. Use generators for large result sets
        4. Monitor memory usage and trigger cleanup
        """
```

## 5. Session Management & State

Sessions provide persistent context and configuration for project work.

### Session Architecture
```python
class SessionManager:
    def create_session(self, name, project_path):
        """
        Session contains:
        - Pinned files (guaranteed inclusion)
        - Custom ranking weights
        - Project-specific configuration
        - Analysis cache keys
        - Tenet definitions (guiding principles)
        """

    def merge_contexts(self, session_name, new_context):
        """
        Intelligently merge new analysis with existing session:

        1. Preserve pinned files
        2. Update file rankings
        3. Maintain tenet priorities
        4. Merge analysis results
        5. Update cache references
        """
```

### Tenets System
```python
class TenetManager:
    """
    Manages 'tenets' - persistent principles that guide context generation

    Examples:
    - "Always include error handling examples"
    - "Prioritize async/await patterns"
    - "Include security considerations"
    - "Focus on performance implications"
    """

    def apply_tenets(self, context_result, active_tenets):
        """
        Inject tenets into generated context:

        1. Analyze context for tenet relevance
        2. Select most applicable tenets
        3. Format tenets appropriately
        4. Insert at strategic positions
        5. Ensure token budget compliance
        """
```

## Integration & Extension Points

### Custom Analyzers
```python
class MyCustomAnalyzer(LanguageAnalyzer):
    def analyze_structure(self, content: str) -> CodeStructure:
        # Implement custom parsing logic
        pass

    def get_language_patterns(self) -> Dict[str, str]:
        return {
            'function_def': r'def\s+(\w+)\s*\(',
            'class_def': r'class\s+(\w+)',
            'import': r'import\s+(.+)',
        }
```

### Custom Ranking Factors
```python
class MyRankingFactor(RankingFactor):
    def calculate(self, file_analysis, prompt_context):
        """
        Add custom ranking logic:
        - Domain-specific patterns
        - Company coding standards
        - Architecture preferences
        - Performance considerations
        """
        return relevance_score
```

### Plugin System
```python
class TenetsPlugin:
    def register_analyzer(self, language: str, analyzer_class):
        """Register custom language analyzer"""

    def register_ranking_factor(self, factor_name: str, factor_class):
        """Register custom ranking factor"""

    def register_output_formatter(self, format_name: str, formatter_class):
        """Register custom output format"""
```

## Performance Characteristics

### Benchmarked Performance (Real-World Codebase)

| Mode | Relative Performance | Key Optimizations |
|------|---------------------|-------------------|
| **Fast** | 100% (baseline) | Lightweight analysis, no corpus, minimal processing |
| **Balanced** | 150% (1.5x slower) | BM25 corpus, word boundaries, intelligent summarization |
| **Thorough** | 400% (4x slower) | ML embeddings, dual algorithms, comprehensive analysis |

### Performance Breakdown by Phase

| Phase | Fast Mode | Balanced Mode | Thorough Mode |
|-------|-----------|---------------|---------------|
| **File Scanning** | ~1s (same) | ~1s (same) | ~1s (same) |
| **Analysis** | Lightweight (0.2s) | Full AST (2-3s) | Full AST (2-3s) |
| **Corpus Building** | Skipped | BM25 only (1s) | BM25 + TF-IDF (5s) |
| **ML Model Loading** | N/A | N/A | ~10s (first run) |
| **Ranking** | Simple (0.02s) | BM25 (0.4s) | ML + Dual (23s) |
| **Aggregation** | Truncation (0.1s) | Summarization (2s) | Deep summarization (3s) |

### Key Performance Improvements

#### Fast Mode with Lightweight Analysis
- **Implementation**: `LightweightAnalyzer` class reads only first 8KB of files
- **Performance**: 10-100x faster than full AST parsing
- **Trade-off**: Less accurate structure analysis but sufficient for ranking
- **Deep Analysis**: Applied only to top 20 files after ranking

### Performance Notes

- Fast mode achieves baseline performance through lightweight analysis
- Balanced mode adds only 50% overhead while providing significantly better accuracy
- Thorough mode's 4x slowdown is justified by ML-powered semantic understanding
- All modes benefit from aggressive caching (cache hits < 500ms)
- First-run ML model loading in thorough mode adds one-time 10s overhead

### Memory Usage

```python
# Typical memory footprint:
base_memory = 50_000_000      # ~50MB base
per_file_overhead = 5_000     # ~5KB per analyzed file
embedding_cache = 200_000_000 # ~200MB for ML embeddings (when used)
```

### Scalability Limits

- **Files**: Tested up to 100K+ files
- **File Size**: Handles up to 10MB individual files
- **Concurrency**: Scales to available CPU cores
- **Memory**: Degrades gracefully with limited RAM

## Language Analyzer Architecture

Tenets provides specialized analyzers for each programming language, extracting structure, complexity, and relationships.

### Analyzer System Overview

```mermaid
graph TB
    subgraph "Base Analyzer Interface"
        BASE[LanguageAnalyzer<br/>Abstract Base]
        EXTRACT[extract_structure()]
        COMPLEX[calculate_complexity()]
        DEPS[trace_dependencies()]
    end

    subgraph "Language-Specific Analyzers"
        PYTHON[Python<br/>Full AST]
        JS[JavaScript<br/>ES6+ support]
        GO[Go<br/>Package detection]
        JAVA[Java<br/>OOP patterns]
        RUST[Rust<br/>Ownership]
        GENERIC[Generic<br/>Pattern-based]
    end

    subgraph "Analysis Output"
        CLASSES[Classes & Methods]
        FUNCTIONS[Functions & Signatures]
        IMPORTS[Import Graph]
        METRICS[Complexity Metrics]
    end

    BASE --> EXTRACT
    BASE --> COMPLEX
    BASE --> DEPS

    BASE --> PYTHON
    BASE --> JS
    BASE --> GO
    BASE --> JAVA
    BASE --> RUST
    BASE --> GENERIC

    PYTHON --> CLASSES
    PYTHON --> FUNCTIONS
    PYTHON --> IMPORTS
    PYTHON --> METRICS

    style BASE fill:#e1f5fe
    style METRICS fill:#e8f5e8
```

### Python Analyzer Deep Dive

```mermaid
graph LR
    subgraph "AST Analysis"
        PARSE[AST Parser]
        VISIT[Node Visitor]
        TABLE[Symbol Table]
    end

    subgraph "Structure Extraction"
        CLASSES[Classes<br/>Inheritance]
        FUNCTIONS[Functions<br/>Async/Sync]
        DECORATORS[Decorators]
        TYPES[Type Hints]
    end

    subgraph "Import Analysis"
        ABS[Absolute Imports]
        REL[Relative Imports]
        GRAPH[Import Graph]
    end

    subgraph "Complexity"
        CYCLO[Cyclomatic]
        COGNITIVE[Cognitive]
        HALSTEAD[Halstead]
    end

    PARSE --> VISIT
    VISIT --> TABLE

    TABLE --> CLASSES
    TABLE --> FUNCTIONS
    TABLE --> DECORATORS
    TABLE --> TYPES

    VISIT --> ABS
    VISIT --> REL
    ABS --> GRAPH
    REL --> GRAPH

    TABLE --> CYCLO
    TABLE --> COGNITIVE
    TABLE --> HALSTEAD
```

### Language Support Matrix

| Language | AST Support | Complexity Analysis | Import Resolution | Special Features |
|----------|------------|-------------------|-------------------|------------------|
| **Python** | ✅ Full | ✅ CC, MI, Halstead | ✅ Complete | Decorators, async/await, type hints |
| **JavaScript/TypeScript** | ✅ Full | ✅ CC, MI | ✅ ES6+ modules | React components, JSX, Node.js |
| **Java** | ✅ Full | ✅ CC, MI | ✅ Package/class | Annotations, Spring framework |
| **C#** | ✅ Full | ✅ CC, MI | ✅ Namespace/using | Attributes, LINQ, async |
| **Go** | ✅ Full | ✅ CC, MI | ✅ Package/import | Goroutines, interfaces |
| **Rust** | ✅ Full | ✅ CC, MI | ✅ Use/extern | Traits, lifetimes, macros |
| **Ruby** | ✅ Full | ✅ CC, MI | ✅ Require/gem | Gems, Rails detection |
| **PHP** | ✅ Full | ✅ CC, MI | ✅ Require/use | Composer, namespaces |
| **C/C++** | 🟨 Regex+ | ✅ Basic | ✅ #include | Headers, templates |
| **Others** | 🟨 Pattern | 🟨 Basic | 🟨 Pattern-based | Extensible patterns |

*CC = Cyclomatic Complexity, MI = Maintainability Index*

This architecture provides the foundation for intelligent, fast, and scalable code intelligence that adapts to any project size and developer workflow.
