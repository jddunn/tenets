# **tenets** Complete Architecture Documentation

**tenets** is a code intelligence platform that analyzes codebases locally to surface relevant files, track development velocity, and build optimal context for both human understanding and AI pair programming - all without making any LLM API calls.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Input Adapter System](#input-adapter-system)
4. [File Discovery & Scanning](#file-discovery--scanning)
5. [Code Analysis Engine](#code-analysis-engine)
6. [Relevance Ranking System](#relevance-ranking-system)
7. [Summarization Engine](#summarization-engine)
8. [Context Management](#context-management)
9. [Session Management](#session-management)
10. [Storage Architecture](#storage-architecture)
11. [Performance Architecture](#performance-architecture)
12. [API Design](#api-design)
13. [Output Formatting](#output-formatting)
14. [Development Tracking](#development-tracking)
15. [Visualization System](#visualization-system)
16. [SaaS Platform Architecture](#saas-platform-architecture)
17. [Security & Privacy](#security--privacy)
18. [Extensibility & Plugins](#extensibility--plugins)
19. [Deployment Options](#deployment-options)
20. [Detailed Roadmap](#detailed-roadmap)

## System Overview

### Core Philosophy

1. **Local-First**: All analysis happens on your machine - no code leaves your system
2. **Output-Only**: Generates context for YOU to paste into LLMs - not an LLM wrapper
3. **Zero-Cost Intelligence**: Core functionality requires no API calls
4. **Progressive Enhancement**: Works with minimal dependencies, scales with optional features
5. **Developer-Centric**: Solves real workflow problems beyond just AI assistance

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                  │
├──────────────────┬───────────────────┬──────────────────────────────────┤
│      CLI         │    Python API     │        Future Web UI             │
│  tenets cmd      │  from tenets      │      tenets.dev SaaS            │
│  Interactive     │  import Tenets    │   Team collaboration            │
│  Pipe-friendly   │  Extensible       │   Real-time updates             │
└──────────────────┴───────────────────┴──────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                   │
├──────────────────┬───────────────────┬──────────────────────────────────┤
│  Input Adapters  │  Prompt Parser    │    External Integrations         │
│  - Text          │  - Keywords       │    - GitHub Issues/PRs          │
│  - URLs          │  - Intent         │    - JIRA Tickets               │
│  - File refs     │  - Task type      │    - GitLab/Bitbucket          │
│  - Patterns      │  - Focus areas    │    - Confluence/Docs           │
└──────────────────┴───────────────────┴──────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS ENGINE                                  │
├──────────────────┬───────────────────┬──────────────────────────────────┤
│  File Scanner    │  Code Analyzer    │    Git Integration              │
│  - Parallel scan │  - AST parsing    │    - History analysis           │
│  - .gitignore    │  - Imports/deps   │    - Blame tracking             │
│  - Custom rules  │  - Complexity     │    - Change velocity            │
│  - Binary detect │  - Language ID    │    - Contributor map            │
└──────────────────┴───────────────────┴──────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENCE LAYER                                  │
├──────────────────┬───────────────────┬──────────────────────────────────┤
│ Relevance Ranker │   NLP Engine      │    Pattern Recognition          │
│ - Multi-factor   │   - YAKE keywords │    - Code patterns              │
│ - Presets        │   - TF-IDF        │    - Architecture detection     │
│ - Custom algos   │   - Embeddings*   │    - Framework identification   │
│ - Configurable   │   (*optional ML)  │    - Hotspot analysis           │
└──────────────────┴───────────────────┴──────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT OPTIMIZATION                                  │
├──────────────────┬───────────────────┬──────────────────────────────────┤
│ Token Manager    │  Summarizer       │    Format Generator             │
│ - Model limits   │  - Local algos    │    - Markdown                   │
│ - Budget calc    │  - Extract method │    - XML (Claude)               │
│ - Cost estimate  │  - LLM enhance*   │    - JSON                       │
│ - Overflow       │  (*optional)      │    - Custom templates           │
└──────────────────┴───────────────────┴──────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER                                     │
├──────────────────┬───────────────────┬──────────────────────────────────┤
│   File Cache     │   Session DB      │    Configuration                │
│ - Content hash   │ - SQLite          │    - Project settings           │
│ - Analysis       │ - History         │    - User preferences           │
│ - Embeddings     │ - Branches        │    - Ranking configs            │
│ - Git data       │ - State           │    - Global ignores             │
└──────────────────┴───────────────────┴──────────────────────────────────┘
```

## Core Components

### Component Overview

| Component | Purpose | Key Features | Dependencies |
|-----------|---------|--------------|--------------|
| Input Adapters | Parse various input sources | GitHub/JIRA/URL support | httpx (optional) |
| File Scanner | Discover relevant files | Parallel, .gitignore aware | pathspec (optional) |
| Code Analyzer | Extract code structure | AST parsing, language detection | ast (built-in) |
| Relevance Ranker | Score file importance | Multi-factor, configurable | sklearn (optional) |
| Summarizer | Condense large files | Local algorithms, LLM optional | tiktoken (optional) |
| Context Manager | Build optimal output | Token budgeting, formatting | - |
| Session Manager | Maintain state | SQLite storage, branching | sqlite3 (built-in) |

## Input Adapter System

### Architecture

```python
class InputAdapter(Protocol):
    """Base protocol for all input adapters."""
    
    @property
    def source_type(self) -> str:
        """Type identifier: text|url|github|jira|gitlab|file"""
        
    @property
    def requires_auth(self) -> bool:
        """Whether this adapter needs credentials."""
        
    def can_handle(self, input_text: str) -> bool:
        """Check if this adapter can process the input."""
        
    def parse(self, input_text: str) -> PromptContext:
        """Parse input into structured context."""
        
    async def fetch_external_context(self) -> Optional[ExternalContext]:
        """Fetch additional context if needed (async for API calls)."""
```

### Adapter Registry

```python
class AdapterRegistry:
    """Manages all input adapters with priority ordering."""
    
    def __init__(self):
        self.adapters = [
            GitHubAdapter(),      # github.com URLs
            JIRAAdapter(),        # JIRA ticket patterns
            GitLabAdapter(),      # GitLab URLs
            ConfluenceAdapter(),  # Confluence pages
            FilePathAdapter(),    # Local/remote files
            URLAdapter(),         # Generic URLs
            TextAdapter(),        # Fallback - handles everything
        ]
    
    def parse_input(self, input_text: str) -> PromptContext:
        """Route to appropriate adapter."""
        for adapter in self.adapters:
            if adapter.can_handle(input_text):
                context = adapter.parse(input_text)
                
                # Fetch external context if available
                if hasattr(adapter, 'fetch_external_context'):
                    external = adapter.fetch_external_context()
                    if external:
                        context.external_context = external
                
                return context
```

### GitHub Adapter Example

```python
class GitHubAdapter(InputAdapter):
    """Handles GitHub issues and PRs."""
    
    patterns = [
        r'github\.com/(?P<owner>[\w-]+)/(?P<repo>[\w-]+)/issues/(?P<number>\d+)',
        r'github\.com/(?P<owner>[\w-]+)/(?P<repo>[\w-]+)/pull/(?P<number>\d+)'
    ]
    
    def parse(self, url: str) -> PromptContext:
        match = self._match_pattern(url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {url}")
            
        # Extract issue/PR details
        issue_data = self._fetch_issue(
            match['owner'], 
            match['repo'], 
            match['number']
        )
        
        # Check for linked issues
        linked_issues = self._find_linked_issues(issue_data['body'])
        
        # Build prompt context
        return PromptContext(
            text=f"{issue_data['title']}\n\n{issue_data['body']}",
            original=url,
            source_type="github",
            source_url=url,
            keywords=self._extract_keywords(issue_data),
            focus_areas=self._extract_labels(issue_data),
            external_context=self._format_github_context(issue_data),
            external_metadata={
                "issue_number": match['number'],
                "linked_issues": linked_issues,
                "labels": issue_data.get('labels', []),
                "milestone": issue_data.get('milestone'),
                "assignees": issue_data.get('assignees', [])
            }
        )
```

## File Discovery & Scanning

### File Scanner Architecture

```python
class FileScanner:
    """High-performance file discovery system."""
    
    def __init__(self, config: TenetsConfig):
        self.config = config
        self.ignore_matcher = IgnoreMatcher()
        self.binary_detector = BinaryDetector()
        
    def scan(
        self,
        paths: List[Path],
        preset: Optional[str] = None,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
        max_depth: Optional[int] = None
    ) -> Generator[ScannedFile, None, None]:
        """Scan paths yielding relevant files."""
        
        # Load ignore patterns
        ignore_patterns = self._load_ignore_patterns(paths)
        
        # Parallel scanning for performance
        with ProcessPoolExecutor(max_workers=self.config.scanner_workers) as executor:
            futures = []
            
            for path in paths:
                if path.is_file():
                    yield self._scan_single_file(path)
                else:
                    future = executor.submit(
                        self._scan_directory,
                        path,
                        ignore_patterns,
                        preset,
                        max_depth
                    )
                    futures.append(future)
            
            # Yield results as they complete
            for future in as_completed(futures):
                yield from future.result()
```

### Ignore System

```python
class IgnoreMatcher:
    """Hierarchical ignore pattern matching."""
    
    def __init__(self):
        self.patterns = {
            'global': self._load_global_ignores(),      # ~/.config/tenets/ignore
            'gitignore': {},                            # .gitignore files
            'dockerignore': {},                         # .dockerignore files
            'tenetsignore': {},                         # .tenetsignore files
            'runtime': set()                            # CLI arguments
        }
    
    def should_ignore(self, path: Path, root: Path) -> bool:
        """Check if path should be ignored."""
        
        # Check in priority order
        rel_path = path.relative_to(root)
        
        # 1. Runtime ignores (highest priority)
        if self._matches_patterns(rel_path, self.patterns['runtime']):
            return True
            
        # 2. .tenetsignore
        if self._check_ignore_file(rel_path, root, '.tenetsignore'):
            return True
            
        # 3. .gitignore (if respect_gitignore is True)
        if self.config.respect_gitignore:
            if self._check_ignore_file(rel_path, root, '.gitignore'):
                return True
                
        # 4. Global ignores
        if self._matches_patterns(rel_path, self.patterns['global']):
            return True
            
        return False
```

### Binary Detection

```python
class BinaryDetector:
    """Detect binary files to skip analysis."""
    
    # Known binary extensions
    BINARY_EXTENSIONS = {
        '.exe', '.dll', '.so', '.dylib', '.a', '.o',
        '.jar', '.class', '.pyc', '.pyo',
        '.zip', '.tar', '.gz', '.bz2', '.7z',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx',
        '.db', '.sqlite', '.bin', '.dat'
    }
    
    def is_binary(self, path: Path) -> bool:
        """Check if file is binary."""
        
        # 1. Check extension
        if path.suffix.lower() in self.BINARY_EXTENSIONS:
            return True
            
        # 2. Check file size (skip very large files)
        if path.stat().st_size > self.config.max_file_size:
            return True
            
        # 3. Sample content check
        try:
            with open(path, 'rb') as f:
                chunk = f.read(8192)
                if not chunk:
                    return False
                    
                # Check for null bytes
                if b'\x00' in chunk:
                    return True
                    
                # Check text ratio
                text_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in (9, 10, 13))
                return (text_chars / len(chunk)) < 0.7
                
        except Exception:
            return True  # Assume binary if can't read
```

## Code Analysis Engine

### Language-Specific Analyzers

```python
class CodeAnalyzer:
    """Orchestrates language-specific code analysis."""
    
    analyzers = {
        '.py': PythonAnalyzer(),
        '.js': JavaScriptAnalyzer(),
        '.ts': TypeScriptAnalyzer(),
        '.java': JavaAnalyzer(),
        '.go': GoAnalyzer(),
        '.rs': RustAnalyzer(),
        '.cpp': CppAnalyzer(),
        '.c': CAnalyzer(),
        '.cs': CSharpAnalyzer(),
        '.rb': RubyAnalyzer(),
        '.php': PHPAnalyzer(),
        '.swift': SwiftAnalyzer(),
        '.kt': KotlinAnalyzer(),
        '.scala': ScalaAnalyzer(),
        '.r': RAnalyzer(),
        '.sql': SQLAnalyzer(),
        '*': GenericAnalyzer()  # Fallback
    }
    
    def analyze_file(
        self,
        file_path: Path,
        content: str,
        deep: bool = False
    ) -> FileAnalysis:
        """Analyze a single file."""
        
        # Select analyzer
        ext = file_path.suffix.lower()
        analyzer = self.analyzers.get(ext, self.analyzers['*'])
        
        # Basic analysis
        analysis = FileAnalysis(
            path=str(file_path),
            language=analyzer.language_name,
            size=len(content),
            lines=content.count('\n') + 1,
            content=content
        )
        
        # Language-specific analysis
        try:
            analysis.imports = analyzer.extract_imports(content)
            analysis.exports = analyzer.extract_exports(content)
            analysis.classes = analyzer.extract_classes(content)
            analysis.functions = analyzer.extract_functions(content)
            analysis.complexity = analyzer.calculate_complexity(content)
            analysis.dependencies = analyzer.trace_dependencies(content)
            
            if deep:
                analysis.ast = analyzer.parse_ast(content)
                analysis.metrics = analyzer.calculate_metrics(content)
                
        except Exception as e:
            logger.warning(f"Analysis failed for {file_path}: {e}")
            
        return analysis
```

### Python Analyzer Example

```python
class PythonAnalyzer(LanguageAnalyzer):
    """Python-specific code analysis using AST."""
    
    language_name = "python"
    
    def extract_imports(self, content: str) -> List[Import]:
        """Extract all imports from Python code."""
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(Import(
                            module=alias.name,
                            alias=alias.asname,
                            line=node.lineno,
                            type="import"
                        ))
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(Import(
                            module=f"{module}.{alias.name}" if module else alias.name,
                            alias=alias.asname,
                            line=node.lineno,
                            type="from",
                            level=node.level  # Relative import level
                        ))
                        
        except SyntaxError:
            # Fallback to regex if AST parsing fails
            imports = self._extract_imports_regex(content)
            
        return imports
    
    def calculate_complexity(self, content: str) -> ComplexityMetrics:
        """Calculate cyclomatic complexity and other metrics."""
        
        tree = ast.parse(content)
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1  # Base complexity
                self.max_depth = 0
                self.current_depth = 0
                
            def visit_If(self, node):
                self.complexity += 1
                self._visit_branch(node)
                
            def visit_While(self, node):
                self.complexity += 1
                self._visit_branch(node)
                
            def visit_For(self, node):
                self.complexity += 1
                self._visit_branch(node)
                
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self._visit_branch(node)
                
            def visit_With(self, node):
                self.complexity += 1
                self._visit_branch(node)
                
            def visit_Assert(self, node):
                self.complexity += 1
                self.generic_visit(node)
                
            def visit_BoolOp(self, node):
                # Each 'and'/'or' adds a decision point
                self.complexity += len(node.values) - 1
                self.generic_visit(node)
                
            def _visit_branch(self, node):
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)
                self.generic_visit(node)
                self.current_depth -= 1
                
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return ComplexityMetrics(
            cyclomatic=visitor.complexity,
            max_depth=visitor.max_depth,
            line_count=content.count('\n') + 1,
            function_count=len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            class_count=len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        )
```

## Relevance Ranking System

### Multi-Factor Ranking Architecture

```python
class RelevanceRanker:
    """Sophisticated multi-factor relevance scoring."""
    
    def __init__(self, config: RankingConfig):
        self.config = config
        self.algorithms = {
            'fast': FastRanking(),
            'balanced': BalancedRanking(),
            'thorough': ThoroughRanking(),
            'custom': CustomRanking(config.custom_weights)
        }
        
    def rank_files(
        self,
        files: List[FileAnalysis],
        prompt_context: PromptContext,
        algorithm: str = 'balanced'
    ) -> List[RankedFile]:
        """Rank files by relevance to prompt."""
        
        ranker = self.algorithms[algorithm]
        
        # Pre-compute shared data
        shared_context = SharedRankingContext(
            prompt_keywords=prompt_context.keywords,
            prompt_embedding=self._get_embedding(prompt_context.text) if algorithm == 'thorough' else None,
            import_graph=self._build_import_graph(files),
            file_statistics=self._compute_statistics(files)
        )
        
        # Score files in parallel
        with ThreadPoolExecutor(max_workers=self.config.ranking_workers) as executor:
            futures = []
            
            for file in files:
                future = executor.submit(
                    ranker.score_file,
                    file,
                    prompt_context,
                    shared_context
                )
                futures.append((file, future))
                
            # Collect results
            ranked_files = []
            for file, future in futures:
                try:
                    score = future.result(timeout=5.0)
                    if score >= self.config.relevance_threshold:
                        ranked_files.append(RankedFile(
                            file=file,
                            score=score,
                            factors=ranker.get_score_breakdown(file, prompt_context)
                        ))
                except Exception as e:
                    logger.warning(f"Failed to rank {file.path}: {e}")
                    
        # Sort by score
        ranked_files.sort(key=lambda x: x.score, reverse=True)
        
        return ranked_files
```

### Ranking Algorithms

#### Fast Ranking (5-10ms per file)

```python
class FastRanking(RankingAlgorithm):
    """Quick keyword and path-based ranking."""
    
    weights = {
        'keyword_match': 0.6,
        'path_relevance': 0.3,
        'file_type': 0.1
    }
    
    def score_file(self, file: FileAnalysis, prompt: PromptContext, context: SharedContext) -> float:
        scores = {}
        
        # Keyword matching (simple contains check)
        keyword_score = 0.0
        content_lower = file.content.lower()
        for keyword in prompt.keywords:
            if keyword.lower() in content_lower:
                keyword_score += 1.0
        scores['keyword_match'] = min(keyword_score / max(len(prompt.keywords), 1), 1.0)
        
        # Path relevance
        path_score = 0.0
        path_lower = file.path.lower()
        for keyword in prompt.keywords:
            if keyword.lower() in path_lower:
                path_score += 0.5
        
        # Special path bonuses
        if any(important in path_lower for important in ['main', 'index', 'app', 'core']):
            path_score += 0.3
            
        scores['path_relevance'] = min(path_score, 1.0)
        
        # File type relevance
        if prompt.task_type == 'test' and 'test' in path_lower:
            scores['file_type'] = 1.0
        elif prompt.task_type == 'config' and any(cfg in path_lower for cfg in ['config', 'settings', 'env']):
            scores['file_type'] = 1.0
        else:
            scores['file_type'] = 0.5
            
        # Weighted sum
        return sum(scores[k] * self.weights[k] for k in scores)
```

#### Balanced Ranking (50-100ms per file)

```python
class BalancedRanking(RankingAlgorithm):
    """Default algorithm balancing multiple factors."""
    
    weights = {
        'keyword_match': 0.25,
        'tfidf_similarity': 0.25,
        'import_centrality': 0.20,
        'path_structure': 0.15,
        'git_recency': 0.15
    }
    
    def score_file(self, file: FileAnalysis, prompt: PromptContext, context: SharedContext) -> float:
        scores = {}
        
        # 1. Enhanced keyword matching with position weighting
        scores['keyword_match'] = self._calculate_keyword_score(file, prompt.keywords)
        
        # 2. TF-IDF similarity
        scores['tfidf_similarity'] = self._calculate_tfidf_score(file.content, prompt.text)
        
        # 3. Import centrality (how many other files depend on this)
        scores['import_centrality'] = self._calculate_import_centrality(file, context.import_graph)
        
        # 4. Path structure analysis
        scores['path_structure'] = self._analyze_path_structure(file.path, prompt)
        
        # 5. Git recency (if available)
        if file.git_info:
            scores['git_recency'] = self._calculate_recency_score(file.git_info)
        else:
            scores['git_recency'] = 0.5  # Neutral if no git info
            
        return sum(scores[k] * self.weights[k] for k in scores)
    
    def _calculate_keyword_score(self, file: FileAnalysis, keywords: List[str]) -> float:
        """Advanced keyword scoring with position and frequency."""
        if not keywords or not file.content:
            return 0.0
            
        score = 0.0
        content_lower = file.content.lower()
        content_lines = content_lower.split('\n')
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Check filename (highest weight)
            if keyword_lower in os.path.basename(file.path).lower():
                score += 0.3
                
            # Check imports (high weight)
            for imp in file.imports:
                if keyword_lower in imp.module.lower():
                    score += 0.2
                    
            # Check class/function names (medium weight)
            for cls in file.classes:
                if keyword_lower in cls.name.lower():
                    score += 0.15
            for func in file.functions:
                if keyword_lower in func.name.lower():
                    score += 0.1
                    
            # Check content with position weighting
            for i, line in enumerate(content_lines):
                if keyword_lower in line:
                    # Earlier lines have higher weight
                    position_weight = 1.0 - (i / len(content_lines)) * 0.5
                    score += 0.05 * position_weight
                    
        return min(score / len(keywords), 1.0)
```

#### Thorough Ranking (200-500ms per file)

```python
class ThoroughRanking(RankingAlgorithm):
    """Deep analysis using ML and comprehensive metrics."""
    
    weights = {
        'semantic_similarity': 0.30,  # Requires ML features
        'keyword_match': 0.15,
        'ast_complexity': 0.15,
        'dependency_depth': 0.15,
        'code_patterns': 0.15,
        'git_activity': 0.10
    }
    
    def __init__(self):
        # Initialize ML model if available
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.ml_available = True
        except ImportError:
            self.ml_available = False
            logger.warning("ML features not available. Install with: pip install tenets[ml]")
    
    def score_file(self, file: FileAnalysis, prompt: PromptContext, context: SharedContext) -> float:
        scores = {}
        
        # 1. Semantic similarity using embeddings
        if self.ml_available and context.prompt_embedding is not None:
            file_embedding = self._get_file_embedding(file)
            scores['semantic_similarity'] = self._cosine_similarity(
                context.prompt_embedding,
                file_embedding
            )
        else:
            scores['semantic_similarity'] = scores.get('tfidf_similarity', 0.5)
            
        # 2. Comprehensive keyword analysis
        scores['keyword_match'] = self._deep_keyword_analysis(file, prompt)
        
        # 3. AST complexity relevance
        scores['ast_complexity'] = self._analyze_ast_relevance(file, prompt)
        
        # 4. Dependency depth
        scores['dependency_depth'] = self._calculate_dependency_depth(file, context)
        
        # 5. Code pattern matching
        scores['code_patterns'] = self._match_code_patterns(file, prompt)
        
        # 6. Git activity analysis
        scores['git_activity'] = self._analyze_git_activity(file, prompt)
        
        return sum(scores[k] * self.weights[k] for k in scores)
```

### Ranking Performance Characteristics

| Algorithm | Speed/File | Accuracy | Memory | CPU | Best Use Case |
|-----------|------------|----------|--------|-----|---------------|
| Fast | 5-10ms | 70-80% | Low (10MB) | Low | Quick edits, simple searches |
| Balanced | 50-100ms | 85-90% | Medium (50MB) | Medium | General development |
| Thorough | 200-500ms | 95%+ | High (200MB) | High | Architecture changes, debugging |

### Custom Ranking Configuration

```yaml
# .tenets/ranking.yml
custom_algorithm:
  name: "security-focused"
  weights:
    security_patterns: 0.35    # Custom factor
    keyword_match: 0.20
    import_graph: 0.20
    auth_relevance: 0.15       # Custom factor
    encryption_usage: 0.10     # Custom factor
    
  patterns:
    security:
      - "auth"
      - "crypto"
      - "password"
      - "token"
      - "permission"
      
  boost_files:
    - "*/security/*"
    - "*/auth/*"
    - "*_auth.py"
    
  penalty_files:
    - "*/test/*"
    - "*/mock/*"
```

## Summarization Engine

### Intelligent File Condensing

```python
class Summarizer:
    """Sophisticated file summarization system."""
    
    def __init__(self, config: SummarizerConfig):
        self.config = config
        self.strategies = {
            'truncate': TruncateStrategy(),
            'extract': ExtractStrategy(),
            'compress': CompressStrategy(),
            'semantic': SemanticStrategy(),  # Requires ML
            'llm': LLMStrategy()  # Optional, requires API key
        }
        
    def summarize_file(
        self,
        file: FileAnalysis,
        max_tokens: int,
        preserve_sections: List[str] = None,
        strategy: str = 'auto'
    ) -> FileSummary:
        """Summarize file to fit within token budget."""
        
        # Check if summarization needed
        current_tokens = self._count_tokens(file.content)
        if current_tokens <= max_tokens:
            return FileSummary(
                content=file.content,
                was_summarized=False,
                original_tokens=current_tokens,
                summary_tokens=current_tokens
            )
            
        # Select strategy
        if strategy == 'auto':
            strategy = self._select_best_strategy(file, max_tokens)
            
        summarizer = self.strategies[strategy]
        
        # Build summary
        summary = summarizer.summarize(
            file=file,
            max_tokens=max_tokens,
            preserve=preserve_sections or self._get_critical_sections(file)
        )
        
        # Add metadata and instructions
        summary.add_metadata({
            'strategy_used': strategy,
            'compression_ratio': summary.summary_tokens / summary.original_tokens,
            'preserved_sections': preserve_sections
        })
        
        # Add AI instructions
        summary.add_instruction(
            f"This file has been summarized from {summary.original_tokens} to "
            f"{summary.summary_tokens} tokens. To see the full file, ask: "
            f"'Show me the complete {file.path}'"
        )
        
        # Add ignored sections list
        if summary.ignored_sections:
            summary.add_instruction(
                f"Sections not included in summary: {', '.join(summary.ignored_sections)}. "
                f"Request specific sections if needed."
            )
            
        return summary
```

### Extract Strategy (Default)

```python
class ExtractStrategy(SummarizationStrategy):
    """Extract key components while preserving structure."""
    
    def summarize(self, file: FileAnalysis, max_tokens: int, preserve: List[str]) -> FileSummary:
        sections = self._extract_sections(file)
        
        # Priority order for inclusion
        priority = [
            'imports',          # Always include
            'exports',          # API surface
            'class_signatures', # Structure
            'function_signatures',
            'docstrings',       # Documentation
            'type_definitions', # Type info
            'constants',        # Configuration
            'complex_logic',    # Key algorithms
            'recent_changes'    # Git context
        ]
        
        # Build summary respecting priority and token budget
        summary_parts = []
        used_tokens = 0
        
        for section_type in priority:
            if section_type in preserve or section_type in ['imports', 'exports']:
                section = sections.get(section_type, '')
                section_tokens = self._count_tokens(section)
                
                if used_tokens + section_tokens <= max_tokens:
                    summary_parts.append(self._format_section(section_type, section))
                    used_tokens += section_tokens
                elif section_type in preserve:
                    # Must include but need to truncate
                    available = max_tokens - used_tokens
                    truncated = self._truncate_to_tokens(section, available)
                    summary_parts.append(self._format_section(section_type, truncated))
                    break
                    
        return FileSummary(
            content='\n\n'.join(summary_parts),
            was_summarized=True,
            original_tokens=self._count_tokens(file.content),
            summary_tokens=used_tokens,
            preserved_sections=preserve,
            strategy='extract'
        )
    
    def _extract_sections(self, file: FileAnalysis) -> Dict[str, str]:
        """Extract different sections from code."""
        sections = {}
        
        if file.language == 'python':
            sections['imports'] = self._extract_python_imports(file)
            sections['exports'] = self._extract_python_exports(file)
            sections['class_signatures'] = self._extract_python_classes(file)
            sections['function_signatures'] = self._extract_python_functions(file)
            sections['docstrings'] = self._extract_python_docstrings(file)
            
        # ... similar for other languages
        
        return sections
```

### LLM-Enhanced Summarization (Optional)

```python
class LLMStrategy(SummarizationStrategy):
    """Use LLM for intelligent summarization (costs money)."""
    
    def __init__(self):
        self.enabled = bool(os.getenv('TENETS_LLM_ENABLED', False))
        self.provider = os.getenv('TENETS_LLM_PROVIDER', 'openai')
        self.model = os.getenv('TENETS_LLM_MODEL', 'gpt-4o-mini')
        self.max_cost = float(os.getenv('TENETS_MAX_LLM_COST_PER_RUN', '0.10'))
        
        if self.enabled:
            self._init_llm_client()
            
    def summarize(self, file: FileAnalysis, max_tokens: int, preserve: List[str]) -> FileSummary:
        if not self.enabled:
            # Fallback to extract strategy
            return ExtractStrategy().summarize(file, max_tokens, preserve)
            
        # Estimate cost
        estimated_cost = self._estimate_cost(file.content)
        if estimated_cost > self.max_cost:
            logger.warning(f"LLM summary would cost ${estimated_cost:.2f}, exceeding limit. Using local strategy.")
            return ExtractStrategy().summarize(file, max_tokens, preserve)
            
        # Build prompt
        prompt = self._build_summary_prompt(file, max_tokens, preserve)
        
        # Call LLM
        try:
            summary_text = self._call_llm(prompt)
            
            return FileSummary(
                content=summary_text,
                was_summarized=True,
                original_tokens=self._count_tokens(file.content),
                summary_tokens=self._count_tokens(summary_text),
                strategy='llm',
                cost=estimated_cost
            )
            
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}. Falling back to local strategy.")
            return ExtractStrategy().summarize(file, max_tokens, preserve)
```

## Context Management

### Context Building Pipeline

```python
class ContextManager:
    """Orchestrates context building within token limits."""
    
    def __init__(self, config: ContextConfig):
        self.config = config
        self.token_counter = TokenCounter()
        self.formatter = ContextFormatter()
        
    def build_context(
        self,
        ranked_files: List[RankedFile],
        prompt_context: PromptContext,
        session: Optional[Session] = None,
        target_model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> ContextResult:
        """Build optimized context for LLM consumption."""
        
        # Determine token budget
        budget = self._calculate_budget(
            prompt_context,
            target_model,
            max_tokens
        )
        
        # Select files that fit
        selected_files = self._select_files_for_budget(
            ranked_files,
            budget,
            session
        )
        
        # Process files (summarize if needed)
        processed_files = self._process_files(
            selected_files,
            budget,
            session
        )
        
        # Format output
        if session and session.is_first_prompt:
            formatted = self._format_initial_context(
                processed_files,
                prompt_context,
                session
            )
        else:
            formatted = self._format_incremental_context(
                processed_files,
                prompt_context,
                session
            )
            
        # Add cost estimate
        cost_estimate = None
        if target_model:
            cost_estimate = self._estimate_cost(
                formatted.content,
                target_model
            )
            
        return ContextResult(
            content=formatted.content,
            format=formatted.format,
            token_count=formatted.tokens,
            files_included=len(processed_files),
            files_summarized=sum(1 for f in processed_files if f.was_summarized),
            cost_estimate=cost_estimate,
            metadata={
                'model': target_model,
                'session_id': session.id if session else None,
                'timestamp': datetime.now().isoformat()
            }
        )
```

### Token Budget Management

```python
class TokenBudget:
    """Manage token allocation for context building."""
    
    def __init__(
        self,
        total_limit: int,
        prompt_tokens: int,
        response_reserve: int = 4000,
        safety_margin: float = 0.95
    ):
        self.total_limit = total_limit
        self.prompt_tokens = prompt_tokens
        self.response_reserve = response_reserve
        self.safety_margin = safety_margin
        
        # Calculate available tokens for context
        self.available = int(
            (total_limit - prompt_tokens - response_reserve) * safety_margin
        )
        
        self.used = 0
        self.allocations = []
        
    def can_fit(self, tokens: int) -> bool:
        """Check if tokens fit in remaining budget."""
        return self.used + tokens <= self.available
        
    def allocate(self, item: str, tokens: int) -> bool:
        """Allocate tokens for an item."""
        if not self.can_fit(tokens):
            return False
            
        self.used += tokens
        self.allocations.append({
            'item': item,
            'tokens': tokens,
            'timestamp': datetime.now()
        })
        return True
        
    def remaining(self) -> int:
        """Get remaining token budget."""
        return self.available - self.used
        
    def utilization(self) -> float:
        """Get budget utilization percentage."""
        return (self.used / self.available) * 100 if self.available > 0 else 0
```

### File Selection Algorithm

```python
def _select_files_for_budget(
    self,
    ranked_files: List[RankedFile],
    budget: TokenBudget,
    session: Optional[Session]
) -> List[SelectedFile]:
    """Select files that fit within token budget."""
    
    selected = []
    
    # Phase 1: Include must-have files
    for file in ranked_files:
        if self._is_must_include(file, session):
            tokens = self.token_counter.count_file(file)
            if budget.can_fit(tokens):
                budget.allocate(file.path, tokens)
                selected.append(SelectedFile(
                    file=file,
                    tokens=tokens,
                    will_summarize=False
                ))
            else:
                # Must include but need to summarize
                summary_tokens = self._estimate_summary_tokens(file)
                if budget.can_fit(summary_tokens):
                    budget.allocate(file.path, summary_tokens)
                    selected.append(SelectedFile(
                        file=file,
                        tokens=summary_tokens,
                        will_summarize=True
                    ))
                    
    # Phase 2: Add high-relevance files
    for file in ranked_files:
        if file in selected:
            continue
            
        if file.score >= self.config.high_relevance_threshold:
            tokens = self.token_counter.count_file(file)
            
            if budget.can_fit(tokens):
                budget.allocate(file.path, tokens)
                selected.append(SelectedFile(
                    file=file,
                    tokens=tokens,
                    will_summarize=False
                ))
            elif budget.remaining() > self.config.min_summary_tokens:
                # Try summarized version
                summary_tokens = min(
                    self._estimate_summary_tokens(file),
                    budget.remaining()
                )
                if budget.can_fit(summary_tokens):
                    budget.allocate(file.path, summary_tokens)
                    selected.append(SelectedFile(
                        file=file,
                        tokens=summary_tokens,
                        will_summarize=True
                    ))
                    
    # Phase 3: Fill remaining space with relevant files
    remaining_files = [f for f in ranked_files if f not in selected]
    
    for file in remaining_files:
        if budget.utilization() >= 95:  # Stop at 95% full
            break
            
        tokens = self.token_counter.count_file(file)
        
        if budget.can_fit(tokens):
            budget.allocate(file.path, tokens)
            selected.append(SelectedFile(
                file=file,
                tokens=tokens,
                will_summarize=False
            ))
            
    return selected
```

## Session Management

### Session Architecture

```python
class Session:
    """Stateful context session with history."""
    
    def __init__(self, name: str, project_root: Path):
        self.id = str(uuid.uuid4())
        self.name = name
        self.project_root = project_root
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Session state
        self.prompts: List[Prompt] = []
        self.contexts: List[ContextResult] = []
        self.shown_files: Set[str] = set()
        self.ignored_files: Set[str] = set()
        self.ai_requests: List[AIRequest] = []
        
        # Flags
        self.is_first_prompt = True
        
        # Initialize storage
        self.db = SessionDB(self.id)
        self.db.create_session(self)
        
    def make_context(self, prompt_text: str) -> ContextResult:
        """Generate context for a prompt."""
        
        # Create prompt object
        prompt = Prompt(
            text=prompt_text,
            timestamp=datetime.now(),
            is_first=self.is_first_prompt
        )
        self.prompts.append(prompt)
        
        # Build context
        if self.is_first_prompt:
            context = self._make_initial_context(prompt)
            self.is_first_prompt = False
        else:
            context = self._make_incremental_context(prompt)
            
        # Save to history
        self.contexts.append(context)
        self.db.save_prompt(prompt)
        self.db.save_context(context)
        
        return context
```

### First vs Subsequent Prompts

```python
def _make_initial_context(self, prompt: Prompt) -> ContextResult:
    """Build comprehensive initial context."""
    
    # Full analysis
    tenets = Tenets(config=self.config)
    analysis = tenets.analyze(self.project_root)
    
    # Rank all files
    ranked_files = tenets.ranker.rank_files(
        analysis.files,
        prompt.context,
        algorithm='balanced'  # Use balanced for initial
    )
    
    # Build context with special formatting
    context_manager = ContextManager(self.config)
    
    # Custom format for initial prompt
    initial_format = InitialContextFormat(
        include_overview=True,
        include_structure=True,
        include_instructions=True,
        summarize_all=True  # Summarize everything initially
    )
    
    result = context_manager.build_context(
        ranked_files=ranked_files,
        prompt_context=prompt.context,
        session=self,
        format_options=initial_format
    )
    
    # The formatted output includes:
    """
    # Context for: {prompt}
    
    ## Project Overview
    - Language: Python (65%), JavaScript (35%)
    - Total files: 1,234
    - Key directories: src/, lib/, tests/
    - Recent activity: 45 commits in last week
    
    ## Relevant Files (Summarized)
    
    ### src/auth/oauth.py (2,100 lines → 200 line summary)
    [Summary focused on OAuth implementation...]
    *To see full file: "Show me src/auth/oauth.py"*
    
    ### src/models/user.py (450 lines → 100 line summary)
    [Summary of user model...]
    *To see full file: "Show me src/models/user.py"*
    
    ## AI Assistant Instructions
    
    This is our initial context. Based on these summaries:
    
    1. **Request specific files**: Tell me which files you need to see in full
       Format: SHOW: path/to/file1.py, path/to/file2.js
       
    2. **Exclude irrelevant files**: Point out files that aren't relevant
       Format: IGNORE: path/to/irrelevant1.py, path/to/irrelevant2.py
       
    3. **Search for patterns**: Ask me to find files containing specific patterns
       Format: FIND: "pattern" in *.py
       
    4. **Get more context**: Request related files, tests, or documentation
    
    ## Quick Commands
    - `tenets show <files>` - Show specific files
    - `tenets find <pattern>` - Search for pattern
    - `tenets ignore <files>` - Add to ignore list
    - `tenets tree <path>` - Show directory structure
    """
    
    return result

def _make_incremental_context(self, prompt: Prompt) -> ContextResult:
    """Build focused incremental context."""
    
    # Only analyze what's changed or newly relevant
    changed_files = self._get_changed_files_since_last_context()
    
    # Re-rank with focus on new prompt
    ranked_files = self._rank_incrementally(
        prompt.context,
        changed_files,
        previous_context=self.contexts[-1] if self.contexts else None
    )
    
    # Build streamlined context
    """
    # Updated Context for: {prompt}
    
    ## New/Changed Since Last Context
    - src/auth/permissions.py (modified)
    - src/api/endpoints.py (new file)
    
    ## Current Focus Files
    [Only files relevant to new prompt]
    
    ## Previously Shown
    - src/auth/oauth.py (full)
    - src/models/user.py (summary)
    
    ## Current Ignore List
    - tests/old_tests.py
    - docs/deprecated/
    """
```

### AI Request Handling

```python
class AIRequestHandler:
    """Process requests from AI assistant."""
    
    def handle_request(self, session: Session, request_text: str) -> ContextResult:
        """Parse and handle AI requests."""
        
        # Parse request type
        request = self._parse_request(request_text)
        
        if request.type == 'show_files':
            return self._handle_show_files(session, request.files)
            
        elif request.type == 'ignore_files':
            return self._handle_ignore_files(session, request.files)
            
        elif request.type == 'find_pattern':
            return self._handle_find_pattern(session, request.pattern)
            
        elif request.type == 'show_structure':
            return self._handle_show_structure(session, request.path)
            
    def _handle_show_files(self, session: Session, files: List[str]) -> ContextResult:
        """Show requested files in full."""
        
        shown_files = []
        not_found = []
        
        for file_path in files:
            full_path = session.project_root / file_path
            
            if full_path.exists():
                content = full_path.read_text()
                shown_files.append({
                    'path': file_path,
                    'content': content,
                    'tokens': self._count_tokens(content)
                })
                session.shown_files.add(file_path)
            else:
                not_found.append(file_path)
                
        # Format response
        """
        # Requested Files
        
        ## src/auth/oauth.py (Full Content)
        ```python
        [Complete file content]
        ```
        
        ## Not Found
        - src/missing/file.py
        
        ## Session Update
        - Total files shown: 15
        - Total tokens used: 45,234
        """
```

## Storage Architecture

### Directory Structure

```
~/.tenets/
├── cache/
│   ├── projects/
│   │   └── {project_hash}/
│   │       ├── metadata.json          # Project metadata
│   │       ├── file_analysis/
│   │       │   ├── {file_hash}.json   # Cached analysis
│   │       │   └── index.db           # File index
│   │       ├── embeddings/            # ML features only
│   │       │   └── {file_hash}.npy
│   │       ├── git/
│   │       │   ├── commits/
│   │       │   │   └── {commit_hash}.json
│   │       │   └── history.db
│   │       └── dependencies/
│   │           └── import_graph.json
│   └── external/                      # Cached external content
│       ├── github/
│       │   └── {issue_hash}.json
│       └── urls/
│           └── {url_hash}.json
├── sessions/
│   ├── active/                        # Active sessions
│   │   └── {session_id}/
│   │       ├── session.db             # SQLite database
│   │       ├── contexts/              # Context snapshots
│   │       │   └── {timestamp}.json
│   │       └── state.json             # Current state
│   └── archived/                      # Completed sessions
├── config/
│   ├── global.yaml                    # User preferences
│   ├── ranking/                       # Custom ranking configs
│   │   ├── security-focused.yaml
│   │   └── performance.yaml
│   ├── ignore_patterns.txt            # Global ignores
│   └── projects/                      # Per-project configs
│       └── {project_hash}/
│           └── config.yaml
├── logs/
│   ├── tenets.log                     # Main log
│   ├── sessions/                      # Session logs
│   │   └── {session_id}.log
│   └── performance.log                # Performance metrics
└── metrics/
    ├── usage.db                       # Usage statistics
    └── velocity/                      # Development velocity
        └── {project_hash}/
            └── metrics.json
```

### Database Schemas

#### Session Database (SQLite)

```sql
-- Main sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    project_root TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSON
);

-- Prompts history
CREATE TABLE prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_type TEXT,  -- initial|incremental|ai_request
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tokens INTEGER,
    keywords JSON,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Context snapshots
CREATE TABLE contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    prompt_id INTEGER NOT NULL,
    content BLOB,  -- Compressed context
    format TEXT,   -- markdown|xml|json
    tokens INTEGER,
    files_included INTEGER,
    files_summarized INTEGER,
    cost_estimate REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
);

-- File states
CREATE TABLE file_states (
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    state TEXT NOT NULL,  -- shown|ignored|summarized|mentioned
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    times_requested INTEGER DEFAULT 0,
    metadata JSON,
    PRIMARY KEY (session_id, file_path),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- AI requests tracking
CREATE TABLE ai_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    request_type TEXT,  -- show_files|ignore_files|find_pattern
    request_data JSON,
    response_tokens INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Session branches (for exploration)
CREATE TABLE session_branches (
    id TEXT PRIMARY KEY,
    parent_session_id TEXT NOT NULL,
    branch_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);
```

#### Cache Database

```sql
-- File analysis cache
CREATE TABLE file_cache (
    file_hash TEXT PRIMARY KEY,  -- SHA256 of content
    file_path TEXT NOT NULL,
    analysis_version TEXT,        -- Version of analyzer used
    analysis_data JSON,           -- Cached analysis results
    file_size INTEGER,
    file_mtime REAL,              -- Modification time
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1
);

-- Create indexes
CREATE INDEX idx_file_path ON file_cache(file_path);
CREATE INDEX idx_last_accessed ON file_cache(last_accessed);

-- Git cache
CREATE TABLE git_cache (
    commit_hash TEXT PRIMARY KEY,
    analysis_data JSON,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- External content cache
CREATE TABLE external_cache (
    url_hash TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    content TEXT,
    headers JSON,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl INTEGER DEFAULT 3600,  -- Time to live in seconds
    metadata JSON
);
```

### Cache Management

```python
class CacheManager:
    """Intelligent caching with TTL and size limits."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.db_path = config.cache_dir / 'cache.db'
        self._init_db()
        
    def get_file_analysis(self, file_path: Path) -> Optional[FileAnalysis]:
        """Get cached analysis if valid."""
        
        # Generate cache key
        file_hash = self._hash_file(file_path)
        
        # Check cache
        with self.db as conn:
            result = conn.execute(
                "SELECT analysis_data, file_mtime FROM file_cache WHERE file_hash = ?",
                (file_hash,)
            ).fetchone()
            
            if result:
                cached_data, cached_mtime = result
                
                # Validate cache
                if file_path.stat().st_mtime == cached_mtime:
                    # Update access time
                    conn.execute(
                        "UPDATE file_cache SET last_accessed = ?, access_count = access_count + 1 WHERE file_hash = ?",
                        (datetime.now(), file_hash)
                    )
                    
                    return FileAnalysis.from_dict(json.loads(cached_data))
                    
        return None
        
    def cache_file_analysis(self, file_path: Path, analysis: FileAnalysis):
        """Cache analysis results."""
        
        file_hash = self._hash_file(file_path)
        
        with self.db as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO file_cache 
                (file_hash, file_path, analysis_version, analysis_data, file_size, file_mtime)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_hash,
                    str(file_path),
                    self.config.analyzer_version,
                    json.dumps(analysis.to_dict()),
                    file_path.stat().st_size,
                    file_path.stat().st_mtime
                )
            )
            
    def cleanup_cache(self):
        """Remove old/unused cache entries."""
        
        # Remove entries older than TTL
        cutoff_date = datetime.now() - timedelta(days=self.config.cache_ttl_days)
        
        with self.db as conn:
            conn.execute(
                "DELETE FROM file_cache WHERE last_accessed < ?",
                (cutoff_date,)
            )
            
        # Check total cache size
        cache_size = self._get_cache_size()
        if cache_size > self.config.max_cache_size:
            # Remove least recently used
            self._evict_lru_entries(cache_size - self.config.max_cache_size)
```

## Performance Architecture

### Optimization Strategies

#### 1. Parallel Processing

```python
class ParallelProcessor:
    """Efficient parallel processing for file analysis."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        
    async def analyze_files_parallel(
        self,
        files: List[Path],
        analyzer: CodeAnalyzer
    ) -> List[FileAnalysis]:
        """Analyze files in parallel with load balancing."""
        
        # Sort files by size for better load distribution
        sorted_files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
        
        # Create batches
        batches = self._create_balanced_batches(sorted_files)
        
        # Process batches in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            futures = [
                executor.submit(self._process_batch, batch, analyzer)
                for batch in batches
            ]
            
            # Progress tracking
            with tqdm(total=len(files), desc="Analyzing files") as pbar:
                for future in as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)
                    pbar.update(len(batch_results))
                    
        return results
    
    def _create_balanced_batches(self, files: List[Path]) -> List[List[Path]]:
        """Create batches with balanced workload."""
        
        # Estimate work per file
        file_weights = [self._estimate_file_weight(f) for f in files]
        
        # Distribute files across batches
        num_batches = self.max_workers
        batches = [[] for _ in range(num_batches)]
        batch_weights = [0] * num_batches
        
        # Assign files to batches using greedy algorithm
        for file, weight in sorted(zip(files, file_weights), key=lambda x: x[1], reverse=True):
            # Find batch with minimum weight
            min_batch_idx = batch_weights.index(min(batch_weights))
            batches[min_batch_idx].append(file)
            batch_weights[min_batch_idx] += weight
            
        return [b for b in batches if b]  # Remove empty batches
    
    def _estimate_file_weight(self, file: Path) -> float:
        """Estimate processing weight for a file."""
        size = file.stat().st_size
        ext = file.suffix.lower()
        
        # Language-specific weights
        complexity_weights = {
            '.py': 1.2,   # AST parsing
            '.js': 1.1,   # Complex syntax
            '.ts': 1.3,   # Type analysis
            '.java': 1.4, # Verbose
            '.cpp': 1.5,  # Complex parsing
            '.go': 1.0,   # Simple syntax
            '.rs': 1.3,   # Ownership analysis
        }
        
        lang_weight = complexity_weights.get(ext, 1.0)
        
        # Size-based weight (non-linear)
        if size < 1000:  # < 1KB
            size_weight = 0.1
        elif size < 10000:  # < 10KB
            size_weight = 0.5
        elif size < 100000:  # < 100KB
            size_weight = 1.0
        else:  # > 100KB
            size_weight = 2.0 + (size / 1_000_000)  # Linear growth for large files
            
        return size_weight * lang_weight
```

#### 2. Streaming Architecture

```python
class StreamingAnalyzer:
    """Stream processing for large codebases."""
    
    def analyze_stream(
        self,
        file_stream: AsyncIterator[Path],
        callback: Optional[Callable[[FileAnalysis], None]] = None
    ) -> AsyncIterator[FileAnalysis]:
        """Stream analysis results as they complete."""
        
        # Pipeline stages
        pipeline = [
            self._read_file_stage,
            self._analyze_code_stage,
            self._calculate_relevance_stage,
            self._enrich_metadata_stage
        ]
        
        # Process stream through pipeline
        stream = file_stream
        for stage in pipeline:
            stream = stage(stream)
            
        # Yield results
        async for result in stream:
            if callback:
                callback(result)
            yield result
    
    async def _read_file_stage(self, stream: AsyncIterator[Path]) -> AsyncIterator[Tuple[Path, str]]:
        """Read file contents asynchronously."""
        async for path in stream:
            try:
                async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    yield (path, content)
            except Exception as e:
                logger.warning(f"Failed to read {path}: {e}")
                continue
```

#### 3. Memory Management

```python
class MemoryManager:
    """Manage memory usage during analysis."""
    
    def __init__(self, max_memory_mb: int = 2048):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.current_usage = 0
        self.gc_threshold = 0.8  # Trigger GC at 80% usage
        
    def check_memory(self):
        """Check current memory usage."""
        process = psutil.Process()
        self.current_usage = process.memory_info().rss
        
        if self.current_usage > self.max_memory * self.gc_threshold:
            self._trigger_cleanup()
            
    def _trigger_cleanup(self):
        """Trigger memory cleanup."""
        # Force garbage collection
        gc.collect()
        
        # Clear caches
        if hasattr(self, 'cache'):
            self.cache.evict_old_entries()
            
        # Log warning
        logger.warning(f"Memory usage high: {self.current_usage / 1024 / 1024:.1f}MB")
```

### Performance Benchmarks

#### File Analysis Performance

| Operation | Small File (<10KB) | Medium File (100KB) | Large File (1MB) |
|-----------|-------------------|--------------------|--------------------|
| Read | 1ms | 5ms | 20ms |
| Parse (Python) | 5ms | 50ms | 500ms |
| Parse (Generic) | 2ms | 10ms | 50ms |
| Relevance Score | 10ms | 30ms | 100ms |
| Total | 18ms | 95ms | 670ms |

#### Scaling Performance

| Codebase Size | Files | Total Size | Analysis Time | Memory Usage |
|---------------|-------|------------|---------------|--------------|
| Small | 100 | 5MB | 2s | 50MB |
| Medium | 1,000 | 50MB | 15s | 200MB |
| Large | 10,000 | 500MB | 2m | 800MB |
| Huge | 100,000 | 5GB | 20m | 3GB |
| Monorepo | 1,000,000 | 50GB | 3h | 8GB |

#### With Caching

| Operation | Cold Cache | Warm Cache | Hot Cache |
|-----------|------------|------------|-----------|
| Small Project | 2s | 0.5s | 0.1s |
| Medium Project | 15s | 3s | 0.5s |
| Large Project | 2m | 20s | 3s |

## API Design

### CLI Interface

```bash
# Primary commands
tenets make-context <prompt> [path] [options]
tenets analyze [path] [options]
tenets track-changes [options]
tenets session <subcommand>
tenets viz <type> [options]
tenets config <subcommand>

# Examples with all options
tenets make-context "implement OAuth2" ./src \
    --ranking thorough \
    --format markdown \
    --model gpt-4o \
    --max-tokens 150000 \
    --output context.md \
    --session my-feature \
    --include "*.py,*.js" \
    --exclude "test_*" \
    --no-git \
    --verbose

# Session management
tenets session start "payment-integration" --branch main
tenets session list --active
tenets session resume payment-integration
tenets session show-files api/payment.py api/stripe.py
tenets session ignore-files old_payment.py legacy/
tenets session export --format json

# Visualization
tenets viz deps . --output dependencies.svg --format graphviz
tenets viz complexity . --threshold 10 --heatmap
tenets viz velocity . --since "3 months" --team
tenets viz changes . --author "john@example.com"

# Configuration
tenets config set ranking.algorithm balanced
tenets config set output.max_tokens 100000
tenets config add ignore "*.generated.ts"
tenets config show --json
```

### Python API

```python
from tenets import Tenets, Config, RankingPreset, OutputFormat
from tenets.models import AnalysisResult, ContextResult, Session

# Basic usage
t = Tenets()

# Simple context generation
result = t.make_context(
    prompt="add caching layer",
    path="./src"
)
print(result.content)  # Ready to paste into LLM

# Advanced configuration
config = Config(
    ranking={
        'algorithm': 'thorough',
        'threshold': 0.3,
        'weights': {
            'semantic': 0.4,
            'keywords': 0.3,
            'structure': 0.3
        }
    },
    output={
        'max_tokens': 150_000,
        'format': 'markdown',
        'include_metadata': True
    },
    cache={
        'enabled': True,
        'ttl_days': 7
    }
)

t = Tenets(config=config)

# Analyze codebase
analysis = t.analyze(
    path=".",
    deep=True,
    include_git=True,
    include_metrics=True
)

print(f"Languages: {analysis.languages}")
print(f"Complexity: {analysis.avg_complexity}")
print(f"Recent changes: {analysis.recent_activity}")

# Session management
session = t.create_session("new-feature")

# First context - comprehensive
ctx1 = session.make_context("design authentication flow")
print(ctx1.content)  # Includes summaries + instructions

# AI requests specific files
ctx2 = session.show_files(["auth/oauth.py", "models/user.py"])

# AI suggests ignores
session.ignore_files(["deprecated/old_auth.py"])

# Next context - incremental
ctx3 = session.make_context("add JWT support")
print(ctx3.content)  # Only new relevant info

# Track development
changes = t.track_changes(since="last-week")
for file in changes.files:
    print(f"{file.path}: +{file.additions} -{file.deletions}")

# Visualize
viz = t.visualize(analysis, type="complexity")
viz.save("complexity.html")

# Custom ranking
@t.register_ranker("ml-focused")
def ml_ranker(file, prompt):
    score = 0.0
    if any(ml_term in file.path for ml_term in ['model', 'train', 'data']):
        score += 0.5
    if 'tensorflow' in file.imports or 'torch' in file.imports:
        score += 0.3
    return score

# Use custom ranker
result = t.make_context(
    "optimize model training",
    ranking="ml-focused"
)
```

### Plugin API

```python
from tenets.plugins import Plugin, register_plugin

@register_plugin("security-scanner")
class SecurityPlugin(Plugin):
    """Add security analysis to tenets."""
    
    def analyze_file(self, file: FileAnalysis) -> dict:
        """Analyze file for security issues."""
        vulnerabilities = []
        
        # Check for hardcoded secrets
        if self._contains_secrets(file.content):
            vulnerabilities.append({
                'type': 'hardcoded_secret',
                'severity': 'high',
                'line': self._find_secret_line(file.content)
            })
            
        # Check for SQL injection
        if self._has_sql_injection_risk(file):
            vulnerabilities.append({
                'type': 'sql_injection',
                'severity': 'medium'
            })
            
        return {'vulnerabilities': vulnerabilities}
    
    def modify_ranking(self, file: FileAnalysis, prompt: PromptContext) -> float:
        """Boost security files for security-related prompts."""
        if any(term in prompt.text.lower() for term in ['security', 'vulnerability', 'auth']):
            if 'security' in file.path or 'auth' in file.path:
                return 1.5  # 50% boost
        return 1.0  # No change
```

## Output Formatting

### Markdown Format (Default)

```markdown
# Context for: implement OAuth2 authentication

*Generated by tenets v0.5.0 at 2024-03-15 10:30:00*

## Task Analysis
- **Type**: Feature Implementation
- **Complexity**: Medium
- **Key Areas**: Authentication, OAuth2, User Management
- **Estimated Scope**: 15-20 files

## Relevant Files (15 found, 12 included)

### Core Implementation

#### src/auth/base.py (Full - 250 lines)
```python
# Complete file content here
```

#### src/auth/oauth2.py (Summarized - 1,200 → 200 lines)
```python
"""OAuth2 implementation with Google, GitHub, Facebook providers.

Key classes:
- OAuth2Handler: Main OAuth flow handler
- TokenManager: Access/refresh token management
- ProviderRegistry: Dynamic provider registration

Key methods:
- authorize(provider, scopes): Initiate OAuth flow
- callback(code, state): Handle provider callback
- refresh_token(refresh_token): Refresh access token
- revoke_token(token): Revoke access

Dependencies: requests, cryptography, jwt
External APIs: Provider OAuth endpoints

[Implementation details condensed - for full code, request: "Show me src/auth/oauth2.py"]
"""
```

### Supporting Files

#### src/models/user.py (Relevant sections - 450 → 100 lines)
```python
class User(BaseModel):
    """User model with OAuth support."""
    
    # OAuth fields
    oauth_providers = JSONField(default=dict)  # {provider: {id, tokens}}
    primary_email = EmailField(unique=True)
    verified_emails = JSONField(default=list)
    
    def link_oauth_account(self, provider: str, oauth_data: dict):
        """Link OAuth provider to user account."""
        # ... implementation ...
```

### Configuration Files

#### config/oauth_providers.yaml (Full - 50 lines)
```yaml
# OAuth provider configuration
providers:
  google:
    client_id: ${GOOGLE_CLIENT_ID}
    client_secret: ${GOOGLE_CLIENT_SECRET}
    authorize_url: "https://accounts.google.com/o/oauth2/v2/auth"
    token_url: "https://oauth2.googleapis.com/token"
    scopes:
      - "openid"
      - "email"
      - "profile"
```

## Git Context

### Recent Changes
- **2 days ago**: oauth2.py - Add token refresh mechanism (user@example.com)
- **1 week ago**: user.py - Add OAuth provider fields (user@example.com)
- **2 weeks ago**: Initial OAuth2 setup (user@example.com)

### Related Branches
- `feature/oauth2-integration` (current)
- `feature/social-login` (merged)

## Instructions for AI Assistant

Based on this context:

1. **Request specific files** if you need full implementation details:
   - Format: `SHOW: path/to/file.py`
   - Example: `SHOW: src/auth/oauth2.py, tests/test_oauth.py`

2. **Exclude irrelevant files** to refine future context:
   - Format: `IGNORE: path/to/file.py`
   - Example: `IGNORE: src/auth/old_oauth.py, docs/deprecated/`

3. **Search for patterns** if something seems missing:
   - Format: `FIND: "pattern" in *.py`
   - Example: `FIND: "OAuth" in tests/*.py`

## Summary Statistics
- **Files analyzed**: 234
- **Files included**: 12 (5 full, 7 summarized)
- **Total tokens**: 14,523 / 100,000 (14.5% of budget)
- **Relevance threshold**: 0.35
- **Analysis time**: 2.3 seconds
```

### XML Format (Claude-Optimized)

```xml
<context>
  <metadata>
    <generator>tenets v0.5.0</generator>
    <timestamp>2024-03-15T10:30:00Z</timestamp>
    <task>implement OAuth2 authentication</task>
  </metadata>
  
  <analysis>
    <keywords>oauth2, authentication, provider, token, user</keywords>
    <complexity>medium</complexity>
    <scope>
      <files_found>234</files_found>
      <files_relevant>15</files_relevant>
      <files_included>12</files_included>
    </scope>
  </analysis>
  
  <files>
    <file path="src/auth/base.py" inclusion="full" lines="250">
      <content><![CDATA[
# Complete file content here
      ]]></content>
    </file>
    
    <file path="src/auth/oauth2.py" inclusion="summary" original_lines="1200" summary_lines="200">
      <summary><![CDATA[
OAuth2 implementation with provider support...
      ]]></summary>
      <request_full>Show me src/auth/oauth2.py</request_full>
    </file>
  </files>
  
  <git_context>
    <recent_commits>
      <commit sha="abc123" date="2024-03-13" author="user@example.com">
        Add token refresh mechanism
      </commit>
    </recent_commits>
  </git_context>
  
  <instructions>
    <request_files>SHOW: path/to/file</request_files>
    <ignore_files>IGNORE: path/to/file</ignore_files>
    <search_pattern>FIND: "pattern" in *.ext</search_pattern>
  </instructions>
</context>
```

### JSON Format (API-Friendly)

```json
{
  "context": {
    "task": "implement OAuth2 authentication",
    "generated_at": "2024-03-15T10:30:00Z",
    "generator": "tenets v0.5.0"
  },
  "analysis": {
    "keywords": ["oauth2", "authentication", "provider", "token", "user"],
    "task_type": "feature",
    "complexity": "medium",
    "estimated_scope": {
      "files": 15,
      "lines_of_code": 3500
    }
  },
  "files": [
    {
      "path": "src/auth/base.py",
      "inclusion": "full",
      "lines": 250,
      "relevance_score": 0.92,
      "content": "# Full file content..."
    },
    {
      "path": "src/auth/oauth2.py",
      "inclusion": "summary",
      "original_lines": 1200,
      "summary_lines": 200,
      "relevance_score": 0.98,
      "summary": "OAuth2 implementation...",
      "request_full_command": "SHOW: src/auth/oauth2.py"
    }
  ],
  "statistics": {
    "files_analyzed": 234,
    "files_included": 12,
    "files_full": 5,
    "files_summarized": 7,
    "total_tokens": 14523,
    "token_budget": 100000,
    "analysis_time_seconds": 2.3
  },
  "instructions": {
    "show_files": "SHOW: file1, file2",
    "ignore_files": "IGNORE: file1, file2",
    "search": "FIND: \"pattern\" in *.ext"
  }
}
```

## Development Tracking

### Velocity Metrics

```python
class VelocityTracker:
    """Track development velocity and patterns."""
    
    def calculate_velocity(
        self,
        repo: git.Repo,
        since: datetime,
        until: datetime = None,
        team: bool = False
    ) -> VelocityMetrics:
        """Calculate velocity metrics for time period."""
        
        metrics = VelocityMetrics()
        
        # Analyze commits
        commits = list(repo.iter_commits(since=since, until=until))
        
        # Basic metrics
        metrics.commit_count = len(commits)
        metrics.active_days = len(set(c.committed_datetime.date() for c in commits))
        
        # Analyze changes
        for commit in commits:
            stats = commit.stats
            metrics.lines_added += stats.total['insertions']
            metrics.lines_deleted += stats.total['deletions']
            metrics.files_changed += stats.total['files']
            
            # Track authors
            author = str(commit.author)
            metrics.contributors[author] = metrics.contributors.get(author, 0) + 1
            
        # Calculate derived metrics
        metrics.avg_commits_per_day = metrics.commit_count / max(metrics.active_days, 1)
        metrics.code_churn = (metrics.lines_added + metrics.lines_deleted) / 2
        
        # Complexity trends
        metrics.complexity_trend = self._analyze_complexity_trend(repo, commits)
        
        # Hotspots
        metrics.hotspots = self._identify_hotspots(commits)
        
        return metrics
```

### Change Pattern Analysis

```python
class ChangeAnalyzer:
    """Analyze patterns in code changes."""
    
    def analyze_patterns(self, repo: git.Repo) -> ChangePatterns:
        """Identify patterns in how code changes."""
        
        patterns = ChangePatterns()
        
        # Time-based patterns
        patterns.busy_hours = self._analyze_commit_times(repo)
        patterns.busy_days = self._analyze_commit_days(repo)
        
        # File coupling
        patterns.coupled_files = self._find_coupled_files(repo)
        
        # Change categories
        patterns.change_types = self._categorize_changes(repo)
        
        return patterns
    
    def _find_coupled_files(self, repo: git.Repo) -> List[FileCoupling]:
        """Find files that often change together."""
        
        # Build co-change matrix
        cochange_counts = defaultdict(lambda: defaultdict(int))
        
        for commit in repo.iter_commits(max_count=1000):
            files = list(commit.stats.files.keys())
            
            # Count co-occurrences
            for i, file1 in enumerate(files):
                for file2 in files[i+1:]:
                    cochange_counts[file1][file2] += 1
                    cochange_counts[file2][file1] += 1
                    
        # Find strong couplings
        couplings = []
        for file1, related in cochange_counts.items():
            for file2, count in related.items():
                if count >= 5:  # Threshold
                    couplings.append(FileCoupling(
                        file1=file1,
                        file2=file2,
                        change_count=count,
                        confidence=count / 10.0  # Simple confidence score
                    ))
                    
        return sorted(couplings, key=lambda c: c.change_count, reverse=True)
```

## Visualization System

### Dependency Graphs

```python
class DependencyVisualizer:
    """Generate dependency visualizations."""
    
    def generate_graph(
        self,
        analysis: AnalysisResult,
        format: str = "svg",
        layout: str = "hierarchical"
    ) -> str:
        """Generate dependency graph."""
        
        # Build graph
        graph = nx.DiGraph()
        
        # Add nodes
        for file in analysis.files:
            graph.add_node(
                file.path,
                language=file.language,
                complexity=file.complexity,
                size=file.size
            )
            
        # Add edges (dependencies)
        for file in analysis.files:
            for imp in file.imports:
                # Resolve import to file
                target = self._resolve_import(imp, analysis)
                if target:
                    graph.add_edge(file.path, target)
                    
        # Generate visualization
        if format == "mermaid":
            return self._to_mermaid(graph)
        elif format == "dot":
            return self._to_graphviz(graph)
        elif format == "d3":
            return self._to_d3_json(graph)
            
    def _to_mermaid(self, graph: nx.DiGraph) -> str:
        """Convert to Mermaid diagram."""
        lines = ["graph TD"]
        
        # Add nodes with styling
        for node, data in graph.nodes(data=True):
            node_id = node.replace("/", "_").replace(".", "_")
            label = os.path.basename(node)
            
            # Style based on language
            if data.get('language') == 'python':
                lines.append(f"    {node_id}[{label}]:::python")
            elif data.get('language') == 'javascript':
                lines.append(f"    {node_id}[{label}]:::javascript")
            else:
                lines.append(f"    {node_id}[{label}]")
                
        # Add edges
        for source, target in graph.edges():
            source_id = source.replace("/", "_").replace(".", "_")
            target_id = target.replace("/", "_").replace(".", "_")
            lines.append(f"    {source_id} --> {target_id}")
            
        # Add styling
        lines.extend([
            "",
            "classDef python fill:#3776ab,stroke:#333,stroke-width:2px;",
            "classDef javascript fill:#f7df1e,stroke:#333,stroke-width:2px;"
        ])
        
        return "\n".join(lines)
```

### Complexity Heatmaps

```python
class ComplexityVisualizer:
    """Visualize code complexity."""
    
    def generate_heatmap(
        self,
        analysis: AnalysisResult,
        metric: str = "cyclomatic"
    ) -> str:
        """Generate complexity heatmap."""
        
        # Group files by directory
        dir_complexities = defaultdict(list)
        
        for file in analysis.files:
            dir_path = os.path.dirname(file.path)
            complexity = file.metrics.get(metric, 0)
            dir_complexities[dir_path].append({
                'file': os.path.basename(file.path),
                'complexity': complexity,
                'size': file.size
            })
            
        # Generate treemap data
        treemap_data = {
            'name': 'root',
            'children': []
        }
        
        for dir_path, files in dir_complexities.items():
            dir_node = {
                'name': dir_path or '.',
                'children': [
                    {
                        'name': f['file'],
                        'value': f['complexity'],
                        'size': f['size']
                    }
                    for f in files
                ]
            }
            treemap_data['children'].append(dir_node)
            
        # Generate HTML with D3.js
        return self._generate_d3_treemap(treemap_data)
```

## SaaS Platform Architecture

### Multi-Tenant Architecture

```python
class TenantManager:
    """Manage multi-tenant isolation."""
    
    def __init__(self):
        self.tenant_registry = {}
        
    def create_tenant(self, org_id: str, plan: str) -> Tenant:
        """Create new tenant with isolated resources."""
        
        tenant = Tenant(
            id=org_id,
            plan=plan,
            storage_quota=self._get_storage_quota(plan),
            user_limit=self._get_user_limit(plan),
            feature_flags=self._get_features(plan)
        )
        
        # Create isolated storage
        tenant.storage_path = Path(f"/data/tenants/{org_id}")
        tenant.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create isolated database
        tenant.db_name = f"tenets_{org_id}"
        self._create_tenant_db(tenant.db_name)
        
        # Create isolated cache
        tenant.cache_prefix = f"tenets:{org_id}:"
        
        self.tenant_registry[org_id] = tenant
        return tenant
```

### Real-Time Collaboration

```python
class CollaborationServer:
    """WebSocket server for real-time collaboration."""
    
    def __init__(self):
        self.sessions = {}
        self.connections = defaultdict(set)
        
    async def handle_connection(self, websocket, path):
        """Handle new WebSocket connection."""
        
        # Authenticate
        auth_data = await self._authenticate(websocket)
        if not auth_data:
            await websocket.close(1008, "Authentication failed")
            return
            
        user_id = auth_data['user_id']
        session_id = auth_data['session_id']
        
        # Join session
        self.connections[session_id].add(websocket)
        
        try:
            # Send current state
            await self._send_session_state(websocket, session_id)
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(
                    websocket,
                    session_id,
                    user_id,
                    json.loads(message)
                )
                
        finally:
            # Clean up
            self.connections[session_id].remove(websocket)
            
    async def _handle_message(self, websocket, session_id, user_id, message):
        """Handle collaboration message."""
        
        msg_type = message['type']
        
        if msg_type == 'context_update':
            # Broadcast context changes
            await self._broadcast_to_session(
                session_id,
                {
                    'type': 'context_update',
                    'user': user_id,
                    'changes': message['changes']
                },
                exclude=websocket
            )
            
        elif msg_type == 'cursor_position':
            # Share cursor positions
            await self._broadcast_to_session(
                session_id,
                {
                    'type': 'cursor_position',
                    'user': user_id,
                    'position': message['position']
                },
                exclude=websocket
            )
```

### API Gateway

```python
class APIGateway:
    """FastAPI-based API gateway."""
    
    def __init__(self):
        self.app = FastAPI(title="tenets API", version="1.0.0")
        self.setup_routes()
        self.setup_middleware()
        
    def setup_routes(self):
        """Setup API routes."""
        
        # Analysis endpoints
        @self.app.post("/api/v1/analyze")
        async def analyze_project(
            project: ProjectAnalysisRequest,
            tenant: Tenant = Depends(get_current_tenant)
        ):
            """Analyze a project."""
            # Rate limiting
            await self.rate_limiter.check(tenant.id)
            
            # Queue analysis job
            job_id = await self.job_queue.submit(
                "analyze_project",
                tenant_id=tenant.id,
                project_data=project.dict()
            )
            
            return {"job_id": job_id, "status": "queued"}
            
        # Context generation
        @self.app.post("/api/v1/context")
        async def generate_context(
            request: ContextRequest,
            tenant: Tenant = Depends(get_current_tenant)
        ):
            """Generate context for prompt."""
            # Check quota
            if not await self.quota_manager.check(tenant, "context_generation"):
                raise HTTPException(429, "Quota exceeded")
                
            # Process request
            result = await self.context_service.generate(
                tenant_id=tenant.id,
                prompt=request.prompt,
                project_id=request.project_id,
                options=request.options
            )
            
            # Track usage
            await self.usage_tracker.track(
                tenant.id,
                "context_generation",
                tokens=result.token_count
            )
            
            return result
```

### Background Workers

```python
class AnalysisWorker:
    """Background worker for analysis jobs."""
    
    def __init__(self):
        self.queue = JobQueue()
        self.storage = DistributedStorage()
        
    async def run(self):
        """Main worker loop."""
        
        while True:
            # Get next job
            job = await self.queue.get_next()
            if not job:
                await asyncio.sleep(1)
                continue
                
            try:
                # Process job
                await self._process_job(job)
                
                # Mark complete
                await self.queue.mark_complete(job.id)
                
            except Exception as e:
                # Handle failure
                await self.queue.mark_failed(
                    job.id,
                    error=str(e)
                )
                
    async def _process_job(self, job: Job):
        """Process analysis job."""
        
        if job.type == "analyze_project":
            # Get project data
            project = await self.storage.get_project(
                job.tenant_id,
                job.data['project_id']
            )
            
            # Run analysis
            analyzer = Analyzer(tenant_id=job.tenant_id)
            result = await analyzer.analyze_async(project.path)
            
            # Store results
            await self.storage.save_analysis(
                job.tenant_id,
                project.id,
                result
            )
            
            # Notify completion
            await self.notifier.notify(
                job.tenant_id,
                "analysis_complete",
                project_id=project.id
            )
```

### Pricing & Billing

```python
class BillingService:
    """Handle subscription billing."""
    
    PLANS = {
        'free': {
            'price': 0,
            'projects': 5,
            'users': 1,
            'storage_gb': 1,
            'features': ['basic_analysis', 'context_generation']
        },
        'pro': {
            'price': 10,  # per user per month
            'projects': -1,  # unlimited
            'users': 10,
            'storage_gb': 50,
            'features': ['all', 'priority_processing', 'api_access']
        },
        'team': {
            'price': 25,
            'projects': -1,
            'users': -1,
            'storage_gb': 500,
            'features': ['all', 'team_collaboration', 'sso', 'audit_logs']
        },
        'enterprise': {
            'price': 'custom',
            'projects': -1,
            'users': -1,
            'storage_gb': -1,
            'features': ['all', 'custom_deployment', 'sla', 'dedicated_support']
        }
    }
    
    async def check_usage(self, tenant: Tenant, resource: str) -> bool:
        """Check if tenant can use resource."""
        
        plan = self.PLANS[tenant.plan]
        
        if resource == 'projects':
            current = await self.count_projects(tenant.id)
            return plan['projects'] == -1 or current < plan['projects']
            
        elif resource == 'storage':
            current_gb = await self.get_storage_usage(tenant.id) / 1_000_000_000
            return plan['storage_gb'] == -1 or current_gb < plan['storage_gb']
            
        elif resource in plan['features']:
            return True
            
        return False
```

## Security & Privacy

### Local Security

```python
class SecurityManager:
    """Handle security for local operations."""
    
    def __init__(self):
        self.secret_patterns = [
            r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([^"\'\s]+)',
            r'(?i)(secret|password|passwd|pwd)\s*[:=]\s*["\']?([^"\'\s]+)',
            r'(?i)(token)\s*[:=]\s*["\']?([^"\'\s]+)',
            r'-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----',
        ]
        
    def scan_file_for_secrets(self, content: str) -> List[SecretMatch]:
        """Scan file content for potential secrets."""
        
        matches = []
        
        for pattern in self.secret_patterns:
            for match in re.finditer(pattern, content):
                matches.append(SecretMatch(
                    type='potential_secret',
                    pattern=pattern,
                    line_number=content[:match.start()].count('\n') + 1,
                    severity='high'
                ))
                
        return matches
    
    def sanitize_output(self, content: str) -> str:
        """Remove sensitive information from output."""
        
        # Replace potential secrets
        for pattern in self.secret_patterns:
            content = re.sub(
                pattern,
                lambda m: m.group(0).split('=')[0] + '=***REDACTED***',
                content
            )
            
        return content
```

### Cloud Security

```python
class CloudSecurityManager:
    """Security for SaaS platform."""
    
    def __init__(self):
        self.encryption_key = self._load_master_key()
        
    def encrypt_at_rest(self, data: bytes, tenant_id: str) -> bytes:
        """Encrypt data for storage."""
        
        # Use tenant-specific key derivation
        tenant_key = self._derive_tenant_key(tenant_id)
        
        # Encrypt with AES-256-GCM
        cipher = Cipher(
            algorithms.AES(tenant_key),
            modes.GCM(os.urandom(12))
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return encryptor.tag + ciphertext
    
    def enforce_access_control(
        self,
        user: User,
        resource: Resource,
        action: str
    ) -> bool:
        """Check access permissions."""
        
        # RBAC check
        if not self.has_role_permission(user.role, resource.type, action):
            return False
            
        # Resource-level check
        if resource.owner_id != user.tenant_id:
            return False
            
        # Additional checks for sensitive operations
        if action in ['delete', 'export']:
            return self.requires_mfa(user) and user.mfa_verified
            
        return True
```

## Extensibility & Plugins

### Plugin System Architecture

```python
class PluginManager:
    """Manage tenets plugins."""
    
    def __init__(self):
        self.plugins = {}
        self.hooks = defaultdict(list)
        
    def register_plugin(self, plugin: Plugin):
        """Register a new plugin."""
        
        # Validate plugin
        if not self._validate_plugin(plugin):
            raise InvalidPluginError(f"Plugin {plugin.name} validation failed")
            
        # Register hooks
        for hook_name, handler in plugin.get_hooks().items():
            self.hooks[hook_name].append((plugin, handler))
            
        # Store plugin
        self.plugins[plugin.name] = plugin
        
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")
        
    def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute all handlers for a hook."""
        
        results = []
        
        for plugin, handler in self.hooks[hook_name]:
            try:
                result = handler(*args, **kwargs)
                results.append((plugin.name, result))
            except Exception as e:
                logger.error(f"Plugin {plugin.name} hook {hook_name} failed: {e}")
                
        return results
```

### Example Plugins

```python
# Django-specific plugin
@register_plugin("django-analyzer")
class DjangoPlugin(Plugin):
    """Enhanced analysis for Django projects."""
    
    name = "django-analyzer"
    version = "1.0.0"
    
    def get_hooks(self):
        return {
            'analyze_file': self.analyze_django_file,
            'score_relevance': self.score_django_relevance,
            'format_context': self.format_django_context
        }
    
    def analyze_django_file(self, file: FileAnalysis) -> dict:
        """Add Django-specific analysis."""
        
        django_data = {}
        
        # Detect Django components
        if 'models.py' in file.path:
            django_data['type'] = 'models'
            django_data['models'] = self._extract_django_models(file.content)
            
        elif 'views.py' in file.path:
            django_data['type'] = 'views'
            django_data['views'] = self._extract_django_views(file.content)
            
        elif 'urls.py' in file.path:
            django_data['type'] = 'urls'
            django_data['urlpatterns'] = self._extract_url_patterns(file.content)
            
        return {'django': django_data}
```

## Deployment Options

### Local Installation

```bash
# From PyPI
pip install tenets

# From source
git clone https://github.com/jddunn/tenets.git
cd tenets
pip install -e .

# With all features
pip install tenets[all]
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install tenets
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

# Create non-root user
RUN useradd -m tenets
USER tenets

ENTRYPOINT ["tenets"]
```

### Kubernetes Deployment

```yaml
# tenets-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tenets-api
  namespace: tenets
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tenets-api
  template:
    metadata:
      labels:
        app: tenets-api
    spec:
      containers:
      - name: api
        image: tenets/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: tenets-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: tenets-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: tenets-api
  namespace: tenets
spec:
  selector:
    app: tenets-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Self-Hosted Enterprise

```python
# enterprise_config.py
class EnterpriseConfig:
    """Configuration for self-hosted enterprise deployment."""
    
    # Authentication
    SSO_ENABLED = True
    SSO_PROVIDER = "saml2"
    SSO_METADATA_URL = "https://idp.company.com/metadata"
    
    # Security
    ENCRYPTION_AT_REST = True
    AUDIT_LOGGING = True
    COMPLIANCE_MODE = "sox"  # sox, hipaa, gdpr
    
    # Infrastructure
    DATABASE_URL = "postgresql://..."
    REDIS_CLUSTER = ["redis-1:6379", "redis-2:6379", "redis-3:6379"]
    S3_BUCKET = "company-tenets-storage"
    
    # Limits
    MAX_FILE_SIZE = 50_000_000  # 50MB
    MAX_PROJECT_SIZE = 10_000_000_000  # 10GB
    MAX_ANALYSIS_TIME = 3600  # 1 hour
    
    # Features
    ENABLED_PLUGINS = [
        "security-scanner",
        "compliance-checker",
        "custom-metrics"
    ]
```

## Detailed Roadmap

### Phase 1: Foundation (Q1 2024)
**Goal**: Production-ready core with essential features

- [x] Core file discovery and scanning
- [x] Basic code analysis (Python focus)
- [x] Multi-factor relevance ranking
- [x] Local summarization
- [x] Git integration
- [ ] Session management implementation
- [ ] Input adapter system completion
- [ ] Comprehensive test suite (>80% coverage)
- [ ] Performance optimization (<5s for 10k files)
- [ ] Documentation website

**Deliverables**:
- v0.1.0 release on PyPI
- Basic CLI fully functional
- Python API stable

### Phase 2: Intelligence Enhancement (Q2 2024)
**Goal**: Best-in-class code understanding

- [ ] Full AST analysis for top 10 languages
- [ ] ML-powered semantic search (optional)
- [ ] Pattern learning from user feedback
- [ ] Smart ignore suggestions
- [ ] Code quality scoring
- [ ] Dependency vulnerability scanning
- [ ] Cross-repository analysis
- [ ] Real-time file watching
- [ ] IDE extensions (VS Code, JetBrains)

**Deliverables**:
- v0.5.0 with ML features
- VS Code extension
- Advanced ranking algorithms

### Phase 3: Collaboration Platform (Q3 2024)
**Goal**: Team features and web platform

- [ ] Web UI launch at tenets.dev
- [ ] Real-time collaborative sessions
- [ ] Team workspaces
- [ ] Shared context libraries
- [ ] Code review integration
- [ ] CI/CD plugins (GitHub Actions, GitLab CI)
- [ ] API access with rate limiting
- [ ] Basic analytics dashboard
- [ ] Slack/Discord integrations

**Deliverables**:
- tenets.dev beta launch
- Team plan available
- REST & GraphQL APIs

### Phase 4: Enterprise & Scale (Q4 2024)
**Goal**: Enterprise-ready platform

- [ ] SSO/SAML support
- [ ] Advanced audit logging
- [ ] Compliance frameworks (SOC2, HIPAA)
- [ ] Self-hosted deployment options
- [ ] White-label capabilities
- [ ] SLA guarantees
- [ ] Advanced security scanning
- [ ] Custom plugin marketplace
- [ ] AI model fine-tuning support
- [ ] Multi-region deployment

**Deliverables**:
- Enterprise plan launch
- Self-hosted option
- 99.9% uptime SLA

### Future Vision (2025+)

1. **AI-Native Development Environment**
   - Integrated IDE with tenets built-in
   - Real-time AI suggestions based on context
   - Automatic refactoring proposals

2. **Code Intelligence API**
   - tenets-as-a-Service for other tools
   - Standardized code analysis protocol
   - Industry-wide adoption

3. **Learning Platform**
   - Understand how developers learn codebases
   - Personalized onboarding paths
   - Knowledge graph of code relationships

4. **Open Ecosystem**
   - Plugin marketplace
   - Community-driven analyzers
   - Integration with all major tools

---