# Code Analysis Engine

## Language Analyzer Architecture

```mermaid
graph TB
    subgraph "Base Analyzer Interface"
        BASE[LanguageAnalyzer<br/>Abstract Base Class]
        EXTRACT_IMP[extract_imports()]
        EXTRACT_EXP[extract_exports()]
        EXTRACT_CLS[extract_classes()]
        EXTRACT_FN[extract_functions()]
        CALC_COMP[calculate_complexity()]
        TRACE_DEP[trace_dependencies()]
    end

    subgraph "Language-Specific Analyzers"
        PYTHON[Python Analyzer<br/>Full AST parsing]
        JAVASCRIPT[JavaScript Analyzer<br/>ES6+ support]
        GOLANG[Go Analyzer<br/>Package detection]
        JAVA[Java Analyzer<br/>OOP patterns]
        RUST[Rust Analyzer<br/>Ownership patterns]
        GENERIC[Generic Analyzer<br/>Pattern-based fallback]
    end

    subgraph "Analysis Features"
        AST[AST Parsing]
        IMPORTS[Import Resolution]
        TYPES[Type Extraction]
        DOCS[Documentation Parsing]
        PATTERNS[Code Patterns]
        COMPLEXITY[Complexity Metrics]
    end

    BASE --> EXTRACT_IMP
    BASE --> EXTRACT_EXP
    BASE --> EXTRACT_CLS
    BASE --> EXTRACT_FN
    BASE --> CALC_COMP
    BASE --> TRACE_DEP

    BASE --> PYTHON
    BASE --> JAVASCRIPT
    BASE --> GOLANG
    BASE --> JAVA
    BASE --> RUST
    BASE --> GENERIC

    PYTHON --> AST
    PYTHON --> IMPORTS
    PYTHON --> TYPES
    PYTHON --> DOCS

    JAVASCRIPT --> PATTERNS
    GOLANG --> PATTERNS
    JAVA --> COMPLEXITY
    RUST --> COMPLEXITY
    GENERIC --> PATTERNS
```

## Python Analyzer Detail

```mermaid
graph LR
    subgraph "Python AST Analysis"
        AST_PARSE[AST Parser]
        NODE_VISIT[Node Visitor]
        SYMBOL_TABLE[Symbol Table]
    end

    subgraph "Code Structure"
        CLASSES[Class Definitions<br/>Inheritance chains]
        FUNCTIONS[Function Definitions<br/>Async detection]
        DECORATORS[Decorator Analysis]
        TYPE_HINTS[Type Hint Extraction]
    end

    subgraph "Import Analysis"
        ABS_IMP[Absolute Imports]
        REL_IMP[Relative Imports]
        STAR_IMP[Star Imports]
        IMPORT_GRAPH[Import Graph Building]
    end

    subgraph "Complexity Metrics"
        CYCLO[Cyclomatic Complexity<br/>+1 for if, for, while]
        COGNITIVE[Cognitive Complexity<br/>Nesting penalties]
        HALSTEAD[Halstead Metrics<br/>Operators/operands]
    end

    AST_PARSE --> NODE_VISIT
    NODE_VISIT --> SYMBOL_TABLE

    SYMBOL_TABLE --> CLASSES
    SYMBOL_TABLE --> FUNCTIONS
    SYMBOL_TABLE --> DECORATORS
    SYMBOL_TABLE --> TYPE_HINTS

    NODE_VISIT --> ABS_IMP
    NODE_VISIT --> REL_IMP
    NODE_VISIT --> STAR_IMP
    ABS_IMP --> IMPORT_GRAPH
    REL_IMP --> IMPORT_GRAPH
    STAR_IMP --> IMPORT_GRAPH

    SYMBOL_TABLE --> CYCLO
    SYMBOL_TABLE --> COGNITIVE
    SYMBOL_TABLE --> HALSTEAD
```