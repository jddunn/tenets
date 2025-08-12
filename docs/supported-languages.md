# Supported Languages

Tenets ships with first-class analyzers for a broad set of ecosystems. Each analyzer extracts structural signals (definitions, imports, dependencies) that feed ranking.

| Language / Tech | Analyzer Class | Extensions | Notes |
|-----------------|----------------|-----------|-------|
| Python | PythonAnalyzer | .py | AST parsing, imports, class/function graph |
| JavaScript / TypeScript* | JavaScriptAnalyzer | .js, .jsx, .ts, .tsx | Lightweight regex/heuristic (TypeScript treated as JS for now) |
| Java | JavaAnalyzer | .java | Package & import extraction |
| Go | GoAnalyzer | .go | Import graph, function signatures |
| C# | CSharpAnalyzer | .cs | Namespace & using directives (great for Unity scripts) |
| C / C++ | CppAnalyzer | .c, .h, .cpp, .hpp | Include graph detection |
| Rust | RustAnalyzer | .rs | Module/use extraction |
| Scala | ScalaAnalyzer | .scala | Object/class/trait discovery |
| Kotlin | KotlinAnalyzer | .kt, .kts | Package & import extraction |
| Swift | SwiftAnalyzer | .swift | Import/use lines |
| PHP | PhpAnalyzer | .php | Namespace/use detection |
| Ruby | RubyAnalyzer | .rb | Class/module defs |
| Dart | DartAnalyzer | .dart | Import and class/function capture |
| GDScript (Godot) | GDScriptAnalyzer | .gd | Signals + extends parsing |
| HTML | HTMLAnalyzer | .html, .htm | Link/script/style references |
| CSS | CSSAnalyzer | .css | @import and rule summarization |
| Generic Text | GenericAnalyzer | * (fallback) | Used when no specific analyzer matches |

*TypeScript currently leverages the JavaScript analyzer (roadmap: richer TS-specific parsing).

## Detection Rules

File extension matching selects the analyzer. Unsupported files fall back to the generic analyzer supplying minimal term frequency and path heuristics.

## Adding a New Language

1. Subclass `LanguageAnalyzer` in `tenets/core/analysis/implementations`
2. Implement `match(path)` and `analyze(content)`
3. Register in the analyzer registry (if dynamic) or import to ensure discovery
4. Add tests under `tests/core/analysis/implementations`
5. Update this page

## Roadmap

Planned enhancements:

- Deeper TypeScript semantic model
- SQL schema/introspection analyzer
- Proto / gRPC IDL support
- Framework-aware weighting (Django, Rails, Spring) optional modules

Got a priority? Open an issue or PR.
