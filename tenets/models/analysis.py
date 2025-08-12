"""Code analysis data models used by the analyzer system.

This module contains all data structures used by the code analysis subsystem,
including file analysis results, project metrics, and dependency graphs.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class ImportInfo:
    """Information about an import statement in code.

    Represents imports across different languages (import, require, include, use).

    Attributes:
        module: The imported module/package name
        alias: Any alias assigned to the import
        line: Line number where import appears
        type: Type of import (import, from, require, include)
        is_relative: Whether this is a relative import
        level: Relative import level (Python), 0 for absolute
        from_module: Module specified in a 'from X import ...' statement
    """

    module: str
    alias: Optional[str] = None
    line: int = 0
    type: str = "import"
    is_relative: bool = False
    # Compatibility: some analyzers provide 'level' for Python relative imports
    level: int = 0
    # Additional metadata for 'from' imports
    from_module: Optional[str] = None
    # Extended optional fields used by analyzers across languages
    is_stdlib: bool = False
    is_stl: bool = False
    conditional: bool = False
    is_project_header: bool = False
    # Project-local include/import (used by some analyzers like Ruby)
    is_project_file: bool = False
    has_include_guard: bool = False
    uses_pragma_once: bool = False
    import_clause: Optional[str] = None
    original_name: Optional[str] = None
    is_wildcard: bool = False
    category: Optional[str] = None
    package: Optional[str] = None
    namespace: Optional[str] = None
    import_type: Optional[str] = None
    is_file_include: bool = False
    is_dynamic: bool = False
    # Ruby/General: whether this load can reload the file
    reloads: bool = False
    # Ruby/General: whether this represents a gem/library import
    is_gem: bool = False
    is_autoload: bool = False
    # Ruby autoload laziness flag
    lazy_load: bool = False
    # C#/Unity specific flags and context
    is_unity: bool = False
    namespace_context: Optional[str] = None
    is_project_reference: bool = False
    is_external: bool = False
    is_module_declaration: bool = False
    is_glob: bool = False
    is_dev_dependency: bool = False
    is_dependency: bool = False
    version: Optional[str] = None
    # Ruby Bundler: indicates Bundler.require was detected
    loads_all_gems: bool = False
    # New compatibility fields referenced by analyzers/tests
    is_renamed: bool = False
    is_given: bool = False
    # GDScript-specific
    is_resource: bool = False
    resource_type: Optional[str] = None
    is_runtime_load: bool = False
    is_inheritance: bool = False
    parent_type: Optional[str] = None
    associated_class: Optional[str] = None
    is_editor_script: bool = False
    # Dart/General package/library flags
    is_package_declaration: bool = False
    is_package: bool = False
    is_dart_core: bool = False
    is_deferred: bool = False
    # Dart show/hide lists default to empty for easier use in tests
    show_symbols: List[str] = field(default_factory=list)
    hide_symbols: List[str] = field(default_factory=list)
    is_part_file: bool = False
    is_library_part: bool = False
    is_library_declaration: bool = False
    package_context: Optional[str] = None
    is_android: bool = False
    # HTML-specific
    integrity: Optional[str] = None
    crossorigin: Optional[str] = None
    is_async: bool = False
    is_defer: bool = False
    is_module: bool = False
    as_type: Optional[str] = None
    # CSS/HTML and other analyzer optional fields
    media_query: Optional[str] = None
    layer: Optional[str] = None
    supports: Optional[str] = None
    config: Optional[str] = None
    visibility: Optional[str] = None
    composes: Optional[str] = None
    # Swift/Apple optional flags
    is_apple: bool = False
    is_apple_framework: bool = False
    # Swift-specific import flags
    is_testable: bool = False
    is_exported: bool = False
    import_kind: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all import information
        """
        return asdict(self)


@dataclass
class ComplexityMetrics:
    """Code complexity metrics for analysis.

    Contains various complexity measurements used to assess code quality
    and maintainability.

    Attributes:
        cyclomatic: McCabe cyclomatic complexity
        cognitive: Cognitive complexity score
        halstead_volume: Halstead volume metric
        halstead_difficulty: Halstead difficulty metric
        maintainability_index: Maintainability index (0-100)
        line_count: Total number of lines
        function_count: Number of functions
        class_count: Number of classes
        max_depth: Maximum nesting depth
        comment_ratio: Ratio of comments to code
        code_lines: Number of actual code lines
        comment_lines: Number of comment lines
        character_count: Total number of characters
        key_count: Number of key/value pairs (for config files)
        section_count: Number of sections (for structured files)
        tag_count: Number of tags (for markup languages)
        header_count: Number of headers (for document files)
        column_count: Number of columns (for tabular data)
        row_count: Number of rows (for tabular data)
    """

    cyclomatic: int = 1
    cognitive: int = 0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    maintainability_index: float = 100.0
    line_count: int = 0
    function_count: int = 0
    class_count: int = 0
    max_depth: int = 0
    comment_ratio: float = 0.0
    code_lines: int = 0
    comment_lines: int = 0
    character_count: int = 0
    key_count: int = 0
    section_count: int = 0
    tag_count: int = 0
    header_count: int = 0
    column_count: int = 0
    row_count: int = 0
    # Extended metrics used by some analyzers
    template_count: int = 0
    template_specializations: int = 0
    macro_count: int = 0
    ifdef_count: int = 0
    include_count: int = 0
    new_count: int = 0
    delete_count: int = 0
    malloc_count: int = 0
    free_count: int = 0
    unique_ptr_count: int = 0
    shared_ptr_count: int = 0
    weak_ptr_count: int = 0
    uses_raii: bool = False
    memory_safety_score: float = 1.0
    interface_count: int = 0
    type_count: int = 0
    enum_count: int = 0
    record_count: int = 0
    method_count: int = 0
    try_blocks: int = 0
    catch_blocks: int = 0
    finally_blocks: int = 0
    throws_declarations: int = 0
    annotation_count: int = 0
    extends_count: int = 0
    implements_count: int = 0
    lambda_count: int = 0
    stream_operations: int = 0
    unsafe_blocks: int = 0
    unsafe_functions: int = 0
    unsafe_traits: int = 0
    unsafe_impl: int = 0
    unsafe_score: int = 0
    lifetime_annotations: int = 0
    lifetime_bounds: int = 0
    generic_types: int = 0
    trait_bounds: int = 0
    async_functions: int = 0
    await_points: int = 0
    result_types: int = 0
    option_types: int = 0
    unwrap_calls: int = 0
    expect_calls: int = 0
    question_marks: int = 0
    macro_invocations: int = 0
    derive_macros: int = 0
    test_count: int = 0
    bench_count: int = 0
    assertion_count: int = 0
    # Swift/Combine and other optional metrics
    combine_operators: int = 0
    # Swift async/await metrics
    task_count: int = 0
    task_groups: int = 0
    await_calls: int = 0
    # Swift optional handling metrics
    optional_types: int = 0
    force_unwraps: int = 0
    optional_chaining: int = 0
    nil_coalescing: int = 0
    guard_statements: int = 0
    if_let_bindings: int = 0
    guard_let_bindings: int = 0
    # Swift property wrappers
    published_wrappers: int = 0
    # Swift Combine
    combine_publishers: int = 0
    combine_subscriptions: int = 0
    # Scala-specific metrics
    trait_count: int = 0
    object_count: int = 0
    case_class_count: int = 0
    match_expressions: int = 0
    case_clauses: int = 0
    pattern_guards: int = 0
    higher_order_functions: int = 0
    for_comprehensions: int = 0
    partial_functions: int = 0
    type_parameters: int = 0
    variance_annotations: int = 0
    type_aliases: int = 0
    existential_types: int = 0
    implicit_defs: int = 0
    implicit_params: int = 0
    implicit_conversions: int = 0
    future_usage: int = 0
    actor_usage: int = 0
    async_await: int = 0
    immutable_collections: int = 0
    mutable_collections: int = 0
    throw_statements: int = 0
    option_usage: int = 0
    either_usage: int = 0
    try_usage: int = 0
    # Additional metrics used by other analyzers/tests
    # GDScript
    signal_count: int = 0
    export_count: int = 0
    onready_count: int = 0
    node_ref_count: int = 0
    get_node_count: int = 0
    connect_count: int = 0
    emit_count: int = 0
    lifecycle_count: int = 0
    rpc_count: int = 0
    typed_vars: int = 0
    typed_funcs: int = 0
    return_types: int = 0
    # HTML security/content metrics
    has_csp: bool = False
    has_integrity_checks: bool = False
    has_https_links: int = 0
    # Kotlin-specific metrics
    delegation_count: int = 0
    lazy_properties: int = 0
    observable_properties: int = 0
    lateinit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all metrics
        """
        return asdict(self)


@dataclass
class FunctionInfo:
    """Information about a function or method.

    Represents functions, methods, procedures across languages.

    Attributes:
        name: Function/method name
        line_start: Starting line number
        line_end: Ending line number
        parameters: List of parameter names
        complexity: Cyclomatic complexity of the function
        line: Compatibility alias for line_start
        end_line: Compatibility alias for line_end
        is_toplevel: Whether function is top-level (for some analyzers)
        args: Argument strings with type hints (analyzer compatibility)
        decorators: Decorators applied to the function
        is_async: Whether the function is async
        docstring: Function docstring
        return_type: Return type annotation
    """

    name: str
    line_start: int = 0
    line_end: int = 0
    parameters: List[str] = field(default_factory=list)
    complexity: int = 1
    # Compatibility fields accepted by analyzers/tests
    line: int = 0
    end_line: int = 0
    is_toplevel: bool = False
    # Extended optional fields
    args: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    docstring: Optional[str] = None
    return_type: Optional[str] = None
    is_constructor: bool = False
    is_abstract: bool = False
    is_static: bool = False
    is_class: bool = False
    is_property: bool = False
    is_private: bool = False
    # Additional flags used by language analyzers
    is_generator: bool = False
    is_exported: bool = False
    is_arrow: bool = False
    generics: Optional[str] = None
    is_inline: bool = False
    is_constexpr: bool = False
    is_template: bool = False
    is_extern: bool = False
    is_unsafe: bool = False
    is_const: bool = False
    is_public: bool = False
    # New compatibility fields for Kotlin/Scala/etc.
    visibility: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    type_parameters: Optional[str] = None
    receiver_type: Optional[str] = None
    is_extension: bool = False
    is_suspend: bool = False
    is_operator: bool = False
    is_infix: bool = False
    is_implicit: bool = False
    is_curried: bool = False
    # Additional optional flags seen in tests
    is_lifecycle: bool = False
    is_virtual: bool = False
    # Swift/Scala compatibility
    access_level: Optional[str] = None
    is_throwing: bool = False
    where_clause: Optional[str] = None

    def __post_init__(self):
        # Map compatibility fields to canonical ones when provided
        if not self.line_start and self.line:
            self.line_start = self.line
        if not self.line_end and self.end_line:
            self.line_end = self.end_line

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing function information
        """
        return asdict(self)


@dataclass
class ClassInfo:
    """Information about a class or similar construct.

    Represents classes, structs, interfaces across languages.

    Attributes:
        name: Class/struct/interface name
        line_start: Starting line number
        line_end: Ending line number
        methods: List of methods in the class
        base_classes: List of base/parent class names
        line: Compatibility alias for line_start
        decorators: Decorator names applied to the class
        docstring: Class docstring
        is_abstract: Whether class is abstract
        metaclass: Metaclass name
        attributes: Collected class attributes
        end_line: Compatibility alias for line_end
        bases: Compatibility alias accepted by some analyzers/tests
    """

    name: str
    line_start: int = 0
    line_end: int = 0
    methods: List[FunctionInfo] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)
    # Compatibility alias used in some tests/analyzers
    line: int = 0
    # Extended optional fields used by analyzers
    decorators: List[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_abstract: bool = False
    metaclass: Optional[str] = None
    attributes: List[Dict[str, Any]] = field(default_factory=list)
    # Additional compatibility to accept `end_line` in constructor
    end_line: int = 0
    # Accept legacy/alternate parameter name for base classes
    bases: List[str] = field(default_factory=list)
    # Extended fields to support multiple analyzers
    fields: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    visibility: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)
    generics: Optional[str] = None
    inner_classes: List[str] = field(default_factory=list)
    struct_type: Optional[str] = None
    is_public: bool = False
    is_struct: bool = False
    is_template: bool = False
    # New compatibility fields used by some analyzers (JS/PHP)
    is_exported: bool = False
    properties: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    # Kotlin/Scala/PHP specific compatibility
    type_parameters: Optional[str] = None
    constructor_params: List[Dict[str, Any]] = field(default_factory=list)
    mixins: List[str] = field(default_factory=list)
    # Ruby-specific module/mixin tracking
    included_modules: List[str] = field(default_factory=list)
    extended_modules: List[str] = field(default_factory=list)
    delegates: Dict[str, str] = field(default_factory=dict)
    companion_object: Optional[Dict[str, Any]] = field(default_factory=dict)
    nested_classes: List[Dict[str, Any]] = field(default_factory=dict)
    has_companion: bool = False
    is_case_class: bool = False
    # Add Kotlin compatibility flag for data classes
    is_data_class: bool = False
    is_sealed: bool = False
    is_enum: bool = False
    is_inner: bool = False
    is_value_class: bool = False
    android_type: Optional[str] = None
    traits_used: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Any] = field(default_factory=list)
    # Added for Dart/Flutter analyzers
    constructors: List[Dict[str, Any]] = field(default_factory=list)
    is_widget: bool = False
    widget_type: Optional[str] = None
    # Swift compatibility fields
    access_level: Optional[str] = None
    is_open: bool = False
    is_final: bool = False
    superclass: Optional[str] = None
    protocols: List[str] = field(default_factory=list)
    nested_types: List[Dict[str, Any]] = field(default_factory=list)
    ui_type: Optional[str] = None
    # GDScript compatibility alias
    is_inner_class: bool = False
    # Ruby singleton class flag
    is_singleton: bool = False
    # Unity/C# specific flags and lists
    is_monobehaviour: bool = False
    is_scriptable_object: bool = False
    unity_methods: List[str] = field(default_factory=list)
    coroutines: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.line_start and self.line:
            self.line_start = self.line
        if not self.line_end and self.end_line:
            self.line_end = self.end_line
        # Map compatibility alias `bases` -> `base_classes` and vice versa
        if not self.base_classes and self.bases:
            self.base_classes = list(self.bases)
        elif self.base_classes and not self.bases:
            self.bases = list(self.base_classes)
        # Keep GDScript flag in sync
        if self.is_inner_class and not self.is_inner:
            self.is_inner = True
        if self.is_inner and not self.is_inner_class:
            self.is_inner_class = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing class information with serialized methods
        """
        data = asdict(self)
        # Keep methods serialized
        data["methods"] = [m.to_dict() if hasattr(m, "to_dict") else m for m in self.methods]
        return data


@dataclass
class CodeStructure:
    """Represents the structure of a code file.

    Contains organized information about code elements found in a file.

    Attributes:
        classes: List of classes in the file
        functions: List of standalone functions
        imports: List of import statements
        file_type: Type of the file (e.g., script, module, package)
        sections: List of sections or blocks in the code
        variables: List of variables used
        constants: List of constants
        todos: List of TODO comments or annotations
        block_count: Total number of code blocks
        indent_levels: Indentation levels used in the code
        type_aliases: List of type alias definitions (Python 3.10+)
    """

    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    file_type: str = "text"
    sections: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    todos: List[Dict[str, Any]] = field(default_factory=list)
    block_count: int = 0
    indent_levels: Dict[str, Any] = field(default_factory=dict)
    type_aliases: List[Dict[str, Any]] = field(default_factory=list)
    # Extended structural fields used by language analyzers
    namespace: Optional[str] = None
    is_unity_script: bool = False
    language_variant: Optional[str] = None
    namespaces: List[Dict[str, Any]] = field(default_factory=list)
    templates: List[Dict[str, Any]] = field(default_factory=list)
    macros: List[Dict[str, Any]] = field(default_factory=list)
    unions: List[Dict[str, Any]] = field(default_factory=list)
    structs: List[Dict[str, Any]] = field(default_factory=list)
    operator_overloads: int = 0
    uses_stl: bool = False
    smart_pointers: List[str] = field(default_factory=list)
    lambda_count: int = 0
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    types: List[Dict[str, Any]] = field(default_factory=list)
    enums: List[Dict[str, Any]] = field(default_factory=list)
    modules: List[Dict[str, Any]] = field(default_factory=list)
    framework: Optional[str] = None
    package: Optional[str] = None
    records: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    anonymous_classes_count: int = 0
    # Additional JS/TS support
    components: List[Dict[str, Any]] = field(default_factory=list)
    # Rust-specific fields (optional)
    is_library: bool = False
    is_binary: bool = False
    # Ruby-specific additions
    modules: List[Dict[str, Any]] = field(default_factory=list)
    aliases: List[Dict[str, Any]] = field(default_factory=list)
    is_test: bool = False
    traits: List[Dict[str, Any]] = field(default_factory=list)
    impl_blocks: List[Dict[str, Any]] = field(default_factory=list)
    modules: List[Dict[str, Any]] = field(default_factory=list)
    statics: List[Dict[str, Any]] = field(default_factory=list)
    derives: List[str] = field(default_factory=list)
    unsafe_blocks: int = 0
    async_functions: int = 0
    test_functions: int = 0
    bench_functions: int = 0
    crate_type: Optional[str] = None
    # Additional Rust metrics counted at structure-level
    await_points: int = 0
    unsafe_functions: int = 0
    # New fields used by other analyzers
    # Kotlin/Scala/General
    objects: List[Dict[str, Any]] = field(default_factory=list)
    scala_version: Optional[int] = None
    given_instances: int = 0
    using_clauses: int = 0
    extension_methods: int = 0
    # Alias for tests/analyzers that use "extension_functions"
    extension_functions: int = 0
    match_expressions: int = 0
    case_statements: int = 0
    for_comprehensions: int = 0
    yield_expressions: int = 0
    implicit_defs: int = 0
    implicit_params: int = 0
    lambda_expressions: int = 0
    partial_functions: int = 0
    # Kotlin Android flags
    is_android: bool = False
    suspend_functions: int = 0
    coroutine_launches: int = 0
    flow_usage: int = 0
    nullable_types: int = 0
    null_assertions: int = 0
    safe_calls: int = 0
    elvis_operators: int = 0
    scope_functions: int = 0
    # GDScript-specific structure
    is_tool_script: bool = False
    class_name: Optional[str] = None
    parent_class: Optional[str] = None
    godot_version: Optional[int] = None
    signals: List[Dict[str, Any]] = field(default_factory=list)
    export_vars: List[Dict[str, Any]] = field(default_factory=list)
    onready_vars: List[Dict[str, Any]] = field(default_factory=list)
    setget_properties: List[Dict[str, Any]] = field(default_factory=list)
    node_references: int = 0
    get_node_calls: int = 0
    connect_calls: int = 0
    emit_signal_calls: int = 0
    is_custom_resource: bool = False
    # Added for Dart analyzer compatibility
    is_flutter: bool = False
    mixins: List[Dict[str, Any]] = field(default_factory=list)
    extensions: List[Dict[str, Any]] = field(default_factory=list)
    typedefs: List[Dict[str, Any]] = field(default_factory=list)
    has_main: bool = False
    is_test_file: bool = False
    # Swift-specific structure
    is_ios: bool = False
    is_swiftui: bool = False
    is_uikit: bool = False
    structs: List[Dict[str, Any]] = field(default_factory=list)
    protocols: List[Dict[str, Any]] = field(default_factory=list)
    actors: List[Dict[str, Any]] = field(default_factory=list)
    # Swift async/concurrency metrics
    task_count: int = 0
    await_count: int = 0
    # Swift optional handling metrics
    optional_count: int = 0
    force_unwrap_count: int = 0
    optional_chaining_count: int = 0
    nil_coalescing_count: int = 0
    guard_count: int = 0
    if_let_count: int = 0
    guard_let_count: int = 0
    actor_count: int = 0
    # Swift UI and framework metrics
    property_wrappers: int = 0
    result_builders: int = 0
    combine_publishers: int = 0
    combine_operators: int = 0
    swiftui_views: int = 0
    view_modifiers: int = 0
    body_count: int = 0
    # HTML document structure metrics used by tests
    form_count: int = 0
    input_count: int = 0
    alt_texts: int = 0

    def __post_init__(self):
        # Keep Kotlin alias fields in sync
        if self.extension_functions and not self.extension_methods:
            self.extension_methods = self.extension_functions
        if self.extension_methods and not self.extension_functions:
            self.extension_functions = self.extension_methods

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all structural information
        """
        base = {
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "imports": [i.to_dict() for i in self.imports],
        }
        # Include extended fields
        base.update(
            {
                "file_type": self.file_type,
                "sections": self.sections,
                "variables": self.variables,
                "constants": self.constants,
                "todos": self.todos,
                "block_count": self.block_count,
                "indent_levels": self.indent_levels,
                "type_aliases": self.type_aliases,
                # Extended
                "language_variant": self.language_variant,
                "namespaces": self.namespaces,
                "templates": self.templates,
                "macros": self.macros,
                "unions": self.unions,
                "operator_overloads": self.operator_overloads,
                "uses_stl": self.uses_stl,
                "smart_pointers": self.smart_pointers,
                "lambda_count": self.lambda_count,
                "interfaces": self.interfaces,
                "types": self.types,
                "enums": self.enums,
                "framework": self.framework,
                "package": self.package,
                "records": self.records,
                "annotations": self.annotations,
                "anonymous_classes_count": self.anonymous_classes_count,
                # JS/TS
                "components": self.components,
                # Rust
                "is_library": self.is_library,
                "is_binary": self.is_binary,
                "is_test": self.is_test,
                "traits": self.traits,
                "impl_blocks": self.impl_blocks,
                "modules": self.modules,
                "statics": self.statics,
                "derives": self.derives,
                "unsafe_blocks": self.unsafe_blocks,
                "async_functions": self.async_functions,
                "test_functions": self.test_functions,
                "bench_functions": self.bench_functions,
                "crate_type": self.crate_type,
                "await_points": self.await_points,
                "unsafe_functions": self.unsafe_functions,
                # Kotlin/Scala/General additions
                "objects": self.objects,
                "scala_version": self.scala_version,
                "given_instances": self.given_instances,
                "using_clauses": self.using_clauses,
                "extension_methods": self.extension_methods,
                "extension_functions": self.extension_functions,
                "match_expressions": self.match_expressions,
                "case_statements": self.case_statements,
                "for_comprehensions": self.for_comprehensions,
                "yield_expressions": self.yield_expressions,
                "implicit_defs": self.implicit_defs,
                "implicit_params": self.implicit_params,
                "lambda_expressions": self.lambda_expressions,
                "partial_functions": self.partial_functions,
                # Kotlin Android metrics
                "is_android": self.is_android,
                "suspend_functions": self.suspend_functions,
                "coroutine_launches": self.coroutine_launches,
                "flow_usage": self.flow_usage,
                "nullable_types": self.nullable_types,
                "null_assertions": self.null_assertions,
                "safe_calls": self.safe_calls,
                "elvis_operators": self.elvis_operators,
                "scope_functions": self.scope_functions,
                # GDScript-specific
                "is_tool_script": self.is_tool_script,
                "class_name": self.class_name,
                "parent_class": self.parent_class,
                "godot_version": self.godot_version,
                "signals": self.signals,
                "export_vars": self.export_vars,
                "onready_vars": self.onready_vars,
                "setget_properties": self.setget_properties,
                "node_references": self.node_references,
                "get_node_calls": self.get_node_calls,
                "connect_calls": self.connect_calls,
                "emit_signal_calls": self.emit_signal_calls,
                "is_custom_resource": self.is_custom_resource,
                # Dart analyzer additions
                "is_flutter": self.is_flutter,
                "mixins": self.mixins,
                "extensions": self.extensions,
                "typedefs": self.typedefs,
                "has_main": self.has_main,
                "is_test_file": self.is_test_file,
                # Swift structure additions
                "is_ios": self.is_ios,
                "is_swiftui": self.is_swiftui,
                "is_uikit": self.is_uikit,
                "structs": self.structs,
                "protocols": self.protocols,
                "actors": self.actors,
                "task_count": self.task_count,
                "await_count": self.await_count,
                "optional_count": self.optional_count,
                "force_unwrap_count": self.force_unwrap_count,
                "optional_chaining_count": self.optional_chaining_count,
                "nil_coalescing_count": self.nil_coalescing_count,
                "guard_count": self.guard_count,
                "if_let_count": self.if_let_count,
                "guard_let_count": self.guard_let_count,
                "actor_count": self.actor_count,
                "property_wrappers": self.property_wrappers,
                "result_builders": self.result_builders,
                "combine_publishers": self.combine_publishers,
                "combine_operators": self.combine_operators,
                "swiftui_views": self.swiftui_views,
                "view_modifiers": self.view_modifiers,
                "body_count": self.body_count,
                # HTML doc structure
                "form_count": self.form_count,
                "input_count": self.input_count,
                "alt_texts": self.alt_texts,
            }
        )
        return base


@dataclass
class FileAnalysis:
    """Complete analysis results for a single file.

    Contains all information extracted from analyzing a source code file,
    including structure, complexity, and metadata.

    Attributes:
        path: File path
        content: File content
        size: File size in bytes
        lines: Number of lines
        language: Programming language
        file_name: Name of the file
        file_extension: File extension
        last_modified: Last modification time
        hash: Content hash
        imports: List of imports
        exports: List of exports
        structure: Code structure information
        complexity: Complexity metrics
        classes: List of classes (convenience accessor)
        functions: List of functions (convenience accessor)
        keywords: Extracted keywords
        relevance_score: Relevance score for ranking
        quality_score: Code quality score
        error: Any error encountered during analysis
    """

    path: str
    content: str = ""
    size: int = 0
    lines: int = 0
    language: str = "unknown"
    file_name: str = ""
    file_extension: str = ""
    last_modified: Optional[datetime] = None
    hash: Optional[str] = None

    # Analysis results
    imports: List[ImportInfo] = field(default_factory=list)
    exports: List[Dict[str, Any]] = field(default_factory=list)
    structure: Optional[CodeStructure] = None
    complexity: Optional[ComplexityMetrics] = None
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # Git information
    git_info: Optional[Dict[str, Any]] = None

    # Ranking/scoring
    relevance_score: float = 0.0
    quality_score: float = 0.0

    # Error handling
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all file analysis data
        """
        data = asdict(self)
        if self.last_modified:
            data["last_modified"] = self.last_modified.isoformat()
        if self.structure:
            data["structure"] = self.structure.to_dict()
        if self.complexity:
            data["complexity"] = self.complexity.to_dict()
        data["imports"] = [i.to_dict() for i in self.imports]
        data["classes"] = [c.to_dict() for c in self.classes]
        data["functions"] = [f.to_dict() for f in self.functions]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileAnalysis":
        """Create FileAnalysis from dictionary.

        Args:
            data: Dictionary containing file analysis data

        Returns:
            FileAnalysis instance
        """
        # Handle datetime conversion
        if "last_modified" in data and isinstance(data["last_modified"], str):
            data["last_modified"] = datetime.fromisoformat(data["last_modified"])

        # Handle nested objects
        if "imports" in data:
            data["imports"] = [
                ImportInfo(**i) if isinstance(i, dict) else i for i in data["imports"]
            ]
        if "classes" in data:
            data["classes"] = [
                ClassInfo(**c) if isinstance(c, dict) else c for c in data["classes"]
            ]
        if "functions" in data:
            data["functions"] = [
                FunctionInfo(**f) if isinstance(f, dict) else f for f in data["functions"]
            ]
        if "structure" in data and isinstance(data["structure"], dict):
            # Reconstruct CodeStructure
            structure_data = data["structure"]
            if "classes" in structure_data:
                structure_data["classes"] = [
                    ClassInfo(**c) if isinstance(c, dict) else c for c in structure_data["classes"]
                ]
            if "functions" in structure_data:
                structure_data["functions"] = [
                    FunctionInfo(**f) if isinstance(f, dict) else f
                    for f in structure_data["functions"]
                ]
            if "imports" in structure_data:
                structure_data["imports"] = [
                    ImportInfo(**i) if isinstance(i, dict) else i for i in structure_data["imports"]
                ]
            data["structure"] = CodeStructure(**structure_data)
        if "complexity" in data and isinstance(data["complexity"], dict):
            data["complexity"] = ComplexityMetrics(**data["complexity"])

        return cls(**data)


@dataclass
class DependencyGraph:
    """Represents project dependency graph.

    Tracks dependencies between files and modules in the project.

    Attributes:
        nodes: Dictionary of node ID to node data
        edges: List of edges (from_id, to_id, edge_data)
        cycles: List of detected dependency cycles
    """

    nodes: Dict[str, Any] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)

    def add_node(self, node_id: str, data: Any) -> None:
        """Add a node to the dependency graph.

        Args:
            node_id: Unique identifier for the node
            data: Node data (typically FileAnalysis)
        """
        self.nodes[node_id] = data

    def add_edge(self, from_id: str, to_id: str, import_info: Optional[ImportInfo] = None) -> None:
        """Add an edge representing a dependency.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            import_info: Optional import information
        """
        self.edges.append((from_id, to_id, import_info))

    def calculate_metrics(self) -> None:
        """Calculate graph metrics like centrality and cycles.

        Updates internal metrics based on current graph structure.
        """
        # This would calculate various graph metrics
        # For now, just detect simple cycles
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing graph structure
        """
        return {
            "nodes": list(self.nodes.keys()),
            "edges": [(e[0], e[1]) for e in self.edges],
            "cycles": self.cycles,
        }


@dataclass
class ProjectAnalysis:
    """Analysis results for an entire project.

    Aggregates file-level analysis into project-wide metrics and insights.

    Attributes:
        path: Project root path
        name: Project name
        files: List of analyzed files
        total_files: Total number of files
        analyzed_files: Number of successfully analyzed files
        failed_files: Number of files that failed analysis
        total_lines: Total lines of code
        total_code_lines: Total non-blank, non-comment lines
        total_comment_lines: Total comment lines
        average_complexity: Average cyclomatic complexity
        total_functions: Total number of functions
        total_classes: Total number of classes
        languages: Language distribution (language -> file count)
        language_distribution: Percentage distribution of languages
        frameworks: Detected frameworks
        project_type: Type of project (web, library, cli, etc.)
        dependency_graph: Project dependency graph
        summary: Project summary dictionary
    """

    path: str
    name: str
    files: List[FileAnalysis] = field(default_factory=list)
    total_files: int = 0
    analyzed_files: int = 0
    failed_files: int = 0

    # Aggregate metrics
    total_lines: int = 0
    total_code_lines: int = 0
    total_comment_lines: int = 0
    average_complexity: float = 0.0
    total_functions: int = 0
    total_classes: int = 0

    # Language info
    languages: Dict[str, int] = field(default_factory=dict)
    language_distribution: Dict[str, float] = field(default_factory=dict)

    # Project info
    frameworks: List[str] = field(default_factory=list)
    project_type: str = "unknown"
    dependency_graph: Optional[DependencyGraph] = None
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing all project analysis data
        """
        data = asdict(self)
        data["files"] = [f.to_dict() for f in self.files]
        if self.dependency_graph:
            data["dependency_graph"] = self.dependency_graph.to_dict()
        return data


@dataclass
class AnalysisReport:
    """Report generated from analysis results.

    Formatted output of analysis results for different consumers.

    Attributes:
        timestamp: When report was generated
        format: Report format (json, html, markdown, csv)
        content: Report content
        statistics: Analysis statistics
        output_path: Where report was saved (if applicable)
    """

    timestamp: datetime = field(default_factory=datetime.now)
    format: str = "json"
    content: str = ""
    statistics: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict containing report information
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "format": self.format,
            "statistics": self.statistics,
            "output_path": self.output_path,
        }
