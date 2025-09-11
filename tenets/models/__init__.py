"""Data models for the Tenets package.

This package contains all data models and structures used throughout
the Tenets codebase, including analysis results, context management,
tenets, and more.

Modules:
    analysis: Code analysis models (ImportInfo, ComplexityMetrics, FunctionInfo, etc.)
    context: Context management models (PromptContext, ContextResult, SessionContext)
    tenet: Tenet-related models (Tenet, TenetCollection, Priority, etc.)
    summary: Summary models (FileSummary, ProjectSummary)
    llm: LLM-related models (ModelPricing, ModelLimits)
"""

# Export commonly used models for convenience
from .analysis import (
    ImportInfo,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
    CodeStructure,
    FileAnalysis,
    DependencyGraph,
    ProjectAnalysis,
    AnalysisReport,
)

from .context import (
    TaskType,
    PromptContext,
    ContextResult,
    SessionContext,
)

from .tenet import (
    Priority,
    TenetStatus,
    TenetCategory,
    TenetMetrics,
    InjectionStrategy,
    Tenet,
    TenetCollection,
)

from .summary import (
    SummaryStrategy,
    SummarySection,
    FileSummary,
    ProjectSummary,
)

from .llm import (
    ModelPricing,
    ModelLimits,
)

__all__ = [
    # Analysis models
    "ImportInfo",
    "ComplexityMetrics",
    "FunctionInfo",
    "ClassInfo",
    "CodeStructure",
    "FileAnalysis",
    "DependencyGraph",
    "ProjectAnalysis",
    "AnalysisReport",
    # Context models
    "TaskType",
    "PromptContext",
    "ContextResult",
    "SessionContext",
    # Tenet models
    "Priority",
    "TenetStatus",
    "TenetCategory",
    "TenetMetrics",
    "InjectionStrategy",
    "Tenet",
    "TenetCollection",
    # Summary models
    "SummaryStrategy",
    "SummarySection",
    "FileSummary",
    "ProjectSummary",
    # LLM models
    "ModelPricing",
    "ModelLimits",
]
