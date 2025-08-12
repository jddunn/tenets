"""Context models for prompt processing and result handling.

This module defines the data structures for managing context throughout
the distillation and instillation process.
"""

from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import hashlib


class TaskType(Enum):
    """Types of tasks detected in prompts."""

    FEATURE = "feature"
    DEBUG = "debug"
    TEST = "test"
    REFACTOR = "refactor"
    UNDERSTAND = "understand"
    REVIEW = "review"
    DOCUMENT = "document"
    OPTIMIZE = "optimize"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    MIGRATION = "migration"
    GENERAL = "general"

    @classmethod
    def from_string(cls, value: str) -> "TaskType":
        """Create TaskType from string value."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.GENERAL


@dataclass
class PromptContext:
    """Context extracted from user prompt.

    Contains all information parsed from the prompt to guide
    file selection and ranking. This is the primary data structure
    that flows through the system after prompt parsing.

    Attributes:
        text: The processed prompt text (cleaned and normalized)
        original: Original input (may be URL or raw text)
        keywords: Extracted keywords for searching
        task_type: Type of task detected
        intent: User intent classification
        entities: Named entities found (classes, functions, modules)
        file_patterns: File patterns to match (*.py, test_*, etc)
        focus_areas: Areas to focus on (auth, api, database, etc)
        temporal_context: Time-related context (recent, yesterday, etc)
        scope: Scope indicators (modules, directories, exclusions)
        external_context: Context from external sources (GitHub, JIRA)
        metadata: Additional metadata for processing
        confidence_scores: Confidence scores for various extractions
        session_id: Associated session if any
        timestamp: When context was created
    """

    text: str
    original: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    task_type: str = "general"
    intent: str = "understand"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)
    temporal_context: Optional[Dict[str, Any]] = None
    scope: Dict[str, Any] = field(default_factory=dict)
    external_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure original is populated for backward compatibility with tests
        if self.original is None:
            self.original = self.text

        # Normalize task type
        if isinstance(self.task_type, str):
            self.task_type = self.task_type.lower()

        # Remove duplicate keywords while preserving order
        seen = set()
        unique_keywords = []
        for kw in self.keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        self.keywords = unique_keywords

        # Initialize default scope if empty
        if not self.scope:
            self.scope = {
                "modules": [],
                "directories": [],
                "specific_files": [],
                "exclusions": [],
                "is_global": False,
                "is_specific": False,
            }

    def add_keyword(self, keyword: str, confidence: float = 1.0) -> None:
        """Add a keyword with confidence score."""
        if keyword and keyword.lower() not in [k.lower() for k in self.keywords]:
            self.keywords.append(keyword)
            self.confidence_scores[f"keyword_{keyword}"] = confidence

    def add_entity(self, name: str, entity_type: str, confidence: float = 1.0) -> None:
        """Add an entity with type and confidence."""
        self.entities.append({"name": name, "type": entity_type, "confidence": confidence})

    def add_focus_area(self, area: str) -> None:
        """Add a focus area if not already present."""
        if area and area not in self.focus_areas:
            self.focus_areas.append(area)

    def merge_with(self, other: "PromptContext") -> "PromptContext":
        """Merge this context with another."""
        # Merge keywords
        for kw in other.keywords:
            self.add_keyword(kw)

        # Merge entities
        self.entities.extend(other.entities)

        # Merge file patterns
        self.file_patterns.extend(
            [fp for fp in other.file_patterns if fp not in self.file_patterns]
        )

        # Merge focus areas
        for area in other.focus_areas:
            self.add_focus_area(area)

        # Merge metadata
        self.metadata.update(other.metadata)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "original": self.original,
            "keywords": self.keywords,
            "task_type": self.task_type,
            "intent": self.intent,
            "entities": self.entities,
            "file_patterns": self.file_patterns,
            "focus_areas": self.focus_areas,
            "temporal_context": self.temporal_context,
            "scope": self.scope,
            "external_context": self.external_context,
            "metadata": self.metadata,
            "confidence_scores": self.confidence_scores,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptContext":
        """Create PromptContext from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def get_hash(self) -> str:
        """Get a hash of this context for caching."""
        key_data = f"{self.text}_{self.task_type}_{sorted(self.keywords)}"
        return hashlib.md5(key_data.encode()).hexdigest()


@dataclass
class ContextResult:
    """Result of context generation.

    Contains the generated context ready for consumption by LLMs
    or other tools. This is the final output of the distillation process.

    Attributes:
        content: The generated context content (preferred alias)
        context: Backward-compatible alias for content
        format: Output format (markdown, xml, json)
        token_count: Number of tokens in context
        files: List of included file paths (preferred alias)
        files_included: Backward-compatible alias for files
        files_summarized: List of summarized file paths
        metadata: Additional metadata about generation
        session_id: Session this belongs to
        timestamp: When context was generated
        statistics: Generation statistics
        prompt_context: Original prompt context
        cost_estimate: Estimated cost for LLM usage
        warnings: Any warnings during generation
        errors: Any errors during generation
    """

    # Support both names for content
    content: Optional[str] = None
    context: Optional[str] = None
    format: str = "markdown"
    token_count: int = 0
    # Support both names for files
    files: List[str] = field(default_factory=list)
    files_included: List[str] = field(default_factory=list)
    files_summarized: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    statistics: Dict[str, Any] = field(default_factory=dict)
    prompt_context: Optional[PromptContext] = None
    cost_estimate: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization processing and alias synchronization."""
        # Synchronize content/context aliases
        if self.content is None and self.context is not None:
            self.content = self.context
        if self.context is None and self.content is not None:
            self.context = self.content

        # Synchronize files/files_included aliases
        if not self.files and self.files_included:
            self.files = list(self.files_included)
        if not self.files_included and self.files:
            self.files_included = list(self.files)

        # Calculate default statistics if not provided
        if not self.statistics:
            self.statistics = {
                "total_files_included": len(self.files_included),
                "total_files_summarized": len(self.files_summarized),
                "token_count": self.token_count,
                "format": self.format,
                "has_warnings": len(self.warnings) > 0,
                "has_errors": len(self.errors) > 0,
            }

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        self.statistics["has_warnings"] = True

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.statistics["has_errors"] = True

    def update_statistics(self, key: str, value: Any) -> None:
        """Update a statistic value."""
        self.statistics[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            # Prefer normalized keys expected by tests
            "content": self.content,
            "format": self.format,
            "token_count": self.token_count,
            "files": list(self.files),
            # Include legacy keys for backward compatibility
            "context": self.context,
            "files_included": list(self.files_included),
            "files_summarized": list(self.files_summarized),
            "metadata": self.metadata,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "statistics": self.statistics,
            "cost_estimate": self.cost_estimate,
            "warnings": self.warnings,
            "errors": self.errors,
        }

        if self.prompt_context:
            data["prompt_context"] = self.prompt_context.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextResult":
        """Create from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        if "prompt_context" in data and isinstance(data["prompt_context"], dict):
            data["prompt_context"] = PromptContext.from_dict(data["prompt_context"])

        # Normalize alias keys on load
        if "context" in data and "content" not in data:
            data["content"] = data["context"]
        if "files_included" in data and "files" not in data:
            data["files"] = data["files_included"]

        return cls(**data)

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save context result to file."""
        path = Path(path)

        if self.format == "json":
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        else:
            with open(path, "w") as f:
                f.write(self.content or "")

    def get_summary(self) -> str:
        """Get a summary of the context result."""
        lines = [
            f"Context Result Summary:",
            f"  Format: {self.format}",
            f"  Token Count: {self.token_count:,}",
            f"  Files Included: {len(self.files_included)}",
            f"  Files Summarized: {len(self.files_summarized)}",
        ]

        if self.cost_estimate:
            lines.append(f"  Estimated Cost: ${self.cost_estimate.get('total_cost', 0):.4f}")

        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")

        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")

        return "\n".join(lines)


@dataclass
class SessionContext:
    """Context for a session.

    Maintains state across multiple prompts in a session for
    incremental context building and state management.

    Attributes:
        session_id: Unique session identifier
        name: Human-readable session name
        project_root: Root path of the project
        shown_files: Files explicitly shown
        ignored_files: Files to ignore
        context_history: History of contexts
        current_focus: Current focus areas
        tenets_applied: Tenets applied in session
        created_at: When session was created
        updated_at: Last update time
        metadata: Session metadata
        ai_requests: History of AI requests
        branch: Git branch if applicable
    """

    session_id: str
    name: str = ""
    project_root: Optional[Path] = None
    shown_files: Set[str] = field(default_factory=set)
    ignored_files: Set[str] = field(default_factory=set)
    context_history: List[ContextResult] = field(default_factory=list)
    current_focus: List[str] = field(default_factory=list)
    tenets_applied: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ai_requests: List[Dict[str, Any]] = field(default_factory=list)
    branch: Optional[str] = None
    # New: pinned files explicitly added via instill --add-file / --add-folder
    pinned_files: Set[str] = field(default_factory=set)

    def add_shown_file(self, file_path: str) -> None:
        """Mark file as shown."""
        self.shown_files.add(file_path)
        if file_path in self.ignored_files:
            self.ignored_files.remove(file_path)
        self.updated_at = datetime.now()

    def add_ignored_file(self, file_path: str) -> None:
        """Mark file as ignored."""
        self.ignored_files.add(file_path)
        if file_path in self.shown_files:
            self.shown_files.remove(file_path)
        self.updated_at = datetime.now()

    def add_context(self, context: ContextResult) -> None:
        """Add context to history."""
        self.context_history.append(context)
        context.session_id = self.session_id
        self.updated_at = datetime.now()

    def add_ai_request(self, request_type: str, request_data: Dict[str, Any]) -> None:
        """Record an AI request."""
        self.ai_requests.append(
            {"type": request_type, "data": request_data, "timestamp": datetime.now().isoformat()}
        )
        self.updated_at = datetime.now()

    def add_pinned_file(self, file_path: str) -> None:
        """Pin a file so it is always considered for future distill operations.

        Args:
            file_path: Absolute or project-relative path to the file.
        """
        self.pinned_files.add(file_path)
        self.updated_at = datetime.now()

    def list_pinned_files(self) -> List[str]:
        """Return pinned file paths."""
        return sorted(self.pinned_files)

    def get_latest_context(self) -> Optional[ContextResult]:
        """Get the most recent context."""
        return self.context_history[-1] if self.context_history else None

    def should_show_file(self, file_path: str) -> bool:
        """Check if file should be shown based on session state."""
        if file_path in self.ignored_files:
            return False
        if file_path in self.shown_files:
            return True
        return None  # No preference

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "project_root": str(self.project_root) if self.project_root else None,
            "shown_files": list(self.shown_files),
            "ignored_files": list(self.ignored_files),
            "context_history": [c.to_dict() for c in self.context_history],
            "current_focus": self.current_focus,
            "tenets_applied": self.tenets_applied,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "ai_requests": self.ai_requests,
            "branch": self.branch,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        """Create from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        if "shown_files" in data:
            data["shown_files"] = set(data["shown_files"])
        if "ignored_files" in data:
            data["ignored_files"] = set(data["ignored_files"])

        if "context_history" in data:
            data["context_history"] = [
                ContextResult.from_dict(c) if isinstance(c, dict) else c
                for c in data["context_history"]
            ]

        if "project_root" in data and data["project_root"]:
            data["project_root"] = Path(data["project_root"])

        return cls(**data)
