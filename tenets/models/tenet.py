"""Tenet (guiding principle) data model.

This module defines the data structures for tenets - the guiding principles
that can be instilled into context to maintain consistency across AI interactions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Priority(Enum):
    """Tenet priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        """Get numerical weight for priority."""
        weights = {
            Priority.LOW: 0.25,
            Priority.MEDIUM: 0.5,
            Priority.HIGH: 0.75,
            Priority.CRITICAL: 1.0,
        }
        return weights[self]


class TenetStatus(Enum):
    """Tenet status in the system."""

    PENDING = "pending"  # Not yet instilled
    INSTILLED = "instilled"  # Applied to context
    ARCHIVED = "archived"  # No longer active


class TenetCategory(Enum):
    """Common tenet categories."""

    ARCHITECTURE = "architecture"
    SECURITY = "security"
    STYLE = "style"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    API_DESIGN = "api_design"
    ERROR_HANDLING = "error_handling"
    QUALITY = "quality"
    CUSTOM = "custom"


@dataclass
class TenetMetrics:
    """Metrics for tracking tenet effectiveness."""

    injection_count: int = 0
    last_injected: Optional[datetime] = None
    contexts_appeared_in: int = 0
    compliance_score: float = 0.0  # 0-1 score of how well it's followed
    reinforcement_needed: bool = False

    def update_injection(self) -> None:
        """Update metrics after injection."""
        self.injection_count += 1
        self.last_injected = datetime.now()
        self.contexts_appeared_in += 1

        # Reduce reinforcement need after injection
        if self.injection_count % 3 == 0:
            self.reinforcement_needed = False


@dataclass
class InjectionStrategy:
    """Strategy for how a tenet should be injected."""

    frequency: str = "adaptive"  # always, adaptive, periodic
    position: str = "strategic"  # top, bottom, strategic
    max_per_context: int = 3
    min_tokens_between: int = 1000

    def should_inject(self, context_length: int, already_injected: int) -> bool:
        """Determine if tenet should be injected."""
        if already_injected >= self.max_per_context:
            return False

        if self.frequency == "always":
            return True
        elif self.frequency == "adaptive":
            # Inject based on context length and previous injections
            if context_length < 5000:
                return already_injected == 0
            elif context_length < 20000:
                return already_injected < 2
            else:
                return already_injected < self.max_per_context

        return False


@dataclass
class Tenet:
    """A guiding principle for code development.

    Tenets are persistent instructions that guide AI interactions to maintain
    consistency across multiple prompts and sessions.

    Attributes:
        id: Unique identifier
        content: The principle text
        priority: Importance level
        category: Classification category
        status: Current status (pending, instilled, archived)
        created_at: When the tenet was created
        instilled_at: When first instilled into context
        updated_at: Last modification time
        session_bindings: Sessions this tenet applies to
        author: Who created the tenet
        metrics: Usage and effectiveness metrics
        injection_strategy: How this tenet should be injected
        metadata: Additional custom data

    Example:
        >>> tenet = Tenet(
        ...     content="Always use type hints in Python code",
        ...     priority=Priority.HIGH,
        ...     category=TenetCategory.STYLE
        ... )
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    priority: Priority = Priority.MEDIUM
    category: Optional[TenetCategory] = None
    status: TenetStatus = TenetStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    instilled_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
    session_bindings: list[str] = field(default_factory=list)
    author: Optional[str] = None
    metrics: TenetMetrics = field(default_factory=TenetMetrics)
    injection_strategy: InjectionStrategy = field(default_factory=InjectionStrategy)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and process after initialization."""
        # Convert string priority to enum if needed
        if isinstance(self.priority, str):
            self.priority = Priority(self.priority.lower())

        # Convert string category to enum if needed
        if isinstance(self.category, str):
            try:
                self.category = TenetCategory(self.category.lower())
            except ValueError:
                self.category = TenetCategory.CUSTOM
                self.metadata["custom_category"] = self.category

        # Ensure content is not empty
        if not self.content.strip():
            raise ValueError("Tenet content cannot be empty")

    def instill(self) -> None:
        """Mark tenet as instilled."""
        self.status = TenetStatus.INSTILLED
        if not self.instilled_at:
            self.instilled_at = datetime.now()
        self.updated_at = datetime.now()

    def archive(self) -> None:
        """Archive this tenet."""
        self.status = TenetStatus.ARCHIVED
        self.updated_at = datetime.now()

    def bind_to_session(self, session_id: str) -> None:
        """Bind tenet to a specific session."""
        if session_id not in self.session_bindings:
            self.session_bindings.append(session_id)
            self.updated_at = datetime.now()

    def unbind_from_session(self, session_id: str) -> None:
        """Remove session binding."""
        if session_id in self.session_bindings:
            self.session_bindings.remove(session_id)
            self.updated_at = datetime.now()

    def applies_to_session(self, session_id: Optional[str]) -> bool:
        """Check if tenet applies to a session."""
        if not self.session_bindings:
            return True  # Global tenet
        return session_id in self.session_bindings if session_id else False

    def should_inject(self, context_length: int, already_injected: int) -> bool:
        """Determine if this tenet should be injected."""
        # Archived tenets are never injected
        if self.status == TenetStatus.ARCHIVED:
            return False

        # Check injection strategy
        if not self.injection_strategy.should_inject(context_length, already_injected):
            return False

        # High priority tenets are injected more frequently
        if self.priority == Priority.CRITICAL:
            return True
        elif self.priority == Priority.HIGH:
            return self.metrics.reinforcement_needed or already_injected == 0
        else:
            return already_injected == 0 or self.metrics.reinforcement_needed

    def format_for_injection(self) -> str:
        """Format tenet content for injection into context."""
        prefix = f"[{self.category.value.upper()}]" if self.category else "[PRINCIPLE]"
        return f"{prefix} {self.content}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "priority": self.priority.value,
            "category": self.category.value if self.category else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "instilled_at": self.instilled_at.isoformat() if self.instilled_at else None,
            "updated_at": self.updated_at.isoformat(),
            "session_bindings": self.session_bindings,
            "author": self.author,
            "metrics": {
                "injection_count": self.metrics.injection_count,
                "last_injected": (
                    self.metrics.last_injected.isoformat() if self.metrics.last_injected else None
                ),
                "contexts_appeared_in": self.metrics.contexts_appeared_in,
                "compliance_score": self.metrics.compliance_score,
                "reinforcement_needed": self.metrics.reinforcement_needed,
            },
            "injection_strategy": {
                "frequency": self.injection_strategy.frequency,
                "position": self.injection_strategy.position,
                "max_per_context": self.injection_strategy.max_per_context,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tenet":
        """Create Tenet from dictionary."""
        # Parse dates
        created_at = (
            datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )
        instilled_at = (
            datetime.fromisoformat(data["instilled_at"]) if data.get("instilled_at") else None
        )
        updated_at = (
            datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now()
        )

        # Parse metrics
        metrics = TenetMetrics()
        if "metrics" in data:
            m = data["metrics"]
            metrics.injection_count = m.get("injection_count", 0)
            metrics.contexts_appeared_in = m.get("contexts_appeared_in", 0)
            metrics.compliance_score = m.get("compliance_score", 0.0)
            metrics.reinforcement_needed = m.get("reinforcement_needed", False)
            if m.get("last_injected"):
                metrics.last_injected = datetime.fromisoformat(m["last_injected"])

        # Parse injection strategy
        strategy = InjectionStrategy()
        if "injection_strategy" in data:
            s = data["injection_strategy"]
            strategy.frequency = s.get("frequency", "adaptive")
            strategy.position = s.get("position", "strategic")
            strategy.max_per_context = s.get("max_per_context", 3)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data["content"],
            priority=Priority(data.get("priority", "medium")),
            category=TenetCategory(data["category"]) if data.get("category") else None,
            status=TenetStatus(data.get("status", "pending")),
            created_at=created_at,
            instilled_at=instilled_at,
            updated_at=updated_at,
            session_bindings=data.get("session_bindings", []),
            author=data.get("author"),
            metrics=metrics,
            injection_strategy=strategy,
            metadata=data.get("metadata", {}),
        )


@dataclass
class TenetCollection:
    """A collection of related tenets."""

    name: str
    description: str = ""
    tenets: list[Tenet] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

    def add_tenet(self, tenet: Tenet) -> None:
        """Add a tenet to the collection."""
        if tenet not in self.tenets:
            self.tenets.append(tenet)

    def remove_tenet(self, tenet_id: str) -> bool:
        """Remove a tenet by ID."""
        for i, tenet in enumerate(self.tenets):
            if tenet.id == tenet_id:
                self.tenets.pop(i)
                return True
        return False

    def get_by_category(self, category: TenetCategory) -> list[Tenet]:
        """Get all tenets of a specific category."""
        return [t for t in self.tenets if t.category == category]

    def get_by_priority(self, priority: Priority) -> list[Tenet]:
        """Get all tenets of a specific priority."""
        return [t for t in self.tenets if t.priority == priority]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "tenets": [t.to_dict() for t in self.tenets],
        }
