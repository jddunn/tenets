"""Summary models for file condensation.

This module defines data structures for managing file summaries
when content needs to be condensed to fit within token limits.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class SummaryStrategy(Enum):
    """Strategies for summarizing files."""

    TRUNCATE = "truncate"  # Simple truncation
    EXTRACT = "extract"  # Extract key sections
    COMPRESS = "compress"  # Aggressive compression
    SEMANTIC = "semantic"  # ML-based semantic summary
    LLM = "llm"  # LLM-generated summary
    HYBRID = "hybrid"  # Combination of strategies

    @classmethod
    def get_default(cls) -> "SummaryStrategy":
        """Get default summarization strategy."""
        return cls.EXTRACT

    def get_priority(self) -> int:
        """Get priority for strategy selection."""
        priorities = {
            cls.LLM: 5,
            cls.SEMANTIC: 4,
            cls.HYBRID: 3,
            cls.EXTRACT: 2,
            cls.COMPRESS: 1,
            cls.TRUNCATE: 0,
        }
        return priorities.get(self, 0)


@dataclass
class SummarySection:
    """A section within a file summary.

    Represents a specific section of code that was extracted
    for inclusion in the summary.

    Attributes:
        name: Section name (e.g., "imports", "class_definitions")
        content: Section content
        line_start: Starting line in original file
        line_end: Ending line in original file
        importance: Importance score (0-1)
        preserved_fully: Whether section was preserved in full
        tokens: Token count for this section
        metadata: Additional section metadata
    """

    name: str
    content: str
    line_start: int = 0
    line_end: int = 0
    importance: float = 1.0
    preserved_fully: bool = True
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization processing."""
        # Calculate line count if not set
        if self.line_end == 0 and self.line_start > 0:
            self.line_end = self.line_start + self.content.count("\n")

        # Estimate tokens if not set
        if self.tokens == 0:
            self.tokens = len(self.content) // 4  # Rough estimate

    def truncate(self, max_tokens: int) -> "SummarySection":
        """Truncate section to fit within token limit."""
        if self.tokens <= max_tokens:
            return self

        # Calculate approximate character limit
        char_limit = (max_tokens * 4) - 50  # Leave buffer for ellipsis

        if len(self.content) <= char_limit:
            return self

        # Truncate content
        truncated_content = self.content[:char_limit] + "\n... [truncated]"

        return SummarySection(
            name=self.name,
            content=truncated_content,
            line_start=self.line_start,
            line_end=self.line_start + truncated_content.count("\n"),
            importance=self.importance,
            preserved_fully=False,
            tokens=max_tokens,
            metadata={**self.metadata, "truncated": True},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SummarySection":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FileSummary:
    """Summary of a file's content.

    Represents a condensed version of a file when the full content
    would exceed token limits. Contains sections, metadata, and
    instructions for AI assistants.

    Attributes:
        content: Summarized content
        was_summarized: Whether content was summarized
        original_tokens: Token count of original
        summary_tokens: Token count of summary
        original_lines: Line count of original
        summary_lines: Line count of summary
        preserved_sections: Sections that were preserved
        ignored_sections: Sections that were omitted
        sections: List of summary sections
        strategy: Strategy used for summarization
        compression_ratio: Ratio of summary to original
        instructions: Instructions for AI about summary
        metadata: Additional metadata
        file_path: Original file path
        timestamp: When summary was created

        Compatibility fields:
        - path: legacy alias for file_path
        - summary: legacy alias for content
        - token_count: legacy alias for summary_tokens
    """

    content: str = ""
    was_summarized: bool = True
    original_tokens: int = 0
    summary_tokens: int = 0
    original_lines: int = 0
    summary_lines: int = 0
    preserved_sections: List[str] = field(default_factory=list)
    ignored_sections: List[str] = field(default_factory=list)
    sections: List[SummarySection] = field(default_factory=list)
    strategy: str = "extract"
    compression_ratio: float = 0.0
    instructions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    # Backward-compat: support `path=` constructor arg used by tests
    path: Optional[str] = None
    # Backward-compat: support `summary=` and `token_count=` args used by tests
    summary: Optional[str] = None
    token_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Post-initialization processing."""
        # Map legacy `path` to `file_path` if provided
        if not self.file_path and self.path:
            self.file_path = self.path

        # Map legacy `summary` to `content` if provided
        if self.summary and not self.content:
            self.content = self.summary

        # Map legacy `token_count` to `summary_tokens` if provided
        if self.token_count and not self.summary_tokens:
            self.summary_tokens = self.token_count
        # Keep token_count mirrored for external reads
        if not self.token_count and self.summary_tokens:
            self.token_count = self.summary_tokens

        # Calculate compression ratio if not set
        if self.original_tokens > 0 and self.compression_ratio == 0:
            self.compression_ratio = self.summary_tokens / self.original_tokens

        # Calculate line counts if not set
        if self.summary_lines == 0:
            self.summary_lines = self.content.count("\n") + 1 if self.content else 0

        # Add default instructions if none provided
        if self.was_summarized and not self.instructions:
            self.add_default_instructions()

    def add_instruction(self, instruction: str) -> None:
        """Add an instruction for the AI about this summary."""
        if instruction and instruction not in self.instructions:
            self.instructions.append(instruction)

    def add_default_instructions(self) -> None:
        """Add default instructions based on summary characteristics."""
        if self.file_path:
            self.add_instruction(
                f"This is a summary of {self.file_path} "
                f"({self.original_lines} lines → {self.summary_lines} lines)"
            )

        if self.compression_ratio < 0.5 and self.original_tokens > 0:
            self.add_instruction(
                f"Significant compression applied ({self.compression_ratio:.1%} of original). "
                f"Request full file if more detail needed."
            )

        if self.ignored_sections:
            self.add_instruction(
                f"Omitted sections: {', '.join(self.ignored_sections)}. "
                f"Request specific sections if needed."
            )

    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata about the summary."""
        self.metadata.update(metadata)

    def add_section(self, section: SummarySection) -> None:
        """Add a section to the summary."""
        self.sections.append(section)
        self.summary_tokens += section.tokens
        self.token_count = self.summary_tokens

        if section.preserved_fully:
            self.preserved_sections.append(section.name)
        else:
            if section.name not in self.ignored_sections:
                self.ignored_sections.append(section.name)

    def get_section(self, name: str) -> Optional[SummarySection]:
        """Get a specific section by name."""
        for section in self.sections:
            if section.name == name:
                return section
        return None

    def merge_sections(self) -> str:
        """Merge all sections into final content."""
        if not self.sections:
            return self.content

        merged = []
        for section in self.sections:
            if section.name:
                merged.append(f"# {section.name}")
            merged.append(section.content)
            merged.append("")  # Empty line between sections

        self.content = "\n".join(merged)
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "was_summarized": self.was_summarized,
            "original_tokens": self.original_tokens,
            "summary_tokens": self.summary_tokens,
            "original_lines": self.original_lines,
            "summary_lines": self.summary_lines,
            "preserved_sections": self.preserved_sections,
            "ignored_sections": self.ignored_sections,
            "sections": [s.to_dict() for s in self.sections],
            "strategy": self.strategy,
            "compression_ratio": self.compression_ratio,
            "instructions": self.instructions,
            "metadata": self.metadata,
            "file_path": self.file_path,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileSummary":
        """Create from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        if "sections" in data:
            data["sections"] = [
                SummarySection.from_dict(s) if isinstance(s, dict) else s for s in data["sections"]
            ]

        return cls(**data)


@dataclass
class ProjectSummary:
    """Summary of an entire project.

    High-level summary for initial context or overview, providing
    a bird's eye view of the project structure and characteristics.

    Attributes:
        name: Project name
        description: Project description
        structure: Project structure overview
        key_files: Most important files
        key_directories: Important directories
        technologies: Technologies used
        frameworks: Frameworks detected
        patterns: Architectural patterns detected
        dependencies: Key dependencies
        statistics: Project statistics
        recent_activity: Recent development activity
        team_info: Team/contributor information
        metadata: Additional metadata
        timestamp: When summary was created
    """

    name: str
    description: str = ""
    structure: Dict[str, Any] = field(default_factory=dict)
    key_files: List[str] = field(default_factory=list)
    key_directories: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recent_activity: Dict[str, Any] = field(default_factory=dict)
    team_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Post-initialization processing."""
        # Initialize default statistics if not provided
        if not self.statistics:
            self.statistics = {
                "total_files": 0,
                "total_lines": 0,
                "languages": {},
                "file_types": {},
            }

    def add_technology(self, tech: str) -> None:
        """Add a detected technology."""
        if tech and tech not in self.technologies:
            self.technologies.append(tech)

    def add_framework(self, framework: str) -> None:
        """Add a detected framework."""
        if framework and framework not in self.frameworks:
            self.frameworks.append(framework)

    def add_pattern(self, pattern: str) -> None:
        """Add a detected architectural pattern."""
        if pattern and pattern not in self.patterns:
            self.patterns.append(pattern)

    def update_statistics(self, key: str, value: Any) -> None:
        """Update a statistic value."""
        self.statistics[key] = value

    def to_markdown(self) -> str:
        """Generate markdown representation of project summary."""
        lines = [f"# Project: {self.name}", ""]

        if self.description:
            lines.extend([self.description, ""])

        # Technologies and frameworks
        if self.technologies or self.frameworks:
            lines.append("## Technologies & Frameworks")
            if self.technologies:
                lines.append(f"**Technologies:** {', '.join(self.technologies)}")
            if self.frameworks:
                lines.append(f"**Frameworks:** {', '.join(self.frameworks)}")
            lines.append("")

        # Key structure
        if self.key_directories:
            lines.append("## Key Directories")
            for dir in self.key_directories[:10]:  # Top 10
                lines.append(f"- `{dir}`")
            lines.append("")

        if self.key_files:
            lines.append("## Key Files")
            for file in self.key_files[:10]:  # Top 10
                lines.append(f"- `{file}`")
            lines.append("")

        # Statistics
        if self.statistics:
            lines.append("## Statistics")
            for key, value in self.statistics.items():
                if isinstance(value, dict):
                    lines.append(f"**{key}:**")
                    for k, v in value.items():
                        lines.append(f"  - {k}: {v}")
                else:
                    lines.append(f"- **{key}:** {value}")
            lines.append("")

        # Patterns
        if self.patterns:
            lines.append("## Architectural Patterns")
            for pattern in self.patterns:
                lines.append(f"- {pattern}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "structure": self.structure,
            "key_files": self.key_files,
            "key_directories": self.key_directories,
            "technologies": self.technologies,
            "frameworks": self.frameworks,
            "patterns": self.patterns,
            "dependencies": self.dependencies,
            "statistics": self.statistics,
            "recent_activity": self.recent_activity,
            "team_info": self.team_info,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectSummary":
        """Create from dictionary."""
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
