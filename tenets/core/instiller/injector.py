"""Tenet injection system.

This module handles the strategic injection of tenets into generated context
to maintain consistency across AI interactions.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tenets.models.context import ContextResult
from tenets.models.tenet import Priority, Tenet
from tenets.utils.logger import get_logger
from tenets.utils.tokens import count_tokens


class InjectionPosition(Enum):
    """Where to inject tenets in the context."""

    TOP = "top"
    BOTTOM = "bottom"
    STRATEGIC = "strategic"
    DISTRIBUTED = "distributed"


@dataclass
class InjectionPoint:
    """A specific point where a tenet can be injected."""

    position: int  # Character position in content
    score: float  # How good this position is (0-1)
    reason: str  # Why this is a good position
    after_section: Optional[str] = None  # Section this comes after


class TenetInjector:
    """Handles strategic injection of tenets into context."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the injector.

        Args:
            config: Injection configuration
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

        # Injection settings
        self.min_distance_between = self.config.get("min_distance_between", 1000)
        self.prefer_natural_breaks = self.config.get("prefer_natural_breaks", True)
        self.reinforce_at_end = self.config.get("reinforce_at_end", True)

    def inject_tenets(
        self,
        content: str,
        tenets: List[Tenet],
        format: str = "markdown",
        context_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Inject tenets into content.

        Args:
            content: The content to inject into
            tenets: List of tenets to inject
            format: Content format (markdown, xml, json)
            context_metadata: Metadata about the context

        Returns:
            Tuple of (modified content, injection metadata)
        """
        if not tenets:
            return content, {"injected_count": 0}

        # Analyze content structure
        structure = self._analyze_content_structure(content, format)

        # Determine injection strategy
        strategy = self._determine_strategy(
            content_length=len(content), tenet_count=len(tenets), structure=structure
        )

        # Find injection points
        injection_points = self._find_injection_points(
            content=content, structure=structure, strategy=strategy, tenet_count=len(tenets)
        )

        # Sort tenets by priority
        sorted_tenets = sorted(
            tenets, key=lambda t: (t.priority.weight, t.metrics.reinforcement_needed), reverse=True
        )

        # Inject tenets
        injected_content = content
        injection_map = []

        for i, (tenet, point) in enumerate(zip(sorted_tenets, injection_points)):
            # Format tenet for injection
            formatted_tenet = self._format_tenet(tenet, format, position=i)

            # Calculate actual position (accounting for previous injections)
            offset = sum(len(inj["content"]) for inj in injection_map)
            actual_position = point.position + offset

            # Inject
            injected_content = (
                injected_content[:actual_position]
                + formatted_tenet
                + injected_content[actual_position:]
            )

            # Track injection
            injection_map.append(
                {
                    "tenet_id": tenet.id,
                    "position": actual_position,
                    "content": formatted_tenet,
                    "reason": point.reason,
                }
            )

        # Add reinforcement section if needed
        if self.reinforce_at_end and len(sorted_tenets) > 3:
            reinforcement = self._create_reinforcement_section(
                sorted_tenets[:3],
                format,  # Top 3 most important
            )
            injected_content += f"\n\n{reinforcement}"

        # Build metadata
        metadata = {
            "injected_count": len(injection_map),
            "strategy": strategy.value,
            "injections": injection_map,
            "token_increase": count_tokens(injected_content) - count_tokens(content),
            "reinforcement_added": self.reinforce_at_end and len(sorted_tenets) > 3,
        }

        return injected_content, metadata

    def _analyze_content_structure(self, content: str, format: str) -> Dict[str, Any]:
        """Analyze content to understand its structure.

        Args:
            content: Content to analyze
            format: Content format

        Returns:
            Structure analysis
        """
        structure = {
            "sections": [],
            "code_blocks": [],
            "natural_breaks": [],
            "total_lines": content.count("\n") + 1,
            "total_chars": len(content),
        }

        if format == "markdown":
            # Find markdown sections
            section_pattern = r"^(#{1,6})\s+(.+)$"
            for match in re.finditer(section_pattern, content, re.MULTILINE):
                structure["sections"].append(
                    {
                        "level": len(match.group(1)),
                        "title": match.group(2),
                        "position": match.start(),
                        "end_position": match.end(),
                    }
                )

            # Find code blocks
            code_pattern = r"```[\s\S]*?```"
            for match in re.finditer(code_pattern, content):
                structure["code_blocks"].append({"start": match.start(), "end": match.end()})

            # Find natural breaks (double newlines)
            break_pattern = r"\n\n+"
            for match in re.finditer(break_pattern, content):
                structure["natural_breaks"].append(match.start())

        elif format == "xml":
            # Find XML tags
            tag_pattern = r"<(\w+)[^>]*>.*?</\1>"
            for match in re.finditer(tag_pattern, content, re.DOTALL):
                structure["sections"].append(
                    {"tag": match.group(1), "position": match.start(), "end_position": match.end()}
                )

        return structure

    def _determine_strategy(
        self, content_length: int, tenet_count: int, structure: Dict[str, Any]
    ) -> InjectionPosition:
        """Determine the best injection strategy.

        Args:
            content_length: Length of content
            tenet_count: Number of tenets to inject
            structure: Content structure analysis

        Returns:
            Injection position strategy
        """
        # For short content or few tenets, use top injection
        if content_length < 5000 or tenet_count <= 2:
            return InjectionPosition.TOP

        # For very long content with many tenets, distribute
        if content_length > 50000 and tenet_count > 5:
            return InjectionPosition.DISTRIBUTED

        # If we have good section structure, use strategic
        if len(structure["sections"]) >= 3:
            return InjectionPosition.STRATEGIC

        # Default to strategic placement
        return InjectionPosition.STRATEGIC

    def _find_injection_points(
        self, content: str, structure: Dict[str, Any], strategy: InjectionPosition, tenet_count: int
    ) -> List[InjectionPoint]:
        """Find optimal injection points.

        Args:
            content: Content to inject into
            structure: Content structure
            strategy: Injection strategy
            tenet_count: Number of injection points needed

        Returns:
            List of injection points
        """
        points = []

        if strategy == InjectionPosition.TOP:
            # Inject all at the top after initial context
            if structure["sections"]:
                # After first section
                first_section = structure["sections"][0]
                position = first_section["end_position"] + 1
                for i in range(tenet_count):
                    points.append(
                        InjectionPoint(
                            position=position,
                            score=1.0 - (i * 0.1),  # Slightly decreasing score
                            reason="top_of_context",
                            after_section=first_section.get("title"),
                        )
                    )
            else:
                # At the very beginning
                for i in range(tenet_count):
                    points.append(
                        InjectionPoint(
                            position=0, score=1.0 - (i * 0.1), reason="beginning_of_context"
                        )
                    )

        elif strategy == InjectionPosition.STRATEGIC:
            # Find strategic positions throughout content
            candidate_points = []

            # After major sections
            for i, section in enumerate(structure["sections"]):
                if i == 0:  # After first section is high priority
                    score = 0.9
                elif section.get("level", 1) <= 2:  # Major sections
                    score = 0.7
                else:
                    score = 0.5

                # Look for natural break after section
                next_break = None
                for break_pos in structure["natural_breaks"]:
                    if break_pos > section["end_position"]:
                        next_break = break_pos
                        break

                position = next_break if next_break else section["end_position"] + 1

                # Avoid code blocks
                in_code = any(
                    cb["start"] <= position <= cb["end"] for cb in structure["code_blocks"]
                )

                if not in_code:
                    candidate_points.append(
                        InjectionPoint(
                            position=position,
                            score=score,
                            reason=f"after_section_{section.get('title', 'unnamed')}",
                            after_section=section.get("title"),
                        )
                    )

            # Add points at natural breaks
            for break_pos in structure["natural_breaks"]:
                # Check if not already near a section
                near_section = any(abs(break_pos - p.position) < 100 for p in candidate_points)

                if not near_section:
                    candidate_points.append(
                        InjectionPoint(position=break_pos, score=0.6, reason="natural_break")
                    )

            # Sort by score and position
            candidate_points.sort(key=lambda p: (-p.score, p.position))

            # Select best points with minimum distance
            selected = []
            for point in candidate_points:
                if len(selected) >= tenet_count:
                    break

                # Check minimum distance from other selected points
                too_close = any(
                    abs(point.position - p.position) < self.min_distance_between for p in selected
                )

                if not too_close:
                    selected.append(point)

            points = selected

        elif strategy == InjectionPosition.DISTRIBUTED:
            # Evenly distribute throughout content
            interval = len(content) // (tenet_count + 1)

            for i in range(tenet_count):
                target_position = interval * (i + 1)

                # Find nearest natural break
                best_break = target_position
                if structure["natural_breaks"]:
                    distances = [(abs(b - target_position), b) for b in structure["natural_breaks"]]
                    distances.sort()
                    if distances and distances[0][0] < interval // 4:
                        best_break = distances[0][1]

                points.append(
                    InjectionPoint(
                        position=best_break, score=0.7, reason=f"distributed_position_{i + 1}"
                    )
                )

        # Ensure we have enough points
        while len(points) < tenet_count:
            # Add at the end
            points.append(InjectionPoint(position=len(content), score=0.5, reason="end_fallback"))

        return points[:tenet_count]

    def _format_tenet(self, tenet: Tenet, format: str, position: int) -> str:
        """Format a tenet for injection.

        Args:
            tenet: Tenet to format
            format: Content format
            position: Position index (for variation)

        Returns:
            Formatted tenet string
        """
        # Base formatting
        if format == "markdown":
            # Vary the formatting to avoid monotony
            if position == 0:
                # First tenet gets emphasis
                prefix = "**🎯 Key Guiding Principle:**"
            elif tenet.priority == Priority.CRITICAL:
                prefix = "**⚠️ Critical Guiding Principle:**"
            elif tenet.priority == Priority.HIGH:
                prefix = "**📌 Important Guiding Principle:**"
            else:
                prefix = "**💡 Guiding Principle:**"

            formatted = f"\n{prefix} {tenet.content}\n"

        elif format == "xml":
            priority_attr = f'priority="{tenet.priority.value}"'
            category_attr = f'category="{tenet.category.value}"' if tenet.category else ""
            formatted = f"\n<guiding_principle {priority_attr} {category_attr}>{tenet.content}</guiding_principle>\n"

        elif format == "json":
            # For JSON, we'll need to handle this differently in practice
            formatted = f"\n/* GUIDING PRINCIPLE: {tenet.content} */\n"

        else:
            # Plain text fallback
            formatted = f"\n[GUIDING PRINCIPLE - {tenet.priority.value.upper()}] {tenet.content}\n"

        return formatted

    def _create_reinforcement_section(self, top_tenets: List[Tenet], format: str) -> str:
        """Create a reinforcement section for key tenets.

        Args:
            top_tenets: Most important tenets to reinforce
            format: Content format

        Returns:
            Reinforcement section string
        """
        if format == "markdown":
            section = "## 🎯 Key Guiding Principles to Remember\n\n"
            section += (
                "As you work with this code, keep these critical guiding principles in mind:\n\n"
            )

            for i, tenet in enumerate(top_tenets, 1):
                icon = "🔴" if tenet.priority == Priority.CRITICAL else "🟡"
                section += f"{i}. {icon} **{tenet.content}**\n"

            section += "\nThese guiding principles should guide all decisions and implementations."

        elif format == "xml":
            section = "<guiding_principles_reinforcement>\n"
            section += "<title>Key Guiding Principles to Remember</title>\n"
            section += "<guiding_principles>\n"

            for tenet in top_tenets:
                section += f'  <guiding_principle priority="{tenet.priority.value}">{tenet.content}</guiding_principle>\n'

            section += "</guiding_principles>\n"
            section += "</guiding_principles_reinforcement>"

        else:
            section = "\n" + "=" * 60 + "\n"
            section += "KEY GUIDING PRINCIPLES TO REMEMBER:\n"
            section += "=" * 60 + "\n\n"

            for i, tenet in enumerate(top_tenets, 1):
                section += (
                    f"{i}. [GUIDING PRINCIPLE - {tenet.priority.value.upper()}] {tenet.content}\n"
                )

        return section

    def calculate_optimal_injection_count(
        self, content_length: int, available_tenets: int, max_token_increase: int = 1000
    ) -> int:
        """Calculate optimal number of tenets to inject.

        Args:
            content_length: Current content length
            available_tenets: Number of available tenets
            max_token_increase: Maximum allowed token increase

        Returns:
            Optimal number of tenets to inject
        """
        # Estimate tokens per tenet (including formatting)
        avg_tenet_tokens = 30

        # Calculate based on content length
        if content_length < 1000:
            base_count = 1
        elif content_length < 5000:
            base_count = 2
        elif content_length < 20000:
            base_count = 3
        elif content_length < 50000:
            base_count = 5
        else:
            base_count = 7

        # Limit by token budget
        max_by_tokens = max_token_increase // avg_tenet_tokens

        # Take minimum of all constraints
        return min(base_count, available_tenets, max_by_tokens)

    def inject_into_context_result(
        self, context_result: ContextResult, tenets: List[Tenet]
    ) -> ContextResult:
        """Inject tenets into a ContextResult object.

        Args:
            context_result: The context result to modify
            tenets: Tenets to inject

        Returns:
            Modified context result
        """
        # Inject into the context content
        modified_content, injection_metadata = self.inject_tenets(
            content=context_result.context,
            tenets=tenets,
            format=context_result.format,
            context_metadata=context_result.metadata,
        )

        # Update the context result
        context_result.context = modified_content

        # Update metadata
        context_result.metadata["tenet_injection"] = injection_metadata
        context_result.metadata["tenets_injected"] = [
            {"id": t.id, "content": t.content, "priority": t.priority.value} for t in tenets
        ]

        # Update token count if available
        if "total_tokens" in context_result.metadata:
            context_result.metadata["total_tokens"] += injection_metadata["token_increase"]

        return context_result
