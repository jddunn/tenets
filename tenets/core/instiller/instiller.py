"""Main instiller module for orchestrating tenet injection into context.

This module provides the high-level interface for the instiller system,
coordinating between the TenetManager and TenetInjector to apply guiding
principles to generated context.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tenets.config import TenetsConfig
from tenets.core.instiller.injector import InjectionPosition, TenetInjector
from tenets.core.instiller.manager import TenetManager
from tenets.models.context import ContextResult
from tenets.models.tenet import Priority, Tenet, TenetStatus
from tenets.utils.logger import get_logger
from tenets.utils.tokens import count_tokens


@dataclass
class InstillationResult:
    """Result of instilling tenets into context.

    This dataclass captures all the information about what tenets were
    instilled, where they were injected, and the overall impact on the context.

    Attributes:
        tenets_instilled: List of tenets that were successfully instilled
        injection_positions: Where each tenet was injected in the content
        token_increase: How many tokens were added by tenet injection
        strategy_used: Which injection strategy was employed
        session: The session this was applied to (if any)
        timestamp: When the instillation occurred
        success: Whether the operation succeeded
        error: Error message if operation failed
        metrics: Additional metrics about the instillation
    """

    tenets_instilled: List[Tenet]
    injection_positions: List[Dict[str, Any]]
    token_increase: int
    strategy_used: str
    session: Optional[str] = None
    timestamp: datetime = None
    success: bool = True
    error: Optional[str] = None
    metrics: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization.

        Returns:
            Dictionary containing all instillation result data
        """
        return {
            "tenets_instilled": [t.to_dict() for t in self.tenets_instilled],
            "injection_positions": self.injection_positions,
            "token_increase": self.token_increase,
            "strategy_used": self.strategy_used,
            "session": self.session,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics,
        }


class Instiller:
    """Main orchestrator for the tenet instillation system.

    The Instiller coordinates between the TenetManager (which manages tenet
    storage and lifecycle) and the TenetInjector (which handles strategic
    injection into content). It provides the high-level interface for applying
    guiding principles to maintain consistency across AI interactions.

    The instiller system is designed to combat "context drift" - the tendency
    for AI assistants to gradually deviate from established coding principles
    over the course of a conversation. By strategically injecting tenets at
    optimal positions in the context, we maintain consistency even across
    long conversations with many prompts.

    Attributes:
        config: Tenets configuration
        logger: Logger instance
        manager: TenetManager for tenet lifecycle
        injector: TenetInjector for strategic injection
        cache: Cache for instillation results
        metrics_tracker: Tracks effectiveness metrics
    """

    def __init__(self, config: TenetsConfig):
        """Initialize the Instiller with configuration.

        Args:
            config: Tenets configuration containing settings for tenet management,
                   injection strategies, and system behavior
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize components
        self.manager = TenetManager(config)
        self.injector = TenetInjector(
            config.tenet_injection_config if hasattr(config, "tenet_injection_config") else None
        )

        # Cache for instillation results
        self._cache = {}

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        self.logger.info(
            "Instiller initialized with %d active tenets", len(self.manager._tenet_cache)
        )

    def instill(
        self,
        context: Union[str, ContextResult],
        session: Optional[str] = None,
        force: bool = False,
        max_tenets: Optional[int] = None,
        strategy: Optional[str] = None,
    ) -> Union[str, ContextResult, InstillationResult]:
        """Instill pending tenets into context.

        This is the main method for applying guiding principles to generated
        context. It retrieves applicable tenets, determines the optimal injection
        strategy, and modifies the context to include the tenets at strategic
        positions.

        Args:
            context: The context to instill tenets into. Can be a string of
                    content or a ContextResult object
            session: Optional session name to filter tenets by session binding
            force: If True, re-instill even already instilled tenets
            max_tenets: Maximum number of tenets to inject (overrides config)
            strategy: Injection strategy to use (overrides automatic selection)

        Returns:
            Modified context with tenets instilled. Returns the same type as
            the input (string or ContextResult), or an InstillationResult if
            detailed results are requested

        Raises:
            ValueError: If context format is invalid
            RuntimeError: If instillation fails due to system error

        Example:
            >>> instiller = Instiller(config)
            >>> context = "# OAuth Implementation\\n\\nThis implements..."
            >>> result = instiller.instill(context, session="oauth-work")
            >>> print(result)  # Context now has tenets injected
        """
        self.logger.info("Beginning tenet instillation for session: %s", session or "global")

        # Extract content and metadata based on input type
        if isinstance(context, str):
            content = context
            format_type = "markdown"
            metadata = {}
            return_context_result = False
        elif isinstance(context, ContextResult):
            content = context.context
            format_type = context.format
            metadata = context.metadata
            return_context_result = True
        else:
            raise ValueError(f"Invalid context type: {type(context)}")

        # Get tenets to instill
        tenets_to_instill = self._get_tenets_for_instillation(
            session=session,
            force=force,
            content_length=len(content),
            max_tenets=max_tenets or self.config.max_tenets_per_context,
        )

        if not tenets_to_instill:
            self.logger.info("No tenets to instill")
            if return_context_result:
                return context
            return content

        self.logger.info("Instilling %d tenets into context", len(tenets_to_instill))

        # Determine injection strategy
        if not strategy:
            strategy = self._determine_injection_strategy(
                content_length=len(content),
                tenet_count=len(tenets_to_instill),
                format_type=format_type,
            )

        # Inject tenets into content
        modified_content, injection_metadata = self.injector.inject_tenets(
            content=content, tenets=tenets_to_instill, format=format_type, context_metadata=metadata
        )

        # Update tenet status and metrics
        for tenet in tenets_to_instill:
            if not force:
                tenet.instill()
            tenet.metrics.update_injection()
            self.manager._save_tenet(tenet)

        # Track metrics
        self.metrics_tracker.record_instillation(
            tenet_count=len(tenets_to_instill),
            token_increase=injection_metadata["token_increase"],
            strategy=strategy,
            session=session,
        )

        # Build result
        result = InstillationResult(
            tenets_instilled=tenets_to_instill,
            injection_positions=injection_metadata.get("injections", []),
            token_increase=injection_metadata["token_increase"],
            strategy_used=strategy,
            session=session,
            metrics={
                "original_length": len(content),
                "modified_length": len(modified_content),
                "format": format_type,
                "reinforcement_added": injection_metadata.get("reinforcement_added", False),
            },
        )

        # Cache result
        cache_key = f"{session or 'global'}_{datetime.now().isoformat()}"
        self._cache[cache_key] = result

        # Return appropriate type
        if return_context_result:
            context.context = modified_content
            context.metadata["tenet_instillation"] = result.to_dict()
            return context
        else:
            return modified_content

    def _get_tenets_for_instillation(
        self, session: Optional[str], force: bool, content_length: int, max_tenets: int
    ) -> List[Tenet]:
        """Get the list of tenets to instill.

        This method retrieves tenets from the manager and filters them based on
        various criteria including session binding, priority, and whether they've
        already been instilled.

        Args:
            session: Optional session to filter by
            force: Whether to include already instilled tenets
            content_length: Length of the content (for determining count)
            max_tenets: Maximum number of tenets to return

        Returns:
            List of Tenet objects ready for injection
        """
        if force:
            # Get all non-archived tenets
            candidates = [
                t for t in self.manager._tenet_cache.values() if t.status != TenetStatus.ARCHIVED
            ]
        else:
            # Get only pending tenets
            candidates = self.manager.get_pending_tenets(session)

        # Filter by session if specified
        if session:
            candidates = [t for t in candidates if t.applies_to_session(session)]

        # Sort by priority and metrics
        candidates.sort(
            key=lambda t: (
                t.priority.weight,
                t.metrics.reinforcement_needed,
                -t.metrics.injection_count,  # Prefer less frequently injected
                t.created_at,
            ),
            reverse=True,
        )

        # Determine optimal count
        optimal_count = self.injector.calculate_optimal_injection_count(
            content_length=content_length, available_tenets=len(candidates), max_token_increase=1000
        )

        # Take the minimum of optimal count and max_tenets
        count = min(optimal_count, max_tenets, len(candidates))

        return candidates[:count]

    def _determine_injection_strategy(
        self, content_length: int, tenet_count: int, format_type: str
    ) -> str:
        """Determine the best injection strategy.

        Analyzes the content and tenet characteristics to select the optimal
        injection strategy for maximum effectiveness while minimizing disruption
        to the natural flow of the content.

        Args:
            content_length: Length of the content in characters
            tenet_count: Number of tenets to inject
            format_type: Format of the content (markdown, xml, json)

        Returns:
            Strategy name: 'top', 'strategic', 'distributed', etc.
        """
        # Short content or few tenets - put at top
        if content_length < 5000 or tenet_count <= 2:
            return "top"

        # Very long content with many tenets - distribute
        if content_length > 50000 and tenet_count > 5:
            return "distributed"

        # XML format works well with strategic placement
        if format_type == "xml":
            return "strategic"

        # Default to strategic for most cases
        return "strategic"

    def analyze_effectiveness(
        self, session: Optional[str] = None, time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze the effectiveness of tenet instillation.

        This method provides insights into how well the tenet system is working,
        including metrics on injection frequency, compliance scores, and areas
        that may need reinforcement.

        Args:
            session: Optional session to analyze
            time_period: Time period to analyze (e.g., "last_week", "last_month")

        Returns:
            Dictionary containing effectiveness analysis including:
            - Total instillations
            - Average token increase
            - Most/least effective tenets
            - Compliance trends
            - Recommendations for improvement
        """
        analysis = self.manager.analyze_tenet_effectiveness()

        # Add instillation metrics
        instillation_metrics = self.metrics_tracker.get_metrics(
            session=session, time_period=time_period
        )

        analysis["instillation_metrics"] = instillation_metrics

        # Calculate compliance trends
        if self._cache:
            recent_results = list(self._cache.values())[-10:]  # Last 10 instillations
            avg_token_increase = sum(r.token_increase for r in recent_results) / len(recent_results)
            analysis["avg_token_increase"] = avg_token_increase

        # Recommendations
        recommendations = []

        if analysis.get("need_reinforcement"):
            recommendations.append(
                f"Consider reinforcing {len(analysis['need_reinforcement'])} tenets that need attention"
            )

        if instillation_metrics.get("avg_tenets_per_context", 0) < 2:
            recommendations.append(
                "Consider increasing the number of tenets per context for better consistency"
            )

        analysis["recommendations"] = recommendations

        return analysis

    def export_instillation_history(
        self, output_path: Union[str, Path], format: str = "json"
    ) -> None:
        """Export the history of tenet instillations.

        Exports detailed records of all tenet instillations for analysis,
        auditing, or transfer to another system.

        Args:
            output_path: Path to save the export
            format: Export format ('json' or 'csv')

        Raises:
            ValueError: If format is not supported
            IOError: If unable to write to output path
        """
        output_path = Path(output_path)

        # Prepare export data
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "version": "1.0",
            "instillations": [result.to_dict() for result in self._cache.values()],
            "metrics": self.metrics_tracker.get_all_metrics(),
        }

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
        elif format == "csv":
            # CSV export for analysis in spreadsheets
            import csv

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Timestamp", "Session", "Tenets Count", "Token Increase", "Strategy"]
                )
                for result in self._cache.values():
                    writer.writerow(
                        [
                            result.timestamp.isoformat(),
                            result.session or "global",
                            len(result.tenets_instilled),
                            result.token_increase,
                            result.strategy_used,
                        ]
                    )
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info("Exported instillation history to %s", output_path)


class MetricsTracker:
    """Tracks metrics for tenet instillation effectiveness.

    This class maintains detailed metrics about how tenets are being used,
    their effectiveness, and patterns in instillation that can inform
    improvements to the system.
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.instillations = []
        self.session_metrics = {}
        self.strategy_usage = {}

    def record_instillation(
        self, tenet_count: int, token_increase: int, strategy: str, session: Optional[str] = None
    ) -> None:
        """Record metrics for an instillation.

        Args:
            tenet_count: Number of tenets instilled
            token_increase: Tokens added by injection
            strategy: Strategy used for injection
            session: Session name if applicable
        """
        record = {
            "timestamp": datetime.now(),
            "tenet_count": tenet_count,
            "token_increase": token_increase,
            "strategy": strategy,
            "session": session,
        }

        self.instillations.append(record)

        # Update session metrics
        session_key = session or "global"
        if session_key not in self.session_metrics:
            self.session_metrics[session_key] = {
                "total_instillations": 0,
                "total_tenets": 0,
                "total_tokens": 0,
            }

        self.session_metrics[session_key]["total_instillations"] += 1
        self.session_metrics[session_key]["total_tenets"] += tenet_count
        self.session_metrics[session_key]["total_tokens"] += token_increase

        # Track strategy usage
        self.strategy_usage[strategy] = self.strategy_usage.get(strategy, 0) + 1

    def get_metrics(
        self, session: Optional[str] = None, time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get metrics for analysis.

        Args:
            session: Optional session to filter by
            time_period: Optional time period to filter

        Returns:
            Dictionary of metrics
        """
        # Filter records
        records = self.instillations
        if session:
            records = [r for r in records if r["session"] == session]

        if not records:
            return {"message": "No instillation records found"}

        return {
            "total_instillations": len(records),
            "total_tenets_instilled": sum(r["tenet_count"] for r in records),
            "total_token_increase": sum(r["token_increase"] for r in records),
            "avg_tenets_per_context": sum(r["tenet_count"] for r in records) / len(records),
            "avg_token_increase": sum(r["token_increase"] for r in records) / len(records),
            "strategy_usage": self.strategy_usage,
            "session_breakdown": self.session_metrics,
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics for export.

        Returns:
            Complete metrics dictionary
        """
        return {
            "instillations": [
                {
                    "timestamp": r["timestamp"].isoformat(),
                    "tenet_count": r["tenet_count"],
                    "token_increase": r["token_increase"],
                    "strategy": r["strategy"],
                    "session": r["session"],
                }
                for r in self.instillations
            ],
            "session_metrics": self.session_metrics,
            "strategy_usage": self.strategy_usage,
        }
