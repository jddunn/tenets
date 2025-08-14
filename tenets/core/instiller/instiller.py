"""Instiller module - Orchestrates intelligent tenet injection into context.

This module provides the main Instiller class that manages the injection of
guiding principles (tenets) into generated context. It supports various
injection strategies including:
- Always inject
- Periodic injection (every Nth time)
- Adaptive injection based on context complexity
- Session-aware smart injection

The instiller tracks injection history, analyzes context complexity using
NLP components, and adapts injection frequency based on session patterns.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tenets.config import TenetsConfig
from tenets.core.instiller.injector import InjectionPosition, TenetInjector
from tenets.core.instiller.manager import TenetManager
from tenets.models.context import ContextResult
from tenets.models.tenet import Priority, Tenet, TenetStatus
from tenets.utils.logger import get_logger


@dataclass
class InjectionHistory:
    """Track injection history for a session.

    Attributes:
        session_id: Session identifier
        total_distills: Total number of distill operations
        total_injections: Total number of tenet injections
        last_injection: Timestamp of last injection
        last_injection_index: Index of last injection (for periodic)
        complexity_scores: List of context complexity scores
        injected_tenets: Set of tenet IDs that have been injected
        reinforcement_count: Count of reinforcement injections
        created_at: When this history was created
        updated_at: Last update timestamp
    """

    session_id: str
    total_distills: int = 0
    total_injections: int = 0
    last_injection: Optional[datetime] = None
    last_injection_index: int = 0
    complexity_scores: List[float] = field(default_factory=list)
    injected_tenets: Set[str] = field(default_factory=set)
    reinforcement_count: int = 0
    # Track whether the system instruction has been injected for this session
    system_instruction_injected: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def should_inject(
        self,
        frequency: str,
        interval: int,
        complexity: float,
        complexity_threshold: float,
        min_session_length: int,
    ) -> Tuple[bool, str]:
        """Determine if tenets should be injected.

        Args:
            frequency: Injection frequency mode
            interval: Injection interval for periodic mode
            complexity: Current context complexity score
            complexity_threshold: Threshold for complexity-based injection
            min_session_length: Minimum session length before injection

        Returns:
            Tuple of (should_inject, reason)
        """
        # Always inject mode
        if frequency == "always":
            return True, "always_mode"

        # Manual mode - never auto-inject
        if frequency == "manual":
            return False, "manual_mode"

        # Check minimum session length
        if self.total_distills < min_session_length and self.total_injections == 0:
            return False, f"session_too_short_{self.total_distills}/{min_session_length}"

        # Periodic injection
        if frequency == "periodic":
            if self.total_distills % interval == 0 and self.total_distills > 0:
                return True, f"periodic_interval_{interval}"
            return False, f"not_at_interval_{self.total_distills % interval}/{interval}"

        # Adaptive injection
        if frequency == "adaptive":
            # First injection after minimum session length
            if self.total_injections == 0 and self.total_distills >= min_session_length:
                return True, "first_adaptive_injection"

            # Complexity-based injection
            if complexity >= complexity_threshold:
                # Check if we've injected recently
                if self.last_injection:
                    time_since_last = datetime.now() - self.last_injection
                    if time_since_last < timedelta(minutes=5):
                        return False, "injected_recently"
                return True, f"high_complexity_{complexity:.2f}"

            # Check for reinforcement interval
            if self.total_injections > 0:
                injections_since_last = self.total_distills - self.last_injection_index
                if injections_since_last >= interval * 2:  # Double interval for adaptive
                    return True, f"reinforcement_needed_{injections_since_last}"

        return False, "no_injection_criteria_met"

    def record_injection(self, tenets: List[Tenet], complexity: float) -> None:
        """Record that an injection occurred.

        Args:
            tenets: List of tenets that were injected
            complexity: Complexity score of the context
        """
        self.total_injections += 1
        self.last_injection = datetime.now()
        self.last_injection_index = self.total_distills
        self.complexity_scores.append(complexity)

        for tenet in tenets:
            self.injected_tenets.add(tenet.id)

        self.updated_at = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get injection statistics for this session.

        Returns:
            Dictionary of statistics
        """
        avg_complexity = (
            sum(self.complexity_scores) / len(self.complexity_scores)
            if self.complexity_scores
            else 0.0
        )

        injection_rate = (
            self.total_injections / self.total_distills if self.total_distills > 0 else 0.0
        )

        return {
            "session_id": self.session_id,
            "total_distills": self.total_distills,
            "total_injections": self.total_injections,
            "injection_rate": injection_rate,
            "average_complexity": avg_complexity,
            "unique_tenets_injected": len(self.injected_tenets),
            "reinforcement_count": self.reinforcement_count,
            "session_duration": (self.updated_at - self.created_at).total_seconds(),
            "last_injection": self.last_injection.isoformat() if self.last_injection else None,
        }


@dataclass
class InstillationResult:
    """Result of a tenet instillation operation.

    Attributes:
        tenets_instilled: List of tenets that were instilled
        injection_positions: Where tenets were injected
        token_increase: Number of tokens added
        strategy_used: Injection strategy that was used
        session: Session identifier if any
        timestamp: When instillation occurred
        success: Whether instillation succeeded
        error_message: Error message if failed
        metrics: Additional metrics from the operation
        complexity_score: Complexity score of the context
        skip_reason: Reason if injection was skipped
    """

    tenets_instilled: List[Tenet]
    injection_positions: List[Dict[str, Any]]
    token_increase: int
    strategy_used: str
    session: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    complexity_score: float = 0.0
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tenets_instilled": [t.to_dict() for t in self.tenets_instilled],
            "injection_positions": self.injection_positions,
            "token_increase": self.token_increase,
            "strategy_used": self.strategy_used,
            "session": self.session,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "complexity_score": self.complexity_score,
            "skip_reason": self.skip_reason,
        }


class ComplexityAnalyzer:
    """Analyze context complexity to guide injection decisions.

    Uses NLP components to analyze:
    - Token count and density
    - Code vs documentation ratio
    - Keyword diversity
    - Structural complexity
    - Topic coherence
    """

    def __init__(self, config: TenetsConfig):
        """Initialize complexity analyzer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize NLP components if available
        self._init_nlp_components()

    def _init_nlp_components(self) -> None:
        """Initialize NLP components for analysis."""
        try:
            from tenets.core.nlp import ML_AVAILABLE, CodeTokenizer, KeywordExtractor

            self.tokenizer = CodeTokenizer(use_stopwords=True)
            self.keyword_extractor = KeywordExtractor(
                use_yake=True, use_stopwords=True, stopword_set="code"
            )
            self.ml_available = ML_AVAILABLE

            if ML_AVAILABLE:
                from tenets.core.nlp import SemanticSimilarity

                self.semantic_analyzer = SemanticSimilarity()
            else:
                self.semantic_analyzer = None

        except ImportError:
            self.logger.warning("NLP components not available for complexity analysis")
            self.tokenizer = None
            self.keyword_extractor = None
            self.semantic_analyzer = None
            self.ml_available = False

    def analyze(self, context: Union[str, ContextResult]) -> float:
        """Analyze context complexity.

        Args:
            context: Context to analyze (string or ContextResult)

        Returns:
            Complexity score between 0 and 1
        """
        if isinstance(context, ContextResult):
            text = context.context
            metadata = context.metadata
        else:
            text = context
            metadata = {}

        if not text:
            return 0.0

        scores = []

        # Length-based complexity
        length_score = min(1.0, len(text) / 50000)  # Normalize to 50k chars
        scores.append(length_score * 0.2)  # 20% weight

        # Token diversity
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            diversity_score = 1.0 - unique_ratio  # Higher repetition = higher complexity
            scores.append(diversity_score * 0.2)  # 20% weight

        # Keyword density
        if self.keyword_extractor:
            keywords = self.keyword_extractor.extract(text, max_keywords=30)
            keyword_density = len(keywords) / max(len(text.split()), 1)
            scores.append(min(1.0, keyword_density * 100) * 0.2)  # 20% weight

        # Code vs documentation ratio
        code_blocks = text.count("```")
        doc_sections = text.count("#") + text.count("##")
        if code_blocks + doc_sections > 0:
            code_ratio = code_blocks / (code_blocks + doc_sections)
            scores.append(code_ratio * 0.2)  # 20% weight
        else:
            scores.append(0.1)  # Default low complexity

        # File count from metadata
        if metadata.get("file_count", 0) > 10:
            file_complexity = min(1.0, metadata["file_count"] / 50)
            scores.append(file_complexity * 0.2)  # 20% weight
        else:
            scores.append(0.1)

        # Calculate weighted average
        if scores:
            complexity = sum(scores)
        else:
            complexity = 0.5  # Default medium complexity

        return min(1.0, max(0.0, complexity))


class MetricsTracker:
    """Track metrics for tenet instillation.

    Tracks:
    - Instillation counts and frequencies
    - Token usage and increases
    - Strategy effectiveness
    - Session-specific metrics
    - Tenet performance
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.instillations: List[Dict[str, Any]] = []
        self.session_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_instillations": 0,
                "total_tenets": 0,
                "total_tokens": 0,
                "strategies_used": defaultdict(int),
                "complexity_scores": [],
            }
        )
        self.strategy_usage: Dict[str, int] = defaultdict(int)
        self.tenet_usage: Dict[str, int] = defaultdict(int)
        self.skip_reasons: Dict[str, int] = defaultdict(int)

    def record_instillation(
        self,
        tenet_count: int,
        token_increase: int,
        strategy: str,
        session: Optional[str] = None,
        complexity: float = 0.0,
        skip_reason: Optional[str] = None,
    ) -> None:
        """Record an instillation event.

        Args:
            tenet_count: Number of tenets instilled
            token_increase: Tokens added
            strategy: Strategy used
            session: Session identifier
            complexity: Context complexity score
            skip_reason: Reason if skipped
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "tenet_count": tenet_count,
            "token_increase": token_increase,
            "strategy": strategy,
            "session": session,
            "complexity": complexity,
            "skip_reason": skip_reason,
        }

        self.instillations.append(record)

        if skip_reason:
            self.skip_reasons[skip_reason] += 1
            return

        self.strategy_usage[strategy] += 1

        if session:
            metrics = self.session_metrics[session]
            metrics["total_instillations"] += 1
            metrics["total_tenets"] += tenet_count
            metrics["total_tokens"] += token_increase
            metrics["strategies_used"][strategy] += 1
            metrics["complexity_scores"].append(complexity)

    def record_tenet_usage(self, tenet_id: str) -> None:
        """Record that a tenet was used.

        Args:
            tenet_id: Tenet identifier
        """
        self.tenet_usage[tenet_id] += 1

    def get_metrics(self, session: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated metrics.

        Args:
            session: Optional session filter

        Returns:
            Dictionary of metrics
        """
        if session:
            records = [r for r in self.instillations if r["session"] == session]
        else:
            records = [r for r in self.instillations if not r.get("skip_reason")]

        if not records:
            return {"message": "No instillation records found"}

        total_tenets = sum(r["tenet_count"] for r in records)
        total_tokens = sum(r["token_increase"] for r in records)
        complexities = [r["complexity"] for r in records if r["complexity"] > 0]

        metrics = {
            "total_instillations": len(records),
            "total_tenets_instilled": total_tenets,
            "total_token_increase": total_tokens,
            "avg_tenets_per_context": total_tenets / len(records) if records else 0,
            "avg_token_increase": total_tokens / len(records) if records else 0,
            "avg_complexity": sum(complexities) / len(complexities) if complexities else 0,
            "strategy_distribution": dict(self.strategy_usage),
            "skip_distribution": dict(self.skip_reasons),
            "top_tenets": sorted(self.tenet_usage.items(), key=lambda x: x[1], reverse=True)[:10],
        }

        if session:
            metrics["session_specific"] = self.session_metrics.get(session, {})

        return metrics

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all tracked metrics for export."""
        return {
            "instillations": self.instillations,
            "session_metrics": dict(self.session_metrics),
            "strategy_usage": dict(self.strategy_usage),
            "tenet_usage": dict(self.tenet_usage),
            "skip_reasons": dict(self.skip_reasons),
            "summary": self.get_metrics(),
        }


class Instiller:
    """Main orchestrator for tenet instillation with smart injection.

    The Instiller manages the entire process of injecting tenets into context,
    including:
    - Tracking injection history per session
    - Analyzing context complexity
    - Determining optimal injection frequency
    - Selecting appropriate tenets
    - Applying injection strategies
    - Recording metrics and effectiveness

    It supports multiple injection modes:
    - Always: Inject into every context
    - Periodic: Inject every Nth distillation
    - Adaptive: Smart injection based on complexity and session
    - Manual: Only inject when explicitly requested
    """

    def __init__(self, config: TenetsConfig):
        """Initialize the Instiller.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = get_logger(__name__)

        # Core components
        self.manager = TenetManager(config)
        self.injector = TenetInjector(config.tenet.injection_config)
        self.complexity_analyzer = ComplexityAnalyzer(config)
        self.metrics_tracker = MetricsTracker()

        # Session tracking
        self.session_histories: Dict[str, InjectionHistory] = {}
        self._load_session_histories()

        # Cache for results
        self._cache: Dict[str, InstillationResult] = {}

        self.logger.info("Instiller initialized with smart injection capabilities")

    def inject_system_instruction(
        self,
        content: str,
        format: str = "markdown",
        session: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Inject system instruction (system prompt) according to config.

        Behavior:
        - If system instruction is disabled or empty, return unchanged.
        - If session provided and once-per-session is enabled, inject only on first distill.
        - If no session, inject on every distill.
        - Placement controlled by system_instruction_position.
        - Formatting controlled by system_instruction_format.

        Returns modified content and metadata about injection.
        """
        cfg = self.config.tenet
        meta: Dict[str, Any] = {
            "system_instruction_enabled": cfg.system_instruction_enabled,
            "system_instruction_injected": False,
        }

        if not cfg.system_instruction_enabled or not cfg.system_instruction:
            meta["reason"] = "disabled_or_empty"
            return content, meta

        # Session-aware check: only once per session
        if session:
            # Ensure a history exists for the session
            if session not in self.session_histories:
                self.session_histories[session] = InjectionHistory(session_id=session)
            hist = self.session_histories[session]
            if cfg.system_instruction_once_per_session and hist.system_instruction_injected:
                meta["reason"] = "already_injected_in_session"
                return content, meta

        instruction = cfg.system_instruction
        formatted_instr = self._format_system_instruction(
            instruction, cfg.system_instruction_format
        )

        # Determine position
        if cfg.system_instruction_position == "top":
            modified = formatted_instr + "\n" + content
            position = "top"
        elif cfg.system_instruction_position == "after_header":
            # After first markdown header or beginning if not found
            header_match = None
            try:
                import re

                # Match first Markdown header line
                header_match = re.search(r"^#+\s+.*$", content, flags=re.MULTILINE)
            except Exception:
                header_match = None
            if header_match:
                idx = header_match.end()
                modified = content[:idx] + "\n\n" + formatted_instr + content[idx:]
                position = "after_header"
            else:
                modified = formatted_instr + "\n" + content
                position = "top_fallback"
        elif cfg.system_instruction_position == "before_content":
            # Before first non-empty line
            lines = content.splitlines()
            i = 0
            while i < len(lines) and not lines[i].strip():
                i += 1
            prefix = "\n".join(lines[:i])
            suffix = "\n".join(lines[i:])
            modified = (
                prefix
                + ("\n" if prefix else "")
                + formatted_instr
                + ("\n" if suffix else "")
                + suffix
            )
            position = "before_content"
        else:
            modified = formatted_instr + "\n" + content
            position = "top_default"

        from tenets.utils.tokens import count_tokens

        meta.update(
            {
                "system_instruction_injected": True,
                "system_instruction_position": position,
                "token_increase": count_tokens(modified) - count_tokens(content),
            }
        )

        # Mark as injected for this session and persist history if applicable
        if session:
            hist = self.session_histories[session]
            hist.system_instruction_injected = True
            hist.updated_at = datetime.now()
            # Best-effort save
            try:
                self._save_session_histories()
            except Exception:
                pass
        return modified, meta

    def _format_system_instruction(self, instruction: str, fmt: str) -> str:
        """Format system instruction according to selected format."""
        if fmt == "markdown":
            return f"<!-- System Instruction -->\n> [!NOTE]\n> {instruction.strip()}\n"
        if fmt == "xml":
            return f"<system-instruction>{instruction.strip()}</system-instruction>\n"
        if fmt == "comment":
            # Use a neutral comment style; markdown HTML comment works across formats
            return f"<!-- {instruction.strip()} -->\n"
        return instruction.strip() + "\n"

    def _load_session_histories(self) -> None:
        """Load session histories from storage."""
        history_file = self.config.cache.directory / "injection_histories.json"

        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                for session_id, history_data in data.items():
                    history = InjectionHistory(
                        session_id=session_id,
                        total_distills=history_data.get("total_distills", 0),
                        total_injections=history_data.get("total_injections", 0),
                        complexity_scores=history_data.get("complexity_scores", []),
                        injected_tenets=set(history_data.get("injected_tenets", [])),
                        reinforcement_count=history_data.get("reinforcement_count", 0),
                        system_instruction_injected=history_data.get(
                            "system_instruction_injected", False
                        ),
                    )

                    if history_data.get("last_injection"):
                        history.last_injection = datetime.fromisoformat(
                            history_data["last_injection"]
                        )

                    # Restore last_injection_index if present
                    if "last_injection_index" in history_data:
                        history.last_injection_index = history_data["last_injection_index"]

                    self.session_histories[session_id] = history

                self.logger.info(f"Loaded {len(self.session_histories)} session histories")
            except Exception as e:
                self.logger.warning(f"Failed to load session histories: {e}")

    def _save_session_histories(self) -> None:
        """Save session histories to storage."""
        history_file = self.config.cache.directory / "injection_histories.json"

        try:
            # Ensure cache directory exists
            history_file.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for session_id, history in self.session_histories.items():
                data[session_id] = {
                    "total_distills": history.total_distills,
                    "total_injections": history.total_injections,
                    "last_injection": (
                        history.last_injection.isoformat() if history.last_injection else None
                    ),
                    "last_injection_index": history.last_injection_index,
                    "complexity_scores": history.complexity_scores,
                    "injected_tenets": list(history.injected_tenets),
                    "reinforcement_count": history.reinforcement_count,
                    "system_instruction_injected": history.system_instruction_injected,
                }

            with open(history_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save session histories: {e}")

    def instill(
        self,
        context: Union[str, ContextResult],
        session: Optional[str] = None,
        force: bool = False,
        strategy: Optional[str] = None,
        max_tenets: Optional[int] = None,
        check_frequency: bool = True,
    ) -> Union[str, ContextResult]:
        """Instill tenets into context with smart injection.

        Args:
            context: Context to inject tenets into
            session: Session identifier for tracking
            force: Force injection regardless of frequency settings
            strategy: Override injection strategy
            max_tenets: Override maximum tenets
            check_frequency: Whether to check injection frequency

        Returns:
            Modified context with tenets injected (if applicable)
        """
        start_time = time.time()

        # Track session if provided
        if session:
            if session not in self.session_histories:
                self.session_histories[session] = InjectionHistory(session_id=session)
            history = self.session_histories[session]
            history.total_distills += 1
        else:
            history = None

        # Extract text and format
        if isinstance(context, ContextResult):
            text = context.context
            format_type = context.format
            is_context_result = True
        else:
            text = context
            format_type = "markdown"
            is_context_result = False

        # Analyze complexity
        complexity = self.complexity_analyzer.analyze(context)
        self.logger.debug(f"Context complexity: {complexity:.2f}")

        # Check if we should inject
        should_inject = force
        skip_reason = None

        if not force and check_frequency and history:
            should_inject, reason = history.should_inject(
                frequency=self.config.tenet.injection_frequency,
                interval=self.config.tenet.injection_interval,
                complexity=complexity,
                complexity_threshold=self.config.tenet.session_complexity_threshold,
                min_session_length=self.config.tenet.min_session_length,
            )

            if not should_inject:
                skip_reason = reason
                self.logger.debug(f"Skipping injection: {reason}")

        # Record metrics even if skipping
        if not should_inject:
            self.metrics_tracker.record_instillation(
                tenet_count=0,
                token_increase=0,
                strategy="skipped",
                session=session,
                complexity=complexity,
                skip_reason=skip_reason,
            )

            # Save histories
            self._save_session_histories()

            return context  # Return unchanged

        # Get tenets for injection
        tenets = self._get_tenets_for_instillation(
            session=session,
            force=force,
            content_length=len(text),
            max_tenets=max_tenets or self.config.tenet.max_per_context,
            history=history,
            complexity=complexity,
        )

        if not tenets:
            self.logger.info("No tenets available for instillation")
            return context

        # Determine injection strategy
        if not strategy:
            strategy = self._determine_injection_strategy(
                content_length=len(text),
                tenet_count=len(tenets),
                format_type=format_type,
                complexity=complexity,
            )

        self.logger.info(
            f"Instilling {len(tenets)} tenets using {strategy} strategy"
            f"{f' for session {session}' if session else ''}"
        )

        # Inject tenets
        modified_text, injection_metadata = self.injector.inject_tenets(
            content=text,
            tenets=tenets,
            format=format_type,
            strategy=strategy,
        )

        # Update tenet metrics
        for tenet in tenets:
            tenet.record_injection()
            self.manager._save_tenet(tenet)
            self.metrics_tracker.record_tenet_usage(tenet.id)

        # Record injection in history
        if history:
            history.record_injection(tenets, complexity)

            # Check for reinforcement
            if (
                self.config.tenet.reinforcement
                and history.total_injections % self.config.tenet.reinforcement_interval == 0
            ):
                history.reinforcement_count += 1
                self.logger.info(f"Reinforcement injection #{history.reinforcement_count}")

        # Create result
        result = InstillationResult(
            tenets_instilled=tenets,
            injection_positions=injection_metadata.get("injections", []),
            token_increase=injection_metadata.get("token_increase", 0),
            strategy_used=strategy,
            session=session,
            complexity_score=complexity,
            metrics={
                "processing_time": time.time() - start_time,
                "complexity": complexity,
                "injection_metadata": injection_metadata,
            },
        )

        # Cache result
        cache_key = f"{session or 'global'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._cache[cache_key] = result

        # Record metrics
        self.metrics_tracker.record_instillation(
            tenet_count=len(tenets),
            token_increase=injection_metadata.get("token_increase", 0),
            strategy=strategy,
            session=session,
            complexity=complexity,
        )

        # Save histories
        self._save_session_histories()

        # Return modified context
        if is_context_result:
            modified_context = ContextResult(
                files=context.files,
                context=modified_text,
                format=context.format,
                metadata={
                    **context.metadata,
                    "tenet_instillation": result.to_dict(),
                    "tenets_injected": [t.id for t in tenets],
                    "injection_complexity": complexity,
                },
            )
            return modified_context
        else:
            return modified_text

    def _get_tenets_for_instillation(
        self,
        session: Optional[str],
        force: bool,
        content_length: int,
        max_tenets: int,
        history: Optional[InjectionHistory],
        complexity: float,
    ) -> List[Tenet]:
        """Get tenets to instill based on context and history.

        Args:
            session: Session identifier
            force: Force all tenets
            content_length: Length of content
            max_tenets: Maximum tenets to return
            history: Injection history for session
            complexity: Context complexity score

        Returns:
            List of tenets to instill
        """
        # Calculate optimal count
        optimal_count = self.injector.calculate_optimal_injection_count(
            content_length=content_length,
            available_tenets=max_tenets,
            max_token_increase=1000,
        )

        # Adjust based on complexity
        if complexity > 0.8:
            # High complexity - reduce tenets
            optimal_count = max(1, optimal_count - 1)
        elif complexity < 0.3:
            # Low complexity - can handle more tenets
            optimal_count = min(max_tenets, optimal_count + 1)

        # Get tenets based on mode
        if force:
            # Get all active tenets
            all_tenets = []
            for tenet_id, tenet in self.manager._tenet_cache.items():
                if tenet.status != TenetStatus.ARCHIVED:
                    if (
                        not session
                        or session in tenet.session_bindings
                        or not tenet.session_bindings
                    ):
                        all_tenets.append(tenet)

            # Sort by priority
            all_tenets.sort(key=lambda t: t.priority.weight, reverse=True)
            return all_tenets[:optimal_count]

        else:
            # Get pending tenets
            pending = self.manager.get_pending_tenets(session)

            # Filter out recently injected tenets if we have history
            if history and history.injected_tenets:
                # Apply decay - tenets become eligible again over time
                decay_threshold = max(3, int(10 * (1 - self.config.tenet.decay_rate)))

                if history.total_distills - history.last_injection_index < decay_threshold:
                    # Filter out recently injected
                    pending = [t for t in pending if t.id not in history.injected_tenets]

            # Sort by priority and metrics
            def tenet_score(t: Tenet) -> float:
                """Calculate tenet importance score."""
                priority_score = t.priority.weight / 10.0

                # Boost if never injected
                if history and t.id not in history.injected_tenets:
                    priority_score *= 1.5

                # Reduce if frequently injected
                injection_penalty = min(0.5, t.metrics.injection_count * 0.05)

                # Boost critical tenets for reinforcement
                if t.priority == Priority.CRITICAL and history:
                    if history.total_injections % self.config.tenet.reinforcement_interval == 0:
                        priority_score *= 2.0

                return priority_score - injection_penalty

            pending.sort(key=tenet_score, reverse=True)

            return pending[:optimal_count]

    def _determine_injection_strategy(
        self,
        content_length: int,
        tenet_count: int,
        format_type: str,
        complexity: float,
    ) -> str:
        """Determine optimal injection strategy.

        Args:
            content_length: Length of content
            tenet_count: Number of tenets to inject
            format_type: Format of content
            complexity: Complexity score

        Returns:
            Strategy name
        """
        # Short content - inject at top
        if content_length < 5000:
            return "top"

        # XML format - use strategic for structure
        if format_type == "xml":
            return "strategic"

        # High complexity - use distributed to spread cognitive load
        if complexity > 0.7:
            return "distributed"

        # Many tenets - distribute them
        if tenet_count > 5:
            return "distributed"

        # Long content with few tenets - strategic placement
        if content_length > 20000 and tenet_count <= 3:
            return "strategic"

        # Default to balanced approach
        return "strategic"

    def get_session_stats(self, session: str) -> Dict[str, Any]:
        """Get statistics for a specific session.

        Args:
            session: Session identifier

        Returns:
            Dictionary of session statistics
        """
        if session not in self.session_histories:
            return {"error": f"No history for session: {session}"}

        history = self.session_histories[session]
        stats = history.get_stats()

        # Add metrics from tracker
        session_metrics = self.metrics_tracker.session_metrics.get(session, {})
        stats.update(session_metrics)

        return stats

    def get_all_session_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all sessions.

        Returns:
            Dictionary mapping session IDs to stats
        """
        all_stats = {}

        for session_id, history in self.session_histories.items():
            all_stats[session_id] = history.get_stats()

        return all_stats

    def analyze_effectiveness(self, session: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the effectiveness of tenet instillation.

        Args:
            session: Optional session to analyze

        Returns:
            Dictionary with analysis results and recommendations
        """
        # Get tenet effectiveness from manager
        tenet_analysis = self.manager.analyze_tenet_effectiveness()

        # Get instillation metrics
        metrics = self.metrics_tracker.get_metrics(session)

        # Session-specific analysis
        session_analysis = {}
        if session and session in self.session_histories:
            session_analysis = self.get_session_stats(session)

        # Generate recommendations
        recommendations = []

        # Check injection frequency
        if metrics.get("total_instillations", 0) > 0:
            avg_complexity = metrics.get("avg_complexity", 0.5)

            if avg_complexity > 0.7:
                recommendations.append(
                    "High average complexity detected. Consider reducing injection frequency "
                    "or using simpler tenets."
                )
            elif avg_complexity < 0.3:
                recommendations.append(
                    "Low average complexity. You could increase injection frequency "
                    "for better reinforcement."
                )

        # Check skip reasons
        skip_dist = metrics.get("skip_distribution", {})
        if skip_dist:
            top_skip = max(skip_dist.items(), key=lambda x: x[1])
            if "session_too_short" in top_skip[0]:
                recommendations.append(
                    f"Many skips due to short sessions. Consider reducing min_session_length "
                    f"(currently {self.config.tenet.min_session_length})."
                )

        # Check tenet usage
        if tenet_analysis.get("need_reinforcement"):
            recommendations.append(
                f"Tenets needing reinforcement: {', '.join(tenet_analysis['need_reinforcement'][:3])}"
            )

        return {
            "tenet_effectiveness": tenet_analysis,
            "instillation_metrics": metrics,
            "session_analysis": session_analysis,
            "recommendations": recommendations,
            "configuration": {
                "injection_frequency": self.config.tenet.injection_frequency,
                "injection_interval": self.config.tenet.injection_interval,
                "complexity_threshold": self.config.tenet.session_complexity_threshold,
                "min_session_length": self.config.tenet.min_session_length,
            },
        }

    def export_instillation_history(
        self,
        output_path: Path,
        format: str = "json",
        session: Optional[str] = None,
    ) -> None:
        """Export instillation history to file.

        Args:
            output_path: Path to output file
            format: Export format (json or csv)
            session: Optional session filter

        Raises:
            ValueError: If format is not supported
        """
        if format == "json":
            # Export as JSON
            data = {
                "exported_at": datetime.now().isoformat(),
                "configuration": {
                    "injection_frequency": self.config.tenet.injection_frequency,
                    "injection_interval": self.config.tenet.injection_interval,
                },
                "metrics": self.metrics_tracker.get_all_metrics(),
                "session_histories": {},
                "cached_results": {},
            }

            # Add session histories
            for sid, history in self.session_histories.items():
                if not session or sid == session:
                    data["session_histories"][sid] = history.get_stats()

            # Add cached results
            for key, result in self._cache.items():
                if not session or result.session == session:
                    data["cached_results"][key] = result.to_dict()

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            # Export as CSV
            import csv

            rows = []
            for record in self.metrics_tracker.instillations:
                if not session or record.get("session") == session:
                    rows.append(
                        {
                            "Timestamp": record["timestamp"],
                            "Session": record.get("session", ""),
                            "Tenets": record["tenet_count"],
                            "Tokens": record["token_increase"],
                            "Strategy": record["strategy"],
                            "Complexity": f"{record.get('complexity', 0):.2f}",
                            "Skip Reason": record.get("skip_reason", ""),
                        }
                    )

            if rows:
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                # Create empty file with headers
                with open(output_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "Timestamp",
                            "Session",
                            "Tenets",
                            "Tokens",
                            "Strategy",
                            "Complexity",
                            "Skip Reason",
                        ]
                    )

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Exported instillation history to {output_path}")

    def reset_session_history(self, session: str) -> bool:
        """Reset injection history for a session.

        Args:
            session: Session identifier

        Returns:
            True if reset, False if session not found
        """
        if session in self.session_histories:
            self.session_histories[session] = InjectionHistory(session_id=session)
            self._save_session_histories()
            self.logger.info(f"Reset injection history for session: {session}")
            return True
        return False

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._cache.clear()
        self.logger.info("Cleared instillation results cache")
