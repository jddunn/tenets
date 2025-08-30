"""Tests for main Instiller orchestrator with smart injection features.

This test suite covers:
- Session-aware injection tracking
- Complexity analysis
- Adaptive injection frequency
- Metrics tracking
- History persistence
- Export functionality
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.instiller.instiller import (
    ComplexityAnalyzer,
    InjectionHistory,
    InstillationResult,
    Instiller,
    MetricsTracker,
)
from tenets.models.context import ContextResult
from tenets.models.tenet import Priority, Tenet


@pytest.fixture
def config():
    """Create test configuration with smart injection settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TenetsConfig()
        config.cache.directory = Path(tmpdir)
        config.tenet.max_per_context = 5
        config.tenet.injection_frequency = "adaptive"
        config.tenet.injection_interval = 3
        config.tenet.session_complexity_threshold = 0.7
        config.tenet.min_session_length = 5
        config.tenet.adaptive_injection = True
        config.tenet.track_injection_history = True
        config.tenet.decay_rate = 0.1
        config.tenet.reinforcement_interval = 10
        yield config


@pytest.fixture
def instiller(config):
    """Create Instiller instance."""
    return Instiller(config)


@pytest.fixture
def sample_tenets():
    """Create sample tenets."""
    return [
        Tenet(content="Use async/await", priority=Priority.HIGH),
        Tenet(content="Add error handling", priority=Priority.CRITICAL),
        Tenet(content="Document functions", priority=Priority.MEDIUM),
        Tenet(content="Optimize queries", priority=Priority.LOW),
    ]


@pytest.fixture
def mock_manager(sample_tenets):
    """Create mock TenetManager."""
    manager = Mock()
    manager._tenet_cache = {t.id: t for t in sample_tenets}
    manager.get_pending_tenets.return_value = sample_tenets
    manager._save_tenet = Mock()
    manager.analyze_tenet_effectiveness.return_value = {
        "total_tenets": len(sample_tenets),
        "need_reinforcement": [],
    }
    return manager


@pytest.fixture
def mock_injector():
    """Create mock TenetInjector."""
    injector = Mock()
    injector.inject_tenets.return_value = (
        "Modified content with tenets",
        {
            "injected_count": 3,
            "token_increase": 150,
            "injections": [{"position": 100}, {"position": 500}, {"position": 900}],
            "reinforcement_added": False,
        },
    )
    injector.calculate_optimal_injection_count.return_value = 3
    return injector


@pytest.fixture
def sample_context():
    """Create sample context for testing."""
    return ContextResult(
        files=["test.py", "utils.py"],
        context="# Test Context\n\nThis is test content for injection.",
        format="markdown",
        metadata={"total_tokens": 500, "file_count": 2},
    )


class TestInjectionHistory:
    """Test suite for InjectionHistory tracking."""

    def test_initialization(self):
        """Test injection history initialization."""
        history = InjectionHistory(session_id="test-session")

        assert history.session_id == "test-session"
        assert history.total_distills == 0
        assert history.total_injections == 0
        assert history.last_injection is None
        assert history.complexity_scores == []
        assert len(history.injected_tenets) == 0

    def test_should_inject_always_mode(self):
        """Test injection decision in always mode."""
        history = InjectionHistory(session_id="test")

        should, reason = history.should_inject(
            frequency="always",
            interval=3,
            complexity=0.5,
            complexity_threshold=0.7,
            min_session_length=5,
        )

        assert should is True
        assert reason == "always_mode"

    def test_should_inject_manual_mode(self):
        """Test injection decision in manual mode."""
        history = InjectionHistory(session_id="test")

        should, reason = history.should_inject(
            frequency="manual",
            interval=3,
            complexity=0.9,
            complexity_threshold=0.7,
            min_session_length=5,
        )

        assert should is False
        assert reason == "manual_mode"

    def test_should_inject_periodic_mode(self):
        """Test injection decision in periodic mode."""
        history = InjectionHistory(session_id="test")

        # Not at interval
        history.total_distills = 2
        should, reason = history.should_inject(
            frequency="periodic",
            interval=3,
            complexity=0.5,
            complexity_threshold=0.7,
            min_session_length=0,
        )
        assert should is False
        assert "not_at_interval" in reason

        # At interval
        history.total_distills = 3
        should, reason = history.should_inject(
            frequency="periodic",
            interval=3,
            complexity=0.5,
            complexity_threshold=0.7,
            min_session_length=0,
        )
        assert should is True
        assert "periodic_interval" in reason

    def test_should_inject_adaptive_mode(self):
        """Test injection decision in adaptive mode."""
        history = InjectionHistory(session_id="test")

        # Session too short
        history.total_distills = 2
        should, reason = history.should_inject(
            frequency="adaptive",
            interval=3,
            complexity=0.8,
            complexity_threshold=0.7,
            min_session_length=5,
        )
        assert should is False
        assert "session_too_short" in reason

        # First injection after minimum length
        history.total_distills = 5
        history.total_injections = 0
        should, reason = history.should_inject(
            frequency="adaptive",
            interval=3,
            complexity=0.5,
            complexity_threshold=0.7,
            min_session_length=5,
        )
        assert should is True
        assert "first_adaptive_injection" in reason

        # High complexity trigger
        history.total_injections = 1
        history.last_injection = datetime.now() - timedelta(minutes=10)
        should, reason = history.should_inject(
            frequency="adaptive",
            interval=3,
            complexity=0.8,
            complexity_threshold=0.7,
            min_session_length=5,
        )
        assert should is True
        assert "high_complexity" in reason

        # Recently injected
        history.last_injection = datetime.now() - timedelta(minutes=2)
        should, reason = history.should_inject(
            frequency="adaptive",
            interval=3,
            complexity=0.8,
            complexity_threshold=0.7,
            min_session_length=5,
        )
        assert should is False
        assert "injected_recently" in reason

    def test_record_injection(self, sample_tenets):
        """Test recording an injection."""
        history = InjectionHistory(session_id="test")
        history.total_distills = 10

        history.record_injection(sample_tenets[:2], complexity=0.75)

        assert history.total_injections == 1
        assert history.last_injection is not None
        assert history.last_injection_index == 10
        assert history.complexity_scores == [0.75]
        assert len(history.injected_tenets) == 2
        assert sample_tenets[0].id in history.injected_tenets
        assert sample_tenets[1].id in history.injected_tenets

    def test_get_stats(self, sample_tenets):
        """Test getting history statistics."""
        history = InjectionHistory(session_id="test")
        history.total_distills = 20
        history.total_injections = 4
        history.complexity_scores = [0.5, 0.6, 0.7, 0.8]
        history.injected_tenets = {t.id for t in sample_tenets}
        history.reinforcement_count = 2

        stats = history.get_stats()

        assert stats["session_id"] == "test"
        assert stats["total_distills"] == 20
        assert stats["total_injections"] == 4
        assert stats["injection_rate"] == 0.2  # 4/20
        assert stats["average_complexity"] == 0.65  # avg of [0.5, 0.6, 0.7, 0.8]
        assert stats["unique_tenets_injected"] == len(sample_tenets)
        assert stats["reinforcement_count"] == 2


class TestComplexityAnalyzer:
    """Test suite for ComplexityAnalyzer."""

    def test_initialization(self, config):
        """Test complexity analyzer initialization."""
        analyzer = ComplexityAnalyzer(config)

        assert analyzer.config == config
        # NLP components should be initialized if available

    def test_analyze_string_content(self, config):
        """Test analyzing string content complexity."""
        analyzer = ComplexityAnalyzer(config)

        # Short content
        complexity = analyzer.analyze("Short text")
        assert 0 <= complexity <= 1
        assert complexity < 0.5  # Should be low for short text

        # Long content
        long_text = "Long text " * 1000
        complexity = analyzer.analyze(long_text)
        assert complexity > 0.2  # Should be higher for long text

    def test_analyze_context_result(self, config, sample_context):
        """Test analyzing ContextResult complexity."""
        analyzer = ComplexityAnalyzer(config)

        complexity = analyzer.analyze(sample_context)

        assert 0 <= complexity <= 1
        # Should consider metadata
        assert complexity > 0  # Has files and content

    def test_analyze_with_code_blocks(self, config):
        """Test complexity with code blocks."""
        analyzer = ComplexityAnalyzer(config)

        content = """
        # Documentation

        Some text here.

        ```python
        def function():
            return 42
        ```

        More text.

        ```javascript
        console.log('test');
        ```
        """

        complexity = analyzer.analyze(content)
        assert complexity > 0.2  # Code blocks increase complexity

    @patch("tenets.core.instiller.instiller.KeywordExtractor")
    def test_analyze_with_nlp_components(self, mock_extractor, config):
        """Test complexity analysis with NLP components."""
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract.return_value = ["keyword1", "keyword2", "keyword3"]
        mock_extractor.return_value = mock_extractor_instance

        analyzer = ComplexityAnalyzer(config)
        analyzer.keyword_extractor = mock_extractor_instance

        complexity = analyzer.analyze("Test content with keywords")

        assert mock_extractor_instance.extract.called
        assert 0 <= complexity <= 1


class TestMetricsTracker:
    """Test suite for MetricsTracker."""

    def test_initialization(self):
        """Test metrics tracker initialization."""
        tracker = MetricsTracker()

        assert tracker.instillations == []
        assert len(tracker.session_metrics) == 0
        assert len(tracker.strategy_usage) == 0
        assert len(tracker.tenet_usage) == 0
        assert len(tracker.skip_reasons) == 0

    def test_record_instillation(self):
        """Test recording an instillation."""
        tracker = MetricsTracker()

        tracker.record_instillation(
            tenet_count=3,
            token_increase=100,
            strategy="strategic",
            session="test-session",
            complexity=0.75,
        )

        assert len(tracker.instillations) == 1
        assert tracker.instillations[0]["tenet_count"] == 3
        assert tracker.instillations[0]["complexity"] == 0.75
        assert tracker.strategy_usage["strategic"] == 1
        assert "test-session" in tracker.session_metrics

    def test_record_skip(self):
        """Test recording a skipped injection."""
        tracker = MetricsTracker()

        tracker.record_instillation(
            tenet_count=0,
            token_increase=0,
            strategy="skipped",
            session="test",
            complexity=0.5,
            skip_reason="session_too_short",
        )

        assert tracker.skip_reasons["session_too_short"] == 1
        assert tracker.strategy_usage.get("skipped", 0) == 0  # Shouldn't count as strategy

    def test_record_tenet_usage(self):
        """Test recording tenet usage."""
        tracker = MetricsTracker()

        tracker.record_tenet_usage("tenet-123")
        tracker.record_tenet_usage("tenet-123")
        tracker.record_tenet_usage("tenet-456")

        assert tracker.tenet_usage["tenet-123"] == 2
        assert tracker.tenet_usage["tenet-456"] == 1

    def test_get_metrics(self):
        """Test getting aggregated metrics."""
        tracker = MetricsTracker()

        # Record some data
        tracker.record_instillation(3, 100, "strategic", "session1", 0.6)
        tracker.record_instillation(2, 80, "top", "session1", 0.7)
        tracker.record_instillation(4, 120, "distributed", "session2", 0.8)
        tracker.record_instillation(0, 0, "skipped", "session2", 0.5, "manual_mode")

        metrics = tracker.get_metrics()

        assert metrics["total_instillations"] == 3  # Excludes skipped
        assert metrics["total_tenets_instilled"] == 9  # 3 + 2 + 4
        assert metrics["total_token_increase"] == 300  # 100 + 80 + 120
        assert metrics["avg_tenets_per_context"] == 3  # 9/3
        assert metrics["avg_complexity"] == 0.7  # (0.6 + 0.7 + 0.8) / 3
        assert "strategic" in metrics["strategy_distribution"]
        assert "manual_mode" in metrics["skip_distribution"]

    def test_get_metrics_by_session(self):
        """Test getting session-specific metrics."""
        tracker = MetricsTracker()

        tracker.record_instillation(3, 100, "strategic", "session1", 0.6)
        tracker.record_instillation(2, 80, "top", "session2", 0.7)

        metrics = tracker.get_metrics(session="session1")

        assert metrics["total_instillations"] == 1
        assert metrics["total_tenets_instilled"] == 3
        assert metrics["total_token_increase"] == 100


class TestInstiller:
    """Test suite for main Instiller class."""

    def test_initialization(self, config):
        """Test Instiller initialization."""
        instiller = Instiller(config)

        assert instiller.config == config
        assert instiller.manager is not None
        assert instiller.injector is not None
        assert instiller.complexity_analyzer is not None
        assert instiller.metrics_tracker is not None
        assert isinstance(instiller.session_histories, dict)

    def test_load_session_histories(self, config):
        """Test loading session histories from disk."""
        # Create a history file
        history_data = {
            "test-session": {
                "total_distills": 10,
                "total_injections": 3,
                "last_injection": datetime.now().isoformat(),
                "complexity_scores": [0.5, 0.6, 0.7],
                "injected_tenets": ["tenet1", "tenet2"],
                "reinforcement_count": 1,
            }
        }

        history_file = config.cache.directory / "injection_histories.json"
        with open(history_file, "w") as f:
            json.dump(history_data, f)

        # Create new instiller to load histories
        instiller = Instiller(config)

        assert "test-session" in instiller.session_histories
        assert instiller.session_histories["test-session"].total_distills == 10
        assert instiller.session_histories["test-session"].total_injections == 3

    def test_save_session_histories(self, instiller):
        """Test saving session histories to disk."""
        # Add a session history
        history = InjectionHistory(session_id="save-test")
        history.total_distills = 5
        history.total_injections = 2
        instiller.session_histories["save-test"] = history

        # Save
        instiller._save_session_histories()

        # Check file exists and contains data
        history_file = instiller.config.cache.directory / "injection_histories.json"
        assert history_file.exists()

        with open(history_file) as f:
            data = json.load(f)
            assert "save-test" in data
            assert data["save-test"]["total_distills"] == 5

    def test_instill_string_always_mode(self, instiller, sample_tenets):
        """Test instilling with always mode."""
        instiller.config.tenet.injection_frequency = "always"

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                mock_inject.return_value = ("Modified", {"injected_count": 2, "token_increase": 50})

                result = instiller.instill("Test content", session="test")

                assert result == "Modified"
                mock_inject.assert_called_once()
                assert "test" in instiller.session_histories

    def test_instill_string_manual_mode(self, instiller):
        """Test instilling with manual mode (should skip)."""
        instiller.config.tenet.injection_frequency = "manual"

        result = instiller.instill("Test content", session="test", force=False)

        assert result == "Test content"  # Unchanged
        assert instiller.metrics_tracker.skip_reasons["manual_mode"] == 1

    def test_instill_periodic_mode(self, instiller, sample_tenets):
        """Test periodic injection."""
        instiller.config.tenet.injection_frequency = "periodic"
        instiller.config.tenet.injection_interval = 3
        instiller.config.tenet.min_session_length = 0

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                mock_inject.return_value = ("Modified", {"injected_count": 2, "token_increase": 50})

                # First two calls should skip
                result1 = instiller.instill("Content 1", session="periodic-test")
                result2 = instiller.instill("Content 2", session="periodic-test")

                # Third call should inject
                result3 = instiller.instill("Content 3", session="periodic-test")

                assert result1 == "Content 1"  # Skipped
                assert result2 == "Content 2"  # Skipped
                assert mock_inject.called  # Should be called on 3rd

    def test_instill_adaptive_mode(self, instiller, sample_tenets):
        """Test adaptive injection based on complexity."""
        instiller.config.tenet.injection_frequency = "adaptive"
        instiller.config.tenet.session_complexity_threshold = 0.7
        instiller.config.tenet.min_session_length = 2

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            with patch.object(instiller.complexity_analyzer, "analyze") as mock_analyze:
                with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                    mock_inject.return_value = (
                        "Modified",
                        {"injected_count": 2, "token_increase": 50},
                    )

                    # Low complexity - should skip
                    mock_analyze.return_value = 0.3
                    result1 = instiller.instill("Simple", session="adaptive-test")

                    # Still building session
                    result2 = instiller.instill("Simple 2", session="adaptive-test")

                    # High complexity after min length - should inject
                    mock_analyze.return_value = 0.8
                    result3 = instiller.instill("Complex content", session="adaptive-test")

                    assert result1 == "Simple"  # Too short
                    assert result2 == "Simple 2"  # First injection
                    assert mock_inject.called  # High complexity triggers

    def test_instill_context_result(self, instiller, sample_context, sample_tenets):
        """Test instilling into ContextResult."""
        instiller.config.tenet.injection_frequency = "always"

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                mock_inject.return_value = ("Modified", {"injected_count": 2, "token_increase": 50})

                result = instiller.instill(sample_context)

                assert isinstance(result, ContextResult)
                assert result.context == "Modified"
                assert "tenet_instillation" in result.metadata
                assert "injection_complexity" in result.metadata

    def test_instill_force_override(self, instiller, sample_tenets):
        """Test force injection overrides frequency settings."""
        instiller.config.tenet.injection_frequency = "manual"

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                mock_inject.return_value = ("Modified", {"injected_count": 2, "token_increase": 50})

                result = instiller.instill("Test", session="test", force=True)

                assert result == "Modified"
                mock_inject.assert_called_once()

    def test_get_tenets_for_instillation(self, instiller, sample_tenets):
        """Test getting tenets with smart selection."""
        history = InjectionHistory(session_id="test")
        history.total_distills = 10
        history.injected_tenets = {sample_tenets[0].id}  # First tenet already injected

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            tenets = instiller._get_tenets_for_instillation(
                session="test",
                force=False,
                content_length=10000,
                max_tenets=3,
                history=history,
                complexity=0.5,
            )

            # Should prioritize uninjected tenets
            assert sample_tenets[0] not in tenets or tenets.index(sample_tenets[0]) > 0

    def test_get_tenets_with_decay(self, instiller, sample_tenets):
        """Test tenet decay over time."""
        history = InjectionHistory(session_id="test")
        history.total_distills = 20
        history.last_injection_index = 5  # Long time ago
        history.injected_tenets = {t.id for t in sample_tenets[:2]}

        instiller.config.tenet.decay_rate = 0.5  # Fast decay

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            tenets = instiller._get_tenets_for_instillation(
                session="test",
                force=False,
                content_length=10000,
                max_tenets=4,
                history=history,
                complexity=0.5,
            )

            # Old injected tenets should be eligible again
            assert len(tenets) > 0

    def test_determine_injection_strategy(self, instiller):
        """Test strategy determination."""
        # Short content
        strategy = instiller._determine_injection_strategy(
            content_length=1000, tenet_count=2, format_type="markdown", complexity=0.5
        )
        assert strategy == "top"

        # High complexity
        strategy = instiller._determine_injection_strategy(
            content_length=10000, tenet_count=3, format_type="markdown", complexity=0.8
        )
        assert strategy == "distributed"

        # XML format
        strategy = instiller._determine_injection_strategy(
            content_length=10000, tenet_count=3, format_type="xml", complexity=0.5
        )
        assert strategy == "strategic"

        # Many tenets
        strategy = instiller._determine_injection_strategy(
            content_length=15000, tenet_count=8, format_type="markdown", complexity=0.5
        )
        assert strategy == "distributed"

    def test_get_session_stats(self, instiller):
        """Test getting session statistics."""
        history = InjectionHistory(session_id="stats-test")
        history.total_distills = 15
        history.total_injections = 5
        history.complexity_scores = [0.5, 0.6, 0.7]
        instiller.session_histories["stats-test"] = history

        stats = instiller.get_session_stats("stats-test")

        assert stats["total_distills"] == 15
        assert stats["total_injections"] == 5
        assert stats["injection_rate"] == 5 / 15

        # Non-existent session
        stats = instiller.get_session_stats("non-existent")
        assert "error" in stats

    def test_get_all_session_stats(self, instiller):
        """Test getting all session statistics."""
        # Add multiple sessions
        for i in range(3):
            history = InjectionHistory(session_id=f"session-{i}")
            history.total_distills = (i + 1) * 5
            instiller.session_histories[f"session-{i}"] = history

        all_stats = instiller.get_all_session_stats()

        assert len(all_stats) == 3
        assert "session-0" in all_stats
        assert "session-1" in all_stats
        assert "session-2" in all_stats

    def test_analyze_effectiveness(self, instiller):
        """Test effectiveness analysis."""
        # Add some metrics
        instiller.metrics_tracker.record_instillation(3, 100, "strategic", "test", 0.8)
        instiller.metrics_tracker.record_instillation(2, 80, "top", "test", 0.6)

        with patch.object(instiller.manager, "analyze_tenet_effectiveness") as mock_analyze:
            mock_analyze.return_value = {
                "total_tenets": 5,
                "need_reinforcement": ["tenet1", "tenet2"],
            }

            analysis = instiller.analyze_effectiveness(session="test")

            assert "tenet_effectiveness" in analysis
            assert "instillation_metrics" in analysis
            assert "recommendations" in analysis
            assert "configuration" in analysis

            # Should have recommendations for high complexity
            assert len(analysis["recommendations"]) > 0

    def test_export_instillation_history_json(self, instiller, tmp_path):
        """Test exporting history as JSON."""
        # Add some data
        history = InjectionHistory(session_id="export-test")
        history.total_distills = 10
        instiller.session_histories["export-test"] = history

        instiller.metrics_tracker.record_instillation(3, 100, "strategic", "export-test", 0.7)

        output_file = tmp_path / "export.json"
        instiller.export_instillation_history(output_file, format="json")

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert "exported_at" in data
            assert "configuration" in data
            assert "metrics" in data
            assert "session_histories" in data
            assert "export-test" in data["session_histories"]

    def test_export_instillation_history_csv(self, instiller, tmp_path):
        """Test exporting history as CSV."""
        instiller.metrics_tracker.record_instillation(3, 100, "strategic", "test", 0.7)
        instiller.metrics_tracker.record_instillation(2, 80, "top", "test", 0.6)

        output_file = tmp_path / "export.csv"
        instiller.export_instillation_history(output_file, format="csv", session="test")

        assert output_file.exists()

        content = output_file.read_text()
        assert "Timestamp" in content
        assert "Session" in content
        assert "test" in content

    def test_export_invalid_format(self, instiller, tmp_path):
        """Test export with invalid format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            instiller.export_instillation_history(tmp_path / "export.txt", format="invalid")

    def test_reset_session_history(self, instiller):
        """Test resetting session history."""
        # Add a session
        history = InjectionHistory(session_id="reset-test")
        history.total_distills = 20
        history.total_injections = 5
        instiller.session_histories["reset-test"] = history

        # Reset it
        result = instiller.reset_session_history("reset-test")

        assert result is True
        assert instiller.session_histories["reset-test"].total_distills == 0
        assert instiller.session_histories["reset-test"].total_injections == 0

        # Reset non-existent
        result = instiller.reset_session_history("non-existent")
        assert result is False

    def test_clear_cache(self, instiller):
        """Test clearing the results cache."""
        # Add some cache entries
        result = InstillationResult(
            tenets_instilled=[], injection_positions=[], token_increase=0, strategy_used="test"
        )
        instiller._cache["test1"] = result
        instiller._cache["test2"] = result

        instiller.clear_cache()

        assert len(instiller._cache) == 0

    def test_reinforcement_injection(self, instiller, sample_tenets):
        """Test reinforcement injection at intervals."""
        instiller.config.tenet.injection_frequency = "always"
        instiller.config.tenet.reinforcement_interval = 3

        history = InjectionHistory(session_id="reinforce-test")
        instiller.session_histories["reinforce-test"] = history

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                mock_inject.return_value = ("Modified", {"injected_count": 2, "token_increase": 50})

                # Inject multiple times
                for i in range(4):
                    instiller.instill("Content", session="reinforce-test", force=True)

                # Check reinforcement count
                assert history.reinforcement_count == 1  # Should reinforce at 3rd injection

    def test_critical_tenet_boost(self, instiller, sample_tenets):
        """Test critical tenets get priority boost."""
        # Set one tenet as critical
        sample_tenets[2].priority = Priority.CRITICAL

        history = InjectionHistory(session_id="boost-test")
        history.total_injections = 5  # For reinforcement check

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            tenets = instiller._get_tenets_for_instillation(
                session="boost-test",
                force=False,
                content_length=5000,
                max_tenets=2,
                history=history,
                complexity=0.5,
            )

            # Critical tenet should be first
            assert tenets[0].priority == Priority.CRITICAL

    def test_skip_low_priority_on_complex(self, instiller, sample_tenets):
        """Test skipping low priority tenets on complex content."""
        instiller.config.tenet.skip_low_priority_on_complex = True

        with patch.object(instiller.manager, "get_pending_tenets", return_value=sample_tenets):
            # High complexity
            tenets = instiller._get_tenets_for_instillation(
                session="complex-test",
                force=False,
                content_length=5000,
                max_tenets=4,
                history=None,
                complexity=0.9,  # Very high
            )

            # Should not include LOW priority tenet
            assert not any(t.priority == Priority.LOW for t in tenets)
