"""Tests for main Instiller orchestrator."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.instiller.instiller import InstillationResult, Instiller, MetricsTracker
from tenets.models.context import ContextResult
from tenets.models.tenet import Priority, Tenet, TenetStatus


@pytest.fixture
def config():
    """Create test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TenetsConfig()
        config.cache_dir = tmpdir
        config.max_tenets_per_context = 5
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
    ]


@pytest.fixture
def mock_manager(sample_tenets):
    """Create mock TenetManager."""
    manager = Mock()
    manager._tenet_cache = {t.id: t for t in sample_tenets}
    manager.get_pending_tenets.return_value = sample_tenets
    manager._save_tenet = Mock()
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
            "injections": [],
            "reinforcement_added": False,
        },
    )
    injector.calculate_optimal_injection_count.return_value = 3
    return injector


class TestInstiller:
    """Test suite for Instiller."""

    def test_initialization(self, config):
        """Test Instiller initialization."""
        instiller = Instiller(config)

        assert instiller.config == config
        assert instiller.manager is not None
        assert instiller.injector is not None
        assert instiller.metrics_tracker is not None

    def test_instill_string_content(self, instiller):
        """Test instilling tenets into string content."""
        with patch.object(instiller.manager, "get_pending_tenets") as mock_get:
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                mock_get.return_value = [Tenet(content="Test tenet", priority=Priority.HIGH)]
                mock_inject.return_value = (
                    "Modified content",
                    {"injected_count": 1, "token_increase": 30, "injections": []},
                )

                result = instiller.instill("Original content")

                assert result == "Modified content"
                mock_inject.assert_called_once()

    def test_instill_context_result(self, instiller):
        """Test instilling into ContextResult."""
        context = ContextResult(
            files=["test.py"],
            context="Original content",
            format="markdown",
            metadata={"key": "value"},
        )

        with patch.object(instiller.manager, "get_pending_tenets") as mock_get:
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                mock_get.return_value = [Tenet(content="Test tenet", priority=Priority.HIGH)]
                mock_inject.return_value = (
                    "Modified content",
                    {"injected_count": 1, "token_increase": 30, "injections": []},
                )

                result = instiller.instill(context)

                assert isinstance(result, ContextResult)
                assert result.context == "Modified content"
                assert "tenet_instillation" in result.metadata

    def test_instill_no_tenets(self, instiller):
        """Test instilling when no tenets available."""
        with patch.object(instiller.manager, "get_pending_tenets") as mock_get:
            mock_get.return_value = []

            result = instiller.instill("Content")

            assert result == "Content"  # Unchanged

    def test_instill_with_session(self, instiller):
        """Test instilling with session filter."""
        with patch.object(instiller, "_get_tenets_for_instillation") as mock_get:
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                tenet = Tenet(content="Session tenet", priority=Priority.HIGH)
                tenet.bind_to_session("test-session")

                mock_get.return_value = [tenet]
                mock_inject.return_value = ("Modified", {"injected_count": 1, "token_increase": 30})

                result = instiller.instill("Content", session="test-session")

                mock_get.assert_called_with(
                    session="test-session",
                    force=False,
                    content_length=7,
                    max_tenets=instiller.config.max_tenets_per_context,
                )

    def test_instill_force_reinstill(self, instiller):
        """Test force reinstilling already instilled tenets."""
        with patch.object(instiller, "_get_tenets_for_instillation") as mock_get:
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                tenet = Tenet(content="Already instilled", priority=Priority.HIGH)
                tenet.status = TenetStatus.INSTILLED

                mock_get.return_value = [tenet]
                mock_inject.return_value = ("Modified", {"injected_count": 1, "token_increase": 30})

                result = instiller.instill("Content", force=True)

                mock_get.assert_called_with(
                    session=None,
                    force=True,
                    content_length=7,
                    max_tenets=instiller.config.max_tenets_per_context,
                )

    def test_instill_custom_strategy(self, instiller):
        """Test instilling with custom strategy."""
        with patch.object(instiller.manager, "get_pending_tenets") as mock_get:
            with patch.object(instiller.injector, "inject_tenets") as mock_inject:
                mock_get.return_value = [Tenet(content="Test", priority=Priority.HIGH)]
                mock_inject.return_value = ("Modified", {"injected_count": 1, "token_increase": 30})

                result = instiller.instill("Content", strategy="distributed")

                # Should not call _determine_injection_strategy
                mock_inject.assert_called()

    def test_get_tenets_for_instillation(self, instiller):
        """Test getting tenets for instillation."""
        pending = [
            Tenet(content="High priority", priority=Priority.HIGH),
            Tenet(content="Critical", priority=Priority.CRITICAL),
            Tenet(content="Low priority", priority=Priority.LOW),
        ]

        with patch.object(instiller.manager, "get_pending_tenets", return_value=pending):
            with patch.object(
                instiller.injector, "calculate_optimal_injection_count", return_value=2
            ):
                tenets = instiller._get_tenets_for_instillation(
                    session=None, force=False, content_length=1000, max_tenets=5
                )

                assert len(tenets) == 2
                # Should be sorted by priority
                assert tenets[0].priority == Priority.CRITICAL
                assert tenets[1].priority == Priority.HIGH

    def test_get_tenets_for_instillation_force(self, instiller):
        """Test getting tenets with force flag."""
        all_tenets = [
            Tenet(content="Active", priority=Priority.HIGH, status=TenetStatus.INSTILLED),
            Tenet(content="Archived", priority=Priority.HIGH, status=TenetStatus.ARCHIVED),
        ]

        with patch.object(instiller.manager, "_tenet_cache", {t.id: t for t in all_tenets}):
            with patch.object(
                instiller.injector, "calculate_optimal_injection_count", return_value=10
            ):
                tenets = instiller._get_tenets_for_instillation(
                    session=None, force=True, content_length=1000, max_tenets=10
                )

                # Should include instilled but not archived
                assert len(tenets) == 1
                assert tenets[0].content == "Active"

    def test_determine_injection_strategy_short(self, instiller):
        """Test strategy determination for short content."""
        strategy = instiller._determine_injection_strategy(
            content_length=1000, tenet_count=2, format_type="markdown"
        )

        assert strategy == "top"

    def test_determine_injection_strategy_long(self, instiller):
        """Test strategy determination for long content."""
        strategy = instiller._determine_injection_strategy(
            content_length=60000, tenet_count=8, format_type="markdown"
        )

        assert strategy == "distributed"

    def test_determine_injection_strategy_xml(self, instiller):
        """Test strategy determination for XML."""
        strategy = instiller._determine_injection_strategy(
            content_length=10000, tenet_count=3, format_type="xml"
        )

        assert strategy == "strategic"

    def test_analyze_effectiveness(self, instiller):
        """Test effectiveness analysis."""
        with patch.object(instiller.manager, "analyze_tenet_effectiveness") as mock_analyze:
            mock_analyze.return_value = {
                "total_tenets": 5,
                "need_reinforcement": ["tenet1", "tenet2"],
            }

            analysis = instiller.analyze_effectiveness()

            assert "instillation_metrics" in analysis
            assert "recommendations" in analysis
            assert len(analysis["recommendations"]) > 0

    def test_export_instillation_history_json(self, instiller, tmp_path):
        """Test exporting history as JSON."""
        # Add some fake history
        result = InstillationResult(
            tenets_instilled=[Tenet(content="Test", priority=Priority.HIGH)],
            injection_positions=[],
            token_increase=50,
            strategy_used="top",
        )
        instiller._cache["test"] = result

        output_file = tmp_path / "history.json"
        instiller.export_instillation_history(output_file, format="json")

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
            assert "instillations" in data
            assert len(data["instillations"]) == 1

    def test_export_instillation_history_csv(self, instiller, tmp_path):
        """Test exporting history as CSV."""
        result = InstillationResult(
            tenets_instilled=[Tenet(content="Test", priority=Priority.HIGH)],
            injection_positions=[],
            token_increase=50,
            strategy_used="top",
            session="test-session",
        )
        instiller._cache["test"] = result

        output_file = tmp_path / "history.csv"
        instiller.export_instillation_history(output_file, format="csv")

        assert output_file.exists()

        content = output_file.read_text()
        assert "Timestamp" in content
        assert "test-session" in content

    def test_export_invalid_format(self, instiller, tmp_path):
        """Test export with invalid format."""
        with pytest.raises(ValueError):
            instiller.export_instillation_history(tmp_path / "output.txt", format="invalid")

    def test_instillation_result(self):
        """Test InstillationResult dataclass."""
        tenets = [Tenet(content="Test", priority=Priority.HIGH)]

        result = InstillationResult(
            tenets_instilled=tenets,
            injection_positions=[{"position": 100}],
            token_increase=50,
            strategy_used="strategic",
            session="test",
        )

        assert result.timestamp is not None
        assert result.success == True
        assert result.metrics is not None

        # Test to_dict
        data = result.to_dict()
        assert "tenets_instilled" in data
        assert "timestamp" in data
        assert data["session"] == "test"


class TestMetricsTracker:
    """Test suite for MetricsTracker."""

    def test_initialization(self):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker()

        assert tracker.instillations == []
        assert tracker.session_metrics == {}
        assert tracker.strategy_usage == {}

    def test_record_instillation(self):
        """Test recording instillation metrics."""
        tracker = MetricsTracker()

        tracker.record_instillation(
            tenet_count=3, token_increase=100, strategy="strategic", session="test-session"
        )

        assert len(tracker.instillations) == 1
        assert tracker.instillations[0]["tenet_count"] == 3
        assert "test-session" in tracker.session_metrics
        assert tracker.strategy_usage["strategic"] == 1

    def test_record_multiple_instillations(self):
        """Test recording multiple instillations."""
        tracker = MetricsTracker()

        tracker.record_instillation(3, 100, "top", "session1")
        tracker.record_instillation(2, 80, "strategic", "session1")
        tracker.record_instillation(4, 120, "top", "session2")

        assert len(tracker.instillations) == 3
        assert tracker.session_metrics["session1"]["total_instillations"] == 2
        assert tracker.session_metrics["session1"]["total_tenets"] == 5
        assert tracker.strategy_usage["top"] == 2
        assert tracker.strategy_usage["strategic"] == 1

    def test_get_metrics_all(self):
        """Test getting all metrics."""
        tracker = MetricsTracker()

        tracker.record_instillation(3, 100, "top")
        tracker.record_instillation(2, 80, "strategic")

        metrics = tracker.get_metrics()

        assert metrics["total_instillations"] == 2
        assert metrics["total_tenets_instilled"] == 5
        assert metrics["total_token_increase"] == 180
        assert metrics["avg_tenets_per_context"] == 2.5
        assert metrics["avg_token_increase"] == 90

    def test_get_metrics_by_session(self):
        """Test getting metrics filtered by session."""
        tracker = MetricsTracker()

        tracker.record_instillation(3, 100, "top", "session1")
        tracker.record_instillation(2, 80, "strategic", "session2")

        metrics = tracker.get_metrics(session="session1")

        assert metrics["total_instillations"] == 1
        assert metrics["total_tenets_instilled"] == 3

    def test_get_metrics_empty(self):
        """Test getting metrics with no data."""
        tracker = MetricsTracker()

        metrics = tracker.get_metrics()

        assert "message" in metrics
        assert metrics["message"] == "No instillation records found"

    def test_get_all_metrics(self):
        """Test getting all metrics for export."""
        tracker = MetricsTracker()

        tracker.record_instillation(3, 100, "top", "test")

        all_metrics = tracker.get_all_metrics()

        assert "instillations" in all_metrics
        assert "session_metrics" in all_metrics
        assert "strategy_usage" in all_metrics
        assert len(all_metrics["instillations"]) == 1
