"""Tests for hotspot detection module."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.examiner.hotspots import (
    FileHotspot,
    HotspotDetector,
    HotspotMetrics,
    HotspotReport,
    ModuleHotspot,
    detect_hotspots,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return TenetsConfig()


@pytest.fixture
def hotspot_detector(config):
    """Create HotspotDetector instance."""
    return HotspotDetector(config)


@pytest.fixture
def sample_file_changes():
    """Create sample file change data."""
    return {
        "hot_file.py": {
            "commit_count": 50,
            "authors": {"alice@example.com", "bob@example.com", "charlie@example.com"},
            "commits": [
                {
                    "sha": f"abc{i:03d}",
                    "date": datetime.now() - timedelta(days=30 - i),
                    "message": f"Fix bug #{i}" if i % 3 == 0 else f"Add feature {i}",
                    "author": f"Author{i % 3}",
                }
                for i in range(50)
            ],
            "lines_added": 500,
            "lines_removed": 300,
            "bug_fixes": 15,
            "refactors": 5,
            "coupled_files": {"related1.py": 10, "related2.py": 8, "related3.py": 5},
            "first_commit": datetime.now() - timedelta(days=90),
            "last_commit": datetime.now() - timedelta(days=1),
        },
        "stable_file.py": {
            "commit_count": 5,
            "authors": {"alice@example.com"},
            "commits": [],
            "lines_added": 100,
            "lines_removed": 20,
            "bug_fixes": 1,
            "refactors": 0,
            "coupled_files": {},
            "first_commit": datetime.now() - timedelta(days=180),
            "last_commit": datetime.now() - timedelta(days=60),
        },
        "complex_file.py": {
            "commit_count": 30,
            "authors": {f"dev{i}@example.com" for i in range(15)},
            "commits": [],
            "lines_added": 1000,
            "lines_removed": 800,
            "bug_fixes": 20,
            "refactors": 10,
            "coupled_files": {f"file{i}.py": i + 1 for i in range(10)},
            "first_commit": datetime.now() - timedelta(days=60),
            "last_commit": datetime.now() - timedelta(days=2),
        },
    }


class TestHotspotMetrics:
    """Test suite for HotspotMetrics dataclass."""

    def test_hotspot_metrics_creation(self):
        """Test creating HotspotMetrics."""
        metrics = HotspotMetrics(
            change_frequency=0.5, commit_count=50, author_count=5, complexity=15.0
        )

        assert metrics.change_frequency == 0.5
        assert metrics.commit_count == 50
        assert metrics.author_count == 5
        assert metrics.complexity == 15.0

    def test_hotspot_score(self):
        """Test hotspot_score calculation."""
        # Low score (stable file)
        stable = HotspotMetrics(
            change_frequency=0.05,
            commit_count=5,
            author_count=2,
            complexity=5.0,
            bug_fix_commits=0,
            coupling=1,
        )
        stable_score = stable.hotspot_score
        assert 0 <= stable_score <= 100
        assert stable_score < 30

        # High score (problematic file)
        problematic = HotspotMetrics(
            change_frequency=2.0,
            commit_count=100,
            author_count=10,
            complexity=25.0,
            bug_fix_commits=40,
            coupling=15,
        )
        problematic_score = problematic.hotspot_score
        assert problematic_score > stable_score
        assert problematic_score > 50

    def test_risk_level(self):
        """Test risk_level property."""
        # Critical risk
        critical = HotspotMetrics()
        critical.change_frequency = 3.0
        critical.complexity = 30.0
        critical.commit_count = 100
        critical.bug_fix_commits = 50
        assert critical.risk_level == "critical"

        # High risk
        high = HotspotMetrics()
        high.change_frequency = 1.0
        high.complexity = 15.0
        high.commit_count = 50
        high.bug_fix_commits = 15
        assert high.risk_level == "high"

        # Medium risk
        medium = HotspotMetrics()
        medium.change_frequency = 0.3
        medium.complexity = 8.0
        assert medium.risk_level == "medium"

        # Low risk
        low = HotspotMetrics()
        assert low.risk_level == "low"

    def test_needs_attention(self):
        """Test needs_attention property."""
        # Needs attention
        critical = HotspotMetrics()
        critical.change_frequency = 2.0
        critical.complexity = 25.0
        critical.commit_count = 50
        critical.bug_fix_commits = 20
        assert critical.needs_attention == True

        # Doesn't need immediate attention
        stable = HotspotMetrics()
        stable.change_frequency = 0.1
        stable.complexity = 5.0
        assert stable.needs_attention == False


class TestFileHotspot:
    """Test suite for FileHotspot dataclass."""

    def test_file_hotspot_creation(self):
        """Test creating FileHotspot."""
        hotspot = FileHotspot(path="src/main.py", name="main.py", size=500, language="Python")

        assert hotspot.path == "src/main.py"
        assert hotspot.name == "main.py"
        assert hotspot.size == 500
        assert hotspot.language == "Python"

    def test_summary(self):
        """Test summary property."""
        hotspot = FileHotspot(path="hot.py", name="hot.py")

        # Set problematic metrics
        hotspot.metrics.change_frequency = 1.5
        hotspot.metrics.complexity = 25.0
        hotspot.metrics.bug_fix_commits = 10
        hotspot.metrics.author_count = 15

        summary = hotspot.summary

        assert "hot.py" in summary
        assert "changes" in summary
        assert "complexity" in summary
        assert "bug fixes" in summary
        assert "authors" in summary

        # Stable file
        stable = FileHotspot(path="stable.py", name="stable.py")
        stable_summary = stable.summary
        assert "stable" in stable_summary


class TestModuleHotspot:
    """Test suite for ModuleHotspot dataclass."""

    def test_module_hotspot_creation(self):
        """Test creating ModuleHotspot."""
        module = ModuleHotspot(path="src/auth", name="auth", file_count=10)

        assert module.path == "src/auth"
        assert module.name == "auth"
        assert module.file_count == 10

    def test_hotspot_density(self):
        """Test hotspot_density property."""
        module = ModuleHotspot(path="src", name="src", file_count=10)

        # Add hotspot files
        for i in range(3):
            module.hotspot_files.append(FileHotspot(path=f"file{i}.py", name=f"file{i}.py"))

        assert module.hotspot_density == 0.3

        # No files
        empty_module = ModuleHotspot(path="empty", name="empty", file_count=0)
        assert empty_module.hotspot_density == 0.0

    def test_module_health(self):
        """Test module_health property."""
        # Healthy module
        healthy = ModuleHotspot(path="healthy", name="healthy", file_count=10, stability_score=80.0)
        assert healthy.module_health == "healthy"

        # Warning module
        warning = ModuleHotspot(path="warning", name="warning", file_count=10, stability_score=50.0)
        for i in range(4):
            warning.hotspot_files.append(FileHotspot(path=f"file{i}.py", name=f"file{i}.py"))
        assert warning.module_health == "warning"

        # Unhealthy module
        unhealthy = ModuleHotspot(
            path="unhealthy", name="unhealthy", file_count=10, stability_score=30.0
        )
        for i in range(6):
            unhealthy.hotspot_files.append(FileHotspot(path=f"file{i}.py", name=f"file{i}.py"))
        assert unhealthy.module_health == "unhealthy"


class TestHotspotReport:
    """Test suite for HotspotReport dataclass."""

    def test_hotspot_report_creation(self):
        """Test creating HotspotReport."""
        report = HotspotReport(
            total_files_analyzed=100, total_hotspots=15, critical_count=3, high_count=5
        )

        assert report.total_files_analyzed == 100
        assert report.total_hotspots == 15
        assert report.critical_count == 3
        assert report.high_count == 5

    def test_total_count(self):
        """Test total_count property."""
        report = HotspotReport()

        for i in range(5):
            report.file_hotspots.append(FileHotspot(path=f"file{i}.py", name=f"file{i}.py"))

        assert report.total_count == 5

    def test_health_score(self):
        """Test health_score property."""
        # Good health
        good_report = HotspotReport(
            total_files_analyzed=100, total_hotspots=5, critical_count=0, high_count=2
        )
        good_score = good_report.health_score
        assert 60 <= good_score <= 100

        # Poor health
        poor_report = HotspotReport(
            total_files_analyzed=100, total_hotspots=30, critical_count=10, high_count=15
        )
        poor_report.coupling_clusters = [["f1", "f2", "f3"] for _ in range(5)]
        poor_score = poor_report.health_score
        assert poor_score < good_score
        assert 0 <= poor_score <= 100

    def test_to_dict(self):
        """Test to_dict conversion."""
        report = HotspotReport(
            total_files_analyzed=50, total_hotspots=10, critical_count=2, high_count=3
        )

        # Add a hotspot
        hotspot = FileHotspot(path="hot.py", name="hot.py")
        hotspot.metrics.hotspot_score = 75.0
        hotspot.metrics.risk_level = "high"
        hotspot.problem_indicators = ["High complexity", "Many bugs"]
        report.file_hotspots.append(hotspot)

        # Add module
        module = ModuleHotspot(path="src", name="src")
        report.module_hotspots.append(module)

        # Add recommendations
        report.recommendations = ["Refactor hot.py"]

        result = report.to_dict()

        assert result["total_files_analyzed"] == 50
        assert result["total_hotspots"] == 10
        assert len(result["hotspot_summary"]) == 1
        assert len(result["module_summary"]) == 1
        assert len(result["recommendations"]) == 1


class TestHotspotDetector:
    """Test suite for HotspotDetector class."""

    def test_initialization(self, config):
        """Test HotspotDetector initialization."""
        detector = HotspotDetector(config)

        assert detector.config == config
        assert detector.git_analyzer is None

    @patch("tenets.core.examiner.hotspots.GitAnalyzer")
    def test_detect_no_repo(self, mock_git_class, hotspot_detector, tmp_path):
        """Test detection with no git repository."""
        mock_git = Mock()
        mock_git.is_repo.return_value = False
        mock_git_class.return_value = mock_git

        report = hotspot_detector.detect(tmp_path)

        assert report.total_files_analyzed == 0
        assert report.total_hotspots == 0

    @patch("tenets.core.examiner.hotspots.GitAnalyzer")
    def test_detect_basic(self, mock_git_class, hotspot_detector, tmp_path):
        """Test basic hotspot detection."""
        mock_git = Mock()
        mock_git.is_repo.return_value = True

        # Create mock commits
        commits = []
        for i in range(20):
            commit = Mock()
            commit.hexsha = f"abc{i:03d}"
            commit.committed_date = (datetime.now() - timedelta(days=20 - i)).timestamp()
            commit.message = "Fix bug" if i % 3 == 0 else "Add feature"
            commit.author = Mock(email=f"dev{i % 3}@example.com")

            # Add file changes
            commit.stats = Mock()
            commit.stats.files = {
                "hot_file.py": {"insertions": 10, "deletions": 5},
                "normal.py": {"insertions": 5, "deletions": 2},
            }
            commits.append(commit)

        mock_git.get_commits_since.return_value = commits
        mock_git_class.return_value = mock_git

        report = hotspot_detector.detect(tmp_path, since_days=30)

        assert report.total_files_analyzed > 0

    def test_analyze_file_changes(self, hotspot_detector):
        """Test file change analysis."""
        hotspot_detector.git_analyzer = Mock()

        # Create mock commits
        commits = []
        for i in range(10):
            commit = Mock()
            commit.hexsha = f"abc{i:03d}"
            commit.committed_date = (datetime.now() - timedelta(days=10 - i)).timestamp()
            commit.message = "Fix critical bug" if i % 2 == 0 else "Refactor code"
            commit.author = Mock(email=f"dev{i % 2}@example.com", name=f"Dev{i % 2}")

            commit.stats = Mock()
            commit.stats.files = {
                "file1.py": {"insertions": 20, "deletions": 10},
                "file2.py": {"insertions": 15, "deletions": 5},
            }
            commits.append(commit)

        hotspot_detector.git_analyzer.get_commits_since.return_value = commits

        file_changes = hotspot_detector._analyze_file_changes(datetime.now() - timedelta(days=30))

        assert "file1.py" in file_changes
        assert "file2.py" in file_changes
        assert file_changes["file1.py"]["commit_count"] == 10
        assert file_changes["file1.py"]["bug_fixes"] == 5
        assert file_changes["file1.py"]["refactors"] == 5
        assert len(file_changes["file1.py"]["authors"]) == 2

    def test_analyze_file_hotspot(self, hotspot_detector, sample_file_changes):
        """Test single file hotspot analysis."""
        change_data = sample_file_changes["hot_file.py"]

        # Create mock analyzed file
        analyzed_file = Mock()
        analyzed_file.path = "hot_file.py"
        analyzed_file.complexity = Mock(cyclomatic=25)
        analyzed_file.lines = 500
        analyzed_file.language = "Python"

        hotspot = hotspot_detector._analyze_file_hotspot(
            "hot_file.py", change_data, [analyzed_file], since_days=90
        )

        assert hotspot.path == "hot_file.py"
        assert hotspot.name == "hot_file.py"
        assert hotspot.metrics.commit_count == 50
        assert hotspot.metrics.author_count == 3
        assert hotspot.metrics.bug_fix_commits == 15
        assert hotspot.metrics.complexity == 25
        assert hotspot.size == 500
        assert hotspot.language == "Python"
        assert len(hotspot.coupled_files) > 0
        assert len(hotspot.problem_indicators) > 0
        assert len(hotspot.recommended_actions) > 0

    def test_identify_problems(self, hotspot_detector):
        """Test problem identification."""
        hotspot = FileHotspot(path="test.py", name="test.py")

        # Set various problematic metrics
        hotspot.metrics.change_frequency = 1.0
        hotspot.metrics.complexity = 25
        hotspot.metrics.bug_fix_commits = 12
        hotspot.metrics.author_count = 15
        hotspot.metrics.coupling = 12
        hotspot.metrics.churn_rate = 15
        hotspot.metrics.recency_days = 3
        hotspot.metrics.commit_count = 10
        hotspot.size = 1200

        problems = hotspot_detector._identify_problems(hotspot)

        assert len(problems) > 0
        assert any("change frequency" in p for p in problems)
        assert any("complexity" in p for p in problems)
        assert any("bug fixes" in p for p in problems)
        assert any("contributors" in p for p in problems)
        assert any("coupled" in p for p in problems)
        assert any("churn" in p for p in problems)
        assert any("large file" in p for p in problems)

    def test_recommend_actions(self, hotspot_detector):
        """Test action recommendation."""
        hotspot = FileHotspot(path="test.py", name="test.py")

        # High complexity
        hotspot.metrics.complexity = 25
        actions = hotspot_detector._recommend_actions(hotspot)
        assert any("refactor" in a.lower() for a in actions)

        # Large file
        hotspot.size = 1500
        actions = hotspot_detector._recommend_actions(hotspot)
        assert any("split" in a.lower() or "smaller" in a.lower() for a in actions)

        # Many bugs
        hotspot.metrics.bug_fix_commits = 10
        actions = hotspot_detector._recommend_actions(hotspot)
        assert any("test" in a.lower() or "review" in a.lower() for a in actions)

        # High coupling
        hotspot.metrics.coupling = 15
        actions = hotspot_detector._recommend_actions(hotspot)
        assert any("coupling" in a.lower() or "abstraction" in a.lower() for a in actions)

    def test_calculate_stability(self, hotspot_detector):
        """Test stability score calculation."""
        # Stable file
        stable = FileHotspot(path="stable.py", name="stable.py")
        stable.metrics.change_frequency = 0.1
        stable.metrics.commit_count = 10
        stable.metrics.bug_fix_commits = 1
        stable.metrics.author_count = 2
        stable.metrics.churn_rate = 1.0
        stable.metrics.recency_days = 60

        stability = hotspot_detector._calculate_stability(stable)
        assert stability > 70

        # Unstable file
        unstable = FileHotspot(path="unstable.py", name="unstable.py")
        unstable.metrics.change_frequency = 2.0
        unstable.metrics.commit_count = 50
        unstable.metrics.bug_fix_commits = 25
        unstable.metrics.author_count = 10
        unstable.metrics.churn_rate = 20.0
        unstable.metrics.recency_days = 2

        instability = hotspot_detector._calculate_stability(unstable)
        assert instability < stability
        assert instability < 50

    def test_detect_coupling_clusters(self, hotspot_detector):
        """Test coupling cluster detection."""
        file_changes = {
            "file1.py": {"coupled_files": {"file2.py": 10, "file3.py": 8, "file4.py": 2}},
            "file2.py": {"coupled_files": {"file1.py": 10, "file3.py": 7, "file4.py": 1}},
            "file3.py": {"coupled_files": {"file1.py": 8, "file2.py": 7, "file4.py": 1}},
            "file4.py": {"coupled_files": {"file1.py": 2, "file2.py": 1, "file3.py": 1}},
            "isolated.py": {"coupled_files": {}},
        }

        clusters = hotspot_detector._detect_coupling_clusters(file_changes)

        assert len(clusters) > 0
        # Should find cluster with file1, file2, file3
        assert any(
            set(["file1.py", "file2.py", "file3.py"]).issubset(set(cluster)) for cluster in clusters
        )

    def test_estimate_remediation_effort(self, hotspot_detector):
        """Test effort estimation."""
        report = HotspotReport()

        # Add hotspots with different risk levels
        for i in range(3):
            hotspot = FileHotspot(path=f"critical{i}.py", name=f"critical{i}.py")
            hotspot.metrics.risk_level = "critical"
            hotspot.size = 800
            hotspot.metrics.complexity = 25
            hotspot.metrics.coupling = 12
            report.file_hotspots.append(hotspot)

        for i in range(5):
            hotspot = FileHotspot(path=f"high{i}.py", name=f"high{i}.py")
            hotspot.metrics.risk_level = "high"
            hotspot.size = 400
            hotspot.metrics.complexity = 15
            report.file_hotspots.append(hotspot)

        effort = hotspot_detector._estimate_remediation_effort(report)

        assert effort > 0
        # Should be reasonable estimate
        assert 50 <= effort <= 500

    def test_generate_recommendations(self, hotspot_detector):
        """Test recommendation generation."""
        report = HotspotReport(critical_count=5, high_count=10)

        # Add coupling clusters
        report.coupling_clusters = [
            ["f1", "f2", "f3"],
            ["f4", "f5"],
            ["f6", "f7", "f8"],
            ["f9", "f10"],
        ]

        # Add unhealthy modules
        for i in range(3):
            module = ModuleHotspot(path=f"module{i}", name=f"module{i}")
            module.module_health = "unhealthy"
            report.module_hotspots.append(module)

        # Add top problems
        report.top_problems = [("High Complexity", 20), ("Frequent Changes", 15), ("Bug Prone", 10)]

        # Set poor health score
        report.health_score = 30.0

        # High effort estimate
        report.estimated_effort = 200.0

        recommendations = hotspot_detector._generate_recommendations(report)

        assert len(recommendations) > 0
        assert any("critical" in r.lower() for r in recommendations)
        assert any("refactoring" in r.lower() for r in recommendations)
        assert any("coupling" in r.lower() for r in recommendations)


class TestDetectHotspotsFunction:
    """Test the detect_hotspots convenience function."""

    @patch("tenets.core.examiner.hotspots.HotspotDetector")
    def test_detect_hotspots_basic(self, mock_detector_class, tmp_path):
        """Test basic hotspot detection."""
        mock_detector = Mock()
        mock_report = HotspotReport(total_hotspots=5)
        mock_detector.detect.return_value = mock_report
        mock_detector_class.return_value = mock_detector

        report = detect_hotspots(tmp_path)

        assert isinstance(report, HotspotReport)
        assert report.total_hotspots == 5
        mock_detector.detect.assert_called_once()

    @patch("tenets.core.examiner.hotspots.HotspotDetector")
    def test_detect_hotspots_with_options(self, mock_detector_class, tmp_path):
        """Test with various options."""
        mock_detector = Mock()
        mock_detector.detect.return_value = HotspotReport()
        mock_detector_class.return_value = mock_detector

        files = [Mock()]
        report = detect_hotspots(tmp_path, files=files, threshold=15)

        assert isinstance(report, HotspotReport)
        mock_detector.detect.assert_called_with(tmp_path, files=files, threshold=15)

    @patch("tenets.core.examiner.hotspots.HotspotDetector")
    def test_detect_hotspots_with_config(self, mock_detector_class, tmp_path, config):
        """Test with custom config."""
        mock_detector = Mock()
        mock_detector.detect.return_value = HotspotReport()
        mock_detector_class.return_value = mock_detector

        report = detect_hotspots(tmp_path, config=config)

        assert isinstance(report, HotspotReport)
        mock_detector_class.assert_called_with(config)
