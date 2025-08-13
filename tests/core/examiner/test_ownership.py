"""Tests for code ownership tracking module."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from tenets.config import TenetsConfig
from tenets.core.examiner.ownership import (
    ContributorInfo,
    FileOwnership,
    OwnershipReport,
    OwnershipTracker,
    track_ownership,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return TenetsConfig()


@pytest.fixture
def ownership_tracker(config):
    """Create OwnershipTracker instance."""
    return OwnershipTracker(config)


@pytest.fixture
def mock_git_analyzer():
    """Create mock GitAnalyzer."""
    analyzer = Mock()
    analyzer.is_repo.return_value = True
    analyzer.get_commits_since.return_value = []
    analyzer.get_tracked_files.return_value = []
    analyzer.get_file_history.return_value = []
    return analyzer


@pytest.fixture
def sample_commits():
    """Create sample commit data."""
    commits = []

    for i in range(10):
        commit = Mock()
        commit.hexsha = f"abc{i:03d}"
        commit.author_name = f"Author{i % 3}"
        commit.author_email = f"author{i % 3}@example.com"
        commit.committed_date = (datetime.now() - timedelta(days=10 - i)).timestamp()
        commit.message = f"Commit {i}"

        # Add file stats
        commit.stats = Mock()
        commit.stats.files = {f"file{i % 4}.py": {"insertions": 10 + i, "deletions": 5}}

        commits.append(commit)

    return commits


class TestContributorInfo:
    """Test suite for ContributorInfo dataclass."""

    def test_contributor_info_creation(self):
        """Test creating ContributorInfo."""
        contributor = ContributorInfo(name="John Doe", email="john@example.com", total_commits=50)

        assert contributor.name == "John Doe"
        assert contributor.email == "john@example.com"
        assert contributor.total_commits == 50

    def test_net_lines_contributed(self):
        """Test net_lines_contributed property."""
        contributor = ContributorInfo(
            name="Test", email="test@example.com", total_lines_added=1000, total_lines_removed=300
        )

        assert contributor.net_lines_contributed == 700

    def test_productivity_score(self):
        """Test productivity_score calculation."""
        contributor = ContributorInfo(
            name="Test",
            email="test@example.com",
            commit_frequency=2.5,
            total_lines_added=5000,
            total_lines_removed=1000,
            active_days=20,
            first_commit_date=datetime.now() - timedelta(days=30),
            last_commit_date=datetime.now(),
        )
        contributor.files_touched = set([f"file{i}.py" for i in range(15)])

        score = contributor.productivity_score
        assert 0 <= score <= 100
        assert score > 50  # Should be relatively productive

    def test_is_active(self):
        """Test is_active property."""
        # Active contributor
        active = ContributorInfo(
            name="Active",
            email="active@example.com",
            last_commit_date=datetime.now() - timedelta(days=10),
        )
        assert active.is_active == True

        # Inactive contributor
        inactive = ContributorInfo(
            name="Inactive",
            email="inactive@example.com",
            last_commit_date=datetime.now() - timedelta(days=60),
        )
        assert inactive.is_active == False

        # No commits
        new_contributor = ContributorInfo(name="New", email="new@example.com")
        assert new_contributor.is_active == False

    def test_expertise_level(self):
        """Test expertise_level property."""
        # Expert
        expert = ContributorInfo(name="Expert", email="expert@example.com", total_commits=600)
        expert.files_touched = set([f"file{i}.py" for i in range(150)])
        assert expert.expertise_level == "expert"

        # Senior
        senior = ContributorInfo(name="Senior", email="senior@example.com", total_commits=150)
        senior.files_touched = set([f"file{i}.py" for i in range(60)])
        assert senior.expertise_level == "senior"

        # Intermediate
        intermediate = ContributorInfo(
            name="Intermediate", email="intermediate@example.com", total_commits=30
        )
        assert intermediate.expertise_level == "intermediate"

        # Junior
        junior = ContributorInfo(name="Junior", email="junior@example.com", total_commits=5)
        assert junior.expertise_level == "junior"


class TestFileOwnership:
    """Test suite for FileOwnership dataclass."""

    def test_file_ownership_creation(self):
        """Test creating FileOwnership."""
        ownership = FileOwnership(
            path="src/main.py", primary_owner="alice@example.com", ownership_percentage=75.0
        )

        assert ownership.path == "src/main.py"
        assert ownership.primary_owner == "alice@example.com"
        assert ownership.ownership_percentage == 75.0

    def test_contributor_count(self):
        """Test contributor_count property."""
        ownership = FileOwnership(path="test.py")
        ownership.contributors = [
            ("alice@example.com", 50),
            ("bob@example.com", 30),
            ("charlie@example.com", 20),
        ]

        assert ownership.contributor_count == 3

    def test_bus_factor(self):
        """Test bus_factor calculation."""
        ownership = FileOwnership(path="test.py", total_changes=100)

        # Single dominant contributor
        ownership.contributors = [("alice@example.com", 90), ("bob@example.com", 10)]
        assert ownership.bus_factor == 1

        # Multiple significant contributors
        ownership.contributors = [
            ("alice@example.com", 40),
            ("bob@example.com", 35),
            ("charlie@example.com", 25),
        ]
        assert ownership.bus_factor == 3

        # No contributors
        ownership.contributors = []
        assert ownership.bus_factor == 0

    def test_risk_level(self):
        """Test risk_level property."""
        ownership = FileOwnership(path="test.py")

        # Critical - orphaned
        ownership.is_orphaned = True
        assert ownership.risk_level == "critical"

        # High - bus factor 1
        ownership.is_orphaned = False
        ownership.contributors = [("alice@example.com", 100)]
        ownership.total_changes = 100
        assert ownership.risk_level == "high"

        # Medium - bus factor 2
        ownership.contributors = [("alice@example.com", 60), ("bob@example.com", 40)]
        assert ownership.risk_level == "medium"

        # Low - bus factor > 2
        ownership.contributors = [
            ("alice@example.com", 40),
            ("bob@example.com", 35),
            ("charlie@example.com", 25),
        ]
        assert ownership.risk_level == "low"


class TestOwnershipReport:
    """Test suite for OwnershipReport dataclass."""

    def test_ownership_report_creation(self):
        """Test creating OwnershipReport."""
        report = OwnershipReport(total_contributors=10, active_contributors=7, bus_factor=3)

        assert report.total_contributors == 10
        assert report.active_contributors == 7
        assert report.bus_factor == 3

    def test_health_score(self):
        """Test health_score property."""
        report = OwnershipReport(risk_score=25.0)
        assert report.health_score == 75.0

        report.risk_score = 80.0
        assert report.health_score == 20.0

        report.risk_score = 0.0
        assert report.health_score == 100.0

    def test_to_dict(self):
        """Test to_dict conversion."""
        report = OwnershipReport(
            total_contributors=5, active_contributors=4, bus_factor=2, risk_score=30.0
        )

        # Add some contributors
        report.contributors = [
            ContributorInfo(name="Alice", email="alice@example.com", total_commits=100),
            ContributorInfo(name="Bob", email="bob@example.com", total_commits=50),
        ]

        result = report.to_dict()

        assert result["total_contributors"] == 5
        assert result["active_contributors"] == 4
        assert result["bus_factor"] == 2
        assert result["risk_score"] == 30.0
        assert len(result["top_contributors"]) == 2


class TestOwnershipTracker:
    """Test suite for OwnershipTracker."""

    def test_initialization(self, config):
        """Test OwnershipTracker initialization."""
        tracker = OwnershipTracker(config)

        assert tracker.config == config
        assert tracker.git_analyzer is None

    @patch("tenets.core.examiner.ownership.GitAnalyzer")
    def test_track_no_repo(self, mock_git_class, ownership_tracker, tmp_path):
        """Test tracking with no git repository."""
        mock_git = Mock()
        mock_git.is_repo.return_value = False
        mock_git_class.return_value = mock_git

        report = ownership_tracker.track(tmp_path)

        assert report.total_contributors == 0
        assert report.total_files_analyzed == 0

    @patch("tenets.core.examiner.ownership.GitAnalyzer")
    def test_track_basic(self, mock_git_class, ownership_tracker, tmp_path):
        """Test basic ownership tracking."""
        # Setup mock git analyzer
        mock_git = Mock()
        mock_git.is_repo.return_value = True

        # Mock commits
        commit1 = Mock()
        commit1.author_email = "alice@example.com"
        commit1.author_name = "Alice"
        commit1.committed_date = datetime.now().timestamp()
        commit1.stats = Mock()
        commit1.stats.files = {"file1.py": {"insertions": 100, "deletions": 20}}

        commit2 = Mock()
        commit2.author_email = "bob@example.com"
        commit2.author_name = "Bob"
        commit2.committed_date = (datetime.now() - timedelta(days=5)).timestamp()
        commit2.stats = Mock()
        commit2.stats.files = {
            "file1.py": {"insertions": 50, "deletions": 10},
            "file2.py": {"insertions": 200, "deletions": 0},
        }

        mock_git.get_commits_since.return_value = [commit1, commit2]
        mock_git.get_tracked_files.return_value = ["file1.py", "file2.py"]
        mock_git.get_file_history.return_value = [commit1, commit2]

        mock_git_class.return_value = mock_git

        report = ownership_tracker.track(tmp_path, since_days=30)

        assert report.total_contributors > 0
        assert len(report.contributors) > 0

    def test_is_bot_account(self, ownership_tracker):
        """Test bot account detection."""
        bot_emails = [
            "dependabot@github.com",
            "renovate-bot@example.com",
            "github-actions[bot]@users.noreply.github.com",
            "ci-automation@example.com",
        ]

        for email in bot_emails:
            assert ownership_tracker._is_bot_account(email, "") == True

        human_emails = ["john.doe@example.com", "alice@company.com", "developer@team.org"]

        for email in human_emails:
            assert ownership_tracker._is_bot_account(email, "") == False

    def test_is_test_file(self, ownership_tracker):
        """Test test file detection."""
        test_files = [
            "test_main.py",
            "main_test.py",
            "tests/test_utils.py",
            "spec/feature.spec.js",
            "file.test.ts",
        ]

        for file_path in test_files:
            assert ownership_tracker._is_test_file(file_path) == True

        non_test_files = ["main.py", "utils.js", "src/app.py", "lib/helper.rb"]

        for file_path in non_test_files:
            assert ownership_tracker._is_test_file(file_path) == False

    def test_ext_to_language(self, ownership_tracker):
        """Test file extension to language mapping."""
        mappings = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".go": "Go",
            ".rs": "Rust",
        }

        for ext, expected_lang in mappings.items():
            assert ownership_tracker._ext_to_language(ext) == expected_lang

        # Unknown extension
        assert ownership_tracker._ext_to_language(".xyz") is None

    def test_calculate_gini_coefficient(self, ownership_tracker):
        """Test Gini coefficient calculation."""
        # Perfect equality
        equal_values = [10, 10, 10, 10]
        gini_equal = ownership_tracker._calculate_gini_coefficient(equal_values)
        assert gini_equal < 0.1

        # Perfect inequality
        unequal_values = [100, 0, 0, 0]
        gini_unequal = ownership_tracker._calculate_gini_coefficient(unequal_values)
        assert gini_unequal > 0.7

        # Empty values
        gini_empty = ownership_tracker._calculate_gini_coefficient([])
        assert gini_empty == 0.0

    def test_interpret_gini(self, ownership_tracker):
        """Test Gini coefficient interpretation."""
        assert ownership_tracker._interpret_gini(0.1) == "Very equal distribution"
        assert ownership_tracker._interpret_gini(0.3) == "Relatively equal distribution"
        assert ownership_tracker._interpret_gini(0.5) == "Moderate inequality"
        assert ownership_tracker._interpret_gini(0.7) == "High inequality"
        assert ownership_tracker._interpret_gini(0.9) == "Very high inequality"

    def test_identify_expertise_areas(self, ownership_tracker):
        """Test expertise area identification."""
        contributor = ContributorInfo(name="Test", email="test@example.com")

        # Add files from different areas
        contributor.files_touched = {
            "src/auth/login.py",
            "src/auth/logout.py",
            "src/auth/session.py",
            "src/db/models.py",
            "tests/test_auth.py",
        }

        contributor.primary_languages = {".py": 10, ".js": 5, ".sql": 2}

        areas = ownership_tracker._identify_expertise_areas(contributor)

        assert "auth" in areas  # Most common directory
        assert "Python" in areas  # Most common language

    @patch("tenets.core.examiner.ownership.GitAnalyzer")
    def test_team_ownership_analysis(self, mock_git_class, ownership_tracker, tmp_path):
        """Test team-level ownership analysis."""
        mock_git = Mock()
        mock_git.is_repo.return_value = True
        mock_git.get_commits_since.return_value = []
        mock_git.get_tracked_files.return_value = []
        mock_git_class.return_value = mock_git

        team_mapping = {
            "frontend": ["alice@example.com", "bob@example.com"],
            "backend": ["charlie@example.com", "dave@example.com"],
        }

        report = ownership_tracker.track(tmp_path, team_mapping=team_mapping)

        assert report.team_ownership is not None
        assert report.team_ownership.teams == team_mapping


class TestTrackOwnershipFunction:
    """Test the track_ownership convenience function."""

    @patch("tenets.core.examiner.ownership.OwnershipTracker")
    def test_track_ownership_basic(self, mock_tracker_class, tmp_path):
        """Test basic ownership tracking."""
        mock_tracker = Mock()
        mock_report = OwnershipReport(total_contributors=5)
        mock_tracker.track.return_value = mock_report
        mock_tracker_class.return_value = mock_tracker

        report = track_ownership(tmp_path, since_days=60)

        assert isinstance(report, OwnershipReport)
        assert report.total_contributors == 5
        mock_tracker.track.assert_called_once()

    @patch("tenets.core.examiner.ownership.OwnershipTracker")
    def test_track_ownership_with_config(self, mock_tracker_class, tmp_path, config):
        """Test with custom config."""
        mock_tracker = Mock()
        mock_tracker.track.return_value = OwnershipReport()
        mock_tracker_class.return_value = mock_tracker

        report = track_ownership(tmp_path, config=config)

        assert isinstance(report, OwnershipReport)
        mock_tracker_class.assert_called_with(config)
