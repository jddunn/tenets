"""Tests for Git chronicle module."""

import subprocess
from datetime import datetime, timedelta

import pytest

from tenets.config import TenetsConfig
from tenets.core.git.chronicle import (
    Chronicle,
    ChronicleReport,
    CommitSummary,
    DayActivity,
    create_chronicle,
)

# Check Git availability
try:
    from git import Repo

    result = subprocess.run(["git", "--version"], capture_output=True, check=True)
    _HAS_GIT = True
except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
    _HAS_GIT = False


@pytest.fixture
def temp_git_repo_with_activity(tmp_path):
    """Create a git repo with varied activity patterns."""
    if not _HAS_GIT:
        pytest.skip("Git not available")

    from git import Repo

    repo = Repo.init(tmp_path)

    # Configure git
    with repo.config_writer() as config:
        config.set_value("user", "name", "Developer")
        config.set_value("user", "email", "dev@example.com")

    # Create commits over several days
    files_created = []

    # Day 1 - High activity
    for i in range(5):
        file_path = tmp_path / f"file_{i}.py"
        file_path.write_text(f"# File {i}\nprint('test')")
        files_created.append(f"file_{i}.py")
        repo.index.add([f"file_{i}.py"])

        # Vary commit types
        if i == 0:
            message = "feat: Initial feature"
        elif i == 1:
            message = "fix: Fix bug in feature"
        elif i == 2:
            message = "docs: Add documentation"
        elif i == 3:
            message = "test: Add tests"
        else:
            message = "refactor: Improve code structure"

        repo.index.commit(message)

    # Day 2 - Moderate activity
    for i in range(2):
        file_path = tmp_path / f"module_{i}.py"
        file_path.write_text(f"# Module {i}")
        repo.index.add([f"module_{i}.py"])
        repo.index.commit(f"feat: Add module {i}")

    # Create a merge commit
    original_branch = repo.active_branch
    new_branch = repo.create_head("feature-branch")
    new_branch.checkout()

    feature_file = tmp_path / "feature.py"
    feature_file.write_text("# Feature code")
    repo.index.add(["feature.py"])
    repo.index.commit("feat: Add feature on branch")

    original_branch.checkout()
    repo.index.merge_tree(new_branch)
    repo.index.commit("Merge feature branch")

    # Add a revert commit
    (tmp_path / "buggy.py").write_text("# Buggy code")
    repo.index.add(["buggy.py"])
    buggy_commit = repo.index.commit("feat: Add buggy feature")

    repo.git.revert(buggy_commit.hexsha, "--no-edit")

    return tmp_path


@pytest.fixture
def config():
    """Create test configuration."""
    return TenetsConfig()


@pytest.fixture
def chronicle(config):
    """Create Chronicle instance."""
    return Chronicle(config)


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestCommitSummary:
    """Test suite for CommitSummary dataclass."""

    def test_commit_summary_creation(self):
        """Test creating CommitSummary."""
        summary = CommitSummary(
            sha="abc123",
            author="Alice",
            email="alice@example.com",
            date=datetime.now(),
            message="Fix critical bug",
            files_changed=3,
            lines_added=50,
            lines_removed=20,
        )

        assert summary.sha == "abc123"
        assert summary.author == "Alice"
        assert summary.files_changed == 3

    def test_net_lines(self):
        """Test net_lines property."""
        summary = CommitSummary(
            sha="abc",
            author="Test",
            email="test@example.com",
            date=datetime.now(),
            message="Test",
            lines_added=100,
            lines_removed=30,
        )

        assert summary.net_lines == 70

    def test_commit_type_conventional(self):
        """Test commit_type detection with conventional commits."""
        test_cases = [
            ("feat: Add new feature", "feat"),
            ("fix: Fix bug", "fix"),
            ("docs: Update README", "docs"),
            ("test: Add unit tests", "test"),
            ("refactor: Clean up code", "refactor"),
            ("style: Format code", "style"),
            ("chore: Update dependencies", "chore"),
            ("feat(scope): Scoped feature", "feat"),
        ]

        for message, expected_type in test_cases:
            summary = CommitSummary(
                sha="abc",
                author="Test",
                email="test@example.com",
                date=datetime.now(),
                message=message,
            )
            assert summary.commit_type == expected_type

    def test_commit_type_heuristic(self):
        """Test commit_type detection with heuristics."""
        test_cases = [
            ("Fixed the login bug", "fix"),
            ("Add new dashboard", "feat"),
            ("Update documentation", "docs"),
            ("Added test coverage", "test"),
            ("Refactored database layer", "refactor"),
            ("Random commit message", "other"),
        ]

        for message, expected_type in test_cases:
            summary = CommitSummary(
                sha="abc",
                author="Test",
                email="test@example.com",
                date=datetime.now(),
                message=message,
            )
            assert summary.commit_type == expected_type

    def test_commit_type_special(self):
        """Test special commit types."""
        # Merge commit
        summary = CommitSummary(
            sha="abc",
            author="Test",
            email="test@example.com",
            date=datetime.now(),
            message="Merge branch 'feature'",
            is_merge=True,
        )
        assert summary.commit_type == "merge"

        # Revert commit
        summary = CommitSummary(
            sha="def",
            author="Test",
            email="test@example.com",
            date=datetime.now(),
            message="Revert previous commit",
            is_revert=True,
        )
        assert summary.commit_type == "revert"

    def test_to_dict(self):
        """Test to_dict conversion."""
        now = datetime.now()
        summary = CommitSummary(
            sha="abc123",
            author="Alice",
            email="alice@example.com",
            date=now,
            message="feat: Add feature",
            files_changed=5,
            lines_added=100,
            lines_removed=50,
            issue_refs=["123", "456"],
            pr_refs=["789"],
        )

        result = summary.to_dict()

        assert result["sha"] == "abc123"
        assert result["author"] == "Alice"
        assert result["type"] == "feat"
        assert result["issue_refs"] == ["123", "456"]
        assert result["pr_refs"] == ["789"]


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestDayActivity:
    """Test suite for DayActivity dataclass."""

    def test_day_activity_creation(self):
        """Test creating DayActivity."""
        activity = DayActivity(
            date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        )

        assert activity.total_commits == 0
        assert len(activity.unique_authors) == 0
        assert activity.lines_added == 0

    def test_net_lines(self):
        """Test net_lines property."""
        activity = DayActivity(
            date=datetime.now(),
            lines_added=200,
            lines_removed=50,
        )

        assert activity.net_lines == 150

    def test_productivity_score(self):
        """Test productivity_score calculation."""
        activity = DayActivity(date=datetime.now())

        # Low activity
        activity.total_commits = 1
        activity.unique_authors.add("Alice")
        activity.lines_added = 10
        activity.lines_removed = 5

        score = activity.productivity_score
        assert 0 <= score <= 100

        # High activity
        activity.total_commits = 15
        activity.unique_authors.update(["Bob", "Charlie"])
        activity.lines_added = 500
        activity.lines_removed = 200
        activity.commit_types = {"feat": 5, "fix": 3}

        high_score = activity.productivity_score
        assert high_score > score
        assert high_score <= 100


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestChronicleReport:
    """Test suite for ChronicleReport dataclass."""

    def test_chronicle_report_creation(self):
        """Test creating ChronicleReport."""
        now = datetime.now()
        report = ChronicleReport(
            period_start=now - timedelta(days=30),
            period_end=now,
        )

        assert report.total_commits == 0
        assert report.total_contributors == 0
        assert len(report.commits) == 0

    def test_most_active_day(self):
        """Test most_active_day property."""
        report = ChronicleReport(
            period_start=datetime.now() - timedelta(days=7),
            period_end=datetime.now(),
        )

        # No activity
        assert report.most_active_day is None

        # Add days with different activity
        for i in range(3):
            day = DayActivity(
                date=datetime.now() - timedelta(days=i),
                total_commits=i * 2 + 1,
            )
            report.daily_activity.append(day)

        most_active = report.most_active_day
        assert most_active.total_commits == 5  # Day 2 has 5 commits

    def test_activity_level(self):
        """Test activity_level property."""
        report = ChronicleReport(
            period_start=datetime.now() - timedelta(days=10),
            period_end=datetime.now(),
        )

        # No activity
        assert report.activity_level == "none"

        # Low activity
        report.total_commits = 20
        report.daily_activity = [
            DayActivity(date=datetime.now() - timedelta(days=i)) for i in range(10)
        ]
        assert report.activity_level == "low"

        # Moderate activity
        report.total_commits = 50
        assert report.activity_level == "moderate"

        # High activity
        report.total_commits = 150
        assert report.activity_level == "high"

    def test_to_dict(self):
        """Test to_dict conversion."""
        now = datetime.now()
        report = ChronicleReport(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_commits=100,
            total_contributors=5,
            summary="Test summary",
        )

        report.commit_type_distribution = {"feat": 40, "fix": 30, "docs": 20, "other": 10}

        report.contributor_stats = {
            "Alice": {"commits": 50},
            "Bob": {"commits": 30},
        }

        result = report.to_dict()

        assert result["period"]["days"] == 30
        assert result["summary"]["total_commits"] == 100
        assert result["summary"]["total_contributors"] == 5
        assert result["commit_types"]["feat"] == 40
        assert len(result["top_contributors"]) == 2


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestChronicle:
    """Test suite for Chronicle class."""

    def test_initialization(self, config):
        """Test Chronicle initialization."""
        chronicle = Chronicle(config)

        assert chronicle.config == config
        assert chronicle.git_analyzer is None

    def test_analyze_basic(self, temp_git_repo_with_activity, chronicle):
        """Test basic chronicle analysis."""
        report = chronicle.analyze(
            temp_git_repo_with_activity,
            since="1 week ago",
            include_stats=True,
        )

        assert report.total_commits > 0
        assert report.total_contributors > 0
        assert len(report.commits) > 0
        assert report.summary != ""

    def test_analyze_with_filters(self, temp_git_repo_with_activity, chronicle):
        """Test analysis with various filters."""
        # Time filter
        report = chronicle.analyze(
            temp_git_repo_with_activity,
            since="2 days ago",
            until="today",
        )

        assert isinstance(report, ChronicleReport)

        # Without merges
        report_no_merges = chronicle.analyze(
            temp_git_repo_with_activity,
            include_merges=False,
        )

        # Should have fewer commits without merges
        assert report_no_merges.total_commits <= report.total_commits

    def test_analyze_no_stats(self, temp_git_repo_with_activity, chronicle):
        """Test analysis without detailed stats."""
        report = chronicle.analyze(
            temp_git_repo_with_activity,
            include_stats=False,
        )

        assert report.total_commits > 0
        # File changes might be empty without stats
        assert len(report.file_change_frequency) == 0

    def test_analyze_non_git_repo(self, tmp_path, chronicle):
        """Test analyzing non-git directory."""
        report = chronicle.analyze(tmp_path)

        assert report.summary == "No git repository found"
        assert report.total_commits == 0

    def test_analyze_empty_period(self, temp_git_repo_with_activity, chronicle):
        """Test analyzing period with no commits."""
        report = chronicle.analyze(
            temp_git_repo_with_activity,
            since="2030-01-01",  # Future date
            until="2030-12-31",
        )

        assert report.total_commits == 0
        assert report.summary == "No commits found in the specified period"

    def test_parse_time_period(self, chronicle):
        """Test time period parsing."""
        # Test relative times
        test_cases = [
            ("1 day ago", "today"),
            ("2 weeks ago", "1 week ago"),
            ("3 months ago", None),
            ("yesterday", "today"),
            ("last week", "today"),
            ("last month", "last week"),
        ]

        for since, until in test_cases:
            start, end = chronicle._parse_time_period(since, until)
            assert isinstance(start, datetime)
            assert isinstance(end, datetime)
            assert start < end

        # Test ISO format
        start, end = chronicle._parse_time_period("2024-01-01", "2024-12-31")
        assert start.year == 2024
        assert start.month == 1
        assert end.year == 2024
        assert end.month == 12

    def test_parse_relative_time(self, chronicle):
        """Test relative time parsing."""
        base = datetime(2024, 6, 15)

        test_cases = [
            ("5 days ago", 5),
            ("2 weeks ago", 14),
            ("1 month ago", 30),
            ("yesterday", 1),
            ("today", 0),
            ("last week", 7),
            ("last month", 30),
        ]

        for time_str, expected_days in test_cases:
            result = chronicle._parse_relative_time(time_str, base)
            actual_days = (base - result).days
            assert abs(actual_days - expected_days) <= 1  # Allow 1 day tolerance

    def test_analyze_daily_activity(self, chronicle):
        """Test daily activity analysis."""
        commits = [
            CommitSummary(
                sha=f"abc{i}",
                author="Test",
                email="test@example.com",
                date=datetime.now() - timedelta(days=i % 3),
                message=f"Commit {i}",
                lines_added=10 * i,
                lines_removed=5 * i,
            )
            for i in range(10)
        ]

        daily_activity = chronicle._analyze_daily_activity(commits)

        assert len(daily_activity) > 0
        assert all(isinstance(d, DayActivity) for d in daily_activity)

        # Check sorting
        for i in range(len(daily_activity) - 1):
            assert daily_activity[i].date <= daily_activity[i + 1].date

    def test_analyze_contributors(self, chronicle):
        """Test contributor analysis."""
        commits = [
            CommitSummary(
                sha=f"abc{i}",
                author="Alice" if i % 2 == 0 else "Bob",
                email=f"{'alice' if i % 2 == 0 else 'bob'}@example.com",
                date=datetime.now() - timedelta(days=i),
                message=f"{'feat' if i % 3 == 0 else 'fix'}: Commit {i}",
                lines_added=10,
                lines_removed=5,
            )
            for i in range(10)
        ]

        contributor_stats = chronicle._analyze_contributors(commits)

        assert "Alice" in contributor_stats
        assert "Bob" in contributor_stats
        assert contributor_stats["Alice"]["commits"] == 5
        assert contributor_stats["Bob"]["commits"] == 5

    def test_identify_hot_periods(self, chronicle):
        """Test identifying hot periods."""
        # Create activity with varying intensity
        daily_activity = []
        for i in range(10):
            activity = DayActivity(date=datetime.now() - timedelta(days=i))
            activity.total_commits = 20 if i in [2, 3, 4] else 2  # Hot period on days 2-4
            daily_activity.append(activity)

        hot_periods = chronicle._identify_hot_periods(daily_activity)

        assert len(hot_periods) > 0
        assert hot_periods[0]["days"] >= 1
        assert hot_periods[0]["commits"] > 0

    def test_identify_quiet_periods(self, chronicle):
        """Test identifying quiet periods."""
        # Create activity with gaps
        daily_activity = [
            DayActivity(date=datetime.now() - timedelta(days=10)),
            DayActivity(date=datetime.now() - timedelta(days=5)),  # 4-day gap
            DayActivity(date=datetime.now()),
        ]

        quiet_periods = chronicle._identify_quiet_periods(daily_activity)

        assert len(quiet_periods) > 0
        assert quiet_periods[0]["days"] >= 3

    def test_identify_significant_events(self, chronicle):
        """Test identifying significant events."""
        commits = [
            CommitSummary(
                sha="abc123",
                author="Test",
                email="test@example.com",
                date=datetime.now(),
                message="Initial commit",
                files_changed=100,
                lines_added=5000,
                tags=["v1.0.0"],
            ),
            CommitSummary(
                sha="def456",
                author="Test",
                email="test@example.com",
                date=datetime.now() - timedelta(days=1),
                message="Minor change",
                files_changed=1,
                lines_added=5,
            ),
        ]

        events = chronicle._identify_significant_events(commits)

        assert len(events) > 0
        # Should identify release and major change
        event_types = [e["type"] for e in events]
        assert (
            "release" in event_types
            or "major_change" in event_types
            or "initial_commit" in event_types
        )

    def test_identify_trends(self, chronicle):
        """Test trend identification."""
        report = ChronicleReport(
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
        )

        # Create increasing activity pattern
        for i in range(14):
            day = DayActivity(date=datetime.now() - timedelta(days=i))
            day.total_commits = 1 if i > 7 else 5  # More commits in recent week
            report.daily_activity.append(day)

        # Set contributor stats
        report.contributor_stats = {
            "Alice": {"last_commit": datetime.now()},
            "Bob": {"last_commit": datetime.now() - timedelta(days=60)},  # Inactive
        }

        # Set commit types
        report.commit_type_distribution = {"fix": 60, "feat": 20, "docs": 20}

        trends = chronicle._identify_trends(report)

        assert len(trends) > 0
        # Should identify some trends based on the data

    def test_generate_summary(self, chronicle):
        """Test summary generation."""
        report = ChronicleReport(
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            total_commits=100,
            total_contributors=5,
        )

        report.contributor_stats = {
            "Alice": {"commits": 50},
            "Bob": {"commits": 30},
        }

        report.commit_type_distribution = {"feat": 40, "fix": 30}

        summary = chronicle._generate_summary(report)

        assert "100 commits" in summary
        assert "5 contributors" in summary
        assert "Alice" in summary


@pytest.mark.skipif(not _HAS_GIT, reason="Git not available")
class TestCreateChronicleFunction:
    """Test the create_chronicle convenience function."""

    def test_create_chronicle_basic(self, temp_git_repo_with_activity):
        """Test basic chronicle creation."""
        report = create_chronicle(temp_git_repo_with_activity)

        assert isinstance(report, ChronicleReport)
        assert report.total_commits > 0

    def test_create_chronicle_with_options(self, temp_git_repo_with_activity):
        """Test chronicle with options."""
        report = create_chronicle(
            temp_git_repo_with_activity,
            since="1 week ago",
            include_stats=False,
            max_commits=5,
        )

        assert report.total_commits <= 5

    def test_create_chronicle_with_config(self, temp_git_repo_with_activity, config):
        """Test with custom config."""
        report = create_chronicle(temp_git_repo_with_activity, config=config)

        assert isinstance(report, ChronicleReport)
