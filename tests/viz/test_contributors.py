"""Tests for contributors visualization module."""

from unittest.mock import Mock, patch

import pytest

from tenets.viz.contributors import ContributorVisualizer


@pytest.fixture
def contributor_visualizer():
    """Create ContributorVisualizer instance."""
    return ContributorVisualizer()


@pytest.fixture
def sample_contributor_data():
    """Create sample contributor data."""
    return {
        "total_contributors": 10,
        "active_contributors": 7,
        "bus_factor": 3,
        "avg_commits_per_contributor": 25.5,
        "contributors": [
            {
                "name": "Alice",
                "email": "alice@example.com",
                "commits": 100,
                "lines": 5000,
                "files": 50,
                "last_commit_days_ago": 5,
            },
            {
                "name": "Bob",
                "email": "bob@example.com",
                "commits": 80,
                "lines": 4000,
                "files": 40,
                "last_commit_days_ago": 10,
            },
            {
                "name": "Charlie",
                "email": "charlie@example.com",
                "commits": 60,
                "lines": 3000,
                "files": 30,
                "last_commit_days_ago": 20,
            },
            {
                "name": "David",
                "email": "david@example.com",
                "commits": 40,
                "lines": 2000,
                "files": 20,
                "last_commit_days_ago": 45,
            },
            {
                "name": "Eve",
                "email": "eve@example.com",
                "commits": 20,
                "lines": 1000,
                "files": 10,
                "last_commit_days_ago": 100,
            },
        ],
        "collaboration_matrix": {
            ("alice@example.com", "bob@example.com"): 15,
            ("alice@example.com", "charlie@example.com"): 10,
            ("bob@example.com", "charlie@example.com"): 8,
        },
    }


class TestContributorVisualizer:
    """Test suite for ContributorVisualizer class."""

    def test_initialization(self):
        """Test ContributorVisualizer initialization."""
        viz = ContributorVisualizer()

        assert viz.chart_config is not None
        assert viz.display_config is not None
        assert viz.terminal_display is not None

    def test_create_contribution_chart(self, contributor_visualizer, sample_contributor_data):
        """Test creating contribution chart."""
        chart = contributor_visualizer.create_contribution_chart(
            sample_contributor_data["contributors"], metric="commits"
        )

        assert chart["type"] == "bar"
        assert "Commits by Contributor" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]) == 5
        assert chart["data"]["labels"][0] == "Alice"
        assert chart["data"]["datasets"][0]["data"] == [100, 80, 60, 40, 20]

    def test_create_contribution_chart_lines_metric(
        self, contributor_visualizer, sample_contributor_data
    ):
        """Test contribution chart with lines metric."""
        chart = contributor_visualizer.create_contribution_chart(
            sample_contributor_data["contributors"], metric="lines", limit=3
        )

        assert "Lines Changed by Contributor" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]) == 3
        assert chart["data"]["datasets"][0]["data"] == [5000, 4000, 3000]

    def test_create_contribution_chart_long_names(self, contributor_visualizer):
        """Test truncating long contributor names."""
        contributors = [{"name": "a" * 30, "commits": 100}, {"name": "Short Name", "commits": 50}]

        chart = contributor_visualizer.create_contribution_chart(contributors)

        assert len(chart["data"]["labels"][0]) == 20
        assert "..." in chart["data"]["labels"][0]
        assert chart["data"]["labels"][1] == "Short Name"

    def test_create_activity_timeline(self, contributor_visualizer):
        """Test creating activity timeline."""
        activity_data = [
            {"date": "2024-01-01", "commits": 10, "contributor": "alice@example.com"},
            {"date": "2024-01-01", "commits": 5, "contributor": "bob@example.com"},
            {"date": "2024-01-02", "commits": 8, "contributor": "alice@example.com"},
            {"date": "2024-01-03", "commits": 12, "contributor": "charlie@example.com"},
        ]

        chart = contributor_visualizer.create_activity_timeline(activity_data)

        assert chart["type"] == "line"
        assert "Contributor Activity Over Time" in chart["options"]["plugins"]["title"]["text"]
        assert chart["data"]["labels"] == ["2024-01-01", "2024-01-02", "2024-01-03"]
        assert len(chart["data"]["datasets"]) == 2
        assert chart["data"]["datasets"][0]["label"] == "Commits"
        assert chart["data"]["datasets"][0]["data"] == [15, 8, 12]
        assert chart["data"]["datasets"][1]["label"] == "Active Contributors"
        assert chart["data"]["datasets"][1]["data"] == [2, 1, 1]

    def test_create_collaboration_network(self, contributor_visualizer):
        """Test creating collaboration network."""
        collaboration_data = {
            ("Alice", "Bob"): 10,
            ("Alice", "Charlie"): 8,
            ("Bob", "Charlie"): 5,
            ("David", "Eve"): 1,  # Below threshold
        }

        chart = contributor_visualizer.create_collaboration_network(
            collaboration_data, min_weight=2
        )

        assert chart["type"] == "network"
        assert "Contributor Collaboration Network" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["nodes"]) == 3  # David and Eve excluded
        assert len(chart["data"]["edges"]) == 3

        # Check node properties
        node_ids = [n["id"] for n in chart["data"]["nodes"]]
        assert "Alice" in node_ids
        assert "Bob" in node_ids
        assert "Charlie" in node_ids
        assert "David" not in node_ids

    def test_create_distribution_pie(self, contributor_visualizer, sample_contributor_data):
        """Test creating contribution distribution pie chart."""
        chart = contributor_visualizer.create_distribution_pie(
            sample_contributor_data["contributors"], metric="commits", top_n=3
        )

        assert chart["type"] == "pie"
        assert "Commits Distribution" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]) == 4  # Top 3 + Others
        assert chart["data"]["labels"][:3] == ["Alice", "Bob", "Charlie"]
        assert chart["data"]["labels"][3] == "Others"
        assert chart["data"]["datasets"][0]["data"] == [100, 80, 60, 60]  # 40+20=60

    def test_create_bus_factor_gauge(self, contributor_visualizer):
        """Test creating bus factor gauge."""
        chart = contributor_visualizer.create_bus_factor_gauge(bus_factor=2, total_contributors=10)

        assert chart["type"] == "doughnut"
        assert "Bus Factor: 2" in chart["options"]["plugins"]["title"]["text"]
        assert chart["data"]["datasets"][0]["data"][0] == 20  # (2/10)*100
        assert chart["options"]["circumference"] == 180
        assert chart["options"]["rotation"] == 270

    @patch("tenets.viz.contributors.TerminalDisplay")
    def test_display_terminal(
        self, mock_display_class, contributor_visualizer, sample_contributor_data
    ):
        """Test terminal display of contributor data."""
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        mock_display.colorize.side_effect = lambda text, color: text
        contributor_visualizer.terminal_display = mock_display

        contributor_visualizer.display_terminal(sample_contributor_data, show_details=True)

        # Verify display methods called
        mock_display.display_header.assert_called_once()
        mock_display.display_metrics.assert_called_once()
        mock_display.display_table.assert_called()

        # Check for warnings
        sample_contributor_data["bus_factor"] = 1
        contributor_visualizer.display_terminal(sample_contributor_data)
        mock_display.display_list.assert_called()

    def test_create_retention_chart(self, contributor_visualizer):
        """Test creating retention chart."""
        retention_data = [
            {"period": "2024-01", "active": 10, "new": 2, "left": 1},
            {"period": "2024-02", "active": 11, "new": 3, "left": 2},
            {"period": "2024-03", "active": 12, "new": 2, "left": 1},
        ]

        chart = contributor_visualizer.create_retention_chart(retention_data)

        assert chart["type"] == "line"
        assert "Contributor Retention" in chart["options"]["plugins"]["title"]["text"]
        assert chart["data"]["labels"] == ["2024-01", "2024-02", "2024-03"]
        assert len(chart["data"]["datasets"]) == 3
        assert chart["data"]["datasets"][0]["label"] == "Active"
        assert chart["data"]["datasets"][0]["data"] == [10, 11, 12]
        assert chart["data"]["datasets"][0]["fill"] == True

    def test_get_activity_indicator(self, contributor_visualizer):
        """Test activity indicator generation."""
        # Mock colorize method
        contributor_visualizer.terminal_display.colorize = lambda text, color: f"[{color}]{text}"

        assert "Active" in contributor_visualizer._get_activity_indicator(5)
        assert "Recent" in contributor_visualizer._get_activity_indicator(20)
        assert "Inactive" in contributor_visualizer._get_activity_indicator(60)
        assert "Dormant" in contributor_visualizer._get_activity_indicator(100)

    @patch("tenets.viz.contributors.TerminalDisplay")
    def test_display_collaboration_matrix(self, mock_display_class, contributor_visualizer):
        """Test displaying collaboration matrix."""
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        contributor_visualizer.terminal_display = mock_display

        matrix = {("alice", "bob"): 10, ("alice", "charlie"): 5, ("bob", "charlie"): 8}

        contributor_visualizer._display_collaboration_matrix(matrix)

        # Verify table display called
        mock_display.display_table.assert_called_once()
        call_args = mock_display.display_table.call_args[0]
        headers = call_args[0]
        rows = call_args[1]

        # Check headers include contributors
        assert "" in headers  # First column
        assert any("alice" in h.lower() for h in headers)
