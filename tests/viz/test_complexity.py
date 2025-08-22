"""Tests for complexity visualization module."""

from unittest.mock import Mock, patch

import pytest

from tenets.viz.base import ChartConfig, ChartType, DisplayConfig
from tenets.viz.complexity import ComplexityVisualizer


@pytest.fixture
def complexity_visualizer():
    """Create ComplexityVisualizer instance."""
    return ComplexityVisualizer()


@pytest.fixture
def sample_complexity_data():
    """Create sample complexity data."""
    return {
        "avg_complexity": 8.5,
        "max_complexity": 35,
        "complex_functions": 15,
        "total_functions": 100,
        "low_complexity_count": 60,
        "medium_complexity_count": 25,
        "high_complexity_count": 10,
        "very_high_complexity_count": 5,
        "distribution": {"low": 60, "medium": 25, "high": 10, "very_high": 5},
        "complex_items": [
            {"name": "complex_function_1", "file": "src/module1.py", "complexity": 35},
            {"name": "complex_function_2", "file": "src/module2.py", "complexity": 28},
            {"name": "complex_function_3", "file": "src/module3.py", "complexity": 22},
            {"name": "complex_function_4", "file": "src/module4.py", "complexity": 18},
            {"name": "complex_function_5", "file": "src/module5.py", "complexity": 15},
        ],
        "recommendations": [
            "Refactor functions with complexity > 20",
            "Add unit tests for complex functions",
            "Consider breaking down large modules",
        ],
    }


class TestComplexityVisualizer:
    """Test suite for ComplexityVisualizer class."""

    def test_initialization(self):
        """Test ComplexityVisualizer initialization."""
        viz = ComplexityVisualizer()

        assert viz.chart_config is not None
        assert viz.display_config is not None
        assert viz.terminal_display is not None

    def test_initialization_with_configs(self):
        """Test initialization with custom configs."""
        chart_config = ChartConfig(type=ChartType.LINE, title="Test")
        display_config = DisplayConfig(use_colors=False)

        viz = ComplexityVisualizer(chart_config, display_config)

        assert viz.chart_config == chart_config
        assert viz.display_config == display_config

    def test_create_distribution_chart(self, complexity_visualizer, sample_complexity_data):
        """Test creating complexity distribution chart."""
        chart = complexity_visualizer.create_distribution_chart(sample_complexity_data)

        assert chart["type"] == "bar"
        assert "Complexity Distribution" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]) == 4
        assert chart["data"]["labels"][0] == "Low (1-5)"
        assert chart["data"]["datasets"][0]["data"] == [60, 25, 10, 5]

    def test_create_distribution_chart_pie(self, complexity_visualizer, sample_complexity_data):
        """Test creating distribution chart as pie chart."""
        chart = complexity_visualizer.create_distribution_chart(
            sample_complexity_data, chart_type=ChartType.PIE
        )

        assert chart["type"] == "pie"
        assert len(chart["data"]["labels"]) == 4

    def test_create_distribution_chart_no_data(self, complexity_visualizer):
        """Test distribution chart with no distribution data."""
        data = {
            "low_complexity_count": 10,
            "medium_complexity_count": 5,
            "high_complexity_count": 3,
            "very_high_complexity_count": 2,
        }

        chart = complexity_visualizer.create_distribution_chart(data)

        assert chart["data"]["datasets"][0]["data"] == [10, 5, 3, 2]

    def test_create_distribution_chart_alternate_keys(self, complexity_visualizer):
        """Test distribution chart with alternate key formats."""
        data = {
            "complexity_distribution": {
                "simple (1-5)": 7,
                "moderate (6-10)": 3,
                "complex (11-20)": 2,
                "very complex (21+)": 1,
            }
        }

        chart = complexity_visualizer.create_distribution_chart(data)

        assert chart["data"]["datasets"][0]["data"] == [7, 3, 2, 1]
        assert chart["data"]["labels"][0] == "Low (1-5)"

    def test_create_top_complex_chart(self, complexity_visualizer, sample_complexity_data):
        """Test creating top complex items chart."""
        chart = complexity_visualizer.create_top_complex_chart(
            sample_complexity_data["complex_items"]
        )

        assert chart["type"] == "horizontal_bar"
        assert "Top 5 Most Complex Functions" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]) == 5
        assert chart["data"]["labels"][0] == "complex_function_1"
        assert chart["data"]["datasets"][0]["data"] == [35, 28, 22, 18, 15]

    def test_create_top_complex_chart_limit(self, complexity_visualizer):
        """Test limiting top complex items."""
        items = [{"name": f"func_{i}", "complexity": 30 - i} for i in range(20)]

        chart = complexity_visualizer.create_top_complex_chart(items, limit=5)

        assert len(chart["data"]["labels"]) == 5
        assert chart["data"]["datasets"][0]["data"][0] == 30

    def test_create_top_complex_chart_long_names(self, complexity_visualizer):
        """Test truncating long function names."""
        items = [{"name": "a" * 40, "complexity": 25}, {"name": "short", "complexity": 20}]

        chart = complexity_visualizer.create_top_complex_chart(items)

        assert len(chart["data"]["labels"][0]) == 30  # Truncated
        assert "..." in chart["data"]["labels"][0]
        assert chart["data"]["labels"][1] == "short"

    def test_create_complexity_heatmap(self, complexity_visualizer):
        """Test creating complexity heatmap."""
        file_complexities = {
            "src/file1.py": [5, 10, 15, 20, 25],
            "src/file2.py": [3, 6, 9, 12],
            "src/very_long_filename_that_needs_truncation.py": [1, 2, 3],
        }

        chart = complexity_visualizer.create_complexity_heatmap(file_complexities, max_functions=10)

        assert chart["type"] == "heatmap"
        assert "Complexity Heatmap" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]["y"]) == 3
        assert len(chart["data"]["labels"]["x"]) == 10
        assert chart["data"]["labels"]["x"][0] == "F1"

    def test_create_trend_chart(self, complexity_visualizer):
        """Test creating complexity trend chart."""
        trend_data = [
            {
                "date": "2024-01",
                "avg_complexity": 7.5,
                "max_complexity": 25,
                "complex_functions": 10,
            },
            {
                "date": "2024-02",
                "avg_complexity": 8.0,
                "max_complexity": 28,
                "complex_functions": 12,
            },
            {
                "date": "2024-03",
                "avg_complexity": 7.8,
                "max_complexity": 30,
                "complex_functions": 11,
            },
        ]

        chart = complexity_visualizer.create_trend_chart(trend_data)

        assert chart["type"] == "line"
        assert "Complexity Trend Over Time" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]) == 3
        assert chart["data"]["labels"] == ["2024-01", "2024-02", "2024-03"]
        assert len(chart["data"]["datasets"]) == 2
        assert chart["data"]["datasets"][0]["label"] == "Average Complexity"
        assert chart["data"]["datasets"][0]["data"] == [7.5, 8.0, 7.8]

    def test_create_trend_chart_empty(self, complexity_visualizer):
        """Test trend chart with empty data."""
        chart = complexity_visualizer.create_trend_chart([])

        assert chart == {}

    def test_create_comparison_chart(self, complexity_visualizer):
        """Test creating comparison chart."""
        current = {"avg_complexity": 8.5, "max_complexity": 30, "complex_functions": 15}
        baseline = {"avg_complexity": 7.0, "max_complexity": 25, "complex_functions": 10}

        chart = complexity_visualizer.create_comparison_chart(current, baseline)

        assert chart["type"] == "bar"
        assert "Complexity Comparison" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]) == 3
        assert chart["data"]["labels"] == ["Average", "Maximum", "Complex Count"]
        assert chart["data"]["datasets"][0]["label"] == "Current"
        assert chart["data"]["datasets"][0]["data"] == [8.5, 30, 15]
        assert chart["data"]["datasets"][1]["label"] == "Baseline"
        assert chart["data"]["datasets"][1]["data"] == [7.0, 25, 10]

    @patch("tenets.viz.complexity.TerminalDisplay")
    def test_display_terminal(
        self, mock_display_class, complexity_visualizer, sample_complexity_data
    ):
        """Test terminal display of complexity data."""
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        complexity_visualizer.terminal_display = mock_display

        complexity_visualizer.display_terminal(sample_complexity_data, show_details=True)

        # Verify display methods called
        mock_display.display_header.assert_called_once()
        mock_display.display_metrics.assert_called_once()
        mock_display.display_distribution.assert_called_once()
        mock_display.display_table.assert_called_once()
        mock_display.display_list.assert_called_once()

    @patch("tenets.viz.complexity.TerminalDisplay")
    def test_display_terminal_no_details(
        self, mock_display_class, complexity_visualizer, sample_complexity_data
    ):
        """Test terminal display without details."""
        mock_display = Mock()
        mock_display_class.return_value = mock_display
        complexity_visualizer.terminal_display = mock_display

        complexity_visualizer.display_terminal(sample_complexity_data, show_details=False)

        # Table should not be displayed
        mock_display.display_table.assert_not_called()

    def test_create_radar_chart(self, complexity_visualizer):
        """Test creating radar chart for metrics."""
        metrics = {
            "cyclomatic": 25,
            "cognitive": 40,
            "halstead": 500,
            "maintainability": 65,
            "lines": 250,
        }

        chart = complexity_visualizer.create_radar_chart(metrics)

        assert chart["type"] == "radar"
        assert "Complexity Metrics Radar" in chart["options"]["plugins"]["title"]["text"]
        assert len(chart["data"]["labels"]) == 5
        assert chart["data"]["labels"][0] == "Cyclomatic"
        assert len(chart["data"]["datasets"]) == 1
        assert chart["data"]["datasets"][0]["label"] == "Current"

    def test_get_risk_level(self, complexity_visualizer):
        """Test risk level determination."""
        assert complexity_visualizer._get_risk_level(3) == "Low"
        assert complexity_visualizer._get_risk_level(8) == "Medium"
        assert complexity_visualizer._get_risk_level(15) == "High"
        assert complexity_visualizer._get_risk_level(25) == "Critical"

    def test_get_risk_color(self, complexity_visualizer):
        """Test risk color mapping."""
        assert complexity_visualizer._get_risk_color("Critical") == "red"
        assert complexity_visualizer._get_risk_color("High") == "yellow"
        assert complexity_visualizer._get_risk_color("Medium") == "cyan"
        assert complexity_visualizer._get_risk_color("Low") == "green"
        assert complexity_visualizer._get_risk_color("Unknown") == "white"

    def test_truncate_path(self, complexity_visualizer):
        """Test path truncation."""
        # Short path - no truncation
        short_path = "src/file.py"
        assert complexity_visualizer._truncate_path(short_path) == short_path

        # Long path - truncation with filename preserved
        long_path = "very/long/path/to/deeply/nested/module/file.py"
        truncated = complexity_visualizer._truncate_path(long_path, max_length=30)
        assert len(truncated) <= 30
        assert "..." in truncated
        assert "file.py" in truncated

        # Very long filename
        long_filename = "a" * 50 + ".py"
        truncated = complexity_visualizer._truncate_path(long_filename, max_length=20)
        assert len(truncated) == 20
        assert truncated.endswith("...")
