"""Tests for viz package initialization and factory functions."""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from tenets.viz import (
    ChartConfig,
    ChartType,
    ComplexityVisualizer,
    ContributorVisualizer,
    CouplingVisualizer,
    DisplayConfig,
    TerminalDisplay,
    combine_visualizations,
    create_chart,
    create_terminal_display,
    create_visualizer,
    detect_visualization_type,
    export_visualization,
)


class TestFactoryFunctions:
    """Test suite for factory functions."""

    def test_create_visualizer_complexity(self):
        """Test creating complexity visualizer."""
        viz = create_visualizer("complexity")
        assert isinstance(viz, ComplexityVisualizer)

    def test_create_visualizer_contributors(self):
        """Test creating contributor visualizer."""
        viz = create_visualizer("contributors")
        assert isinstance(viz, ContributorVisualizer)

    def test_create_visualizer_coupling(self):
        """Test creating coupling visualizer."""
        viz = create_visualizer("coupling")
        assert isinstance(viz, CouplingVisualizer)

    def test_create_visualizer_with_configs(self):
        """Test creating visualizer with custom configs."""
        chart_config = ChartConfig(type=ChartType.LINE, title="Test")
        display_config = DisplayConfig(use_colors=False)

        viz = create_visualizer("complexity", chart_config, display_config)

        assert isinstance(viz, ComplexityVisualizer)
        assert viz.chart_config == chart_config
        assert viz.display_config == display_config

    def test_create_visualizer_invalid_type(self):
        """Test creating visualizer with invalid type."""
        with pytest.raises(ValueError, match="Unknown visualizer type"):
            create_visualizer("invalid")

    def test_create_chart_bar(self):
        """Test creating bar chart."""
        chart = create_chart("bar", {"labels": ["A", "B"], "values": [1, 2]}, title="Test Chart")

        assert chart["type"] == "bar"
        assert chart["options"]["plugins"]["title"]["text"] == "Test Chart"
        assert chart["data"]["labels"] == ["A", "B"]

    def test_create_chart_with_enum(self):
        """Test creating chart with ChartType enum."""
        chart = create_chart(
            ChartType.LINE, {"labels": ["Jan", "Feb"], "datasets": []}, title="Line Chart"
        )

        assert chart["type"] == "line"
        assert chart["options"]["plugins"]["title"]["text"] == "Line Chart"

    def test_create_chart_invalid_type(self):
        """Test creating chart with invalid type."""
        with pytest.raises(ValueError, match="Unknown chart type"):
            create_chart("invalid", {})

    def test_create_terminal_display(self):
        """Test creating terminal display."""
        display = create_terminal_display()
        assert isinstance(display, TerminalDisplay)

    def test_create_terminal_display_with_config(self):
        """Test creating terminal display with config."""
        config = DisplayConfig(use_colors=False, max_width=80)
        display = create_terminal_display(config)

        assert isinstance(display, TerminalDisplay)
        # Config would be used internally


class TestDetectVisualizationType:
    """Test suite for visualization type detection."""

    def test_detect_complexity(self):
        """Test detecting complexity visualization."""
        data = {"complexity": 10, "avg_complexity": 8.5, "complex_functions": []}
        assert detect_visualization_type(data) == "complexity"

    def test_detect_contributors(self):
        """Test detecting contributor visualization."""
        data = {"contributors": [], "total_contributors": 10, "bus_factor": 3}
        assert detect_visualization_type(data) == "contributors"

    def test_detect_hotspots(self):
        """Test detecting hotspot visualization."""
        data = {"hotspots": [], "risk_score": 45.5, "critical_count": 5}
        assert detect_visualization_type(data) == "hotspots"

    def test_detect_momentum(self):
        """Test detecting momentum visualization."""
        data = {"velocity": 25, "sprint": "Sprint 5", "velocity_trend": []}
        assert detect_visualization_type(data) == "momentum"

    def test_detect_dependencies(self):
        """Test detecting dependency visualization."""
        data = {"dependencies": [], "circular_dependencies": [], "dependency_graph": {}}
        assert detect_visualization_type(data) == "dependencies"

    def test_detect_coupling(self):
        """Test detecting coupling visualization."""
        data = {"coupling": 5, "afferent_coupling": 3, "instability": 0.5}
        assert detect_visualization_type(data) == "coupling"

    def test_detect_from_list(self):
        """Test detecting from list data."""
        data = [{"complexity": 10, "cyclomatic": 15}, {"complexity": 8, "cyclomatic": 12}]
        assert detect_visualization_type(data) == "complexity"

        data = [{"author": "Alice", "commits": 100}, {"contributor": "Bob", "commits": 50}]
        assert detect_visualization_type(data) == "contributors"

    def test_detect_custom(self):
        """Test detecting custom/unknown visualization."""
        data = {"unknown": "data", "random": 123}
        assert detect_visualization_type(data) == "custom"

        data = []
        assert detect_visualization_type(data) == "custom"

        data = "string data"
        assert detect_visualization_type(data) == "custom"


class TestExportVisualization:
    """Test suite for visualization export."""

    @patch("builtins.open", new_callable=mock_open)
    def test_export_json(self, mock_file):
        """Test exporting as JSON."""
        visualization = {"type": "bar", "data": {"labels": ["A"], "values": [1]}}

        export_visualization(visualization, "chart.json", format="json")

        mock_file.assert_called_once_with(Path("chart.json"), "w")
        handle = mock_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        # Verify JSON structure
        parsed = json.loads(written_content)
        assert parsed["type"] == "bar"

    @patch("builtins.open", new_callable=mock_open)
    def test_export_html(self, mock_file):
        """Test exporting as HTML."""
        visualization = {"type": "bar", "data": {}}
        config = ChartConfig(type=ChartType.BAR, title="Test Chart")

        export_visualization(visualization, "chart.html", format="html", config=config)

        mock_file.assert_called_once_with(Path("chart.html"), "w")
        handle = mock_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        # Verify HTML structure
        assert "<!DOCTYPE html>" in written_content
        assert "Chart.js" in written_content
        assert "Test Chart" in written_content

    def test_export_unsupported_format(self):
        """Test exporting with unsupported format."""
        visualization = {"type": "bar"}

        with pytest.raises(NotImplementedError, match="SVG export requires"):
            export_visualization(visualization, "chart.svg", format="svg")

        with pytest.raises(ValueError, match="Unsupported export format"):
            export_visualization(visualization, "chart.xyz", format="xyz")


class TestCombineVisualizations:
    """Test suite for combining visualizations."""

    def test_combine_grid_layout(self):
        """Test combining with grid layout."""
        viz1 = {"type": "bar", "data": {}}
        viz2 = {"type": "line", "data": {}}

        dashboard = combine_visualizations([viz1, viz2], layout="grid", title="Dashboard")

        assert dashboard["type"] == "dashboard"
        assert dashboard["title"] == "Dashboard"
        assert dashboard["layout"] == "grid"
        assert len(dashboard["visualizations"]) == 2
        assert dashboard["options"]["responsive"] == True

    def test_combine_vertical_layout(self):
        """Test combining with vertical layout."""
        visualizations = [
            {"type": "bar", "data": {}},
            {"type": "pie", "data": {}},
            {"type": "line", "data": {}},
        ]

        dashboard = combine_visualizations(visualizations, layout="vertical")

        assert dashboard["layout"] == "vertical"
        assert len(dashboard["visualizations"]) == 3

    def test_combine_default_title(self):
        """Test combining with default title."""
        dashboard = combine_visualizations([{"type": "bar"}])

        assert dashboard["title"] == "Dashboard"


class TestPrivateFunctions:
    """Test suite for private helper functions."""

    def test_generate_html_visualization(self):
        """Test HTML generation for single visualization."""
        from tenets.viz import _generate_html_visualization

        visualization = {"type": "bar", "data": {"labels": ["A"], "values": [1]}}
        config = ChartConfig(type=ChartType.BAR, title="Test", width=1000, height=600)

        html = _generate_html_visualization(visualization, config)

        assert "<!DOCTYPE html>" in html
        assert "Test" in html
        assert "1060px" in html  # width + padding
        assert "600px" in html
        assert json.dumps(visualization) in html

    def test_generate_dashboard_html(self):
        """Test HTML generation for dashboard."""
        from tenets.viz import _generate_dashboard_html

        dashboard = {
            "type": "dashboard",
            "title": "My Dashboard",
            "layout": "grid",
            "visualizations": [{"type": "bar", "data": {}}, {"type": "line", "data": {}}],
        }
        config = ChartConfig(type=ChartType.BAR)

        html = _generate_dashboard_html(dashboard, config)

        assert "<!DOCTYPE html>" in html
        assert "My Dashboard" in html
        assert "charts-grid" in html
        assert "chart0" in html
        assert "chart1" in html
        assert "Chart(ctx0" in html
        assert "Chart(ctx1" in html


class TestModuleImports:
    """Test module imports and public API."""

    def test_all_exports(self):
        """Test __all__ exports are available."""
        from tenets import viz

        expected_exports = [
            "BaseVisualizer",
            "ChartConfig",
            "ChartType",
            "ColorPalette",
            "DisplayConfig",
            "DisplayFormat",
            "ComplexityVisualizer",
            "ContributorVisualizer",
            "CouplingVisualizer",
            "TerminalDisplay",
            "ProgressDisplay",
            "create_visualizer",
            "create_chart",
            "create_terminal_display",
            "detect_visualization_type",
            "export_visualization",
            "combine_visualizations",
        ]

        for export in expected_exports:
            assert hasattr(viz, export)

    def test_version(self):
        """Test version is defined."""
        from tenets import viz

        assert hasattr(viz, "__version__")
        assert isinstance(viz.__version__, str)
