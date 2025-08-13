"""Tests for base visualization module."""

import json
from unittest.mock import mock_open, patch

import pytest

from tenets.viz.base import (
    BaseVisualizer,
    ChartConfig,
    ChartType,
    ColorPalette,
    DisplayConfig,
    DisplayFormat,
)


@pytest.fixture
def chart_config():
    """Create test chart configuration."""
    return ChartConfig(type=ChartType.BAR, title="Test Chart", width=800, height=400)


@pytest.fixture
def display_config():
    """Create test display configuration."""
    return DisplayConfig(use_colors=True, use_unicode=True, max_width=120)


@pytest.fixture
def base_visualizer(chart_config, display_config):
    """Create BaseVisualizer instance."""
    return BaseVisualizer(chart_config, display_config)


class TestChartType:
    """Test suite for ChartType enum."""

    def test_chart_types(self):
        """Test available chart types."""
        assert ChartType.BAR.value == "bar"
        assert ChartType.LINE.value == "line"
        assert ChartType.PIE.value == "pie"
        assert ChartType.SCATTER.value == "scatter"
        assert ChartType.RADAR.value == "radar"
        assert ChartType.GAUGE.value == "gauge"
        assert ChartType.HEATMAP.value == "heatmap"
        assert ChartType.TREEMAP.value == "treemap"
        assert ChartType.NETWORK.value == "network"
        assert ChartType.BUBBLE.value == "bubble"


class TestDisplayFormat:
    """Test suite for DisplayFormat enum."""

    def test_display_formats(self):
        """Test available display formats."""
        assert DisplayFormat.TERMINAL.value == "terminal"
        assert DisplayFormat.HTML.value == "html"
        assert DisplayFormat.JSON.value == "json"
        assert DisplayFormat.MARKDOWN.value == "markdown"
        assert DisplayFormat.SVG.value == "svg"
        assert DisplayFormat.PNG.value == "png"


class TestChartConfig:
    """Test suite for ChartConfig dataclass."""

    def test_chart_config_creation(self):
        """Test creating ChartConfig."""
        config = ChartConfig(type=ChartType.LINE, title="Line Chart", width=1000, height=600)

        assert config.type == ChartType.LINE
        assert config.title == "Line Chart"
        assert config.width == 1000
        assert config.height == 600
        assert config.interactive == True
        assert config.show_legend == True

    def test_chart_config_defaults(self):
        """Test ChartConfig default values."""
        config = ChartConfig(type=ChartType.PIE)

        assert config.title == ""
        assert config.width == 800
        assert config.height == 400
        assert config.theme == "light"
        assert config.animation == True
        assert config.responsive == True
        assert "png" in config.export_options
        assert "svg" in config.export_options

    def test_chart_config_custom_colors(self):
        """Test custom color configuration."""
        colors = ["#ff0000", "#00ff00", "#0000ff"]
        config = ChartConfig(type=ChartType.BAR, colors=colors)

        assert config.colors == colors


class TestDisplayConfig:
    """Test suite for DisplayConfig dataclass."""

    def test_display_config_creation(self):
        """Test creating DisplayConfig."""
        config = DisplayConfig(use_colors=False, use_unicode=False, max_width=80, max_rows=30)

        assert config.use_colors == False
        assert config.use_unicode == False
        assert config.max_width == 80
        assert config.max_rows == 30

    def test_display_config_defaults(self):
        """Test DisplayConfig default values."""
        config = DisplayConfig()

        assert config.use_colors == True
        assert config.use_unicode == True
        assert config.max_width == 120
        assert config.max_rows == 50
        assert config.truncate == True
        assert config.show_progress == True
        assert config.style == "detailed"


class TestColorPalette:
    """Test suite for ColorPalette class."""

    def test_default_palette(self):
        """Test default color palette."""
        palette = ColorPalette.get_palette("default")

        assert isinstance(palette, list)
        assert len(palette) == 10
        assert all(color.startswith("#") for color in palette)

    def test_severity_palette(self):
        """Test severity color palette."""
        assert ColorPalette.SEVERITY["critical"] == "#dc2626"
        assert ColorPalette.SEVERITY["high"] == "#ea580c"
        assert ColorPalette.SEVERITY["medium"] == "#ca8a04"
        assert ColorPalette.SEVERITY["low"] == "#16a34a"
        assert ColorPalette.SEVERITY["info"] == "#0891b2"

    def test_health_palette(self):
        """Test health color palette."""
        assert ColorPalette.HEALTH["excellent"] == "#10b981"
        assert ColorPalette.HEALTH["good"] == "#84cc16"
        assert ColorPalette.HEALTH["fair"] == "#f59e0b"
        assert ColorPalette.HEALTH["poor"] == "#f97316"
        assert ColorPalette.HEALTH["critical"] == "#ef4444"

    def test_monochrome_palette(self):
        """Test monochrome color palette."""
        palette = ColorPalette.get_palette("monochrome")

        assert isinstance(palette, list)
        assert len(palette) == 8
        assert all(color.startswith("#") for color in palette)

    def test_get_palette(self):
        """Test getting palettes by name."""
        default = ColorPalette.get_palette("default")
        monochrome = ColorPalette.get_palette("monochrome")
        severity = ColorPalette.get_palette("severity")
        health = ColorPalette.get_palette("health")

        assert default == ColorPalette.DEFAULT
        assert monochrome == ColorPalette.MONOCHROME
        assert len(severity) == len(ColorPalette.SEVERITY)
        assert len(health) == len(ColorPalette.HEALTH)

        # Unknown palette returns default
        unknown = ColorPalette.get_palette("unknown")
        assert unknown == ColorPalette.DEFAULT

    def test_get_color(self):
        """Test getting color for value."""
        # Severity colors
        assert ColorPalette.get_color("critical", "severity") == "#dc2626"
        assert ColorPalette.get_color("high", "severity") == "#ea580c"

        # Health colors
        assert ColorPalette.get_color("excellent", "health") == "#10b981"
        assert ColorPalette.get_color("poor", "health") == "#f97316"

        # Default colors with cycling
        assert ColorPalette.get_color(0, "default") == ColorPalette.DEFAULT[0]
        assert ColorPalette.get_color(10, "default") == ColorPalette.DEFAULT[0]  # Cycles
        assert ColorPalette.get_color(5, "default") == ColorPalette.DEFAULT[5]

        # Unknown category returns first default
        assert ColorPalette.get_color("test", "unknown") == ColorPalette.DEFAULT[0]

    def test_interpolate_color(self):
        """Test color interpolation."""
        # Test midpoint
        color = ColorPalette.interpolate_color(50, 0, 100)
        assert color.startswith("#")
        assert len(color) == 7

        # Test extremes
        start_color = ColorPalette.interpolate_color(0, 0, 100)
        end_color = ColorPalette.interpolate_color(100, 0, 100)

        # Test clamping
        below_min = ColorPalette.interpolate_color(-10, 0, 100)
        above_max = ColorPalette.interpolate_color(110, 0, 100)
        assert below_min == start_color
        assert above_max == end_color

        # Test equal min/max
        equal = ColorPalette.interpolate_color(50, 50, 50)
        assert equal.startswith("#")

    def test_hex_to_rgb(self):
        """Test hex to RGB conversion."""
        rgb = ColorPalette._hex_to_rgb("#ff0000")
        assert rgb == (255, 0, 0)

        rgb = ColorPalette._hex_to_rgb("#00ff00")
        assert rgb == (0, 255, 0)

        rgb = ColorPalette._hex_to_rgb("#0000ff")
        assert rgb == (0, 0, 255)

        # Test without hash
        rgb = ColorPalette._hex_to_rgb("ffffff")
        assert rgb == (255, 255, 255)


class TestBaseVisualizer:
    """Test suite for BaseVisualizer class."""

    def test_initialization(self, chart_config, display_config):
        """Test BaseVisualizer initialization."""
        viz = BaseVisualizer(chart_config, display_config)

        assert viz.chart_config == chart_config
        assert viz.display_config == display_config
        assert isinstance(viz.color_palette, list)
        assert len(viz.color_palette) > 0

    def test_initialization_defaults(self):
        """Test BaseVisualizer with default configs."""
        viz = BaseVisualizer()

        assert viz.chart_config.type == ChartType.BAR
        assert viz.display_config.use_colors == True
        assert viz.color_palette == ColorPalette.DEFAULT

    def test_create_bar_chart(self, base_visualizer):
        """Test creating bar chart."""
        data = {"labels": ["A", "B", "C"], "values": [10, 20, 30]}

        chart = base_visualizer.create_chart(ChartType.BAR, data)

        assert chart["type"] == "bar"
        assert chart["data"]["labels"] == ["A", "B", "C"]
        assert chart["data"]["datasets"][0]["data"] == [10, 20, 30]
        assert "backgroundColor" in chart["data"]["datasets"][0]

    def test_create_line_chart(self, base_visualizer):
        """Test creating line chart."""
        data = {
            "labels": ["Jan", "Feb", "Mar"],
            "datasets": [
                {"label": "Series 1", "data": [10, 15, 20]},
                {"label": "Series 2", "data": [5, 10, 15]},
            ],
        }

        chart = base_visualizer.create_chart(ChartType.LINE, data)

        assert chart["type"] == "line"
        assert chart["data"]["labels"] == ["Jan", "Feb", "Mar"]
        assert len(chart["data"]["datasets"]) == 2
        assert chart["data"]["datasets"][0]["label"] == "Series 1"
        assert chart["data"]["datasets"][0]["data"] == [10, 15, 20]

    def test_create_pie_chart(self, base_visualizer):
        """Test creating pie chart."""
        data = {"labels": ["Red", "Blue", "Green"], "values": [30, 40, 30]}

        chart = base_visualizer.create_chart(ChartType.PIE, data)

        assert chart["type"] == "pie"
        assert chart["data"]["labels"] == ["Red", "Blue", "Green"]
        assert chart["data"]["datasets"][0]["data"] == [30, 40, 30]
        assert "backgroundColor" in chart["data"]["datasets"][0]

    def test_create_scatter_chart(self, base_visualizer):
        """Test creating scatter chart."""
        data = {"points": [(1, 2), (3, 4), (5, 6)]}

        chart = base_visualizer.create_chart(ChartType.SCATTER, data)

        assert chart["type"] == "scatter"
        assert len(chart["data"]["datasets"][0]["data"]) == 3
        assert chart["data"]["datasets"][0]["data"][0] == {"x": 1, "y": 2}
        assert chart["data"]["datasets"][0]["data"][1] == {"x": 3, "y": 4}

    def test_create_radar_chart(self, base_visualizer):
        """Test creating radar chart."""
        data = {
            "labels": ["Speed", "Power", "Defense"],
            "datasets": [{"label": "Player 1", "data": [80, 70, 90]}],
        }

        chart = base_visualizer.create_chart(ChartType.RADAR, data)

        assert chart["type"] == "radar"
        assert chart["data"]["labels"] == ["Speed", "Power", "Defense"]
        assert chart["data"]["datasets"][0]["label"] == "Player 1"
        assert chart["data"]["datasets"][0]["data"] == [80, 70, 90]

    def test_create_gauge_chart(self, base_visualizer):
        """Test creating gauge chart."""
        data = {"value": 75, "max": 100}

        chart = base_visualizer.create_chart(ChartType.GAUGE, data)

        assert chart["type"] == "doughnut"
        assert chart["data"]["datasets"][0]["data"] == [75, 25]
        assert chart["options"]["circumference"] == 180
        assert chart["options"]["rotation"] == 270

    def test_create_heatmap(self, base_visualizer):
        """Test creating heatmap."""
        data = {
            "matrix": [[1, 2, 3], [4, 5, 6]],
            "x_labels": ["A", "B", "C"],
            "y_labels": ["X", "Y"],
        }

        chart = base_visualizer.create_chart(ChartType.HEATMAP, data)

        assert chart["type"] == "heatmap"
        assert chart["data"]["labels"]["x"] == ["A", "B", "C"]
        assert chart["data"]["labels"]["y"] == ["X", "Y"]
        assert len(chart["data"]["datasets"][0]["data"]) == 6

    def test_create_treemap(self, base_visualizer):
        """Test creating treemap."""
        data = {
            "tree": [
                {
                    "name": "Root",
                    "value": 100,
                    "children": [{"name": "Child1", "value": 40}, {"name": "Child2", "value": 60}],
                }
            ]
        }

        chart = base_visualizer.create_chart(ChartType.TREEMAP, data)

        assert chart["type"] == "treemap"
        assert chart["data"]["datasets"][0]["tree"] == data["tree"]

    def test_create_network_graph(self, base_visualizer):
        """Test creating network graph."""
        data = {
            "nodes": [{"id": "A", "label": "Node A"}, {"id": "B", "label": "Node B"}],
            "edges": [{"source": "A", "target": "B", "weight": 1}],
        }

        chart = base_visualizer.create_chart(ChartType.NETWORK, data)

        assert chart["type"] == "network"
        assert chart["data"]["nodes"] == data["nodes"]
        assert chart["data"]["edges"] == data["edges"]
        assert chart["options"]["layout"] == "force"

    def test_create_bubble_chart(self, base_visualizer):
        """Test creating bubble chart."""
        data = {"points": [(1, 2, 5), (3, 4, 10), (5, 6, 15)]}

        chart = base_visualizer.create_chart(ChartType.BUBBLE, data)

        assert chart["type"] == "bubble"
        assert len(chart["data"]["datasets"][0]["data"]) == 3
        assert chart["data"]["datasets"][0]["data"][0] == {"x": 1, "y": 2, "r": 5}

    def test_format_number(self, base_visualizer):
        """Test number formatting."""
        # Basic formatting
        assert base_visualizer.format_number(1234.567) == "1,234.57"
        assert base_visualizer.format_number(1234.567, precision=1) == "1,234.6"
        assert base_visualizer.format_number(1234) == "1,234"

        # Without thousands separator
        assert base_visualizer.format_number(1234.567, use_thousands=False) == "1234.57"

        # Small numbers
        assert base_visualizer.format_number(123.456) == "123.46"
        assert base_visualizer.format_number(0.123) == "0.12"

        # Negative numbers
        assert base_visualizer.format_number(-1234.567) == "-1,234.57"

    def test_format_percentage(self, base_visualizer):
        """Test percentage formatting."""
        # 0-1 range
        assert base_visualizer.format_percentage(0.75) == "75.0%"
        assert base_visualizer.format_percentage(0.125, precision=2) == "12.50%"

        # 0-100 range
        assert base_visualizer.format_percentage(75) == "75.0%"
        assert base_visualizer.format_percentage(12.5, precision=2) == "12.50%"

        # With sign
        assert base_visualizer.format_percentage(0.25, include_sign=True) == "+25.0%"
        assert base_visualizer.format_percentage(-0.25, include_sign=True) == "-25.0%"

        # Edge cases
        assert base_visualizer.format_percentage(0) == "0.0%"
        assert base_visualizer.format_percentage(1) == "100.0%"
        assert base_visualizer.format_percentage(-0.5) == "-50.0%"

    @patch("builtins.open", new_callable=mock_open)
    def test_export_chart_json(self, mock_file, base_visualizer, tmp_path):
        """Test exporting chart as JSON."""
        chart_config = {"type": "bar", "data": {"labels": ["A"], "values": [1]}}
        output_path = tmp_path / "chart.json"

        result = base_visualizer.export_chart(chart_config, output_path, "json")

        assert result == output_path
        mock_file.assert_called_once_with(output_path, "w")
        handle = mock_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        # Verify JSON structure
        parsed = json.loads(written_content)
        assert parsed["type"] == "bar"

    @patch("builtins.open", new_callable=mock_open)
    def test_export_chart_html(self, mock_file, base_visualizer, tmp_path):
        """Test exporting chart as HTML."""
        chart_config = {"type": "bar", "data": {"labels": ["A"], "values": [1]}}
        output_path = tmp_path / "chart.html"

        result = base_visualizer.export_chart(chart_config, output_path, "html")

        assert result == output_path
        mock_file.assert_called_once_with(output_path, "w")
        handle = mock_file()
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)

        # Verify HTML structure
        assert "<!DOCTYPE html>" in written_content
        assert '<canvas id="chart">' in written_content
        assert "Chart.js" in written_content

    def test_export_chart_unsupported(self, base_visualizer, tmp_path):
        """Test exporting chart with unsupported format."""
        chart_config = {"type": "bar"}
        output_path = tmp_path / "chart.pdf"

        with pytest.raises(ValueError, match="Unsupported export format"):
            base_visualizer.export_chart(chart_config, output_path, "pdf")

    def test_get_chart_options(self, base_visualizer, chart_config):
        """Test chart options generation."""
        options = base_visualizer._get_chart_options(chart_config)

        assert options["responsive"] == True
        assert options["maintainAspectRatio"] == False
        assert options["animation"]["duration"] == 1000
        assert options["plugins"]["title"]["display"] == True
        assert options["plugins"]["title"]["text"] == "Test Chart"
        assert options["plugins"]["legend"]["display"] == True

        # Test with animation disabled
        chart_config.animation = False
        options = base_visualizer._get_chart_options(chart_config)
        assert options["animation"]["duration"] == 0

    def test_generate_standalone_html(self, base_visualizer):
        """Test standalone HTML generation."""
        chart_config = {"type": "bar", "data": {}}
        html = base_visualizer._generate_standalone_html(chart_config)

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert '<canvas id="chart">' in html
        assert "new Chart(" in html
        assert json.dumps(chart_config) in html
