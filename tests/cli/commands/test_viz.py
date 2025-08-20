"""Unit tests for the viz CLI command.

Tests cover all visualization functionality including:
- Data loading from JSON/CSV
- Visualization type detection
- Chart generation
- Custom visualizations
- Output formats (terminal, HTML, SVG, PNG)
- Interactive mode
- Field mapping
- Error handling
"""

import csv
import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tenets.cli.commands.viz import viz


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for testing."""
    return {
        "complexity": {
            "avg_complexity": 3.5,
            "max_complexity": 15,
            "complex_items": [
                {"file": "src/core.py", "complexity": 15},
                {"file": "src/api.py", "complexity": 12},
            ],
        },
        "contributors": [
            {"name": "Alice", "commits": 60, "lines": 2000},
            {"name": "Bob", "commits": 45, "lines": 1500},
        ],
    }


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return [
        {"name": "Module A", "score": "85", "category": "core"},
        {"name": "Module B", "score": "72", "category": "api"},
        {"name": "Module C", "score": "90", "category": "core"},
    ]


class TestVizDataLoading:
    """Test data loading functionality."""

    def test_load_json_file(self, runner, sample_json_data, tmp_path):
        """Test loading JSON data file."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file)])

                assert result.exit_code == 0
                assert (
                    "Visualization saved to:" in result.stdout
                    or "Visualization Generated" in result.stdout
                )

    def test_load_csv_file(self, runner, sample_csv_data, tmp_path):
        """Test loading CSV data file."""
        data_file = tmp_path / "data.csv"

        # Write CSV data
        with open(data_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "score", "category"])
            writer.writeheader()
            writer.writerows(sample_csv_data)

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file)])

                assert result.exit_code == 0

    def test_load_unknown_format_as_json(self, runner, sample_json_data, tmp_path):
        """Test loading unknown format attempts JSON first."""
        data_file = tmp_path / "data.txt"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file)])

                assert result.exit_code == 0

    def test_load_file_not_exists(self, runner):
        """Test error when file doesn't exist."""
        result = runner.invoke(viz, ["nonexistent.json"])

        assert result.exit_code != 0
        assert "does not exist" in result.stdout.lower() or "invalid" in result.stdout.lower()


class TestVizTypeDetection:
    """Test visualization type auto-detection."""

    def test_detect_complexity_viz(self, runner, tmp_path):
        """Test auto-detecting complexity visualization."""
        data = {"complexity": {"avg_complexity": 3.5}, "complex_items": []}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.ComplexityVisualizer") as mock_viz:
                mock_viz.return_value.create_distribution_chart.return_value = {}

                result = runner.invoke(viz, [str(data_file), "--type", "auto"])

                assert result.exit_code == 0
                mock_viz.assert_called()

    def test_detect_contributors_viz(self, runner, tmp_path):
        """Test auto-detecting contributors visualization."""
        data = {"contributors": [{"name": "Alice", "commits": 50}]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.ContributorVisualizer") as mock_viz:
                mock_viz.return_value.create_contribution_chart.return_value = {}

                result = runner.invoke(viz, [str(data_file), "--type", "auto"])

                assert result.exit_code == 0
                mock_viz.assert_called()

    def test_detect_hotspots_viz(self, runner, tmp_path):
        """Test auto-detecting hotspots visualization."""
        data = {"hotspots": [{"file": "src/core.py", "risk": 5}]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.HotspotVisualizer") as mock_viz:
                mock_viz.return_value.create_hotspot_bubble.return_value = {}

                result = runner.invoke(viz, [str(data_file), "--type", "auto"])

                assert result.exit_code == 0
                mock_viz.assert_called()

    def test_detect_momentum_viz(self, runner, tmp_path):
        """Test auto-detecting momentum visualization."""
        data = {"velocity": [20, 25, 30], "momentum": {"trend": 15}}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.MomentumVisualizer") as mock_viz:
                mock_viz.return_value.create_velocity_chart.return_value = {}

                result = runner.invoke(viz, [str(data_file), "--type", "auto"])

                assert result.exit_code == 0
                mock_viz.assert_called()


class TestVizSpecificTypes:
    """Test specific visualization types."""

    def test_complexity_visualization(self, runner, sample_json_data, tmp_path):
        """Test complexity visualization."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.ComplexityVisualizer") as mock_viz:
                mock_viz.return_value.create_distribution_chart.return_value = {}
                mock_viz.return_value.display_terminal = MagicMock()

                result = runner.invoke(viz, [str(data_file), "--type", "complexity"])

                assert result.exit_code == 0
                mock_viz.assert_called()
                mock_viz.return_value.display_terminal.assert_called()

    def test_dependencies_visualization(self, runner, tmp_path):
        """Test dependencies visualization."""
        data = {"dependencies": [{"from": "A", "to": "B"}]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.DependencyVisualizer") as mock_viz:
                mock_viz.return_value.create_dependency_graph.return_value = {}
                mock_viz.return_value.display_terminal = MagicMock()

                result = runner.invoke(viz, [str(data_file), "--type", "dependencies"])

                assert result.exit_code == 0
                mock_viz.assert_called()

    def test_coupling_visualization(self, runner, tmp_path):
        """Test coupling visualization."""
        data = {"coupling_data": [{"module": "A", "coupling": 5}]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.CouplingVisualizer") as mock_viz:
                mock_viz.return_value.create_coupling_network.return_value = {}
                mock_viz.return_value.display_terminal = MagicMock()

                result = runner.invoke(viz, [str(data_file), "--type", "coupling"])

                assert result.exit_code == 0
                mock_viz.assert_called()


class TestVizCustomVisualization:
    """Test custom visualization functionality."""

    def test_custom_bar_chart(self, runner, sample_csv_data, tmp_path):
        """Test creating custom bar chart."""
        data_file = tmp_path / "data.csv"

        with open(data_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "score", "category"])
            writer.writeheader()
            writer.writerows(sample_csv_data)

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {}

                result = runner.invoke(
                    viz,
                    [
                        str(data_file),
                        "--type",
                        "custom",
                        "--chart",
                        "bar",
                        "--label-field",
                        "name",
                        "--value-field",
                        "score",
                    ],
                )

                assert result.exit_code == 0
                mock_viz.return_value.create_chart.assert_called()

    def test_custom_line_chart(self, runner, tmp_path):
        """Test creating custom line chart."""
        data = [
            {"date": "2024-01", "value": 100},
            {"date": "2024-02", "value": 120},
            {"date": "2024-03", "value": 110},
        ]
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {}

                result = runner.invoke(
                    viz,
                    [
                        str(data_file),
                        "--type",
                        "custom",
                        "--chart",
                        "line",
                        "--x-field",
                        "date",
                        "--y-field",
                        "value",
                    ],
                )

                assert result.exit_code == 0

    def test_custom_scatter_plot(self, runner, tmp_path):
        """Test creating custom scatter plot."""
        data = [{"x": 10, "y": 20}, {"x": 15, "y": 25}, {"x": 20, "y": 18}]
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {}

                result = runner.invoke(
                    viz,
                    [
                        str(data_file),
                        "--type",
                        "custom",
                        "--chart",
                        "scatter",
                        "--x-field",
                        "x",
                        "--y-field",
                        "y",
                    ],
                )

                assert result.exit_code == 0

    def test_custom_pie_chart(self, runner, sample_csv_data, tmp_path):
        """Test creating custom pie chart."""
        data_file = tmp_path / "data.csv"

        with open(data_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "score", "category"])
            writer.writeheader()
            writer.writerows(sample_csv_data)

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {}

                result = runner.invoke(
                    viz,
                    [
                        str(data_file),
                        "--type",
                        "custom",
                        "--chart",
                        "pie",
                        "--label-field",
                        "name",
                        "--value-field",
                        "score",
                    ],
                )

                assert result.exit_code == 0


class TestVizChartOptions:
    """Test chart configuration options."""

    def test_chart_with_title(self, runner, sample_json_data, tmp_path):
        """Test setting chart title."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file), "--title", "My Custom Chart"])

                assert result.exit_code == 0

    def test_chart_dimensions(self, runner, sample_json_data, tmp_path):
        """Test setting chart dimensions."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file), "--width", "1200", "--height", "600"])

                assert result.exit_code == 0

    def test_data_limit(self, runner, tmp_path):
        """Test limiting data points."""
        data = {"items": [{"value": i} for i in range(100)]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file), "--limit", "10"])

                assert result.exit_code == 0


class TestVizOutputFormats:
    """Test different output formats."""

    def test_terminal_output(self, runner, sample_json_data, tmp_path):
        """Test terminal output format."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.ComplexityVisualizer") as mock_viz:
                mock_viz.return_value.display_terminal = MagicMock()

                result = runner.invoke(viz, [str(data_file), "--format", "terminal"])

                assert result.exit_code == 0
                mock_viz.return_value.display_terminal.assert_called()

    def test_json_output(self, runner, sample_json_data, tmp_path):
        """Test JSON output format."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {"type": "bar", "data": {}}

                result = runner.invoke(viz, [str(data_file), "--format", "json"])

                assert result.exit_code == 0
                # Should output valid JSON
                output_data = json.loads(result.stdout)
                assert "type" in output_data

    def test_json_output_to_file(self, runner, sample_json_data, tmp_path):
        """Test JSON output to file."""
        data_file = tmp_path / "data.json"
        output_file = tmp_path / "viz.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {"type": "bar"}

                result = runner.invoke(
                    viz, [str(data_file), "--format", "json", "--output", str(output_file)]
                )

                assert result.exit_code == 0
                assert output_file.exists()

    def test_html_output(self, runner, sample_json_data, tmp_path):
        """Test HTML output format."""
        data_file = tmp_path / "data.json"
        output_file = tmp_path / "chart.html"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {"type": "bar", "data": {}}

                result = runner.invoke(
                    viz, [str(data_file), "--format", "html", "--output", str(output_file)]
                )

                assert result.exit_code == 0
                assert output_file.exists()

                # Check HTML content
                html_content = output_file.read_text()
                assert "<!DOCTYPE html>" in html_content
                assert "Chart.js" in html_content

    def test_svg_output_not_implemented(self, runner, sample_json_data, tmp_path):
        """Test SVG output (not yet implemented)."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file), "--format", "svg"])

                assert result.exit_code == 0
                assert "SVG export not yet implemented" in result.stdout

    def test_png_output_not_implemented(self, runner, sample_json_data, tmp_path):
        """Test PNG output (not yet implemented)."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file), "--format", "png"])

                assert result.exit_code == 0
                assert "PNG export not yet implemented" in result.stdout


class TestVizInteractiveMode:
    """Test interactive visualization mode."""

    @patch("webbrowser.open")
    @patch("tempfile.NamedTemporaryFile")
    def test_interactive_mode(
        self, mock_tempfile, mock_browser, runner, sample_json_data, tmp_path
    ):
        """Test launching interactive visualization."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        # Mock temp file
        mock_file = MagicMock()
        mock_file.name = "/tmp/viz.html"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {"type": "bar"}

                result = runner.invoke(viz, [str(data_file), "--interactive"])

                assert result.exit_code == 0
                assert "Launching interactive mode" in result.stdout
                assert "Opened in browser" in result.stdout
                mock_browser.assert_called_once()


class TestVizErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_file(self, runner, tmp_path):
        """Test error with invalid JSON."""
        data_file = tmp_path / "invalid.json"
        data_file.write_text("not valid json{")

        with patch("tenets.cli.commands.viz.get_logger"):
            result = runner.invoke(viz, [str(data_file)])

            assert result.exit_code != 0
            assert "Visualization failed" in result.stdout

    def test_missing_required_fields(self, runner, tmp_path):
        """Test error when required fields are missing."""
        data = [{"name": "A"}, {"name": "B"}]  # Missing value field
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                # Simulate error when value field is missing
                mock_viz.return_value.create_chart.side_effect = KeyError("value")

                result = runner.invoke(
                    viz,
                    [
                        str(data_file),
                        "--type",
                        "custom",
                        "--chart",
                        "bar",
                        "--label-field",
                        "name",
                        "--value-field",
                        "value",  # This field doesn't exist
                    ],
                )

                assert result.exit_code != 0

    def test_empty_data_file(self, runner, tmp_path):
        """Test handling empty data file."""
        data_file = tmp_path / "empty.json"
        data_file.write_text("[]")

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = runner.invoke(viz, [str(data_file)])

                # Should handle gracefully
                assert result.exit_code == 0


class TestVizSummaryOutput:
    """Test summary output for visualizations."""

    def test_visualization_summary(self, runner, sample_json_data, tmp_path):
        """Test visualization summary output."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {
                    "type": "bar",
                    "data": {"datasets": [{"data": [1, 2, 3]}]},
                }

                result = runner.invoke(viz, [str(data_file), "--type", "custom"])

                assert result.exit_code == 0
                assert "Custom Visualization Generated" in result.stdout
                assert "Type:" in result.stdout
                assert "Datasets:" in result.stdout
