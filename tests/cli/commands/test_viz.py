"""Unit tests for the viz CLI command.

Tests cover all visualization functionality including:
- Dependency graph generation
- Project type detection  
- Multiple output formats (ASCII, SVG, PNG, HTML, JSON, DOT)
- Clustering and aggregation
- Complexity visualization
- Contributor visualization
- Error handling
"""

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest
from typer.testing import CliRunner

from click.testing import CliRunner as ClickRunner
from tenets.cli.app import app
from tenets.cli.commands.viz import viz_app, viz, aggregate_dependencies, get_aggregate_key


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()

@pytest.fixture
def click_runner():
    """Create a Click CLI test runner for the viz command."""
    return ClickRunner()


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
def sample_dependency_graph():
    """Create sample dependency graph for testing."""
    return {
        "src/main.py": ["src/utils.py", "src/config.py"],
        "src/utils.py": ["src/config.py"],
        "src/api/routes.py": ["src/utils.py", "src/models.py"],
        "tests/test_main.py": ["src/main.py"],
    }


@pytest.fixture
def sample_project_info():
    """Create sample project info for testing."""
    return {
        "type": "python_project",
        "languages": {"python": 85.0, "yaml": 15.0},
        "frameworks": ["flask"],
        "entry_points": ["src/main.py", "setup.py"],
        "structure": {
            "directories": {"src": "Source code", "tests": "Tests"},
            "test_directories": ["tests"],
        }
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

    def test_load_json_file(self, click_runner, sample_json_data, tmp_path):
        """Test loading JSON data file."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file)])

                assert result.exit_code == 0
                assert (
                    "Visualization saved to:" in result.stdout
                    or "Visualization Generated" in result.stdout
                )

    def test_load_csv_file(self, click_runner, sample_csv_data, tmp_path):
        """Test loading CSV data file."""
        data_file = tmp_path / "data.csv"

        # Write CSV data
        with open(data_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "score", "category"])
            writer.writeheader()
            writer.writerows(sample_csv_data)

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file)])

                assert result.exit_code == 0

    def test_load_unknown_format_as_json(self, click_runner, sample_json_data, tmp_path):
        """Test loading unknown format attempts JSON first."""
        data_file = tmp_path / "data.txt"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file)])

                assert result.exit_code == 0

    def test_load_file_not_exists(self, click_runner):
        """Test error when file doesn't exist."""
        result = click_runner.invoke(viz, ["nonexistent.json"])

        assert result.exit_code != 0
        assert "does not exist" in result.stdout.lower() or "invalid" in result.stdout.lower()


class TestVizTypeDetection:
    """Test visualization type auto-detection."""

    def test_detect_complexity_viz(self, click_runner, tmp_path):
        """Test auto-detecting complexity visualization."""
        data = {"complexity": {"avg_complexity": 3.5}, "complex_items": []}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.ComplexityVisualizer") as mock_viz:
                mock_viz.return_value.create_distribution_chart.return_value = {}

                result = click_runner.invoke(viz, [str(data_file), "--type", "auto"])

                assert result.exit_code == 0
                mock_viz.assert_called()

    def test_detect_contributors_viz(self, click_runner, tmp_path):
        """Test auto-detecting contributors visualization."""
        data = {"contributors": [{"name": "Alice", "commits": 50}]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.ContributorVisualizer") as mock_viz:
                mock_viz.return_value.create_contribution_chart.return_value = {}

                result = click_runner.invoke(viz, [str(data_file), "--type", "auto"])

                assert result.exit_code == 0
                mock_viz.assert_called()

    def test_detect_hotspots_viz(self, click_runner, tmp_path):
        """Test auto-detecting hotspots visualization."""
        data = {"hotspots": [{"file": "src/core.py", "risk": 5}]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.HotspotVisualizer") as mock_viz:
                mock_viz.return_value.create_hotspot_bubble.return_value = {}

                result = click_runner.invoke(viz, [str(data_file), "--type", "auto"])

                assert result.exit_code == 0
                mock_viz.assert_called()

    def test_detect_momentum_viz(self, click_runner, tmp_path):
        """Test auto-detecting momentum visualization."""
        data = {"velocity": [20, 25, 30], "momentum": {"trend": 15}}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.MomentumVisualizer") as mock_viz:
                mock_viz.return_value.create_velocity_chart.return_value = {}

                result = click_runner.invoke(viz, [str(data_file), "--type", "auto"])

                assert result.exit_code == 0
                mock_viz.assert_called()


class TestVizSpecificTypes:
    """Test specific visualization types."""

    def test_complexity_visualization(self, click_runner, sample_json_data, tmp_path):
        """Test complexity visualization."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.ComplexityVisualizer") as mock_viz:
                mock_viz.return_value.create_distribution_chart.return_value = {}
                mock_viz.return_value.display_terminal = MagicMock()

                result = click_runner.invoke(viz, [str(data_file), "--type", "complexity"])

                assert result.exit_code == 0
                mock_viz.assert_called()
                mock_viz.return_value.display_terminal.assert_called()

    def test_dependencies_visualization(self, click_runner, tmp_path):
        """Test dependencies visualization."""
        data = {"dependencies": [{"from": "A", "to": "B"}]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.DependencyVisualizer") as mock_viz:
                mock_viz.return_value.create_dependency_graph.return_value = {}
                mock_viz.return_value.display_terminal = MagicMock()

                result = click_runner.invoke(viz, [str(data_file), "--type", "dependencies"])

                assert result.exit_code == 0
                mock_viz.assert_called()

    def test_coupling_visualization(self, click_runner, tmp_path):
        """Test coupling visualization."""
        data = {"coupling_data": [{"module": "A", "coupling": 5}]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.CouplingVisualizer") as mock_viz:
                mock_viz.return_value.create_coupling_network.return_value = {}
                mock_viz.return_value.display_terminal = MagicMock()

                result = click_runner.invoke(viz, [str(data_file), "--type", "coupling"])

                assert result.exit_code == 0
                mock_viz.assert_called()


class TestVizCustomVisualization:
    """Test custom visualization functionality."""

    def test_custom_bar_chart(self, click_runner, sample_csv_data, tmp_path):
        """Test creating custom bar chart."""
        data_file = tmp_path / "data.csv"

        with open(data_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "score", "category"])
            writer.writeheader()
            writer.writerows(sample_csv_data)

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {}

                result = click_runner.invoke(
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

    def test_custom_line_chart(self, click_runner, tmp_path):
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

                result = click_runner.invoke(
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

    def test_custom_scatter_plot(self, click_runner, tmp_path):
        """Test creating custom scatter plot."""
        data = [{"x": 10, "y": 20}, {"x": 15, "y": 25}, {"x": 20, "y": 18}]
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {}

                result = click_runner.invoke(
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

    def test_custom_pie_chart(self, click_runner, sample_csv_data, tmp_path):
        """Test creating custom pie chart."""
        data_file = tmp_path / "data.csv"

        with open(data_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "score", "category"])
            writer.writeheader()
            writer.writerows(sample_csv_data)

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {}

                result = click_runner.invoke(
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

    def test_chart_with_title(self, click_runner, sample_json_data, tmp_path):
        """Test setting chart title."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file), "--title", "My Custom Chart"])

                assert result.exit_code == 0

    def test_chart_dimensions(self, click_runner, sample_json_data, tmp_path):
        """Test setting chart dimensions."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file), "--width", "1200", "--height", "600"])

                assert result.exit_code == 0

    def test_data_limit(self, click_runner, tmp_path):
        """Test limiting data points."""
        data = {"items": [{"value": i} for i in range(100)]}
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file), "--limit", "10"])

                assert result.exit_code == 0


class TestVizOutputFormats:
    """Test different output formats."""

    def test_terminal_output(self, click_runner, sample_json_data, tmp_path):
        """Test terminal output format."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.ComplexityVisualizer") as mock_viz:
                mock_viz.return_value.display_terminal = MagicMock()

                result = click_runner.invoke(viz, [str(data_file), "--format", "terminal"])

                assert result.exit_code == 0
                mock_viz.return_value.display_terminal.assert_called()

    def test_json_output(self, click_runner, sample_json_data, tmp_path):
        """Test JSON output format."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {"type": "bar", "data": {}}

                result = click_runner.invoke(viz, [str(data_file), "--format", "json"])

                assert result.exit_code == 0
                # Should output valid JSON
                output_data = json.loads(result.stdout)
                assert "type" in output_data

    def test_json_output_to_file(self, click_runner, sample_json_data, tmp_path):
        """Test JSON output to file."""
        data_file = tmp_path / "data.json"
        output_file = tmp_path / "viz.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {"type": "bar"}

                result = click_runner.invoke(
                    viz, [str(data_file), "--format", "json", "--output", str(output_file)]
                )

                assert result.exit_code == 0
                assert output_file.exists()

    def test_html_output(self, click_runner, sample_json_data, tmp_path):
        """Test HTML output format."""
        data_file = tmp_path / "data.json"
        output_file = tmp_path / "chart.html"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {"type": "bar", "data": {}}

                result = click_runner.invoke(
                    viz, [str(data_file), "--format", "html", "--output", str(output_file)]
                )

                assert result.exit_code == 0
                assert output_file.exists()

                # Check HTML content
                html_content = output_file.read_text()
                assert "<!DOCTYPE html>" in html_content
                assert "Chart.js" in html_content

    def test_svg_output_not_implemented(self, click_runner, sample_json_data, tmp_path):
        """Test SVG output (not yet implemented)."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file), "--format", "svg"])

                assert result.exit_code == 0
                assert "SVG export not yet implemented" in result.stdout

    def test_png_output_not_implemented(self, click_runner, sample_json_data, tmp_path):
        """Test PNG output (not yet implemented)."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file), "--format", "png"])

                assert result.exit_code == 0
                assert "PNG export not yet implemented" in result.stdout


class TestVizInteractiveMode:
    """Test interactive visualization mode."""

    @patch("webbrowser.open")
    @patch("tempfile.NamedTemporaryFile")
    def test_interactive_mode(
        self, mock_tempfile, mock_browser, click_runner, sample_json_data, tmp_path
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

                result = click_runner.invoke(viz, [str(data_file), "--interactive"])

                assert result.exit_code == 0
                assert "Launching interactive mode" in result.stdout
                assert "Opened in browser" in result.stdout
                mock_browser.assert_called_once()


class TestVizErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_file(self, click_runner, tmp_path):
        """Test error with invalid JSON."""
        data_file = tmp_path / "invalid.json"
        data_file.write_text("not valid json{")

        with patch("tenets.cli.commands.viz.get_logger"):
            result = click_runner.invoke(viz, [str(data_file)])

            assert result.exit_code != 0
            assert "Visualization failed" in result.stdout

    def test_missing_required_fields(self, click_runner, tmp_path):
        """Test error when required fields are missing."""
        data = [{"name": "A"}, {"name": "B"}]  # Missing value field
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                # Simulate error when value field is missing
                mock_viz.return_value.create_chart.side_effect = KeyError("value")

                result = click_runner.invoke(
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

    def test_empty_data_file(self, click_runner, tmp_path):
        """Test handling empty data file."""
        data_file = tmp_path / "empty.json"
        data_file.write_text("[]")

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer"):
                result = click_runner.invoke(viz, [str(data_file)])

                # Should handle gracefully
                assert result.exit_code == 0


class TestVizSummaryOutput:
    """Test summary output for visualizations."""

    def test_visualization_summary(self, click_runner, sample_json_data, tmp_path):
        """Test visualization summary output."""
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(sample_json_data))

        with patch("tenets.cli.commands.viz.get_logger"):
            with patch("tenets.cli.commands.viz.BaseVisualizer") as mock_viz:
                mock_viz.return_value.create_chart.return_value = {
                    "type": "bar",
                    "data": {"datasets": [{"data": [1, 2, 3]}]},
                }

                result = click_runner.invoke(viz, [str(data_file), "--type", "custom"])

                assert result.exit_code == 0
                assert "Custom Visualization Generated" in result.stdout
                assert "Type:" in result.stdout
                assert "Datasets:" in result.stdout


class TestVizDepsCommand:
    """Test the viz deps command for dependency visualization."""
    
    def test_deps_basic(self, runner, tmp_path):
        """Test basic dependency visualization."""
        with patch("tenets.cli.commands.viz.ProjectDetector") as mock_detector:
            with patch("tenets.cli.commands.viz.FileScanner") as mock_scanner:
                with patch("tenets.cli.commands.viz.CodeAnalyzer") as mock_analyzer:
                    # Mock project detection
                    mock_detector.return_value.detect_project.return_value = {
                        "type": "python_project",
                        "languages": {"python": 100.0},
                        "frameworks": [],
                        "entry_points": ["main.py"],
                    }
                    
                    # Mock file scanning
                    mock_scanner.return_value.scan.return_value = [
                        Path("main.py"),
                        Path("utils.py"),
                    ]
                    
                    # Mock analysis
                    mock_analysis = MagicMock()
                    mock_analysis.structure.imports = [
                        MagicMock(module="utils", from_module=None),
                    ]
                    mock_analyzer.return_value.analyze_file.return_value = mock_analysis
                    
                    result = runner.invoke(app, ["viz", "deps", str(tmp_path)])
                    
                    assert result.exit_code == 0
                    assert "Detected project type: python_project" in result.stdout
    
    def test_deps_with_output_formats(self, runner, tmp_path, sample_dependency_graph):
        """Test dependency visualization with different output formats."""
        formats = ["json", "dot", "html", "svg", "png"]
        
        for format in formats:
            output_file = tmp_path / f"deps.{format}"
            
            with patch("tenets.cli.commands.viz.ProjectDetector"):
                with patch("tenets.cli.commands.viz.FileScanner"):
                    with patch("tenets.cli.commands.viz.CodeAnalyzer"):
                        with patch("tenets.cli.commands.viz.GraphGenerator") as mock_gen:
                            mock_gen.return_value.generate_graph.return_value = str(output_file)
                            
                            result = runner.invoke(
                                app,
                                ["viz", "deps", str(tmp_path), 
                                 "--format", format,
                                 "--output", str(output_file)]
                            )
                            
                            # Should call generator with correct format
                            if format != "ascii":
                                mock_gen.return_value.generate_graph.assert_called()
    
    def test_deps_aggregation_levels(self, runner, sample_dependency_graph, sample_project_info):
        """Test dependency aggregation at different levels."""
        levels = ["file", "module", "package"]
        
        for level in levels:
            # Test aggregation function
            if level != "file":
                aggregated = aggregate_dependencies(
                    sample_dependency_graph,
                    level,
                    sample_project_info
                )
                
                if level == "module":
                    # Should aggregate to module level
                    assert "src" in aggregated or "src.api" in aggregated
                elif level == "package":
                    # Should aggregate to package level
                    assert "src" in aggregated or "tests" in aggregated
    
    def test_deps_clustering(self, runner, tmp_path):
        """Test dependency visualization with clustering."""
        cluster_options = ["directory", "module", "package"]
        
        for cluster_by in cluster_options:
            with patch("tenets.cli.commands.viz.GraphGenerator") as mock_gen:
                result = runner.invoke(
                    app,
                    ["viz", "deps", str(tmp_path),
                     "--cluster-by", cluster_by,
                     "--format", "json"]
                )
                
                # Generator should be called with cluster_by option
                # (Would need more mocking to fully test)
    
    def test_deps_max_nodes(self, runner, tmp_path):
        """Test dependency visualization with node limit."""
        with patch("tenets.cli.commands.viz.GraphGenerator") as mock_gen:
            result = runner.invoke(
                app,
                ["viz", "deps", str(tmp_path),
                 "--max-nodes", "50",
                 "--format", "json"]
            )
            
            # Generator should be called with max_nodes option
    
    def test_deps_layouts(self, runner, tmp_path):
        """Test different graph layouts."""
        layouts = ["hierarchical", "circular", "shell", "kamada"]
        
        for layout in layouts:
            with patch("tenets.cli.commands.viz.GraphGenerator") as mock_gen:
                result = runner.invoke(
                    app,
                    ["viz", "deps", str(tmp_path),
                     "--layout", layout,
                     "--format", "svg"]
                )
                
                # Generator should be called with layout option
    
    def test_aggregate_dependencies_function(self, sample_dependency_graph, sample_project_info):
        """Test the aggregate_dependencies helper function."""
        # Test module-level aggregation
        module_aggregated = aggregate_dependencies(
            sample_dependency_graph,
            "module",
            sample_project_info
        )
        
        assert "src" in module_aggregated
        assert "src.api" in module_aggregated
        assert "tests" in module_aggregated
        
        # Test package-level aggregation
        package_aggregated = aggregate_dependencies(
            sample_dependency_graph,
            "package",
            sample_project_info
        )
        
        assert "src" in package_aggregated
        assert "tests" in package_aggregated
        
        # No self-dependencies
        assert "src" not in package_aggregated.get("src", [])
    
    def test_get_aggregate_key_function(self, sample_project_info):
        """Test the get_aggregate_key helper function."""
        # Test module level
        assert get_aggregate_key("src/utils/helpers.py", "module", sample_project_info) == "src.utils"
        assert get_aggregate_key("main.py", "module", sample_project_info) == "root"
        
        # Test package level
        assert get_aggregate_key("src/utils/helpers.py", "package", sample_project_info) == "src"
        assert get_aggregate_key("tests/test_main.py", "package", sample_project_info) == "tests"
        
        # Test with module names (dotted notation)
        assert get_aggregate_key("os.path", "module", sample_project_info) == "os"
    
    def test_deps_no_dependencies_found(self, runner, tmp_path):
        """Test handling when no dependencies are found."""
        with patch("tenets.cli.commands.viz.ProjectDetector"):
            with patch("tenets.cli.commands.viz.FileScanner") as mock_scanner:
                with patch("tenets.cli.commands.viz.CodeAnalyzer") as mock_analyzer:
                    mock_scanner.return_value.scan.return_value = [Path("main.py")]
                    mock_analyzer.return_value.analyze_file.return_value = MagicMock(
                        structure=None
                    )
                    
                    result = runner.invoke(app, ["viz", "deps", str(tmp_path)])
                    
                    assert "No dependencies found" in result.stdout
    
    def test_deps_ascii_output(self, runner, tmp_path):
        """Test ASCII tree output for terminal."""
        with patch("tenets.cli.commands.viz.ProjectDetector"):
            with patch("tenets.cli.commands.viz.FileScanner") as mock_scanner:
                with patch("tenets.cli.commands.viz.CodeAnalyzer") as mock_analyzer:
                    mock_scanner.return_value.scan.return_value = [Path("main.py")]
                    
                    mock_analysis = MagicMock()
                    mock_analysis.structure.imports = [
                        MagicMock(module="utils"),
                        MagicMock(module="config"),
                    ]
                    mock_analyzer.return_value.analyze_file.return_value = mock_analysis
                    
                    result = runner.invoke(
                        app,
                        ["viz", "deps", str(tmp_path), "--format", "ascii"]
                    )
                    
                    assert "Dependency Graph:" in result.stdout
                    assert "└─>" in result.stdout  # Tree character
