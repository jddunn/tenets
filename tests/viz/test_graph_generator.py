"""Tests for graph generation and visualization."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tenets.viz.graph_generator import GraphGenerator


class TestGraphGenerator:
    """Test the GraphGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a GraphGenerator instance."""
        return GraphGenerator()
    
    @pytest.fixture
    def sample_dependency_graph(self):
        """Create a sample dependency graph."""
        return {
            "main.py": ["utils.py", "config.py", "models.py"],
            "utils.py": ["config.py"],
            "models.py": ["utils.py"],
            "config.py": [],
            "tests/test_main.py": ["main.py", "utils.py"],
        }
    
    @pytest.fixture
    def sample_project_info(self):
        """Create sample project info."""
        return {
            "type": "python_project",
            "languages": {"python": 100.0},
            "frameworks": [],
            "entry_points": ["main.py"],
        }
    
    def test_process_graph_basic(self, generator, sample_dependency_graph):
        """Test basic graph processing."""
        processed = generator._process_graph(sample_dependency_graph)
        
        # Check nodes are extracted
        assert len(processed["nodes"]) > 0
        node_ids = [n["id"] for n in processed["nodes"]]
        assert "main.py" in node_ids
        assert "utils.py" in node_ids
        
        # Check edges are created
        assert len(processed["edges"]) > 0
        # Should have edge from main.py to utils.py
        assert any(
            e["source"] == "main.py" and e["target"] == "utils.py"
            for e in processed["edges"]
        )
        
        # Check metadata
        assert processed["metadata"]["total_nodes"] == len(processed["nodes"])
        assert processed["metadata"]["total_edges"] == len(processed["edges"])
    
    def test_process_graph_with_max_nodes(self, generator, sample_dependency_graph):
        """Test graph processing with max_nodes limit."""
        processed = generator._process_graph(
            sample_dependency_graph,
            max_nodes=3
        )
        
        # Should limit to 3 nodes
        assert len(processed["nodes"]) <= 3
        
        # Most connected nodes should be kept (main.py, utils.py, config.py)
        node_ids = [n["id"] for n in processed["nodes"]]
        assert "main.py" in node_ids  # Has 3 connections
        assert "utils.py" in node_ids  # Has connections
    
    def test_cluster_nodes(self, generator, sample_dependency_graph, sample_project_info):
        """Test node clustering."""
        clustered = generator._cluster_nodes(
            sample_dependency_graph,
            "directory",
            sample_project_info
        )
        
        assert "nodes" in clustered
        assert "edges" in clustered
        assert "clusters" in clustered
        
        # Should cluster test_main.py under "tests"
        assert clustered["clusters"].get("tests/test_main.py") == "tests"
        # Others should be in "root"
        assert clustered["clusters"].get("main.py") == "root"
    
    def test_get_node_label(self, generator):
        """Test node label generation."""
        # Should simplify paths
        assert generator._get_node_label("src/utils/helpers.py") == "helpers.py"
        assert generator._get_node_label("main.py") == "main.py"
        assert generator._get_node_label("path/to/file.js") == "file.js"
    
    def test_get_node_type(self, generator):
        """Test node type detection."""
        assert generator._get_node_type("main.py", None) == "python"
        assert generator._get_node_type("app.js", None) == "javascript"
        assert generator._get_node_type("Main.java", None) == "java"
        assert generator._get_node_type("main.go", None) == "go"
        assert generator._get_node_type("main.rs", None) == "rust"
        assert generator._get_node_type("unknown.xyz", None) == "unknown"
    
    def test_generate_json(self, generator, sample_dependency_graph, tmp_path):
        """Test JSON output generation."""
        output_path = tmp_path / "graph.json"
        
        processed = generator._process_graph(sample_dependency_graph)
        result = generator._generate_json(processed, output_path)
        
        # Check file was created
        assert output_path.exists()
        assert result == str(output_path)
        
        # Check JSON content
        with open(output_path) as f:
            data = json.load(f)
        
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
    
    def test_generate_dot(self, generator, sample_dependency_graph, tmp_path):
        """Test DOT format generation."""
        output_path = tmp_path / "graph.dot"
        
        processed = generator._process_graph(sample_dependency_graph)
        result = generator._generate_dot(processed, output_path)
        
        # Check file was created
        assert output_path.exists()
        assert result == str(output_path)
        
        # Check DOT content
        content = output_path.read_text()
        assert "digraph Dependencies" in content
        assert "main.py" in content
        assert "->" in content  # Has edges
    
    def test_generate_basic_html(self, generator, sample_dependency_graph, tmp_path):
        """Test basic HTML generation."""
        output_path = tmp_path / "graph.html"
        
        processed = generator._process_graph(sample_dependency_graph)
        
        # Mock plotly not available
        generator._plotly_available = False
        
        result = generator._generate_basic_html(processed, output_path)
        
        # Check file was created
        assert output_path.exists()
        assert result == str(output_path)
        
        # Check HTML content
        content = output_path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Dependency Graph" in content
        assert "d3js.org" in content  # Uses D3.js
        assert "const data =" in content  # Has data embedded
    
    def test_generate_graph_with_clustering(self, generator, sample_dependency_graph, tmp_path):
        """Test graph generation with clustering."""
        output_path = tmp_path / "graph.json"
        
        result = generator.generate_graph(
            dependency_graph=sample_dependency_graph,
            output_path=output_path,
            format="json",
            cluster_by="directory"
        )
        
        # Check clustering was applied
        with open(output_path) as f:
            data = json.load(f)
        
        assert data["metadata"]["clustered"] == True
        # Should have cluster information
        assert any("cluster" in node for node in data["nodes"])
    
    def test_get_cluster(self, generator):
        """Test cluster assignment."""
        project_info = {"type": "python_project"}
        
        # Directory clustering
        assert generator._get_cluster("src/utils.py", "directory", project_info) == "src"
        assert generator._get_cluster("main.py", "directory", project_info) == "root"
        
        # Module clustering for Python
        assert generator._get_cluster("src/utils/helpers.py", "module", project_info) == "src.utils"
        
        # Package clustering
        assert generator._get_cluster("mypackage/submodule/file.py", "package", project_info) == "mypackage"
    
    @patch('tenets.viz.graph_generator.GraphGenerator._generate_graphviz_image')
    def test_generate_image_fallback(self, mock_graphviz, generator, sample_dependency_graph, tmp_path):
        """Test image generation fallback."""
        # Simulate no visualization libraries available
        generator._graphviz_available = False
        generator._networkx_available = False
        generator._matplotlib_available = False
        
        output_path = tmp_path / "graph.svg"
        processed = generator._process_graph(sample_dependency_graph)
        
        # Should fall back to DOT format
        with patch.object(generator, '_generate_dot') as mock_dot:
            mock_dot.return_value = str(output_path.with_suffix('.dot'))
            result = generator._generate_image(processed, output_path, "svg", "hierarchical")
            mock_dot.assert_called_once()
    
    def test_aggregate_dependencies_module_level(self):
        """Test aggregating dependencies to module level."""
        from tenets.cli.commands.viz import aggregate_dependencies
        
        dependency_graph = {
            "src/models/user.py": ["src/utils/helpers.py", "config.py"],
            "src/models/post.py": ["src/utils/helpers.py"],
            "src/utils/helpers.py": ["config.py"],
            "tests/test_models.py": ["src/models/user.py"],
        }
        
        project_info = {"type": "python_project"}
        
        aggregated = aggregate_dependencies(dependency_graph, "module", project_info)
        
        # Should aggregate to module level
        assert "src.models" in aggregated
        assert "src.utils" in aggregated
        assert "tests" in aggregated
        
        # Check dependencies are aggregated
        assert "src.utils" in aggregated["src.models"]
        assert "root" in aggregated["src.models"]  # config.py goes to root
    
    def test_aggregate_dependencies_package_level(self):
        """Test aggregating dependencies to package level."""
        from tenets.cli.commands.viz import aggregate_dependencies
        
        dependency_graph = {
            "package1/module1/file1.py": ["package2/module2/file2.py"],
            "package1/module2/file3.py": ["package2/module1/file4.py"],
            "package2/module1/file4.py": ["package1/module1/file1.py"],
        }
        
        project_info = {"type": "python_project"}
        
        aggregated = aggregate_dependencies(dependency_graph, "package", project_info)
        
        # Should aggregate to package level
        assert "package1" in aggregated
        assert "package2" in aggregated
        
        # Check circular dependency between packages
        assert "package2" in aggregated["package1"]
        assert "package1" in aggregated["package2"]
    
    def test_empty_dependency_graph(self, generator):
        """Test handling of empty dependency graph."""
        processed = generator._process_graph({})
        
        assert processed["nodes"] == []
        assert processed["edges"] == []
        assert processed["metadata"]["total_nodes"] == 0
        assert processed["metadata"]["total_edges"] == 0