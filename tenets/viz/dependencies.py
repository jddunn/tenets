"""Dependency graph visualization for codebases.

This module creates visual representations of import dependencies and
module relationships within a codebase, helping identify architecture
patterns, circular dependencies, and coupling issues.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.utils.logger import get_logger

from .base import (
    MATPLOTLIB_AVAILABLE,
    NETWORKX_AVAILABLE,
    PLOTLY_AVAILABLE,
    ColorScheme,
    VisualizationBase,
    VisualizationFormat,
    create_graph_layout,
    format_size,
    truncate_text,
)


@dataclass
class DependencyNode:
    """A node in the dependency graph.

    Attributes:
        path: File path
        name: Display name
        imports: Set of imported modules
        imported_by: Set of modules that import this
        size: File size in bytes
        language: Programming language
        metadata: Additional metadata
    """

    path: str
    name: str
    imports: Set[str] = None
    imported_by: Set[str] = None
    size: int = 0
    language: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.imports is None:
            self.imports = set()
        if self.imported_by is None:
            self.imported_by = set()
        if self.metadata is None:
            self.metadata = {}

    @property
    def in_degree(self) -> int:
        """Number of modules that import this one."""
        return len(self.imported_by)

    @property
    def out_degree(self) -> int:
        """Number of modules this one imports."""
        return len(self.imports)

    @property
    def coupling(self) -> int:
        """Total coupling (in + out degree)."""
        return self.in_degree + self.out_degree


class DependencyGraph(VisualizationBase):
    """Visualize code dependencies as a graph.

    Creates various visualizations of import relationships including:
    - Full dependency graph
    - Circular dependencies
    - Module clusters
    - Dependency layers
    - Import hotspots
    """

    def __init__(
        self,
        title: str = "Dependency Graph",
        color_scheme: Optional[ColorScheme] = None,
        format: VisualizationFormat = VisualizationFormat.AUTO,
    ):
        """Initialize dependency graph.

        Args:
            title: Graph title
            color_scheme: Color scheme
            format: Output format
        """
        super().__init__(title, color_scheme, format)
        self.logger = get_logger(__name__)
        self.nodes: Dict[str, DependencyNode] = {}
        self.circular_deps: List[List[str]] = []
        self.clusters: Dict[str, Set[str]] = {}

    def add_file(
        self,
        file_path: str,
        imports: List[str],
        size: int = 0,
        language: str = "",
        metadata: Optional[Dict] = None,
    ):
        """Add a file to the dependency graph.

        Args:
            file_path: Path to file
            imports: List of imported modules
            size: File size
            language: Programming language
            metadata: Additional metadata
        """
        # Normalize path
        file_path = str(Path(file_path))
        name = Path(file_path).stem

        # Create or update node
        if file_path not in self.nodes:
            self.nodes[file_path] = DependencyNode(
                path=file_path, name=name, size=size, language=language, metadata=metadata or {}
            )

        # Add imports
        for imp in imports:
            # Try to resolve import to file in graph
            resolved = self._resolve_import(imp, file_path)
            if resolved and resolved in self.nodes:
                self.nodes[file_path].imports.add(resolved)
                self.nodes[resolved].imported_by.add(file_path)

    def _resolve_import(self, import_name: str, from_file: str) -> Optional[str]:
        """Resolve import name to file path.

        Args:
            import_name: Import module name
            from_file: File doing the importing

        Returns:
            Resolved file path or None
        """
        # Simple resolution - look for matching file names
        for node_path in self.nodes:
            node_name = Path(node_path).stem

            # Direct match
            if node_name == import_name:
                return node_path

            # Match last part of import path
            import_parts = import_name.split(".")
            if import_parts[-1] == node_name:
                return node_path

        return None

    def analyze(self):
        """Analyze the dependency graph.

        Identifies patterns like circular dependencies, clusters, and layers.
        """
        self._find_circular_dependencies()
        self._identify_clusters()
        self._calculate_layers()

    def _find_circular_dependencies(self):
        """Find circular dependencies using DFS."""
        visited = set()
        rec_stack = set()
        self.circular_deps = []

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.nodes[node].imports:
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    self.circular_deps.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node, [])

    def _identify_clusters(self):
        """Identify module clusters based on directory structure."""
        self.clusters = defaultdict(set)

        for node_path in self.nodes:
            # Use parent directory as cluster
            parent = str(Path(node_path).parent)
            self.clusters[parent].add(node_path)

    def _calculate_layers(self):
        """Calculate dependency layers (topological levels)."""
        # Find nodes with no dependencies (layer 0)
        layer = 0
        remaining = set(self.nodes.keys())

        while remaining:
            # Find nodes that only depend on already-layered nodes
            current_layer = []

            for node in remaining:
                deps = self.nodes[node].imports & remaining
                if not deps:
                    current_layer.append(node)

            if not current_layer:
                # Remaining nodes have circular dependencies
                for node in remaining:
                    self.nodes[node].metadata["layer"] = -1
                break

            for node in current_layer:
                self.nodes[node].metadata["layer"] = layer
                remaining.remove(node)

            layer += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.nodes:
            return {}

        total_imports = sum(len(n.imports) for n in self.nodes.values())
        max_in = max((n.in_degree for n in self.nodes.values()), default=0)
        max_out = max((n.out_degree for n in self.nodes.values()), default=0)

        # Find most imported/importing
        most_imported = max(self.nodes.values(), key=lambda n: n.in_degree, default=None)
        most_importing = max(self.nodes.values(), key=lambda n: n.out_degree, default=None)

        return {
            "total_files": len(self.nodes),
            "total_dependencies": total_imports,
            "circular_dependencies": len(self.circular_deps),
            "clusters": len(self.clusters),
            "max_in_degree": max_in,
            "max_out_degree": max_out,
            "most_imported": most_imported.name if most_imported else None,
            "most_importing": most_importing.name if most_importing else None,
            "avg_coupling": sum(n.coupling for n in self.nodes.values()) / len(self.nodes),
        }

    def _render_ascii(self) -> str:
        """Render dependency graph as ASCII."""
        lines = []
        lines.append("=" * 80)
        lines.append(f" {self.title} ".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Statistics
        stats = self.get_statistics()
        lines.append(f"Files: {stats.get('total_files', 0)}")
        lines.append(f"Dependencies: {stats.get('total_dependencies', 0)}")
        lines.append(f"Circular deps: {stats.get('circular_dependencies', 0)}")
        lines.append("")

        # Top imported modules
        lines.append("Most Imported Modules:")
        lines.append("-" * 40)

        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.in_degree, reverse=True)[:10]

        for node in sorted_nodes:
            if node.in_degree > 0:
                lines.append(f"  {node.name:30s} <- {node.in_degree:3d} imports")

        lines.append("")

        # Top importing modules
        lines.append("Most Importing Modules:")
        lines.append("-" * 40)

        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.out_degree, reverse=True)[:10]

        for node in sorted_nodes:
            if node.out_degree > 0:
                lines.append(f"  {node.name:30s} -> {node.out_degree:3d} imports")

        # Circular dependencies
        if self.circular_deps:
            lines.append("")
            lines.append("Circular Dependencies Detected:")
            lines.append("-" * 40)

            for i, cycle in enumerate(self.circular_deps[:5], 1):
                cycle_str = " -> ".join(Path(p).stem for p in cycle)
                lines.append(f"  {i}. {cycle_str}")

        return "\n".join(lines)

    def _render_html(self, width: int, height: int) -> str:
        """Render as interactive HTML using Plotly or D3."""
        if not PLOTLY_AVAILABLE:
            return super()._render_html(width, height)

        import plotly.graph_objs as go
        import plotly.offline as offline

        # Create nodes and edges
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        edge_x = []
        edge_y = []

        # Get layout
        nodes = list(self.nodes.keys())
        edges = []
        for node in self.nodes.values():
            for imp in node.imports:
                edges.append((node.path, imp))

        positions = create_graph_layout(nodes, edges, "spring")

        # Add nodes
        for node_path, (x, y) in positions.items():
            node = self.nodes[node_path]
            node_x.append(x)
            node_y.append(y)

            # Create hover text
            hover = f"{node.name}<br>"
            hover += f"Imports: {node.out_degree}<br>"
            hover += f"Imported by: {node.in_degree}<br>"
            hover += f"Size: {format_size(node.size)}"
            node_text.append(hover)

            # Size based on coupling
            node_size.append(10 + node.coupling * 2)

            # Color based on layer or in-degree
            layer = node.metadata.get("layer", 0)
            node_color.append(layer if layer >= 0 else -1)

        # Add edges
        for source, target in edges:
            if source in positions and target in positions:
                x0, y0 = positions[source]
                x1, y1 = positions[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[Path(n).stem for n in nodes],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale="YlOrRd",
                size=node_size,
                color=node_color,
                colorbar=dict(thickness=15, title="Layer", xanchor="left", titleside="right"),
                line_width=2,
            ),
            hovertext=node_text,
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=self.title,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height,
            ),
        )

        # Generate HTML
        return offline.plot(fig, output_type="div", include_plotlyjs="cdn")

    def _render_svg(self, width: int, height: int, dpi: int) -> str:
        """Render as SVG using matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            return super()._render_svg(width, height, dpi)

        from io import StringIO

        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        # Create figure
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

        # Get layout
        nodes = list(self.nodes.keys())
        edges = []
        for node in self.nodes.values():
            for imp in node.imports:
                edges.append((node.path, imp))

        positions = create_graph_layout(nodes, edges, "spring")

        # Draw edges
        for source, target in edges:
            if source in positions and target in positions:
                x0, y0 = positions[source]
                x1, y1 = positions[target]
                ax.plot([x0, x1], [y0, y1], "k-", alpha=0.3, linewidth=0.5)

        # Draw nodes
        for node_path, (x, y) in positions.items():
            node = self.nodes[node_path]

            # Size based on coupling
            size = 100 + node.coupling * 50

            # Color based on in-degree
            color_intensity = min(1.0, node.in_degree / 10)
            color = plt.cm.YlOrRd(color_intensity)

            ax.scatter(x, y, s=size, c=[color], alpha=0.8, edgecolors="black")

            # Add label for important nodes
            if node.coupling > 5:
                ax.annotate(node.name, (x, y), fontsize=8, ha="center", va="bottom")

        ax.set_title(self.title, fontsize=14, fontweight="bold")
        ax.axis("off")

        # Convert to SVG
        buffer = StringIO()
        plt.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()


def create_dependency_graph(
    files: List[Any], title: str = "Dependency Graph", max_nodes: int = 100, format: str = "auto"
) -> DependencyGraph:
    """Create a dependency graph from files.

    Args:
        files: List of FileAnalysis objects
        title: Graph title
        max_nodes: Maximum nodes to display
        format: Output format

    Returns:
        DependencyGraph instance
    """
    graph = DependencyGraph(title=title, format=VisualizationFormat(format))

    # Add files to graph
    for file in files[:max_nodes]:
        imports = []

        # Extract imports based on file type
        if hasattr(file, "imports"):
            imports = [
                str(imp.module) if hasattr(imp, "module") else str(imp) for imp in file.imports
            ]

        graph.add_file(file_path=file.path, imports=imports, size=file.size, language=file.language)

    # Analyze graph
    graph.analyze()

    return graph
