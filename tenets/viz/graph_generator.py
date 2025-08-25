"""Graph generation for dependency visualization.

This module provides graph generation using pure Python libraries
that can be installed via pip without system dependencies.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from tenets.utils.logger import get_logger


class GraphGenerator:
    """Generates various graph visualizations for dependencies."""
    
    def __init__(self):
        """Initialize the graph generator."""
        self.logger = get_logger(__name__)
        self._networkx_available = False
        self._matplotlib_available = False
        self._graphviz_available = False
        self._plotly_available = False
        
        # Try importing optional dependencies
        try:
            import networkx as nx
            self._networkx_available = True
            self.nx = nx
        except ImportError:
            self.logger.debug("NetworkX not available - install with: pip install networkx")
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            self._matplotlib_available = True
            self.plt = plt
        except ImportError:
            self.logger.debug("Matplotlib not available - install with: pip install matplotlib")
        
        try:
            import graphviz
            self._graphviz_available = True
            self.graphviz = graphviz
        except ImportError:
            self.logger.debug("Graphviz not available - install with: pip install graphviz")
        
        try:
            import plotly.graph_objects as go
            self._plotly_available = True
            self.go = go
        except ImportError:
            self.logger.debug("Plotly not available - install with: pip install plotly")
    
    def generate_graph(
        self,
        dependency_graph: Dict[str, List[str]],
        output_path: Optional[Path] = None,
        format: str = "svg",
        layout: str = "hierarchical",
        cluster_by: Optional[str] = None,
        max_nodes: Optional[int] = None,
        project_info: Optional[Dict] = None,
    ) -> str:
        """Generate a dependency graph visualization.
        
        Args:
            dependency_graph: Dictionary of node -> list of dependencies
            output_path: Where to save the output
            format: Output format (svg, png, html, json, dot)
            layout: Graph layout algorithm
            cluster_by: How to cluster nodes (module, directory, package)
            max_nodes: Maximum number of nodes to display
            project_info: Project detection information
            
        Returns:
            Path to the generated file or visualization data
        """
        # Process the graph
        processed_graph = self._process_graph(
            dependency_graph, 
            cluster_by=cluster_by,
            max_nodes=max_nodes,
            project_info=project_info
        )
        
        # Generate based on format
        if format == "json":
            return self._generate_json(processed_graph, output_path)
        elif format == "dot":
            return self._generate_dot(processed_graph, output_path)
        elif format == "html":
            return self._generate_html(processed_graph, output_path, layout)
        elif format in ["svg", "png", "pdf"]:
            return self._generate_image(processed_graph, output_path, format, layout)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _process_graph(
        self,
        dependency_graph: Dict[str, List[str]],
        cluster_by: Optional[str] = None,
        max_nodes: Optional[int] = None,
        project_info: Optional[Dict] = None,
    ) -> Dict:
        """Process and potentially simplify the dependency graph."""
        processed = {
            "nodes": [],
            "edges": [],
            "clusters": {},
            "metadata": {}
        }
        
        # Group by clusters if requested
        if cluster_by:
            clustered = self._cluster_nodes(dependency_graph, cluster_by, project_info)
            processed["clusters"] = clustered["clusters"]
            nodes = clustered["nodes"]
            edges = clustered["edges"]
        else:
            # Extract unique nodes
            nodes = set(dependency_graph.keys())
            for deps in dependency_graph.values():
                nodes.update(deps)
            
            # Create edges
            edges = []
            for source, targets in dependency_graph.items():
                for target in targets:
                    edges.append({"source": source, "target": target})
        
        # Apply node limit if specified
        if max_nodes and len(nodes) > max_nodes:
            # Keep most connected nodes
            node_connections = defaultdict(int)
            for edge in edges:
                node_connections[edge["source"]] += 1
                node_connections[edge["target"]] += 1
            
            # Sort by connection count
            sorted_nodes = sorted(
                nodes,
                key=lambda n: node_connections.get(n, 0),
                reverse=True
            )[:max_nodes]
            
            nodes = set(sorted_nodes)
            edges = [
                e for e in edges
                if e["source"] in nodes and e["target"] in nodes
            ]
        
        # Convert to node list with metadata
        for node in nodes:
            node_data = {
                "id": node,
                "label": self._get_node_label(node),
                "type": self._get_node_type(node, project_info),
            }
            
            if cluster_by and node in processed["clusters"]:
                node_data["cluster"] = processed["clusters"][node]
            
            processed["nodes"].append(node_data)
        
        processed["edges"] = edges
        processed["metadata"] = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "clustered": cluster_by is not None,
            "project_type": project_info.get("type") if project_info else None,
        }
        
        return processed
    
    def _cluster_nodes(
        self,
        dependency_graph: Dict[str, List[str]],
        cluster_by: str,
        project_info: Optional[Dict] = None,
    ) -> Dict:
        """Cluster nodes by specified criteria."""
        clusters = {}
        cluster_graph = defaultdict(set)
        
        for source, targets in dependency_graph.items():
            source_cluster = self._get_cluster(source, cluster_by, project_info)
            clusters[source] = source_cluster
            
            for target in targets:
                target_cluster = self._get_cluster(target, cluster_by, project_info)
                clusters[target] = target_cluster
                
                # Add edge between clusters
                if source_cluster != target_cluster:
                    cluster_graph[source_cluster].add(target_cluster)
        
        # Create cluster-level nodes and edges
        nodes = set(clusters.values())
        edges = []
        for source, targets in cluster_graph.items():
            for target in targets:
                edges.append({"source": source, "target": target})
        
        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
        }
    
    def _get_cluster(self, node: str, cluster_by: str, project_info: Optional[Dict]) -> str:
        """Get cluster name for a node."""
        path_parts = node.replace("\\", "/").split("/")
        
        if cluster_by == "directory":
            # Group by immediate parent directory
            if len(path_parts) > 1:
                return path_parts[-2]
            return "root"
        elif cluster_by == "module":
            # Group by top-level module/package
            if project_info and project_info.get("type", "").startswith("python"):
                # Python module grouping
                if len(path_parts) > 1:
                    # Find package root (directory with __init__.py)
                    for i in range(len(path_parts) - 1):
                        potential_module = "/".join(path_parts[:i+1])
                        if potential_module:
                            return potential_module.replace("/", ".")
                return "root"
            else:
                # Default to top-level directory
                if len(path_parts) > 1:
                    return path_parts[0]
                return "root"
        elif cluster_by == "package":
            # Group by package (language-specific)
            if len(path_parts) > 1:
                return path_parts[0]
            return "root"
        
        return "default"
    
    def _get_node_label(self, node: str) -> str:
        """Get display label for a node."""
        # Simplify long paths
        if "/" in node or "\\" in node:
            return Path(node).name
        return node
    
    def _get_node_type(self, node: str, project_info: Optional[Dict]) -> str:
        """Determine node type based on file/module."""
        if node.endswith((".py", ".pyw")):
            return "python"
        elif node.endswith((".js", ".jsx", ".ts", ".tsx")):
            return "javascript"
        elif node.endswith((".java",)):
            return "java"
        elif node.endswith((".go",)):
            return "go"
        elif node.endswith((".rs",)):
            return "rust"
        elif node.endswith((".cpp", ".cc", ".cxx", ".hpp", ".h")):
            return "cpp"
        elif node.endswith((".cs",)):
            return "csharp"
        elif node.endswith((".rb",)):
            return "ruby"
        elif node.endswith((".php",)):
            return "php"
        else:
            return "unknown"
    
    def _generate_json(self, processed_graph: Dict, output_path: Optional[Path]) -> str:
        """Generate JSON output."""
        json_data = json.dumps(processed_graph, indent=2)
        
        if output_path:
            output_path = Path(output_path).with_suffix(".json")
            with open(output_path, "w") as f:
                f.write(json_data)
            return str(output_path)
        
        return json_data
    
    def _generate_dot(self, processed_graph: Dict, output_path: Optional[Path]) -> str:
        """Generate Graphviz DOT format."""
        lines = ["digraph Dependencies {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style=rounded];')
        
        # Add clusters if present
        clusters = defaultdict(list)
        for node in processed_graph["nodes"]:
            if "cluster" in node:
                clusters[node["cluster"]].append(node)
        
        if clusters:
            for i, (cluster_name, cluster_nodes) in enumerate(clusters.items()):
                lines.append(f'  subgraph cluster_{i} {{')
                lines.append(f'    label="{cluster_name}";')
                lines.append('    style=filled;')
                lines.append('    color=lightgrey;')
                for node in cluster_nodes:
                    node_id = node["id"].replace('"', '\\"')
                    node_label = node["label"].replace('"', '\\"')
                    lines.append(f'    "{node_id}" [label="{node_label}"];')
                lines.append('  }')
        else:
            # Add nodes
            for node in processed_graph["nodes"]:
                node_id = node["id"].replace('"', '\\"')
                node_label = node["label"].replace('"', '\\"')
                lines.append(f'  "{node_id}" [label="{node_label}"];')
        
        # Add edges
        for edge in processed_graph["edges"]:
            source = edge["source"].replace('"', '\\"')
            target = edge["target"].replace('"', '\\"')
            lines.append(f'  "{source}" -> "{target}";')
        
        lines.append("}")
        dot_content = "\n".join(lines)
        
        if output_path:
            output_path = Path(output_path).with_suffix(".dot")
            with open(output_path, "w") as f:
                f.write(dot_content)
            return str(output_path)
        
        return dot_content
    
    def _generate_html(self, processed_graph: Dict, output_path: Optional[Path], layout: str) -> str:
        """Generate interactive HTML visualization."""
        if self._plotly_available:
            return self._generate_plotly_html(processed_graph, output_path, layout)
        else:
            return self._generate_basic_html(processed_graph, output_path)
    
    def _generate_plotly_html(self, processed_graph: Dict, output_path: Optional[Path], layout: str) -> str:
        """Generate interactive Plotly visualization."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create network layout
        if self._networkx_available:
            G = self.nx.DiGraph()
            for edge in processed_graph["edges"]:
                G.add_edge(edge["source"], edge["target"])
            
            # Choose layout algorithm
            if layout == "hierarchical":
                pos = self.nx.spring_layout(G, k=2, iterations=50)
            elif layout == "circular":
                pos = self.nx.circular_layout(G)
            elif layout == "shell":
                pos = self.nx.shell_layout(G)
            else:
                pos = self.nx.spring_layout(G)
        else:
            # Simple grid layout without networkx
            pos = {}
            nodes = processed_graph["nodes"]
            n = len(nodes)
            cols = math.ceil(math.sqrt(n))
            for i, node in enumerate(nodes):
                row = i // cols
                col = i % cols
                pos[node["id"]] = (col, row)
        
        # Create edge traces
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in processed_graph["edges"]:
            if edge["source"] in pos and edge["target"] in pos:
                x0, y0 = pos[edge["source"]]
                x1, y1 = pos[edge["target"]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
        
        # Create node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    # Plotly >=5 uses nested title for colorbars
                    title=dict(text='Connections', side='right'),
                    xanchor='left',
                ),
            ),
            text=[],
            textposition="top center"
        )
        
        # Add node positions and info
        for node in processed_graph["nodes"]:
            if node["id"] in pos:
                x, y = pos[node["id"]]
                node_trace['x'] += (x,)
                node_trace['y'] += (y,)
                node_trace['text'] += (node["label"],)
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                # Plotly >=5: title is an object with text/font, titlefont_size is deprecated
                title=dict(text='Dependency Graph', font=dict(size=16)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )
        )
        
        # Generate HTML
        html_content = fig.to_html(include_plotlyjs='cdn')
        
        if output_path:
            output_path = Path(output_path).with_suffix(".html")
            with open(output_path, "w") as f:
                f.write(html_content)
            return str(output_path)
        
        return html_content
    
    def _generate_basic_html(self, processed_graph: Dict, output_path: Optional[Path]) -> str:
        """Generate basic HTML with D3.js visualization."""
        html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Dependency Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        #graph {{ border: 1px solid #ccc; }}
        .node {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .node-label {{ font-size: 12px; pointer-events: none; }}
        #info {{ margin-top: 20px; }}
        .stats {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Dependency Graph</h1>
    <div class="stats">
        <strong>Nodes:</strong> {node_count} | 
        <strong>Edges:</strong> {edge_count} |
        <strong>Project Type:</strong> {project_type}
    </div>
    <svg id="graph" width="1200" height="800"></svg>
    <div id="info"></div>
    
    <script>
        const data = {graph_data};
        
        const width = 1200;
        const height = 800;
        
        const svg = d3.select("#graph");
        
        // Create force simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.edges).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        // Add arrow markers for directed edges
        svg.append("defs").selectAll("marker")
            .data(["arrow"])
            .enter().append("marker")
            .attr("id", d => d)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 8)
            .attr("markerHeight", 8)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");
        
        // Add links
        const link = svg.append("g")
            .selectAll("line")
            .data(data.edges)
            .enter().append("line")
            .attr("class", "link")
            .attr("marker-end", "url(#arrow)");
        
        // Add nodes
        const node = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", 8)
            .attr("fill", d => {{
                const colors = {{
                    "python": "#3776ab",
                    "javascript": "#f7df1e",
                    "java": "#007396",
                    "go": "#00add8",
                    "rust": "#dea584",
                    "cpp": "#00599c",
                    "unknown": "#888"
                }};
                return colors[d.type] || colors.unknown;
            }})
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Add labels
        const label = svg.append("g")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .attr("dx", 12)
            .attr("dy", 4)
            .text(d => d.label);
        
        // Add hover info
        node.on("mouseover", function(event, d) {{
            d3.select("#info").html(`
                <strong>Node:</strong> ${{d.id}}<br>
                <strong>Type:</strong> ${{d.type}}<br>
                <strong>Cluster:</strong> ${{d.cluster || "none"}}
            `);
        }});
        
        // Update positions on tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>"""
        
        # Format the template
        html_content = html_template.format(
            node_count=len(processed_graph["nodes"]),
            edge_count=len(processed_graph["edges"]),
            project_type=processed_graph["metadata"].get("project_type", "unknown"),
            graph_data=json.dumps(processed_graph)
        )
        
        if output_path:
            output_path = Path(output_path).with_suffix(".html")
            with open(output_path, "w") as f:
                f.write(html_content)
            return str(output_path)
        
        return html_content
    
    def _generate_image(
        self,
        processed_graph: Dict,
        output_path: Optional[Path],
        format: str,
        layout: str
    ) -> str:
        """Generate static image (SVG, PNG, PDF)."""
        if self._graphviz_available:
            return self._generate_graphviz_image(processed_graph, output_path, format, layout)
        elif self._networkx_available and self._matplotlib_available:
            return self._generate_networkx_image(processed_graph, output_path, format, layout)
        else:
            # Fall back to DOT format
            self.logger.warning(
                f"Neither graphviz nor networkx+matplotlib available for {format} generation. "
                "Install with: pip install graphviz or pip install networkx matplotlib"
            )
            return self._generate_dot(processed_graph, output_path)
    
    def _generate_graphviz_image(
        self,
        processed_graph: Dict,
        output_path: Optional[Path],
        format: str,
        layout: str
    ) -> str:
        """Generate image using Graphviz."""
        import graphviz
        
        # Create graph
        if layout == "hierarchical":
            dot = graphviz.Digraph(engine='dot')
        elif layout == "circular":
            dot = graphviz.Digraph(engine='circo')
        elif layout == "radial":
            dot = graphviz.Digraph(engine='twopi')
        else:
            dot = graphviz.Digraph(engine='neato')
        
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        # Add clusters if present
        clusters = defaultdict(list)
        for node in processed_graph["nodes"]:
            if "cluster" in node:
                clusters[node["cluster"]].append(node)
        
        if clusters:
            for cluster_name, cluster_nodes in clusters.items():
                with dot.subgraph(name=f'cluster_{cluster_name}') as c:
                    c.attr(label=cluster_name)
                    c.attr(style='filled', color='lightgrey')
                    for node in cluster_nodes:
                        c.node(node["id"], node["label"])
        else:
            # Add nodes
            for node in processed_graph["nodes"]:
                # Color by type
                colors = {
                    "python": "lightblue",
                    "javascript": "lightyellow", 
                    "java": "lightcoral",
                    "go": "lightgreen",
                    "rust": "wheat",
                    "cpp": "lavender",
                    "unknown": "lightgray"
                }
                color = colors.get(node["type"], "lightgray")
                dot.node(node["id"], node["label"], fillcolor=color)
        
        # Add edges
        for edge in processed_graph["edges"]:
            dot.edge(edge["source"], edge["target"])
        
        # Render
        if output_path:
            output_path = Path(output_path).with_suffix(f".{format}")
            dot.render(output_path.with_suffix(''), format=format, cleanup=True)
            return str(output_path)
        else:
            return dot.source
    
    def _generate_networkx_image(
        self,
        processed_graph: Dict,
        output_path: Optional[Path],
        format: str,
        layout: str
    ) -> str:
        """Generate image using NetworkX and Matplotlib."""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in processed_graph["nodes"]:
            G.add_node(node["id"], label=node["label"], type=node["type"])
        
        # Add edges
        for edge in processed_graph["edges"]:
            G.add_edge(edge["source"], edge["target"])
        
        # Choose layout
        if layout == "hierarchical":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Color nodes by type
        color_map = {
            "python": "#3776ab",
            "javascript": "#f7df1e",
            "java": "#007396",
            "go": "#00add8",
            "rust": "#dea584",
            "cpp": "#00599c",
            "unknown": "#888888"
        }
        
        node_colors = [
            color_map.get(G.nodes[node].get("type", "unknown"), "#888888")
            for node in G.nodes()
        ]
        
        # Draw graph
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=10,
            alpha=0.5,
            connectionstyle="arc3,rad=0.1"
        )
        
        # Add labels
        labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=8,
            font_family='sans-serif'
        )
        
        plt.title("Dependency Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        if output_path:
            output_path = Path(output_path).with_suffix(f".{format}")
            plt.savefig(output_path, format=format, dpi=150, bbox_inches='tight')
            plt.close()
            return str(output_path)
        else:
            # Return as bytes or similar
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format=format, dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            return buf.getvalue()