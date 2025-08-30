"""Graph generation for dependency visualization.

Pure-Python backends (pip-installable) are preferred:
- Plotly + Kaleido for static/interactive graphs
- NetworkX + Matplotlib as a fallback
- Graphviz only if available (requires system binaries)
- DOT/HTML text fallback otherwise
"""

from __future__ import annotations

import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tenets.utils.logger import get_logger


class GraphGenerator:
    """Generates various graph visualizations for dependencies."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        # Capability flags
        self._networkx_available = False
        self._matplotlib_available = False
        self._graphviz_available = False
        self._plotly_available = False
        self._kaleido_available = False

        # Optional imports (best-effort)
        try:
            import networkx as nx  # type: ignore

            self.nx = nx
            self._networkx_available = True
        except Exception:
            self.logger.debug("NetworkX not available - pip install networkx")

        try:
            import matplotlib  # type: ignore

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore

            self.plt = plt
            self._matplotlib_available = True
        except Exception:
            self.logger.debug("Matplotlib not available - pip install matplotlib")

        try:
            import graphviz  # type: ignore

            self.graphviz = graphviz
            self._graphviz_available = True
        except Exception:
            self.logger.debug("Graphviz not available - pip install graphviz (and install system Graphviz)")

        try:
            import plotly.graph_objects as go  # type: ignore

            self.go = go
            self._plotly_available = True
        except Exception:
            self.logger.debug("Plotly not available - pip install plotly")

        try:
            import kaleido  # noqa: F401  # type: ignore

            self._kaleido_available = True
        except Exception:
            self._kaleido_available = False

    # ------------- Public API -------------
    def generate_graph(
        self,
        dependency_graph: Dict[str, List[str]],
        output_path: Optional[Path] = None,
        format: str = "svg",
        layout: str = "hierarchical",
        cluster_by: Optional[str] = None,
        max_nodes: Optional[int] = None,
        project_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a dependency graph visualization.

        Args:
            dependency_graph: node -> list of dependencies
            output_path: where to save; if None, return string content
            format: svg, png, pdf, html, json, dot
            layout: layout hint (hierarchical, circular, shell, kamada)
            cluster_by: module, directory, package
            max_nodes: optional cap on number of nodes
            project_info: optional project metadata
        """
        processed = self._process_graph(
            dependency_graph,
            cluster_by=cluster_by,
            max_nodes=max_nodes,
            project_info=project_info,
        )

        if format == "json":
            return self._generate_json(processed, output_path)
        if format == "dot":
            return self._generate_dot(processed, output_path)
        if format == "html":
            return self._generate_html(processed, output_path, layout)
        if format in ("svg", "png", "pdf"):
            return self._generate_image(processed, output_path, format, layout)
        raise ValueError(f"Unsupported format: {format}")

    # ------------- Graph processing -------------
    def _process_graph(
        self,
        dependency_graph: Dict[str, List[str]],
        cluster_by: Optional[str] = None,
        max_nodes: Optional[int] = None,
        project_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        processed: Dict[str, Any] = {"nodes": [], "edges": [], "clusters": {}, "metadata": {}}

        if cluster_by:
            clustered = self._cluster_nodes(dependency_graph, cluster_by, project_info)
            processed["clusters"] = clustered["clusters"]
            nodes: Set[str] = clustered["nodes"]
            edges: List[Dict[str, str]] = clustered["edges"]
        else:
            nodes = set(dependency_graph.keys())
            for deps in dependency_graph.values():
                nodes.update(deps)
            edges = []
            for src, tgts in dependency_graph.items():
                for tgt in tgts:
                    edges.append({"source": src, "target": tgt})

        if max_nodes and len(nodes) > max_nodes:
            degree = defaultdict(int)
            for e in edges:
                degree[e["source"]] += 1
                degree[e["target"]] += 1
            top = sorted(nodes, key=lambda n: degree.get(n, 0), reverse=True)[:max_nodes]
            nodes = set(top)
            edges = [e for e in edges if e["source"] in nodes and e["target"] in nodes]

        for node in nodes:
            info: Dict[str, Any] = {
                "id": node,
                "label": self._get_node_label(node),
                "type": self._get_node_type(node, project_info),
            }
            if cluster_by and node in processed["clusters"]:
                info["cluster"] = processed["clusters"][node]
            processed["nodes"].append(info)

        processed["edges"] = edges
        processed["metadata"] = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "clustered": cluster_by is not None,
            "project_type": (project_info or {}).get("type"),
        }
        return processed

    def _cluster_nodes(
        self,
        dependency_graph: Dict[str, List[str]],
        cluster_by: str,
        project_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        clusters: Dict[str, str] = {}
        edges: List[Dict[str, str]] = []
        for src, tgts in dependency_graph.items():
            clusters[src] = self._get_cluster(src, cluster_by, project_info)
            for tgt in tgts:
                clusters[tgt] = self._get_cluster(tgt, cluster_by, project_info)
                edges.append({"source": src, "target": tgt})
        nodes: Set[str] = set(clusters.keys())
        return {"nodes": nodes, "edges": edges, "clusters": clusters}

    def _get_cluster(self, node: str, cluster_by: str, project_info: Optional[Dict[str, Any]]) -> str:
        parts = node.replace("\\", "/").split("/")
        if cluster_by == "directory":
            return parts[-2] if len(parts) > 1 else "root"
        if cluster_by == "module":
            if project_info and str(project_info.get("type", "")).startswith("python"):
                return ".".join(parts[:-1]) if len(parts) > 1 else "root"
            return parts[0] if parts else "root"
        if cluster_by == "package":
            return parts[0] if parts else "root"
        return "default"

    def _get_node_label(self, node: str) -> str:
        return Path(node).name if ("/" in node or "\\" in node) else node

    def _get_node_type(self, node: str, project_info: Optional[Dict[str, Any]]) -> str:
        if node.endswith((".py", ".pyw")):
            return "python"
        if node.endswith((".js", ".jsx", ".ts", ".tsx")):
            return "javascript"
        if node.endswith((".java",)):
            return "java"
        if node.endswith((".go",)):
            return "go"
        if node.endswith((".rs",)):
            return "rust"
        if node.endswith((".cpp", ".cc", ".cxx", ".hpp", ".h")):
            return "cpp"
        if node.endswith((".cs",)):
            return "csharp"
        if node.endswith((".rb",)):
            return "ruby"
        if node.endswith((".php",)):
            return "php"
        return "unknown"

    # ------------- Renderers -------------
    def _generate_json(self, processed_graph: Dict[str, Any], output_path: Optional[Path]) -> str:
        data = json.dumps(processed_graph, indent=2)
        if output_path:
            path = Path(output_path).with_suffix(".json")
            Path(path).write_text(data, encoding="utf-8")
            return str(path)
        return data

    def _generate_dot(self, processed_graph: Dict[str, Any], output_path: Optional[Path]) -> str:
        lines: List[str] = ["digraph Dependencies {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=rounded];")

        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in processed_graph["nodes"]:
            if "cluster" in node:
                clusters[node["cluster"]].append(node)

        if clusters:
            for i, (cluster_name, cluster_nodes) in enumerate(clusters.items()):
                lines.append(f"  subgraph cluster_{i} {{")
                lines.append(f'    label="{cluster_name}";')
                lines.append("    style=filled;")
                lines.append("    color=lightgrey;")
                for node in cluster_nodes:
                    nid = str(node["id"]).replace('"', '\\"')
                    nlabel = str(node["label"]).replace('"', '\\"')
                    lines.append(f'    "{nid}" [label="{nlabel}"];')
                lines.append("  }")
        else:
            for node in processed_graph["nodes"]:
                nid = str(node["id"]).replace('"', '\\"')
                nlabel = str(node["label"]).replace('"', '\\"')
                lines.append(f'  "{nid}" [label="{nlabel}"];')

        for edge in processed_graph["edges"]:
            s = str(edge["source"]).replace('"', '\\"')
            t = str(edge["target"]).replace('"', '\\"')
            lines.append(f'  "{s}" -> "{t}";')

        lines.append("}")
        dot = "\n".join(lines)
        if output_path:
            path = Path(output_path).with_suffix(".dot")
            Path(path).write_text(dot, encoding="utf-8")
            return str(path)
        return dot

    def _generate_html(self, processed_graph: Dict[str, Any], output_path: Optional[Path], layout: str) -> str:
        if self._plotly_available:
            return self._generate_plotly_html(processed_graph, output_path, layout)
        return self._generate_basic_html(processed_graph, output_path)

    def _generate_plotly_html(self, processed_graph: Dict[str, Any], output_path: Optional[Path], layout: str) -> str:
        go = self.go
        
        # Calculate graph metrics for dynamic spacing
        num_nodes = len(processed_graph["nodes"])
        num_edges = len(processed_graph["edges"])
        graph_density = num_edges / max(1, num_nodes * (num_nodes - 1))  # Directed graph density
        
        # Calculate maximum text length for spacing considerations
        max_label_length = max((len(n["label"]) for n in processed_graph["nodes"]), default=10)
        
        # Dynamic spacing parameters based on graph size and density
        # More nodes = tighter spacing, but never too tight
        base_k = max(1.0, 10.0 / math.sqrt(max(1, num_nodes)))  # Inverse sqrt for smooth scaling
        k_factor = base_k * (1 + graph_density * 2)  # Increase spacing for denser graphs
        
        # Scale factor increases for smaller graphs, decreases for larger ones
        scale_factor = max(0.5, min(3.0, 50.0 / max(10, num_nodes)))
        
        # Iterations based on graph size (more iterations for larger graphs to converge)
        iterations = min(200, max(50, num_nodes * 2))

        if self._networkx_available:
            G = self.nx.DiGraph()
            for e in processed_graph["edges"]:
                G.add_edge(e["source"], e["target"])
            # Dynamic layout parameters based on graph metrics
            if layout == "hierarchical":
                pos = self.nx.spring_layout(G, k=k_factor, iterations=iterations, scale=scale_factor)
            elif layout == "circular":
                # Circular layout with dynamic radius
                radius_scale = scale_factor * math.sqrt(num_nodes / 10)
                pos = self.nx.circular_layout(G, scale=radius_scale)
            elif layout == "shell":
                # Shell layout with dynamic scaling
                pos = self.nx.shell_layout(G, scale=scale_factor * 1.5)
            else:
                # Default with dynamic parameters
                pos = self.nx.spring_layout(G, k=k_factor * 0.7, iterations=iterations, scale=scale_factor)
        else:
            pos: Dict[str, Any] = {}
            nodes = processed_graph["nodes"]
            n = len(nodes)
            # Dynamic grid spacing based on node count and label sizes
            cols = max(1, int(math.ceil(math.sqrt(max(1, n)))))
            rows = math.ceil(n / cols)
            
            # Calculate spacing to fit in viewport with text
            spacing_x = max(1.5, 10.0 / cols)  # Horizontal spacing
            spacing_y = max(1.5, 8.0 / rows)   # Vertical spacing
            
            # Adjust for label length
            spacing_x *= (1 + max_label_length / 20)
            
            for i, node in enumerate(nodes):
                row = i // cols
                col = i % cols
                # Center the grid
                x = (col - cols/2) * spacing_x
                y = (row - rows/2) * spacing_y
                pos[node["id"]] = (x, y)

        edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
        for e in processed_graph["edges"]:
            if e["source"] in pos and e["target"] in pos:
                x0, y0 = pos[e["source"]]
                x1, y1 = pos[e["target"]]
                edge_trace["x"] += (x0, x1, None)
                edge_trace["y"] += (y0, y1, None)

        # Calculate node degrees for sizing
        node_degrees = {}
        for e in processed_graph["edges"]:
            node_degrees[e["source"]] = node_degrees.get(e["source"], 0) + 1
            node_degrees[e["target"]] = node_degrees.get(e["target"], 0) + 1
        
        # Dynamic node sizing based on graph size and degree
        min_node_size = max(8, 30 / math.sqrt(max(1, num_nodes)))  # Smaller min for large graphs
        max_node_size = min(60, 200 / math.sqrt(max(1, num_nodes)))  # Smaller max for large graphs
        
        # Calculate node sizes with dynamic range
        max_degree = max(node_degrees.values()) if node_degrees else 1
        node_sizes = []
        for n in processed_graph["nodes"]:
            degree = node_degrees.get(n["id"], 0)
            # Logarithmic scaling for better distribution
            size_factor = math.log(degree + 1) / math.log(max_degree + 1) if max_degree > 0 else 0
            size = min_node_size + (max_node_size - min_node_size) * size_factor
            node_sizes.append(size)
        
        # Dynamic text size based on number of nodes and viewport
        base_font_size = max(6, min(14, 200 / max(10, num_nodes)))
        # Adjust font size based on label length
        avg_label_length = sum(len(n["label"]) for n in processed_graph["nodes"]) / max(1, num_nodes)
        if avg_label_length > 15:
            base_font_size *= 0.8  # Smaller font for longer labels
        
        # Calculate hover text with more info
        hover_texts = []
        for n in processed_graph["nodes"]:
            degree = node_degrees.get(n["id"], 0)
            hover_text = f"<b>{n['label']}</b><br>"
            hover_text += f"Type: {n.get('type', 'unknown')}<br>"
            hover_text += f"Connections: {degree}<br>"
            if 'cluster' in n:
                hover_text += f"Cluster: {n['cluster']}"
            hover_texts.append(hover_text)
        
        node_trace = self.go.Scatter(
            x=[pos.get(n["id"], (0, 0))[0] for n in processed_graph["nodes"]],
            y=[pos.get(n["id"], (0, 0))[1] for n in processed_graph["nodes"]],
            mode="markers+text",
            hoverinfo="text",
            hovertext=hover_texts,
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=node_sizes,  # Dynamic sizing
                color=[node_degrees.get(n["id"], 0) for n in processed_graph["nodes"]],
                colorbar=dict(title=dict(text="Connections", side="right"), thickness=15),
                line=dict(width=max(0.5, 2 / math.sqrt(max(1, num_nodes))), color='white')  # Dynamic border
            ),
            text=[n["label"] for n in processed_graph["nodes"]],
            textposition="top center",
            textfont=dict(size=base_font_size)  # Dynamic font size
        )
        
        # Dynamic title size based on graph size
        title_size = max(12, min(20, 24 - num_nodes / 50))
        
        # Dynamic margins based on label sizes
        margin_padding = max(10, min(40, max_label_length * 2))

        fig = self.go.Figure(
            data=[edge_trace, node_trace],
            layout=self.go.Layout(
                title=dict(
                    text=f"Dependency Graph ({num_nodes} nodes, {num_edges} edges)",
                    font=dict(size=title_size)
                ),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=margin_padding, l=margin_padding, r=margin_padding, t=margin_padding + 20),
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    scaleanchor="y",  # Keep aspect ratio
                    scaleratio=1
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False
                ),
                paper_bgcolor="white",
                plot_bgcolor="white",
                autosize=True,  # Auto-resize to container
                width=max(800, min(1600, 50 * math.sqrt(num_nodes))),  # Dynamic width
                height=max(600, min(1200, 40 * math.sqrt(num_nodes))),  # Dynamic height
            ),
        )

        html = fig.to_html(include_plotlyjs="cdn")
        if output_path:
            path = Path(output_path).with_suffix(".html")
            Path(path).write_text(html, encoding="utf-8")
            return str(path)
        return html

    def _generate_basic_html(self, processed_graph: Dict[str, Any], output_path: Optional[Path]) -> str:
        # Calculate graph metrics for dynamic sizing
        num_nodes = len(processed_graph["nodes"])
        num_edges = len(processed_graph["edges"])
        max_label_length = max((len(n["label"]) for n in processed_graph["nodes"]), default=10)
        
        # Dynamic dimensions based on graph size
        svg_width = max(800, min(1600, 100 + num_nodes * 15))
        svg_height = max(600, min(1200, 100 + num_nodes * 10))
        
        # Dynamic force parameters
        link_distance = max(50, min(200, 500 / math.sqrt(max(1, num_nodes))))
        charge_strength = max(-1500, min(-200, -5000 / math.sqrt(max(1, num_nodes))))
        collision_radius = max(20, min(80, 200 / math.sqrt(max(1, num_nodes))))
        
        # Dynamic font size
        font_size = max(8, min(14, 150 / max(10, num_nodes)))
        
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset=\"utf-8\" />
    <title>Dependency Graph</title>
    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
    #graph {{ border: 1px solid #ccc; }}
    .node {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
    .link {{ stroke: #999; stroke-opacity: 0.6; }}
    .node-label {{ font-size: {font_size}px; pointer-events: none; }}
    .stats {{ background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
    #controls {{ margin-bottom: 10px; }}
    button {{ margin-right: 10px; padding: 5px 10px; }}
    </style>
    </head>
<body>
    <h1>Dependency Graph</h1>
    <div class=\"stats\">
        <strong>Nodes:</strong> {node_count} |
        <strong>Edges:</strong> {edge_count} |
        <strong>Density:</strong> {density:.3f} |
        <strong>Project Type:</strong> {project_type}
    </div>
    <div id=\"controls\">
        <button onclick=\"resetView()\">Reset View</button>
        <button onclick=\"toggleLabels()\">Toggle Labels</button>
    </div>
    <svg id=\"graph\" width=\"{svg_width}\" height=\"{svg_height}\"></svg>
    <div id=\"info\"></div>

    <script>
    const data = {graph_data};
    const width = {svg_width}, height = {svg_height};
    const svg = d3.select('#graph');
    let showLabels = true;
    
    // Calculate node degrees for sizing
    const nodeDegrees = {{}};
    data.edges.forEach(e => {{
        nodeDegrees[e.source] = (nodeDegrees[e.source] || 0) + 1;
        nodeDegrees[e.target] = (nodeDegrees[e.target] || 0) + 1;
    }});
    
    // Add degree info to nodes
    data.nodes.forEach(n => {{
        n.degree = nodeDegrees[n.id] || 0;
    }});
    
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges).id(d => d.id).distance({link_distance}))
      .force('charge', d3.forceManyBody().strength({charge_strength}))
      .force('center', d3.forceCenter(width/2, height/2))
      .force('collision', d3.forceCollide().radius({collision_radius}))
      .force('x', d3.forceX(width/2).strength(0.05))
      .force('y', d3.forceY(height/2).strength(0.05));

    svg.append('defs').selectAll('marker')
      .data(['arrow']).enter().append('marker')
      .attr('id', d => d).attr('viewBox', '0 -5 10 10')
      .attr('refX', 20).attr('refY', 0)
      .attr('markerWidth', 8).attr('markerHeight', 8)
      .attr('orient', 'auto')
      .append('path').attr('d', 'M0,-5L10,0L0,5').attr('fill', '#999');

    const link = svg.append('g').selectAll('line')
      .data(data.edges).enter().append('line')
      .attr('class', 'link')
      .attr('marker-end', 'url(#arrow)');

    // Dynamic node sizing based on degree
    const maxDegree = Math.max(...Object.values(nodeDegrees), 1);
    const minNodeSize = Math.max(4, 20 / Math.sqrt(Math.max(1, data.nodes.length)));
    const maxNodeSize = Math.min(20, 100 / Math.sqrt(Math.max(1, data.nodes.length)));
    
    const nodeScale = d => {{
        const degree = nodeDegrees[d.id] || 0;
        const scale = Math.sqrt(degree + 1) / Math.sqrt(maxDegree + 1);
        return minNodeSize + (maxNodeSize - minNodeSize) * scale;
    }};

    const node = svg.append('g').selectAll('circle')
      .data(data.nodes).enter().append('circle')
      .attr('class', 'node')
      .attr('r', nodeScale)
      .attr('fill', d => ({{python:'#3776ab',javascript:'#f7df1e',java:'#007396',go:'#00add8',rust:'#dea584',cpp:'#00599c'}}[d.type] || '#888'))
      .call(d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended));

    const label = svg.append('g').selectAll('text')
      .data(data.nodes).enter().append('text')
      .attr('class', 'node-label')
      .attr('dx', d => nodeScale(d) + 5)
      .attr('dy', 4)
      .text(d => d.label)
      .style('display', showLabels ? 'block' : 'none');

    node.on('mouseover', (event, d) => {{
      document.getElementById('info').innerHTML = `
        <strong>Node:</strong> ${{d.id}}<br/>
        <strong>Type:</strong> ${{d.type}}<br/>
        <strong>Cluster:</strong> ${{(d.cluster || 'none')}}
      `;
    }});

    simulation.on('tick', () => {{
      link.attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
      label.attr('x', d => d.x).attr('y', d => d.y);
    }});

    function dragstarted(event, d) {{ 
        if(!event.active) simulation.alphaTarget(0.3).restart(); 
        d.fx = d.x; 
        d.fy = d.y; 
    }}
    
    function dragged(event, d) {{ 
        d.fx = event.x; 
        d.fy = event.y; 
    }}
    
    function dragended(event, d) {{ 
        if(!event.active) simulation.alphaTarget(0); 
        d.fx = null; 
        d.fy = null; 
    }}
    
    function resetView() {{
        simulation.alpha(1).restart();
    }}
    
    function toggleLabels() {{
        showLabels = !showLabels;
        label.style('display', showLabels ? 'block' : 'none');
    }}
    </script>
  </body>
 </html>"""
        # Calculate density for display
        graph_density = num_edges / max(1, num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        html = html_template.format(
            node_count=num_nodes,
            edge_count=num_edges,
            density=graph_density,
            project_type=(processed_graph.get("metadata") or {}).get("project_type", "unknown"),
            graph_data=json.dumps(processed_graph),
            svg_width=svg_width,
            svg_height=svg_height,
            font_size=font_size,
            link_distance=link_distance,
            charge_strength=charge_strength,
            collision_radius=collision_radius,
        )
        if output_path:
            path = Path(output_path).with_suffix(".html")
            Path(path).write_text(html, encoding="utf-8")
            return str(path)
        return html

    def _generate_image(self, processed_graph: Dict[str, Any], output_path: Optional[Path], format: str, layout: str) -> str:
        """Generate static image (SVG, PNG, PDF) with pip-first strategy.

        Order: Plotly+Kaleido -> NetworkX+Matplotlib -> Graphviz -> DOT fallback.
        """
        if self._plotly_available and self._kaleido_available:
            try:
                return self._generate_plotly_image(processed_graph, output_path, format)
            except Exception as e:
                self.logger.warning(f"Plotly static export failed, trying other backends: {e}")

        if self._networkx_available and self._matplotlib_available:
            try:
                return self._generate_networkx_image(processed_graph, output_path, format, layout)
            except Exception as e:
                self.logger.warning(f"Matplotlib export failed, trying Graphviz: {e}")

        if self._graphviz_available:
            try:
                return self._generate_graphviz_image(processed_graph, output_path, format, layout)
            except Exception as e:
                self.logger.warning(f"Graphviz export failed, falling back to DOT: {e}")

        self.logger.warning(
            "No image backends available (plotly+kaleido | networkx+matplotlib | graphviz). "
            "Falling back to DOT. Install with: pip install 'plotly kaleido' or 'networkx matplotlib'"
        )
        return self._generate_dot(processed_graph, output_path)

    def _generate_plotly_image(self, processed_graph: Dict[str, Any], output_path: Optional[Path], format: str) -> str:
        go = self.go

        # Try to use NetworkX for better layouts if available
        if self._networkx_available:
            G = self.nx.DiGraph()
            for e in processed_graph["edges"]:
                G.add_edge(e["source"], e["target"])
            # Use spring layout with good spacing
            pos = self.nx.spring_layout(G, k=3, iterations=75, scale=2.0)
        else:
            nodes = processed_graph["nodes"]
            n = len(nodes) or 1
            cols = max(1, int(math.ceil(math.sqrt(n))))
            spacing = 3.0  # Much better spacing
            pos: Dict[str, Any] = {}
            for i, node in enumerate(nodes):
                row = i // cols
                col = i % cols
                pos[node["id"]] = (col * spacing, row * spacing)

        # Calculate metrics for dynamic sizing
        num_nodes = len(nodes)
        num_edges = len(processed_graph["edges"])
        max_label_length = max((len(n["label"]) for n in nodes), default=10)
        
        edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
        for e in processed_graph["edges"]:
            if e["source"] in pos and e["target"] in pos:
                x0, y0 = pos[e["source"]]
                x1, y1 = pos[e["target"]]
                edge_trace["x"] += (x0, x1, None)
                edge_trace["y"] += (y0, y1, None)

        # Calculate node degrees for sizing
        node_degrees = {}
        for e in processed_graph["edges"]:
            node_degrees[e["source"]] = node_degrees.get(e["source"], 0) + 1
            node_degrees[e["target"]] = node_degrees.get(e["target"], 0) + 1
        
        # Dynamic node sizing
        min_size = max(5, 20 / math.sqrt(max(1, num_nodes)))
        max_size = min(30, 100 / math.sqrt(max(1, num_nodes)))
        max_degree = max(node_degrees.values()) if node_degrees else 1
        
        node_sizes = []
        for n in nodes:
            degree = node_degrees.get(n["id"], 0)
            size_factor = math.log(degree + 1) / math.log(max_degree + 1) if max_degree > 0 else 0
            size = min_size + (max_size - min_size) * size_factor
            node_sizes.append(size)
        
        # Dynamic font size
        font_size = max(8, min(12, 150 / max(10, num_nodes)))

        node_trace = self.go.Scatter(
            x=[pos.get(n["id"], (0, 0))[0] for n in nodes],
            y=[pos.get(n["id"], (0, 0))[1] for n in nodes],
            mode="markers+text",
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color=[node_degrees.get(n["id"], 0) for n in nodes],
                colorscale="YlGnBu",
                showscale=True,
                colorbar=dict(title="Connections", thickness=15),
                line=dict(width=1, color='white')
            ),
            text=[n["label"] for n in nodes],
            textposition="top center",
            textfont=dict(size=font_size),
        )
        
        # Dynamic dimensions
        width = max(800, min(1600, 50 * math.sqrt(num_nodes)))
        height = max(600, min(1200, 40 * math.sqrt(num_nodes)))
        margin = max(20, min(50, max_label_length * 2))

        fig = self.go.Figure(
            data=[edge_trace, node_trace],
            layout=self.go.Layout(
                title=dict(text=f"Dependency Graph ({num_nodes} nodes)", font=dict(size=14)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=margin, l=margin, r=margin, t=margin + 20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor="white",
                plot_bgcolor="white",
                width=width,
                height=height,
            ),
        )

        path = Path(output_path or Path("dependency_graph")).with_suffix(f".{format}")
        fig.write_image(str(path))
        return str(path)

    def _generate_graphviz_image(self, processed_graph: Dict[str, Any], output_path: Optional[Path], format: str, layout: str) -> str:
        graphviz = self.graphviz

        if layout == "hierarchical":
            dot = graphviz.Digraph(engine="dot")
        elif layout == "circular":
            dot = graphviz.Digraph(engine="circo")
        elif layout == "radial":
            dot = graphviz.Digraph(engine="twopi")
        else:
            dot = graphviz.Digraph(engine="neato")

        dot.attr(rankdir="LR")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in processed_graph["nodes"]:
            if "cluster" in node:
                clusters[node["cluster"]].append(node)

        if clusters:
            for name, cluster_nodes in clusters.items():
                with dot.subgraph(name=f"cluster_{name}") as c:
                    c.attr(label=name)
                    c.attr(style="filled", color="lightgrey")
                    for node in cluster_nodes:
                        c.node(node["id"], node["label"])
        else:
            for node in processed_graph["nodes"]:
                colors = {
                    "python": "lightblue",
                    "javascript": "lightyellow",
                    "java": "lightcoral",
                    "go": "lightgreen",
                    "rust": "wheat",
                    "cpp": "lavender",
                    "unknown": "lightgray",
                }
                color = colors.get(node.get("type", "unknown"), "lightgray")
                dot.node(node["id"], node["label"], fillcolor=color)

        for e in processed_graph["edges"]:
            dot.edge(e["source"], e["target"])

        path = Path(output_path or Path("dependency_graph")).with_suffix(f".{format}")
        dot.render(path.with_suffix(""), format=format, cleanup=True)
        return str(path)

    def _generate_networkx_image(self, processed_graph: Dict[str, Any], output_path: Optional[Path], format: str, layout: str) -> str:
        plt = self.plt
        nx = self.nx

        G = nx.DiGraph()
        for node in processed_graph["nodes"]:
            G.add_node(node["id"], label=node["label"], type=node["type"]) 
        for e in processed_graph["edges"]:
            G.add_edge(e["source"], e["target"])

        # Calculate dynamic parameters
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        
        # Dynamic spacing based on graph size
        base_k = max(1.0, 10.0 / math.sqrt(max(1, num_nodes)))
        scale_factor = max(0.5, min(3.0, 50.0 / max(10, num_nodes)))
        iterations = min(200, max(50, num_nodes * 2))
        
        # Improved layout parameters with dynamic spacing
        if layout == "hierarchical":
            pos = nx.spring_layout(G, k=base_k, iterations=iterations, scale=scale_factor)
        elif layout == "circular":
            radius_scale = scale_factor * math.sqrt(num_nodes / 10)
            pos = nx.circular_layout(G, scale=radius_scale)
        elif layout == "shell":
            pos = nx.shell_layout(G, scale=scale_factor * 1.5)
        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(G, scale=scale_factor)
        else:
            pos = nx.spring_layout(G, k=base_k * 0.7, iterations=iterations, scale=scale_factor)

        # Dynamic figure size based on graph size
        fig_width = max(10, min(20, 8 + math.sqrt(num_nodes)))
        fig_height = max(8, min(16, 6 + math.sqrt(num_nodes)))
        plt.figure(figsize=(fig_width, fig_height))
        
        color_map = {
            "python": "#3776ab",
            "javascript": "#f7df1e",
            "java": "#007396",
            "go": "#00add8",
            "rust": "#dea584",
            "cpp": "#00599c",
            "unknown": "#888888",
        }
        node_colors = [color_map.get(G.nodes[n].get("type", "unknown"), "#888888") for n in G.nodes()]
        
        # Dynamic node sizes based on degree and graph size
        min_node_size = max(100, 500 / math.sqrt(max(1, num_nodes)))
        max_node_size = min(2000, 5000 / math.sqrt(max(1, num_nodes)))
        max_degree = max((G.degree(n) for n in G.nodes()), default=1)
        
        node_sizes = []
        for n in G.nodes():
            degree = G.degree(n)
            size_factor = math.log(degree + 1) / math.log(max_degree + 1) if max_degree > 0 else 0
            size = min_node_size + (max_node_size - min_node_size) * size_factor
            node_sizes.append(size)

        # Dynamic font size
        font_size = max(6, min(12, 150 / max(10, num_nodes)))
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, linewidths=2, edgecolors='white')
        
        # Dynamic edge width
        edge_width = max(0.5, min(2, 30 / max(10, num_nodes)))
        arrow_size = max(8, min(20, 200 / max(10, num_nodes)))
        
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="gray",
            arrows=True,
            arrowsize=arrow_size,
            alpha=0.5,
            connectionstyle="arc3,rad=0.1",
            width=edge_width,
        )
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_family="sans-serif", font_weight='bold')

        plt.title("Dependency Graph", fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        path = Path(output_path or Path("dependency_graph")).with_suffix(f".{format}")
        plt.savefig(path, format=format, dpi=150, bbox_inches="tight")
        plt.close()
        return str(path)
