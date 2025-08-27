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

        if self._networkx_available:
            G = self.nx.DiGraph()
            for e in processed_graph["edges"]:
                G.add_edge(e["source"], e["target"])
            if layout == "hierarchical":
                pos = self.nx.spring_layout(G, k=2, iterations=50)
            elif layout == "circular":
                pos = self.nx.circular_layout(G)
            elif layout == "shell":
                pos = self.nx.shell_layout(G)
            else:
                pos = self.nx.spring_layout(G)
        else:
            pos: Dict[str, Any] = {}
            nodes = processed_graph["nodes"]
            n = len(nodes)
            cols = max(1, int(math.ceil(math.sqrt(max(1, n)))))
            for i, node in enumerate(nodes):
                row = i // cols
                col = i % cols
                pos[node["id"]] = (col, row)

        edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
        for e in processed_graph["edges"]:
            if e["source"] in pos and e["target"] in pos:
                x0, y0 = pos[e["source"]]
                x1, y1 = pos[e["target"]]
                edge_trace["x"] += (x0, x1, None)
                edge_trace["y"] += (y0, y1, None)

        node_trace = self.go.Scatter(
            x=[pos.get(n["id"], (0, 0))[0] for n in processed_graph["nodes"]],
            y=[pos.get(n["id"], (0, 0))[1] for n in processed_graph["nodes"]],
            mode="markers+text",
            hoverinfo="text",
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=10,
                colorbar=dict(title=dict(text="Connections", side="right"), thickness=15),
            ),
            text=[n["label"] for n in processed_graph["nodes"]],
            textposition="top center",
        )

        fig = self.go.Figure(
            data=[edge_trace, node_trace],
            layout=self.go.Layout(
                title=dict(text="Dependency Graph", font=dict(size=16)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor="white",
                plot_bgcolor="white",
            ),
        )

        html = fig.to_html(include_plotlyjs="cdn")
        if output_path:
            path = Path(output_path).with_suffix(".html")
            Path(path).write_text(html, encoding="utf-8")
            return str(path)
        return html

    def _generate_basic_html(self, processed_graph: Dict[str, Any], output_path: Optional[Path]) -> str:
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
    .node-label {{ font-size: 12px; pointer-events: none; }}
    .stats {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
    </style>
    </head>
<body>
    <h1>Dependency Graph</h1>
    <div class=\"stats\">
        <strong>Nodes:</strong> {node_count} |
        <strong>Edges:</strong> {edge_count} |
        <strong>Project Type:</strong> {project_type}
    </div>
    <svg id=\"graph\" width=\"1200\" height=\"800\"></svg>
    <div id=\"info\"></div>

    <script>
    const data = {graph_data};
    const width = 1200, height = 800;
    const svg = d3.select('#graph');
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width/2, height/2))
      .force('collision', d3.forceCollide().radius(30));

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

    const node = svg.append('g').selectAll('circle')
      .data(data.nodes).enter().append('circle')
      .attr('class', 'node').attr('r', 8)
      .attr('fill', d => ({{python:'#3776ab',javascript:'#f7df1e',java:'#007396',go:'#00add8',rust:'#dea584',cpp:'#00599c'}}[d.type] || '#888'))
      .call(d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended));

    const label = svg.append('g').selectAll('text')
      .data(data.nodes).enter().append('text')
      .attr('class', 'node-label').attr('dx', 12).attr('dy', 4)
      .text(d => d.label);

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

    function dragstarted(event, d) {{ if(!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
    function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
    function dragended(event, d) {{ if(!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}
    </script>
  </body>
 </html>"""
        html = html_template.format(
            node_count=len(processed_graph["nodes"]),
            edge_count=len(processed_graph["edges"]),
            project_type=(processed_graph.get("metadata") or {}).get("project_type", "unknown"),
            graph_data=json.dumps(processed_graph),
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

        nodes = processed_graph["nodes"]
        n = len(nodes) or 1
        cols = max(1, int(math.ceil(math.sqrt(n))))
        pos: Dict[str, Any] = {}
        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            pos[node["id"]] = (col, row)

        edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
        for e in processed_graph["edges"]:
            if e["source"] in pos and e["target"] in pos:
                x0, y0 = pos[e["source"]]
                x1, y1 = pos[e["target"]]
                edge_trace["x"] += (x0, x1, None)
                edge_trace["y"] += (y0, y1, None)

        node_trace = self.go.Scatter(
            x=[pos.get(n["id"], (0, 0))[0] for n in nodes],
            y=[pos.get(n["id"], (0, 0))[1] for n in nodes],
            mode="markers+text",
            hoverinfo="text",
            marker=dict(size=10, color="#1f77b4"),
            text=[n["label"] for n in nodes],
            textposition="top center",
        )

        fig = self.go.Figure(
            data=[edge_trace, node_trace],
            layout=self.go.Layout(
                title=dict(text="Dependency Graph", font=dict(size=16)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                paper_bgcolor="white",
                plot_bgcolor="white",
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

        plt.figure(figsize=(16, 12))
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

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="gray",
            arrows=True,
            arrowsize=10,
            alpha=0.5,
            connectionstyle="arc3,rad=0.1",
        )
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="sans-serif")

        plt.title("Dependency Graph", fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        path = Path(output_path or Path("dependency_graph")).with_suffix(f".{format}")
        plt.savefig(path, format=format, dpi=150, bbox_inches="tight")
        plt.close()
        return str(path)
