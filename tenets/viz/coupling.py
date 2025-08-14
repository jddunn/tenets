"""Coupling visualization module.

This module provides visualization capabilities for code coupling metrics,
including afferent/efferent coupling, instability, and coupling networks.
"""

from typing import Any, Dict, List, Optional

from .base import BaseVisualizer, ChartConfig, ChartType, ColorPalette, DisplayConfig
from .displays import TerminalDisplay


class CouplingVisualizer(BaseVisualizer):
    """Visualizer for coupling metrics.

    Creates visualizations for coupling analysis including dependency
    graphs, coupling matrices, and stability charts.
    """

    def __init__(
        self,
        chart_config: Optional[ChartConfig] = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """Initialize coupling visualizer.

        Args:
            chart_config: Chart configuration
            display_config: Display configuration
        """
        super().__init__(chart_config, display_config)
        self.terminal_display = TerminalDisplay(display_config)

    def create_coupling_network(
        self, coupling_data: Dict[str, Dict[str, int]], min_coupling: int = 1, max_nodes: int = 50
    ) -> Dict[str, Any]:
        """Create coupling network graph.

        Args:
            coupling_data: Dictionary of module -> {coupled_module: strength}
            min_coupling: Minimum coupling strength to show
            max_nodes: Maximum nodes to display

        Returns:
            Dict[str, Any]: Network graph configuration
        """
        # Build nodes and edges
        nodes_set = set()
        edges = []
        node_coupling = {}

        for module, coupled_modules in coupling_data.items():
            for coupled_module, strength in coupled_modules.items():
                if strength >= min_coupling:
                    nodes_set.add(module)
                    nodes_set.add(coupled_module)
                    edges.append({"source": module, "target": coupled_module, "weight": strength})

                    # Track total coupling per node
                    node_coupling[module] = node_coupling.get(module, 0) + strength
                    node_coupling[coupled_module] = node_coupling.get(coupled_module, 0) + strength

        # Limit nodes if necessary
        if len(nodes_set) > max_nodes:
            # Keep nodes with highest coupling
            sorted_nodes = sorted(nodes_set, key=lambda n: node_coupling.get(n, 0), reverse=True)[
                :max_nodes
            ]
            nodes_set = set(sorted_nodes)

            # Filter edges
            edges = [e for e in edges if e["source"] in nodes_set and e["target"] in nodes_set]

        # Create node list with sizing and coloring
        nodes = []
        for node_id in nodes_set:
            coupling_strength = node_coupling.get(node_id, 0)

            # Size based on coupling
            size = min(50, 10 + coupling_strength)

            # Color based on coupling level
            if coupling_strength > 20:
                color = ColorPalette.SEVERITY["critical"]
            elif coupling_strength > 10:
                color = ColorPalette.SEVERITY["high"]
            elif coupling_strength > 5:
                color = ColorPalette.SEVERITY["medium"]
            else:
                color = ColorPalette.HEALTH["good"]

            nodes.append(
                {
                    "id": node_id,
                    "label": self._truncate_module_name(node_id),
                    "size": size,
                    "color": color,
                }
            )

        config = ChartConfig(type=ChartType.NETWORK, title="Module Coupling Network")

        return self.create_chart(
            ChartType.NETWORK, {"nodes": nodes, "edges": edges, "layout": "force"}, config
        )

    def create_coupling_matrix(
        self, modules: List[str], coupling_matrix: List[List[int]]
    ) -> Dict[str, Any]:
        """Create coupling matrix heatmap.

        Args:
            modules: List of module names
            coupling_matrix: 2D matrix of coupling values

        Returns:
            Dict[str, Any]: Heatmap configuration
        """
        # Truncate module names for display
        labels = [self._truncate_module_name(m) for m in modules]

        config = ChartConfig(type=ChartType.HEATMAP, title="Module Coupling Matrix")

        return self.create_chart(
            ChartType.HEATMAP,
            {"matrix": coupling_matrix, "x_labels": labels, "y_labels": labels},
            config,
        )

    def create_instability_chart(
        self, instability_data: List[Dict[str, Any]], limit: int = 20
    ) -> Dict[str, Any]:
        """Create instability chart for modules.

        Args:
            instability_data: List of modules with instability metrics
            limit: Maximum modules to show

        Returns:
            Dict[str, Any]: Scatter plot configuration
        """
        # Sort by instability
        sorted_data = sorted(instability_data, key=lambda x: x.get("instability", 0), reverse=True)[
            :limit
        ]

        points = []
        labels = []

        for module in sorted_data:
            efferent = module.get("efferent_coupling", 0)
            afferent = module.get("afferent_coupling", 0)
            points.append((efferent, afferent))
            labels.append(self._truncate_module_name(module.get("name", "")))

        # Add ideal line (main sequence)
        max_coupling = max(
            max(p[0] for p in points) if points else 10, max(p[1] for p in points) if points else 10
        )

        config = ChartConfig(type=ChartType.SCATTER, title="Instability vs Abstractness")

        chart_config = self.create_chart(ChartType.SCATTER, {"points": points}, config)

        # Add main sequence line
        chart_config["data"]["datasets"].append(
            {
                "type": "line",
                "label": "Main Sequence",
                "data": [{"x": 0, "y": max_coupling}, {"x": max_coupling, "y": 0}],
                "borderColor": "rgba(128, 128, 128, 0.5)",
                "borderDash": [5, 5],
                "fill": False,
                "pointRadius": 0,
            }
        )

        # Add labels to points
        if labels:
            chart_config["options"]["plugins"]["tooltip"] = {
                "callbacks": {
                    "label": f"function(context) {{ "
                    f"var labels = {labels}; "
                    f"return labels[context.dataIndex] + "
                    f'": (" + context.parsed.x + ", " + context.parsed.y + ")"; }}'
                }
            }

        return chart_config

    def create_coupling_trend(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create coupling trend chart over time.

        Args:
            trend_data: List of data points with date and metrics

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        labels = []
        avg_coupling = []
        max_coupling = []
        highly_coupled = []

        for point in trend_data:
            labels.append(point.get("date", ""))
            avg_coupling.append(point.get("avg_coupling", 0))
            max_coupling.append(point.get("max_coupling", 0))
            highly_coupled.append(point.get("highly_coupled_modules", 0))

        datasets = [
            {
                "label": "Average Coupling",
                "data": avg_coupling,
                "borderColor": ColorPalette.DEFAULT[0],
                "fill": False,
            },
            {
                "label": "Max Coupling",
                "data": max_coupling,
                "borderColor": ColorPalette.SEVERITY["high"],
                "fill": False,
            },
            {
                "label": "Highly Coupled Modules",
                "data": highly_coupled,
                "borderColor": ColorPalette.SEVERITY["medium"],
                "fill": False,
                "yAxisID": "y1",
            },
        ]

        config = ChartConfig(type=ChartType.LINE, title="Coupling Trend Over Time")

        chart_config = self.create_chart(
            ChartType.LINE, {"labels": labels, "datasets": datasets}, config
        )

        # Add dual y-axis
        chart_config["options"]["scales"] = {
            "y": {
                "type": "linear",
                "display": True,
                "position": "left",
                "title": {"display": True, "text": "Coupling Value"},
            },
            "y1": {
                "type": "linear",
                "display": True,
                "position": "right",
                "title": {"display": True, "text": "Module Count"},
                "grid": {"drawOnChartArea": False},
            },
        }

        return chart_config

    def create_dependency_sunburst(self, hierarchy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create dependency sunburst chart.

        Args:
            hierarchy_data: Hierarchical dependency data

        Returns:
            Dict[str, Any]: Sunburst/treemap configuration
        """
        # Flatten hierarchy for treemap
        flat_data = self._flatten_hierarchy(hierarchy_data)

        config = ChartConfig(type=ChartType.TREEMAP, title="Dependency Structure")

        return self.create_chart(ChartType.TREEMAP, {"tree": flat_data}, config)

    def display_terminal(self, coupling_data: Dict[str, Any], show_details: bool = True) -> None:
        """Display coupling analysis in terminal.

        Args:
            coupling_data: Coupling analysis data
            show_details: Whether to show detailed breakdown
        """
        # Display header
        self.terminal_display.display_header("Coupling Analysis", style="double")

        # Display summary metrics
        summary_data = {
            "Average Coupling": self.format_number(
                coupling_data.get("avg_coupling", 0), precision=2
            ),
            "Max Coupling": coupling_data.get("max_coupling", 0),
            "Highly Coupled": coupling_data.get("highly_coupled_count", 0),
            "Total Modules": coupling_data.get("total_modules", 0),
        }

        self.terminal_display.display_metrics(summary_data, title="Summary")

        # Display highly coupled modules
        if show_details and "highly_coupled" in coupling_data:
            headers = ["Module", "Afferent", "Efferent", "Instability", "Risk"]
            rows = []

            for module in coupling_data["highly_coupled"][:10]:
                instability = module.get("instability", 0)
                risk = self._get_coupling_risk(instability)

                rows.append(
                    [
                        self._truncate_module_name(module.get("name", "")),
                        str(module.get("afferent_coupling", 0)),
                        str(module.get("efferent_coupling", 0)),
                        self.format_number(instability, precision=2),
                        self.terminal_display.colorize(risk, self._get_risk_color(risk)),
                    ]
                )

            self.terminal_display.display_table(headers, rows, title="Highly Coupled Modules")

        # Display coupling distribution
        if "distribution" in coupling_data:
            self.terminal_display.display_distribution(
                coupling_data["distribution"],
                title="Coupling Distribution",
                labels=["Low (0-2)", "Medium (3-5)", "High (6-10)", "Very High (>10)"],
            )

        # Display recommendations
        if "recommendations" in coupling_data:
            self.terminal_display.display_list(
                coupling_data["recommendations"], title="Recommendations", style="numbered"
            )

    def create_afferent_efferent_chart(
        self, modules: List[Dict[str, Any]], limit: int = 15
    ) -> Dict[str, Any]:
        """Create afferent vs efferent coupling chart.

        Args:
            modules: List of modules with coupling metrics
            limit: Maximum modules to show

        Returns:
            Dict[str, Any]: Grouped bar chart configuration
        """
        # Sort by total coupling
        sorted_modules = sorted(
            modules,
            key=lambda m: m.get("afferent_coupling", 0) + m.get("efferent_coupling", 0),
            reverse=True,
        )[:limit]

        labels = []
        afferent = []
        efferent = []

        for module in sorted_modules:
            labels.append(self._truncate_module_name(module.get("name", "")))
            afferent.append(module.get("afferent_coupling", 0))
            efferent.append(module.get("efferent_coupling", 0))

        datasets = [
            {
                "label": "Afferent (Incoming)",
                "data": afferent,
                "backgroundColor": ColorPalette.DEFAULT[0],
            },
            {
                "label": "Efferent (Outgoing)",
                "data": efferent,
                "backgroundColor": ColorPalette.DEFAULT[1],
            },
        ]

        config = ChartConfig(type=ChartType.BAR, title="Afferent vs Efferent Coupling")

        return {
            "type": "bar",
            "data": {"labels": labels, "datasets": datasets},
            "options": self._get_chart_options(config),
        }

    def _truncate_module_name(self, name: str, max_length: int = 25) -> str:
        """Truncate module name for display.

        Args:
            name: Module name
            max_length: Maximum length

        Returns:
            str: Truncated name
        """
        if len(name) <= max_length:
            return name

        # Try to keep the most important parts
        parts = name.split(".")
        if len(parts) > 1:
            # Keep first and last parts
            first = parts[0]
            last = parts[-1]
            if len(first) + len(last) + 3 < max_length:
                return f"{first}...{last}"

        return name[: max_length - 3] + "..."

    def _get_coupling_risk(self, instability: float) -> str:
        """Get risk level for instability value.

        Args:
            instability: Instability value (0-1)

        Returns:
            str: Risk level
        """
        if instability > 0.8:
            return "Critical"
        elif instability > 0.6:
            return "High"
        elif instability > 0.4:
            return "Medium"
        else:
            return "Low"

    def _get_risk_color(self, risk: str) -> str:
        """Get color for risk level.

        Args:
            risk: Risk level

        Returns:
            str: Color name
        """
        colors = {"Critical": "red", "High": "yellow", "Medium": "cyan", "Low": "green"}
        return colors.get(risk, "white")

    def _flatten_hierarchy(
        self, data: Dict[str, Any], parent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Flatten hierarchical data for treemap.

        Args:
            data: Hierarchical data
            parent: Parent name

        Returns:
            List[Dict[str, Any]]: Flattened data
        """
        result = []

        current = {
            "parent": parent,
            "name": data.get("name", "Unknown"),
            "value": data.get("value", 1),
        }
        result.append(current)

        if "children" in data:
            for child in data["children"]:
                result.extend(self._flatten_hierarchy(child, current["name"]))

        return result
