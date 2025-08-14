"""Dependencies visualization module.

This module provides visualization capabilities for dependency analysis,
including dependency graphs, circular dependencies, and package structure.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseVisualizer, ChartConfig, ChartType, ColorPalette, DisplayConfig
from .displays import TerminalDisplay


class DependencyVisualizer(BaseVisualizer):
    """Visualizer for dependency metrics.

    Creates visualizations for dependency analysis including dependency
    trees, circular dependency detection, and package relationships.
    """

    def __init__(
        self,
        chart_config: Optional[ChartConfig] = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """Initialize dependency visualizer.

        Args:
            chart_config: Chart configuration
            display_config: Display configuration
        """
        super().__init__(chart_config, display_config)
        self.terminal_display = TerminalDisplay(display_config)

    def create_dependency_graph(
        self,
        dependencies: Dict[str, List[str]],
        highlight_circular: bool = True,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        """Create dependency graph visualization.

        Args:
            dependencies: Dictionary of module -> [dependencies]
            highlight_circular: Whether to highlight circular dependencies
            max_nodes: Maximum nodes to display

        Returns:
            Dict[str, Any]: Network graph configuration
        """
        # Build nodes and edges
        nodes_set = set()
        edges = []

        # Track circular dependencies
        circular_pairs = set()
        if highlight_circular:
            circular_pairs = self._find_circular_dependencies(dependencies)

        for module, deps in dependencies.items():
            nodes_set.add(module)
            for dep in deps:
                nodes_set.add(dep)

                # Check if this is a circular dependency
                is_circular = (module, dep) in circular_pairs or (dep, module) in circular_pairs

                edges.append(
                    {
                        "source": module,
                        "target": dep,
                        "color": ColorPalette.SEVERITY["critical"] if is_circular else None,
                        "style": "dashed" if is_circular else "solid",
                    }
                )

        # Limit nodes if necessary
        if len(nodes_set) > max_nodes:
            # Keep nodes with most connections
            node_connections = {}
            for module, deps in dependencies.items():
                node_connections[module] = node_connections.get(module, 0) + len(deps)
                for dep in deps:
                    node_connections[dep] = node_connections.get(dep, 0) + 1

            sorted_nodes = sorted(
                nodes_set, key=lambda n: node_connections.get(n, 0), reverse=True
            )[:max_nodes]
            nodes_set = set(sorted_nodes)

            # Filter edges
            edges = [e for e in edges if e["source"] in nodes_set and e["target"] in nodes_set]

        # Create node list
        nodes = []
        for node_id in nodes_set:
            # Determine node type and color
            is_external = self._is_external_dependency(node_id)
            has_circular = any(node_id in pair for pair in circular_pairs)

            if has_circular:
                color = ColorPalette.SEVERITY["critical"]
            elif is_external:
                color = ColorPalette.DEFAULT[2]  # Green for external
            else:
                color = ColorPalette.DEFAULT[0]  # Blue for internal

            nodes.append(
                {
                    "id": node_id,
                    "label": self._truncate_package_name(node_id),
                    "color": color,
                    "shape": "box" if is_external else "circle",
                }
            )

        config = ChartConfig(type=ChartType.NETWORK, title="Dependency Graph")

        return self.create_chart(
            ChartType.NETWORK, {"nodes": nodes, "edges": edges, "layout": "hierarchical"}, config
        )

    def create_dependency_tree(
        self, tree_data: Dict[str, Any], max_depth: int = 5
    ) -> Dict[str, Any]:
        """Create dependency tree visualization.

        Args:
            tree_data: Hierarchical dependency data
            max_depth: Maximum tree depth to display

        Returns:
            Dict[str, Any]: Treemap configuration
        """
        # Flatten tree to specified depth
        flat_data = self._flatten_tree(tree_data, max_depth)

        config = ChartConfig(type=ChartType.TREEMAP, title="Dependency Tree")

        return self.create_chart(ChartType.TREEMAP, {"tree": flat_data}, config)

    def create_package_sunburst(self, package_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create package structure sunburst chart.

        Args:
            package_data: Hierarchical package data

        Returns:
            Dict[str, Any]: Sunburst/treemap configuration
        """
        flat_data = self._flatten_tree(package_data)

        # Color by depth level
        for i, item in enumerate(flat_data):
            depth = item.get("depth", 0)
            item["color"] = ColorPalette.DEFAULT[depth % len(ColorPalette.DEFAULT)]

        config = ChartConfig(type=ChartType.TREEMAP, title="Package Structure")

        return self.create_chart(ChartType.TREEMAP, {"tree": flat_data}, config)

    def create_circular_dependencies_chart(self, circular_deps: List[List[str]]) -> Dict[str, Any]:
        """Create circular dependencies visualization.

        Args:
            circular_deps: List of circular dependency chains

        Returns:
            Dict[str, Any]: Network graph configuration
        """
        # Build graph from circular chains
        nodes_set = set()
        edges = []

        for chain in circular_deps:
            for i, module in enumerate(chain):
                nodes_set.add(module)
                if i < len(chain) - 1:
                    edges.append(
                        {
                            "source": module,
                            "target": chain[i + 1],
                            "color": ColorPalette.SEVERITY["critical"],
                            "style": "solid",
                            "arrows": "to",
                        }
                    )

        # Create nodes with critical coloring
        nodes = [
            {
                "id": node,
                "label": self._truncate_package_name(node),
                "color": ColorPalette.SEVERITY["critical"],
                "shape": "circle",
            }
            for node in nodes_set
        ]

        config = ChartConfig(type=ChartType.NETWORK, title="Circular Dependencies")

        return self.create_chart(
            ChartType.NETWORK, {"nodes": nodes, "edges": edges, "layout": "circular"}, config
        )

    def create_dependency_matrix(
        self, modules: List[str], dependency_matrix: List[List[bool]]
    ) -> Dict[str, Any]:
        """Create dependency matrix visualization.

        Args:
            modules: List of module names
            dependency_matrix: Boolean matrix of dependencies

        Returns:
            Dict[str, Any]: Heatmap configuration
        """
        # Convert boolean to numeric
        numeric_matrix = [[1 if dep else 0 for dep in row] for row in dependency_matrix]

        labels = [self._truncate_package_name(m) for m in modules]

        config = ChartConfig(type=ChartType.HEATMAP, title="Dependency Matrix")

        return self.create_chart(
            ChartType.HEATMAP,
            {"matrix": numeric_matrix, "x_labels": labels, "y_labels": labels},
            config,
        )

    def create_layer_violations_chart(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create layer violation visualization.

        Args:
            violations: List of layer violations

        Returns:
            Dict[str, Any]: Chart configuration
        """
        # Group violations by type
        violation_types = {}
        for violation in violations:
            vtype = violation.get("type", "Unknown")
            violation_types[vtype] = violation_types.get(vtype, 0) + 1

        labels = list(violation_types.keys())
        values = list(violation_types.values())

        # Color based on severity
        colors = []
        for label in labels:
            if "critical" in label.lower():
                colors.append(ColorPalette.SEVERITY["critical"])
            elif "high" in label.lower():
                colors.append(ColorPalette.SEVERITY["high"])
            else:
                colors.append(ColorPalette.SEVERITY["medium"])

        config = ChartConfig(
            type=ChartType.BAR, title="Architecture Layer Violations", colors=colors
        )

        return self.create_chart(ChartType.BAR, {"labels": labels, "values": values}, config)

    def display_terminal(self, dependency_data: Dict[str, Any], show_details: bool = True) -> None:
        """Display dependency analysis in terminal.

        Args:
            dependency_data: Dependency analysis data
            show_details: Whether to show detailed breakdown
        """
        # Display header
        self.terminal_display.display_header("Dependency Analysis", style="double")

        # Display summary metrics
        summary_data = {
            "Total Modules": dependency_data.get("total_modules", 0),
            "Total Dependencies": dependency_data.get("total_dependencies", 0),
            "External Dependencies": dependency_data.get("external_dependencies", 0),
            "Circular Dependencies": dependency_data.get("circular_count", 0),
        }

        self.terminal_display.display_metrics(summary_data, title="Summary")

        # Display circular dependencies warning
        if dependency_data.get("circular_count", 0) > 0:
            self.terminal_display.display_warning(
                f"⚠️  Found {dependency_data['circular_count']} circular dependencies!"
            )

            if show_details and "circular_chains" in dependency_data:
                for i, chain in enumerate(dependency_data["circular_chains"][:5], 1):
                    chain_str = " → ".join(chain[:5])
                    if len(chain) > 5:
                        chain_str += f" → ... ({len(chain) - 5} more)"
                    print(f"  {i}. {chain_str}")

        # Display most dependent modules
        if show_details and "most_dependent" in dependency_data:
            headers = ["Module", "Dependencies", "Dependents", "Coupling"]
            rows = []

            for module in dependency_data["most_dependent"][:10]:
                coupling = module.get("dependencies", 0) + module.get("dependents", 0)
                rows.append(
                    [
                        self._truncate_package_name(module.get("name", "")),
                        str(module.get("dependencies", 0)),
                        str(module.get("dependents", 0)),
                        str(coupling),
                    ]
                )

            self.terminal_display.display_table(headers, rows, title="Most Dependent Modules")

        # Display external dependencies
        if "external" in dependency_data:
            self._display_external_dependencies(dependency_data["external"])

        # Display recommendations
        if "recommendations" in dependency_data:
            self.terminal_display.display_list(
                dependency_data["recommendations"], title="Recommendations", style="numbered"
            )

    def create_dependency_trend(self, trend_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create dependency trend chart over time.

        Args:
            trend_data: List of data points with date and metrics

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        labels = []
        total_deps = []
        external_deps = []
        circular_deps = []

        for point in trend_data:
            labels.append(point.get("date", ""))
            total_deps.append(point.get("total_dependencies", 0))
            external_deps.append(point.get("external_dependencies", 0))
            circular_deps.append(point.get("circular_dependencies", 0))

        datasets = [
            {
                "label": "Total Dependencies",
                "data": total_deps,
                "borderColor": ColorPalette.DEFAULT[0],
                "fill": False,
            },
            {
                "label": "External Dependencies",
                "data": external_deps,
                "borderColor": ColorPalette.DEFAULT[1],
                "fill": False,
            },
            {
                "label": "Circular Dependencies",
                "data": circular_deps,
                "borderColor": ColorPalette.SEVERITY["critical"],
                "fill": False,
                "yAxisID": "y1",
            },
        ]

        config = ChartConfig(type=ChartType.LINE, title="Dependency Trends")

        chart_config = self.create_chart(
            ChartType.LINE, {"labels": labels, "datasets": datasets}, config
        )

        # Add dual y-axis
        chart_config["options"]["scales"] = {
            "y": {"type": "linear", "display": True, "position": "left"},
            "y1": {
                "type": "linear",
                "display": True,
                "position": "right",
                "grid": {"drawOnChartArea": False},
            },
        }

        return chart_config

    def _find_circular_dependencies(
        self, dependencies: Dict[str, List[str]]
    ) -> Set[Tuple[str, str]]:
        """Find circular dependencies in dependency graph.

        Args:
            dependencies: Dependency dictionary

        Returns:
            Set[Tuple[str, str]]: Set of circular dependency pairs
        """
        circular_pairs = set()

        def dfs(module: str, path: List[str], visited: Set[str]):
            if module in path:
                # Found a cycle
                cycle_start = path.index(module)
                for i in range(cycle_start, len(path)):
                    j = (i + 1) % (len(path) - cycle_start + 1) + cycle_start - 1
                    if j < len(path):
                        circular_pairs.add(tuple(sorted([path[i], path[j]])))
                return

            if module in visited:
                return

            visited.add(module)
            path.append(module)

            for dep in dependencies.get(module, []):
                dfs(dep, path.copy(), visited.copy())

            path.pop()

        for module in dependencies:
            dfs(module, [], set())

        return circular_pairs

    def _is_external_dependency(self, package: str) -> bool:
        """Check if package is external dependency.

        Args:
            package: Package name

        Returns:
            bool: True if external
        """
        # Simple heuristic - no dots or starts with common external prefixes
        external_prefixes = ["numpy", "pandas", "scipy", "django", "flask"]

        if "." not in package:
            return True

        for prefix in external_prefixes:
            if package.startswith(prefix):
                return True

        return False

    def _truncate_package_name(self, name: str, max_length: int = 30) -> str:
        """Truncate package name for display.

        Args:
            name: Package name
            max_length: Maximum length

        Returns:
            str: Truncated name
        """
        if len(name) <= max_length:
            return name

        # Try to keep important parts
        parts = name.split(".")
        if len(parts) > 2:
            # Keep first and last parts
            return f"{parts[0]}...{parts[-1]}"

        return name[: max_length - 3] + "..."

    def _flatten_tree(
        self,
        tree: Dict[str, Any],
        max_depth: int = -1,
        current_depth: int = 0,
        parent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Flatten tree structure for visualization.

        Args:
            tree: Tree structure
            max_depth: Maximum depth (-1 for unlimited)
            current_depth: Current depth
            parent: Parent name

        Returns:
            List[Dict[str, Any]]: Flattened tree
        """
        if max_depth != -1 and current_depth >= max_depth:
            return []

        result = []

        current = {
            "parent": parent,
            "name": tree.get("name", "Unknown"),
            "value": tree.get("value", 1),
            "depth": current_depth,
        }
        result.append(current)

        if "children" in tree:
            for child in tree["children"]:
                result.extend(
                    self._flatten_tree(child, max_depth, current_depth + 1, current["name"])
                )

        return result

    def _display_external_dependencies(self, external_deps: List[Dict[str, Any]]) -> None:
        """Display external dependencies in terminal.

        Args:
            external_deps: List of external dependencies
        """
        if not external_deps:
            return

        headers = ["Package", "Version", "License", "Usage"]
        rows = []

        for dep in external_deps[:15]:
            rows.append(
                [
                    dep.get("name", "Unknown"),
                    dep.get("version", "-"),
                    dep.get("license", "-"),
                    str(dep.get("usage_count", 0)),
                ]
            )

        self.terminal_display.display_table(headers, rows, title="External Dependencies")
