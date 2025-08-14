"""Contributors visualization module.

This module provides visualization capabilities for contributor metrics,
including contribution distribution, collaboration patterns, and
contributor activity visualizations.
"""

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseVisualizer, ChartConfig, ChartType, ColorPalette, DisplayConfig
from .displays import TerminalDisplay


class ContributorVisualizer(BaseVisualizer):
    """Visualizer for contributor metrics.

    Creates visualizations for contributor analysis including activity
    charts, collaboration networks, and contribution distributions.
    """

    def __init__(
        self,
        chart_config: Optional[ChartConfig] = None,
        display_config: Optional[DisplayConfig] = None,
    ):
        """Initialize contributor visualizer.

        Args:
            chart_config: Chart configuration
            display_config: Display configuration
        """
        super().__init__(chart_config, display_config)
        self.terminal_display = TerminalDisplay(display_config)

    def create_contribution_chart(
        self, contributors: List[Dict[str, Any]], metric: str = "commits", limit: int = 10
    ) -> Dict[str, Any]:
        """Create contributor contribution chart.

        Args:
            contributors: List of contributor data
            metric: Metric to visualize (commits, lines, files)
            limit: Maximum contributors to show

        Returns:
            Dict[str, Any]: Chart configuration
        """
        # Sort by metric
        sorted_contributors = sorted(contributors, key=lambda x: x.get(metric, 0), reverse=True)[
            :limit
        ]

        labels = []
        values = []

        for contributor in sorted_contributors:
            name = contributor.get("name", contributor.get("email", "Unknown"))
            # Truncate long names
            if len(name) > 20:
                name = name[:17] + "..."
            labels.append(name)
            values.append(contributor.get(metric, 0))

        # Color based on contribution level
        total = sum(values) if values else 1
        colors = []
        for value in values:
            percentage = (value / total) * 100 if total > 0 else 0
            if percentage > 30:
                colors.append(ColorPalette.HEALTH["excellent"])
            elif percentage > 15:
                colors.append(ColorPalette.HEALTH["good"])
            elif percentage > 5:
                colors.append(ColorPalette.HEALTH["fair"])
            else:
                colors.append(ColorPalette.DEFAULT[len(colors) % len(ColorPalette.DEFAULT)])

        title_map = {
            "commits": "Commits by Contributor",
            "lines": "Lines Changed by Contributor",
            "files": "Files Touched by Contributor",
        }

        config = ChartConfig(
            type=ChartType.BAR,
            title=title_map.get(metric, f"{metric.title()} by Contributor"),
            colors=colors,
        )

        return self.create_chart(ChartType.BAR, {"labels": labels, "values": values}, config)

    def create_activity_timeline(
        self, activity_data: List[Dict[str, Any]], contributor: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create contributor activity timeline.

        Args:
            activity_data: Activity data points with dates
            contributor: Specific contributor to highlight

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        # Group by date
        date_activity = {}

        for activity in activity_data:
            date = activity.get("date", "")
            if not date:
                continue

            if date not in date_activity:
                date_activity[date] = {"commits": 0, "contributors": set()}

            date_activity[date]["commits"] += activity.get("commits", 0)
            if "contributor" in activity:
                date_activity[date]["contributors"].add(activity["contributor"])

        # Sort by date
        sorted_dates = sorted(date_activity.keys())

        labels = sorted_dates
        commits = [date_activity[d]["commits"] for d in sorted_dates]
        active_contributors = [len(date_activity[d]["contributors"]) for d in sorted_dates]

        datasets = [
            {
                "label": "Commits",
                "data": commits,
                "borderColor": ColorPalette.DEFAULT[0],
                "yAxisID": "y",
            },
            {
                "label": "Active Contributors",
                "data": active_contributors,
                "borderColor": ColorPalette.DEFAULT[1],
                "yAxisID": "y1",
            },
        ]

        config = ChartConfig(type=ChartType.LINE, title="Contributor Activity Over Time")

        chart_config = self.create_chart(
            ChartType.LINE, {"labels": labels, "datasets": datasets}, config
        )

        # Add dual y-axis configuration
        chart_config["options"]["scales"] = {
            "y": {
                "type": "linear",
                "display": True,
                "position": "left",
                "title": {"display": True, "text": "Commits"},
            },
            "y1": {
                "type": "linear",
                "display": True,
                "position": "right",
                "title": {"display": True, "text": "Contributors"},
                "grid": {"drawOnChartArea": False},
            },
        }

        return chart_config

    def create_collaboration_network(
        self, collaboration_data: Dict[Tuple[str, str], int], min_weight: int = 2
    ) -> Dict[str, Any]:
        """Create collaboration network graph.

        Args:
            collaboration_data: Dictionary of (contributor1, contributor2) -> weight
            min_weight: Minimum collaboration weight to include

        Returns:
            Dict[str, Any]: Network graph configuration
        """
        # Build nodes and edges
        nodes = set()
        edges = []

        for (contributor1, contributor2), weight in collaboration_data.items():
            if weight >= min_weight:
                nodes.add(contributor1)
                nodes.add(contributor2)
                edges.append({"source": contributor1, "target": contributor2, "weight": weight})

        # Create node list with sizing based on degree
        node_degree = {}
        for edge in edges:
            node_degree[edge["source"]] = node_degree.get(edge["source"], 0) + edge["weight"]
            node_degree[edge["target"]] = node_degree.get(edge["target"], 0) + edge["weight"]

        node_list = []
        for node in nodes:
            degree = node_degree.get(node, 1)
            node_list.append(
                {
                    "id": node,
                    "label": node[:20] + "..." if len(node) > 20 else node,
                    "size": min(50, 10 + degree * 2),
                }
            )

        config = ChartConfig(type=ChartType.NETWORK, title="Contributor Collaboration Network")

        return self.create_chart(
            ChartType.NETWORK, {"nodes": node_list, "edges": edges, "layout": "force"}, config
        )

    def create_distribution_pie(
        self, contributors: List[Dict[str, Any]], metric: str = "commits", top_n: int = 5
    ) -> Dict[str, Any]:
        """Create contribution distribution pie chart.

        Args:
            contributors: List of contributor data
            metric: Metric to visualize
            top_n: Number of top contributors to show individually

        Returns:
            Dict[str, Any]: Pie chart configuration
        """
        # Sort by metric
        sorted_contributors = sorted(contributors, key=lambda x: x.get(metric, 0), reverse=True)

        labels = []
        values = []

        # Add top contributors
        for contributor in sorted_contributors[:top_n]:
            name = contributor.get("name", contributor.get("email", "Unknown"))
            if len(name) > 15:
                name = name[:12] + "..."
            labels.append(name)
            values.append(contributor.get(metric, 0))

        # Add "Others" if there are more contributors
        if len(sorted_contributors) > top_n:
            others_value = sum(c.get(metric, 0) for c in sorted_contributors[top_n:])
            if others_value > 0:
                labels.append("Others")
                values.append(others_value)

        config = ChartConfig(type=ChartType.PIE, title=f"{metric.title()} Distribution")

        return self.create_chart(ChartType.PIE, {"labels": labels, "values": values}, config)

    def create_bus_factor_gauge(self, bus_factor: int, total_contributors: int) -> Dict[str, Any]:
        """Create bus factor gauge chart.

        Args:
            bus_factor: Current bus factor
            total_contributors: Total number of contributors

        Returns:
            Dict[str, Any]: Gauge chart configuration
        """
        # Calculate percentage (higher is better)
        percentage = (bus_factor / max(1, total_contributors)) * 100

        # Determine color based on bus factor
        if bus_factor <= 1:
            color = ColorPalette.SEVERITY["critical"]
        elif bus_factor <= 2:
            color = ColorPalette.SEVERITY["high"]
        elif bus_factor <= 3:
            color = ColorPalette.SEVERITY["medium"]
        else:
            color = ColorPalette.HEALTH["excellent"]

        config = ChartConfig(
            type=ChartType.GAUGE, title=f"Bus Factor: {bus_factor}", colors=[color]
        )

        return self.create_chart(ChartType.GAUGE, {"value": percentage, "max": 100}, config)

    def display_terminal(self, contributor_data: Dict[str, Any], show_details: bool = True) -> None:
        """Display contributor analysis in terminal.

        Args:
            contributor_data: Contributor analysis data
            show_details: Whether to show detailed breakdown
        """
        # Display header
        self.terminal_display.display_header("Contributor Analysis", style="double")

        # Display summary metrics
        summary_data = {
            "Total Contributors": contributor_data.get("total_contributors", 0),
            "Active Contributors": contributor_data.get("active_contributors", 0),
            "Bus Factor": contributor_data.get("bus_factor", 0),
            "Avg Commits/Contributor": self.format_number(
                contributor_data.get("avg_commits_per_contributor", 0), precision=1
            ),
        }

        self.terminal_display.display_metrics(summary_data, title="Summary")

        # Display top contributors table
        if show_details and "contributors" in contributor_data:
            headers = ["Contributor", "Commits", "Lines", "Files", "Activity"]
            rows = []

            for contributor in contributor_data["contributors"][:10]:
                activity = self._get_activity_indicator(
                    contributor.get("last_commit_days_ago", 999)
                )
                rows.append(
                    [
                        contributor.get("name", "Unknown")[:30],
                        str(contributor.get("commits", 0)),
                        self.format_number(contributor.get("lines", 0)),
                        str(contributor.get("files", 0)),
                        activity,
                    ]
                )

            self.terminal_display.display_table(headers, rows, title="Top Contributors")

        # Display collaboration matrix if available
        if "collaboration_matrix" in contributor_data:
            self._display_collaboration_matrix(contributor_data["collaboration_matrix"])

        # Display warnings
        warnings = []
        bus_factor = contributor_data.get("bus_factor", 0)
        if bus_factor <= 1:
            warnings.append("⚠️  Critical: Bus factor is 1 - single point of failure")
        elif bus_factor <= 2:
            warnings.append("⚠️  Warning: Low bus factor - knowledge concentration risk")

        if warnings:
            self.terminal_display.display_list(warnings, title="Warnings", style="bullet")

    def create_retention_chart(self, retention_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create contributor retention chart.

        Args:
            retention_data: Retention data over time

        Returns:
            Dict[str, Any]: Line chart configuration
        """
        labels = []
        active = []
        new = []
        left = []

        for point in retention_data:
            labels.append(point.get("period", ""))
            active.append(point.get("active", 0))
            new.append(point.get("new", 0))
            left.append(point.get("left", 0))

        datasets = [
            {
                "label": "Active",
                "data": active,
                "borderColor": ColorPalette.HEALTH["excellent"],
                "backgroundColor": ColorPalette.HEALTH["excellent"] + "20",
                "fill": True,
            },
            {
                "label": "New",
                "data": new,
                "borderColor": ColorPalette.HEALTH["good"],
                "fill": False,
            },
            {
                "label": "Left",
                "data": left,
                "borderColor": ColorPalette.SEVERITY["high"],
                "fill": False,
            },
        ]

        config = ChartConfig(type=ChartType.LINE, title="Contributor Retention")

        return self.create_chart(ChartType.LINE, {"labels": labels, "datasets": datasets}, config)

    def _get_activity_indicator(self, days_ago: int) -> str:
        """Get activity indicator for days since last commit.

        Args:
            days_ago: Days since last commit

        Returns:
            str: Activity indicator
        """
        if days_ago <= 7:
            return self.terminal_display.colorize("●", "green") + " Active"
        elif days_ago <= 30:
            return self.terminal_display.colorize("●", "yellow") + " Recent"
        elif days_ago <= 90:
            return self.terminal_display.colorize("●", "red") + " Inactive"
        else:
            return self.terminal_display.colorize("○", "white") + " Dormant"

    def _display_collaboration_matrix(self, matrix: Dict[Tuple[str, str], int]) -> None:
        """Display collaboration matrix in terminal.

        Args:
            matrix: Collaboration matrix data
        """
        # Get unique contributors
        contributors = set()
        for (c1, c2), _ in matrix.items():
            contributors.add(c1)
            contributors.add(c2)

        # Limit to top contributors
        contributor_list = sorted(contributors)[:8]

        if not contributor_list:
            return

        # Build matrix display
        headers = [""] + [c[:8] for c in contributor_list]
        rows = []

        for c1 in contributor_list:
            row = [c1[:8]]
            for c2 in contributor_list:
                if c1 == c2:
                    row.append("-")
                else:
                    key = tuple(sorted([c1, c2]))
                    value = matrix.get(key, 0)
                    if value > 0:
                        row.append(str(value))
                    else:
                        row.append("")
            rows.append(row)

        self.terminal_display.display_table(
            headers, rows, title="Collaboration Matrix (File Co-changes)"
        )
