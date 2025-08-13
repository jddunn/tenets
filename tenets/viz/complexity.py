"""Code complexity visualization.

This module creates visual representations of code complexity metrics,
helping identify areas that need refactoring and maintenance hotspots.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tenets.utils.logger import get_logger

from .base import (
    ColorScheme,
    VisualizationBase,
    VisualizationFormat,
    format_size,
    truncate_text,
    MATPLOTLIB_AVAILABLE,
    PLOTLY_AVAILABLE,
)


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a file or function.

    Attributes:
        path: File path
        name: Display name
        cyclomatic: Cyclomatic complexity
        cognitive: Cognitive complexity
        halstead: Halstead complexity metrics
        lines: Lines of code
        maintainability_index: Maintainability score (0-100)
        functions: List of function complexities
    """

    path: str
    name: str
    cyclomatic: int = 0
    cognitive: int = 0
    halstead: Dict[str, float] = None
    lines: int = 0
    maintainability_index: float = 100.0
    functions: List["FunctionComplexity"] = None

    def __post_init__(self):
        if self.halstead is None:
            self.halstead = {}
        if self.functions is None:
            self.functions = []

    @property
    def risk_level(self) -> str:
        """Get risk level based on cyclomatic complexity."""
        if self.cyclomatic <= 5:
            return "low"
        elif self.cyclomatic <= 10:
            return "medium"
        elif self.cyclomatic <= 20:
            return "high"
        else:
            return "very high"

    @property
    def color(self) -> str:
        """Get color based on risk level."""
        risk_colors = {
            "low": "#73AB84",  # Green
            "medium": "#F18F01",  # Orange
            "high": "#C73E1D",  # Red
            "very high": "#8B0000",  # Dark red
        }
        return risk_colors.get(self.risk_level, "#95A5A6")


@dataclass
class FunctionComplexity:
    """Complexity metrics for a function.

    Attributes:
        name: Function name
        line_start: Starting line number
        line_end: Ending line number
        cyclomatic: Cyclomatic complexity
        cognitive: Cognitive complexity
        parameters: Number of parameters
        lines: Lines of code
    """

    name: str
    line_start: int
    line_end: int
    cyclomatic: int = 0
    cognitive: int = 0
    parameters: int = 0
    lines: int = 0

    @property
    def complexity_per_line(self) -> float:
        """Get complexity per line of code."""
        if self.lines == 0:
            return 0.0
        return self.cyclomatic / self.lines


class ComplexityHeatmap(VisualizationBase):
    """Visualize code complexity as a heatmap.

    Creates various visualizations of complexity metrics including:
    - File-level heatmap
    - Function-level breakdown
    - Treemap of complexity
    - Complexity trends
    - Risk matrix
    """

    def __init__(
        self,
        title: str = "Code Complexity Heatmap",
        color_scheme: Optional[ColorScheme] = None,
        format: VisualizationFormat = VisualizationFormat.AUTO,
    ):
        """Initialize complexity heatmap.

        Args:
            title: Visualization title
            color_scheme: Color scheme
            format: Output format
        """
        super().__init__(title, color_scheme, format)
        self.logger = get_logger(__name__)
        self.files: List[ComplexityMetrics] = []
        self.max_complexity = 0

    def add_file(
        self,
        file_path: str,
        cyclomatic: int = 0,
        cognitive: int = 0,
        lines: int = 0,
        functions: Optional[List[Dict]] = None,
    ):
        """Add a file with complexity metrics.

        Args:
            file_path: Path to file
            cyclomatic: Cyclomatic complexity
            cognitive: Cognitive complexity
            lines: Lines of code
            functions: Function-level metrics
        """
        name = Path(file_path).name

        # Create metrics
        metrics = ComplexityMetrics(
            path=file_path, name=name, cyclomatic=cyclomatic, cognitive=cognitive, lines=lines
        )

        # Add function metrics
        if functions:
            for func_data in functions:
                func = FunctionComplexity(
                    name=func_data.get("name", "unknown"),
                    line_start=func_data.get("line_start", 0),
                    line_end=func_data.get("line_end", 0),
                    cyclomatic=func_data.get("cyclomatic", 0),
                    cognitive=func_data.get("cognitive", 0),
                    parameters=func_data.get("parameters", 0),
                    lines=func_data.get("lines", 0),
                )
                metrics.functions.append(func)

        # Calculate maintainability index
        metrics.maintainability_index = self._calculate_maintainability(metrics)

        self.files.append(metrics)
        self.max_complexity = max(self.max_complexity, cyclomatic)

    def _calculate_maintainability(self, metrics: ComplexityMetrics) -> float:
        """Calculate maintainability index.

        Based on Halstead volume, cyclomatic complexity, and lines of code.

        Args:
            metrics: Complexity metrics

        Returns:
            Maintainability index (0-100)
        """
        # Simplified maintainability index calculation
        # MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC)

        if metrics.lines == 0:
            return 100.0

        # Estimate Halstead volume if not provided
        volume = metrics.halstead.get("volume", metrics.lines * 10)

        mi = 171
        if volume > 0:
            mi -= 5.2 * math.log(volume)
        mi -= 0.23 * metrics.cyclomatic
        if metrics.lines > 0:
            mi -= 16.2 * math.log(metrics.lines)

        # Normalize to 0-100
        mi = max(0, min(100, mi))

        return mi

    def get_statistics(self) -> Dict[str, Any]:
        """Get complexity statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.files:
            return {}

        total_cyclomatic = sum(f.cyclomatic for f in self.files)
        avg_cyclomatic = total_cyclomatic / len(self.files)

        total_lines = sum(f.lines for f in self.files)

        # Risk distribution
        risk_counts = {"low": 0, "medium": 0, "high": 0, "very high": 0}
        for f in self.files:
            risk_counts[f.risk_level] += 1

        # Find most complex
        most_complex = max(self.files, key=lambda f: f.cyclomatic)

        # Find least maintainable
        least_maintainable = min(self.files, key=lambda f: f.maintainability_index)

        return {
            "total_files": len(self.files),
            "total_lines": total_lines,
            "total_complexity": total_cyclomatic,
            "avg_complexity": avg_cyclomatic,
            "max_complexity": self.max_complexity,
            "risk_distribution": risk_counts,
            "most_complex_file": most_complex.name,
            "least_maintainable": least_maintainable.name,
            "avg_maintainability": sum(f.maintainability_index for f in self.files)
            / len(self.files),
        }

    def _render_ascii(self) -> str:
        """Render complexity as ASCII chart."""
        lines = []
        lines.append("=" * 80)
        lines.append(f" {self.title} ".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Statistics
        stats = self.get_statistics()
        lines.append(f"Files analyzed: {stats.get('total_files', 0)}")
        lines.append(f"Total lines: {stats.get('total_lines', 0):,}")
        lines.append(f"Average complexity: {stats.get('avg_complexity', 0):.1f}")
        lines.append(f"Average maintainability: {stats.get('avg_maintainability', 0):.1f}/100")
        lines.append("")

        # Risk distribution
        risk_dist = stats.get("risk_distribution", {})
        lines.append("Risk Distribution:")
        lines.append("-" * 40)

        for risk, count in risk_dist.items():
            if count > 0:
                bar_length = int(count / max(risk_dist.values()) * 30)
                bar = "█" * bar_length
                lines.append(f"  {risk:10s}: {bar} {count}")

        lines.append("")

        # Top complex files
        lines.append("Most Complex Files:")
        lines.append("-" * 40)

        sorted_files = sorted(self.files, key=lambda f: f.cyclomatic, reverse=True)[:10]

        for f in sorted_files:
            # Create complexity bar
            bar_length = (
                int(f.cyclomatic / self.max_complexity * 30) if self.max_complexity > 0 else 0
            )
            bar = "█" * bar_length

            # Add risk indicator
            risk_indicator = {"low": "●", "medium": "◐", "high": "◑", "very high": "●"}.get(
                f.risk_level, "○"
            )

            lines.append(f"  {risk_indicator} {f.name:30s} {bar} {f.cyclomatic:3d}")

        lines.append("")

        # Least maintainable files
        lines.append("Least Maintainable Files:")
        lines.append("-" * 40)

        sorted_files = sorted(self.files, key=lambda f: f.maintainability_index)[:5]

        for f in sorted_files:
            mi_bar_length = int((100 - f.maintainability_index) / 100 * 20)
            mi_bar = "▓" * mi_bar_length
            lines.append(f"  {f.name:30s} {mi_bar} {f.maintainability_index:.1f}/100")

        return "\n".join(lines)

    def _render_html(self, width: int, height: int) -> str:
        """Render as interactive HTML heatmap."""
        if not PLOTLY_AVAILABLE:
            return super()._render_html(width, height)

        import plotly.graph_objs as go
        import plotly.offline as offline

        # Prepare data for treemap
        labels = []
        parents = []
        values = []
        colors = []
        text = []

        # Add root
        labels.append("All Files")
        parents.append("")
        values.append(0)  # Will be sum of children
        colors.append(0)
        text.append(f"{len(self.files)} files")

        # Group files by directory
        dirs = {}
        for f in self.files:
            dir_path = str(Path(f.path).parent)
            if dir_path not in dirs:
                dirs[dir_path] = []
            dirs[dir_path].append(f)

        # Add directories and files
        for dir_path, dir_files in dirs.items():
            # Add directory
            dir_name = Path(dir_path).name or "root"
            labels.append(dir_name)
            parents.append("All Files")
            values.append(sum(f.cyclomatic for f in dir_files))
            colors.append(sum(f.cyclomatic for f in dir_files) / len(dir_files))
            text.append(f"{len(dir_files)} files")

            # Add files in directory
            for f in dir_files:
                labels.append(f.name)
                parents.append(dir_name)
                values.append(f.cyclomatic)
                colors.append(f.cyclomatic)

                # Create hover text
                hover = f"Complexity: {f.cyclomatic}<br>"
                hover += f"Lines: {f.lines}<br>"
                hover += f"Maintainability: {f.maintainability_index:.1f}/100<br>"
                hover += f"Risk: {f.risk_level}"
                text.append(hover)

        # Create treemap
        fig = go.Figure(
            go.Treemap(
                labels=labels,
                parents=parents,
                values=values,
                text=text,
                textinfo="label",
                marker=dict(
                    colorscale="RdYlGn_r", cmid=10, colorbar=dict(title="Complexity"), colors=colors
                ),
                hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
            )
        )

        fig.update_layout(
            title=self.title, width=width, height=height, margin=dict(t=50, l=0, r=0, b=0)
        )

        # Generate HTML
        return offline.plot(fig, output_type="div", include_plotlyjs="cdn")

    def _render_svg(self, width: int, height: int, dpi: int) -> str:
        """Render as SVG heatmap."""
        if not MATPLOTLIB_AVAILABLE:
            return super()._render_svg(width, height, dpi)

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import StringIO

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / dpi, height / dpi), dpi=dpi)

        # Left plot: Complexity vs Lines scatter
        if self.files:
            x = [f.lines for f in self.files]
            y = [f.cyclomatic for f in self.files]
            colors = [f.cyclomatic for f in self.files]

            scatter = ax1.scatter(
                x, y, c=colors, cmap="RdYlGn_r", s=100, alpha=0.6, edgecolors="black"
            )
            ax1.set_xlabel("Lines of Code")
            ax1.set_ylabel("Cyclomatic Complexity")
            ax1.set_title("Complexity vs Size")

            # Add colorbar
            plt.colorbar(scatter, ax=ax1, label="Complexity")

            # Add risk zones
            ax1.axhline(y=10, color="orange", linestyle="--", alpha=0.5, label="Medium risk")
            ax1.axhline(y=20, color="red", linestyle="--", alpha=0.5, label="High risk")
            ax1.legend(loc="upper left")

        # Right plot: Risk distribution pie chart
        stats = self.get_statistics()
        risk_dist = stats.get("risk_distribution", {})

        if risk_dist:
            sizes = list(risk_dist.values())
            labels = list(risk_dist.keys())
            colors_map = {
                "low": "#73AB84",
                "medium": "#F18F01",
                "high": "#C73E1D",
                "very high": "#8B0000",
            }
            colors = [colors_map.get(label, "#95A5A6") for label in labels]

            # Filter out zero values
            filtered = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
            if filtered:
                labels, sizes, colors = zip(*filtered)

                ax2.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"fontsize": 10},
                )
                ax2.set_title("Risk Distribution")

        # Overall title
        fig.suptitle(self.title, fontsize=14, fontweight="bold")

        # Convert to SVG
        buffer = StringIO()
        plt.tight_layout()
        plt.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()


def create_complexity_heatmap(
    files: List[Any],
    title: str = "Code Complexity",
    threshold: Optional[int] = None,
    format: str = "auto",
) -> ComplexityHeatmap:
    """Create a complexity heatmap from files.

    Args:
        files: List of FileAnalysis objects
        title: Visualization title
        threshold: Complexity threshold for filtering
        format: Output format

    Returns:
        ComplexityHeatmap instance
    """
    heatmap = ComplexityHeatmap(title=title, format=VisualizationFormat(format))

    # Add files with complexity metrics
    for file in files:
        # Skip if below threshold
        if hasattr(file, "complexity") and file.complexity:
            cyclomatic = getattr(file.complexity, "cyclomatic", 0)

            if threshold and cyclomatic < threshold:
                continue

            # Extract function metrics if available
            functions = []
            if hasattr(file, "functions"):
                for func in file.functions:
                    if hasattr(func, "complexity"):
                        functions.append(
                            {
                                "name": getattr(func, "name", "unknown"),
                                "line_start": getattr(func, "line_start", 0),
                                "line_end": getattr(func, "line_end", 0),
                                "cyclomatic": getattr(func.complexity, "cyclomatic", 0),
                                "lines": getattr(func, "lines", 0),
                            }
                        )

            heatmap.add_file(
                file_path=file.path,
                cyclomatic=cyclomatic,
                cognitive=getattr(file.complexity, "cognitive", 0),
                lines=file.lines,
                functions=functions,
            )

    return heatmap


def analyze_complexity_trends(
    historical_data: List[Dict[str, Any]], title: str = "Complexity Trends"
) -> str:
    """Analyze complexity trends over time.

    Args:
        historical_data: List of historical complexity measurements
        title: Chart title

    Returns:
        Trend analysis as string or HTML
    """
    if not historical_data:
        return "No historical data available"

    # Simple ASCII trend
    lines = []
    lines.append(f"{title}")
    lines.append("=" * 40)

    for entry in historical_data:
        date = entry.get("date", "unknown")
        complexity = entry.get("avg_complexity", 0)

        # Create bar
        bar_length = int(complexity)
        bar = "█" * bar_length

        lines.append(f"{date}: {bar} {complexity:.1f}")

    return "\n".join(lines)
