"""Contributor activity visualization.

This module visualizes contributor patterns, showing who works on what parts
of the codebase, collaboration patterns, and code ownership distribution.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.utils.logger import get_logger

from .base import (
    MATPLOTLIB_AVAILABLE,
    PLOTLY_AVAILABLE,
    ColorScheme,
    VisualizationBase,
    VisualizationFormat,
    format_size,
    truncate_text,
)


@dataclass
class ContributorStats:
    """Statistics for a contributor.

    Attributes:
        name: Contributor name/email
        commits: Total number of commits
        lines_added: Total lines added
        lines_removed: Total lines removed
        files_touched: Set of files modified
        first_commit: First commit timestamp
        last_commit: Last commit timestamp
        active_days: Set of days with activity
        languages: Languages worked with
        co_authors: Other contributors worked with
    """

    name: str
    commits: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    files_touched: Set[str] = None
    first_commit: Optional[datetime] = None
    last_commit: Optional[datetime] = None
    active_days: Set[str] = None
    languages: Dict[str, int] = None
    co_authors: Set[str] = None

    def __post_init__(self):
        if self.files_touched is None:
            self.files_touched = set()
        if self.active_days is None:
            self.active_days = set()
        if self.languages is None:
            self.languages = {}
        if self.co_authors is None:
            self.co_authors = set()

    @property
    def net_lines(self) -> int:
        """Net lines contributed."""
        return self.lines_added - self.lines_removed

    @property
    def productivity(self) -> float:
        """Average lines per commit."""
        if self.commits == 0:
            return 0.0
        return self.net_lines / self.commits

    @property
    def activity_span(self) -> int:
        """Days between first and last commit."""
        if not self.first_commit or not self.last_commit:
            return 0
        return (self.last_commit - self.first_commit).days

    @property
    def consistency(self) -> float:
        """Consistency score (active days / total span)."""
        if self.activity_span == 0:
            return 1.0 if self.active_days else 0.0
        return len(self.active_days) / max(1, self.activity_span)

    @property
    def expertise_breadth(self) -> int:
        """Number of different file types worked on."""
        extensions = set()
        for file_path in self.files_touched:
            ext = Path(file_path).suffix
            if ext:
                extensions.add(ext)
        return len(extensions)


@dataclass
class TeamDynamics:
    """Team collaboration dynamics.

    Attributes:
        collaboration_matrix: Who works with whom
        shared_files: Files worked on by multiple people
        knowledge_silos: Files only one person knows
        bus_factor: Files at risk if someone leaves
    """

    collaboration_matrix: Dict[Tuple[str, str], int] = None
    shared_files: Dict[str, Set[str]] = None
    knowledge_silos: Dict[str, str] = None
    bus_factor: Dict[str, int] = None

    def __post_init__(self):
        if self.collaboration_matrix is None:
            self.collaboration_matrix = {}
        if self.shared_files is None:
            self.shared_files = defaultdict(set)
        if self.knowledge_silos is None:
            self.knowledge_silos = {}
        if self.bus_factor is None:
            self.bus_factor = defaultdict(int)


class ContributorGraph(VisualizationBase):
    """Visualize contributor activity and patterns.

    Creates visualizations showing:
    - Contributor activity over time
    - Code ownership distribution
    - Collaboration patterns
    - Knowledge distribution
    - Bus factor analysis
    """

    def __init__(
        self,
        title: str = "Contributor Activity",
        color_scheme: Optional[ColorScheme] = None,
        format: VisualizationFormat = VisualizationFormat.AUTO,
        active_threshold: int = 30,  # Days to consider active
    ):
        """Initialize contributor graph.

        Args:
            title: Graph title
            color_scheme: Color scheme
            format: Output format
            active_threshold: Days since last commit to be "active"
        """
        super().__init__(title, color_scheme, format)
        self.logger = get_logger(__name__)
        self.active_threshold = active_threshold
        self.contributors: Dict[str, ContributorStats] = {}
        self.team_dynamics = TeamDynamics()
        self.timeline: List[Dict[str, Any]] = []

    def add_commit(
        self,
        author: str,
        timestamp: datetime,
        files: List[str],
        lines_added: int = 0,
        lines_removed: int = 0,
        co_authors: Optional[List[str]] = None,
        message: str = "",
    ):
        """Add a commit to contributor analysis.

        Args:
            author: Commit author
            timestamp: Commit timestamp
            files: Files changed
            lines_added: Lines added
            lines_removed: Lines removed
            co_authors: Co-authors on commit
            message: Commit message
        """
        # Initialize contributor if needed
        if author not in self.contributors:
            self.contributors[author] = ContributorStats(name=author)

        contributor = self.contributors[author]

        # Update stats
        contributor.commits += 1
        contributor.lines_added += lines_added
        contributor.lines_removed += lines_removed
        contributor.files_touched.update(files)

        # Update timestamps
        if not contributor.first_commit or timestamp < contributor.first_commit:
            contributor.first_commit = timestamp
        if not contributor.last_commit or timestamp > contributor.last_commit:
            contributor.last_commit = timestamp

        # Track active days
        day_key = timestamp.strftime("%Y-%m-%d")
        contributor.active_days.add(day_key)

        # Track languages
        for file_path in files:
            ext = Path(file_path).suffix
            if ext:
                contributor.languages[ext] = contributor.languages.get(ext, 0) + 1

        # Track co-authors
        if co_authors:
            contributor.co_authors.update(co_authors)
            # Update collaboration matrix
            for co_author in co_authors:
                pair = tuple(sorted([author, co_author]))
                self.team_dynamics.collaboration_matrix[pair] = (
                    self.team_dynamics.collaboration_matrix.get(pair, 0) + 1
                )

        # Update shared files
        for file_path in files:
            self.team_dynamics.shared_files[file_path].add(author)

        # Add to timeline
        self.timeline.append(
            {
                "author": author,
                "timestamp": timestamp,
                "files": files,
                "lines_added": lines_added,
                "lines_removed": lines_removed,
                "message": message,
            }
        )

    def analyze_team_dynamics(self):
        """Analyze team collaboration patterns."""
        # Identify knowledge silos (files only one person has touched)
        for file_path, authors in self.team_dynamics.shared_files.items():
            if len(authors) == 1:
                sole_author = list(authors)[0]
                self.team_dynamics.knowledge_silos[file_path] = sole_author
                self.team_dynamics.bus_factor[sole_author] += 1

    def get_active_contributors(self) -> List[ContributorStats]:
        """Get currently active contributors.

        Returns:
            List of active contributors
        """
        now = datetime.now()
        threshold_date = now - timedelta(days=self.active_threshold)

        active = []
        for contributor in self.contributors.values():
            if contributor.last_commit and contributor.last_commit >= threshold_date:
                active.append(contributor)

        return sorted(active, key=lambda c: c.commits, reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get contributor statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.contributors:
            return {}

        total_commits = sum(c.commits for c in self.contributors.values())
        total_lines = sum(c.net_lines for c in self.contributors.values())

        # Top contributors
        top_by_commits = max(self.contributors.values(), key=lambda c: c.commits)
        top_by_lines = max(self.contributors.values(), key=lambda c: c.net_lines)

        # Activity metrics
        active = self.get_active_contributors()

        # Collaboration metrics
        max_collab = 0
        top_pair = None
        for pair, count in self.team_dynamics.collaboration_matrix.items():
            if count > max_collab:
                max_collab = count
                top_pair = pair

        # Bus factor analysis
        high_risk_contributors = [
            (author, count) for author, count in self.team_dynamics.bus_factor.items() if count > 10
        ]

        return {
            "total_contributors": len(self.contributors),
            "active_contributors": len(active),
            "total_commits": total_commits,
            "total_lines_changed": total_lines,
            "top_contributor_by_commits": top_by_commits.name,
            "top_contributor_by_lines": top_by_lines.name,
            "most_collaborative_pair": top_pair,
            "collaboration_count": max_collab,
            "knowledge_silos": len(self.team_dynamics.knowledge_silos),
            "high_risk_contributors": high_risk_contributors,
            "avg_commits_per_contributor": total_commits / len(self.contributors),
        }

    def _render_ascii(self) -> str:
        """Render contributor stats as ASCII."""
        lines = []
        lines.append("=" * 80)
        lines.append(f" {self.title} ".center(80))
        lines.append("=" * 80)
        lines.append("")

        # Statistics
        stats = self.get_statistics()
        lines.append(f"Total contributors: {stats.get('total_contributors', 0)}")
        lines.append(f"Active contributors: {stats.get('active_contributors', 0)}")
        lines.append(f"Total commits: {stats.get('total_commits', 0)}")
        lines.append("")

        # Top contributors table
        lines.append("Top Contributors:")
        lines.append("-" * 80)
        lines.append(f"{'Name':<30} {'Commits':<10} {'Lines':<10} {'Files':<10} {'Active':<10}")
        lines.append("-" * 80)

        sorted_contributors = sorted(
            self.contributors.values(), key=lambda c: c.commits, reverse=True
        )[:10]

        for contributor in sorted_contributors:
            # Check if active
            is_active = "Yes" if contributor in self.get_active_contributors() else "No"

            lines.append(
                f"{truncate_text(contributor.name, 30):<30} "
                f"{contributor.commits:<10d} "
                f"{contributor.net_lines:<10d} "
                f"{len(contributor.files_touched):<10d} "
                f"{is_active:<10}"
            )

        lines.append("")

        # Activity distribution
        lines.append("Activity Distribution:")
        lines.append("-" * 40)

        # Group by activity level
        very_active = [c for c in self.contributors.values() if c.commits > 100]
        active = [c for c in self.contributors.values() if 10 < c.commits <= 100]
        occasional = [c for c in self.contributors.values() if c.commits <= 10]

        lines.append(f"  Very active (>100 commits):  {len(very_active):3d} contributors")
        lines.append(f"  Active (11-100 commits):     {len(active):3d} contributors")
        lines.append(f"  Occasional (≤10 commits):    {len(occasional):3d} contributors")
        lines.append("")

        # Collaboration patterns
        if self.team_dynamics.collaboration_matrix:
            lines.append("Top Collaborations:")
            lines.append("-" * 40)

            sorted_collabs = sorted(
                self.team_dynamics.collaboration_matrix.items(), key=lambda x: x[1], reverse=True
            )[:5]

            for (author1, author2), count in sorted_collabs:
                lines.append(
                    f"  {truncate_text(author1, 20)} + {truncate_text(author2, 20)}: {count} commits"
                )

        # Bus factor warnings
        high_risk = stats.get("high_risk_contributors", [])
        if high_risk:
            lines.append("")
            lines.append("⚠️  Bus Factor Risks:")
            lines.append("-" * 40)
            for author, file_count in high_risk[:5]:
                lines.append(f"  {truncate_text(author, 30)}: sole knowledge of {file_count} files")

        return "\n".join(lines)

    def _render_html(self, width: int, height: int) -> str:
        """Render as interactive HTML charts."""
        if not PLOTLY_AVAILABLE:
            return super()._render_html(width, height)

        import plotly.graph_objs as go
        import plotly.offline as offline
        from plotly.subplots import make_subplots

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Commits by Contributor",
                "Lines Changed Over Time",
                "Language Distribution",
                "Collaboration Network",
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}], [{"type": "pie"}, {"type": "scatter"}]],
        )

        # Sort contributors
        sorted_contributors = sorted(
            self.contributors.values(), key=lambda c: c.commits, reverse=True
        )[:15]

        # 1. Commits bar chart
        fig.add_trace(
            go.Bar(
                x=[c.name for c in sorted_contributors],
                y=[c.commits for c in sorted_contributors],
                name="Commits",
                marker_color=self.color_scheme.primary,
            ),
            row=1,
            col=1,
        )

        # 2. Activity timeline
        if self.timeline:
            # Group by day
            daily_activity = defaultdict(int)
            for event in self.timeline:
                day = event["timestamp"].strftime("%Y-%m-%d")
                daily_activity[day] += event["lines_added"] + event["lines_removed"]

            dates = sorted(daily_activity.keys())
            values = [daily_activity[d] for d in dates]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    mode="lines",
                    name="Lines Changed",
                    line=dict(color=self.color_scheme.secondary),
                ),
                row=1,
                col=2,
            )

        # 3. Language distribution pie chart
        all_languages = defaultdict(int)
        for contributor in self.contributors.values():
            for lang, count in contributor.languages.items():
                all_languages[lang] += count

        if all_languages:
            fig.add_trace(
                go.Pie(
                    labels=list(all_languages.keys()),
                    values=list(all_languages.values()),
                    name="Languages",
                ),
                row=2,
                col=1,
            )

        # 4. Collaboration network (simplified)
        if self.team_dynamics.collaboration_matrix:
            # Create collaboration strength scatter
            x = []
            y = []
            sizes = []
            texts = []

            for i, c1 in enumerate(sorted_contributors[:10]):
                for j, c2 in enumerate(sorted_contributors[:10]):
                    if i < j:
                        pair = tuple(sorted([c1.name, c2.name]))
                        strength = self.team_dynamics.collaboration_matrix.get(pair, 0)
                        if strength > 0:
                            x.append(i)
                            y.append(j)
                            sizes.append(strength * 10)
                            texts.append(f"{c1.name}<br>{c2.name}<br>{strength} commits")

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=sizes, color=self.color_scheme.info, opacity=0.6),
                    text=texts,
                    hoverinfo="text",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(title_text=self.title, showlegend=False, height=height, width=width)

        # Generate HTML
        return offline.plot(fig, output_type="div", include_plotlyjs="cdn")

    def _render_svg(self, width: int, height: int, dpi: int) -> str:
        """Render as SVG charts."""
        if not MATPLOTLIB_AVAILABLE:
            return super()._render_svg(width, height, dpi)

        from io import StringIO

        import matplotlib.pyplot as plt

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(width / dpi, height / dpi), dpi=dpi
        )

        # Sort contributors
        sorted_contributors = sorted(
            self.contributors.values(), key=lambda c: c.commits, reverse=True
        )[:10]

        if sorted_contributors:
            # 1. Commits bar chart
            names = [c.name.split("@")[0][:10] for c in sorted_contributors]  # Truncate names
            commits = [c.commits for c in sorted_contributors]

            ax1.bar(range(len(names)), commits, color=self.color_scheme.primary)
            ax1.set_xticks(range(len(names)))
            ax1.set_xticklabels(names, rotation=45, ha="right")
            ax1.set_ylabel("Commits")
            ax1.set_title("Top Contributors")

            # 2. Productivity scatter
            x = [c.commits for c in sorted_contributors]
            y = [c.productivity for c in sorted_contributors]
            s = [len(c.files_touched) * 10 for c in sorted_contributors]

            ax2.scatter(x, y, s=s, alpha=0.6, color=self.color_scheme.secondary)
            ax2.set_xlabel("Commits")
            ax2.set_ylabel("Lines per Commit")
            ax2.set_title("Productivity Analysis")

            # 3. Activity heatmap (simplified)
            # Create activity matrix (contributor x time period)
            weeks = 12  # Last 12 weeks
            activity_matrix = []

            for contributor in sorted_contributors[:5]:
                week_activity = [0] * weeks
                if contributor.last_commit:
                    for day in contributor.active_days:
                        # Simple week calculation
                        week_idx = hash(day) % weeks
                        week_activity[week_idx] += 1
                activity_matrix.append(week_activity)

            if activity_matrix:
                im = ax3.imshow(activity_matrix, cmap="YlOrRd", aspect="auto")
                ax3.set_yticks(range(len(sorted_contributors[:5])))
                ax3.set_yticklabels([c.name.split("@")[0][:10] for c in sorted_contributors[:5]])
                ax3.set_xlabel("Week")
                ax3.set_title("Activity Heatmap")
                plt.colorbar(im, ax=ax3)

            # 4. Language distribution
            all_languages = defaultdict(int)
            for contributor in self.contributors.values():
                for lang, count in contributor.languages.items():
                    all_languages[lang] += count

            if all_languages:
                languages = list(all_languages.keys())[:5]
                counts = [all_languages[l] for l in languages]

                ax4.pie(counts, labels=languages, autopct="%1.1f%%")
                ax4.set_title("Language Distribution")

        # Overall title
        fig.suptitle(self.title, fontsize=14, fontweight="bold")

        # Convert to SVG
        buffer = StringIO()
        plt.tight_layout()
        plt.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()


def analyze_contributors(
    commits: List[Dict[str, Any]], title: str = "Contributor Analysis", active_only: bool = False
) -> ContributorGraph:
    """Analyze contributors from commit data.

    Args:
        commits: List of commit dictionaries
        title: Graph title
        active_only: Only show active contributors

    Returns:
        ContributorGraph instance
    """
    graph = ContributorGraph(title=title)

    # Add commits to graph
    for commit in commits:
        graph.add_commit(
            author=commit.get("author", "unknown"),
            timestamp=commit.get("timestamp", datetime.now()),
            files=commit.get("files", []),
            lines_added=commit.get("lines_added", 0),
            lines_removed=commit.get("lines_removed", 0),
            co_authors=commit.get("co_authors"),
            message=commit.get("message", ""),
        )

    # Analyze team dynamics
    graph.analyze_team_dynamics()

    return graph
