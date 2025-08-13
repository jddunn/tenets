"""File coupling visualization.

This module visualizes files that frequently change together, helping identify
tight coupling and areas that should potentially be refactored together.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tenets.utils.logger import get_logger

from .base import (
    ColorScheme,
    VisualizationBase,
    VisualizationFormat,
    create_graph_layout,
    truncate_text,
    MATPLOTLIB_AVAILABLE,
    NETWORKX_AVAILABLE,
    PLOTLY_AVAILABLE,
)


@dataclass
class FileCoupling:
    """Coupling information between two files.
    
    Attributes:
        file1: First file path
        file2: Second file path
        change_count: Number of times changed together
        confidence: Confidence score (0-1)
        commits: List of commit hashes where both changed
        temporal_distance: Average time between changes
    """
    
    file1: str
    file2: str
    change_count: int = 0
    confidence: float = 0.0
    commits: List[str] = None
    temporal_distance: float = 0.0
    
    def __post_init__(self):
        if self.commits is None:
            self.commits = []
            
    @property
    def strength(self) -> float:
        """Get coupling strength combining count and confidence."""
        return self.change_count * self.confidence
        
    @property
    def coupling_type(self) -> str:
        """Categorize coupling type."""
        if self.change_count >= 10 and self.confidence >= 0.8:
            return "strong"
        elif self.change_count >= 5 and self.confidence >= 0.5:
            return "moderate"
        elif self.change_count >= 2:
            return "weak"
        else:
            return "minimal"


@dataclass
class FileChangeHistory:
    """Change history for a file.
    
    Attributes:
        path: File path
        total_changes: Total number of changes
        commits: List of commits
        authors: Set of authors who changed the file
        first_change: First change timestamp
        last_change: Last change timestamp
        coupled_files: Files frequently changed with this one
    """
    
    path: str
    total_changes: int = 0
    commits: List[Dict[str, Any]] = None
    authors: Set[str] = None
    first_change: Optional[datetime] = None
    last_change: Optional[datetime] = None
    coupled_files: Dict[str, int] = None
    
    def __post_init__(self):
        if self.commits is None:
            self.commits = []
        if self.authors is None:
            self.authors = set()
        if self.coupled_files is None:
            self.coupled_files = {}
            
    @property
    def change_frequency(self) -> float:
        """Get average changes per day."""
        if not self.first_change or not self.last_change:
            return 0.0
            
        days = (self.last_change - self.first_change).days
        if days == 0:
            return self.total_changes
            
        return self.total_changes / days


class CouplingGraph(VisualizationBase):
    """Visualize file coupling patterns.
    
    Creates visualizations showing which files frequently change together,
    helping identify:
    - Tightly coupled modules
    - Hidden dependencies
    - Refactoring candidates
    - Team boundaries
    """
    
    def __init__(
        self,
        title: str = "File Coupling Graph",
        color_scheme: Optional[ColorScheme] = None,
        format: VisualizationFormat = VisualizationFormat.AUTO,
        min_coupling: int = 2
    ):
        """Initialize coupling graph.
        
        Args:
            title: Graph title
            color_scheme: Color scheme
            format: Output format
            min_coupling: Minimum changes together to show coupling
        """
        super().__init__(title, color_scheme, format)
        self.logger = get_logger(__name__)
        self.min_coupling = min_coupling
        self.file_history: Dict[str, FileChangeHistory] = {}
        self.couplings: List[FileCoupling] = []
        self.commit_data: List[Dict[str, Any]] = []
        
    def add_commit(
        self,
        commit_hash: str,
        files: List[str],
        author: str,
        timestamp: datetime,
        message: str = ""
    ):
        """Add a commit to the coupling analysis.
        
        Args:
            commit_hash: Commit hash
            files: Files changed in commit
            author: Commit author
            timestamp: Commit timestamp
            message: Commit message
        """
        commit_data = {
            'hash': commit_hash,
            'files': files,
            'author': author,
            'timestamp': timestamp,
            'message': message
        }
        self.commit_data.append(commit_data)
        
        # Update file histories
        for file_path in files:
            if file_path not in self.file_history:
                self.file_history[file_path] = FileChangeHistory(path=file_path)
                
            history = self.file_history[file_path]
            history.total_changes += 1
            history.commits.append(commit_data)
            history.authors.add(author)
            
            # Update timestamps
            if not history.first_change or timestamp < history.first_change:
                history.first_change = timestamp
            if not history.last_change or timestamp > history.last_change:
                history.last_change = timestamp
                
            # Track coupled files
            for other_file in files:
                if other_file != file_path:
                    if other_file not in history.coupled_files:
                        history.coupled_files[other_file] = 0
                    history.coupled_files[other_file] += 1
                    
    def analyze_coupling(self):
        """Analyze coupling patterns from commit data."""
        # Find file pairs that change together
        pair_counts = defaultdict(int)
        pair_commits = defaultdict(list)
        
        for commit in self.commit_data:
            files = commit['files']
            # Generate all pairs
            for i, file1 in enumerate(files):
                for file2 in files[i + 1:]:
                    pair = tuple(sorted([file1, file2]))
                    pair_counts[pair] += 1
                    pair_commits[pair].append(commit['hash'])
                    
        # Create coupling objects
        self.couplings = []
        
        for (file1, file2), count in pair_counts.items():
            if count >= self.min_coupling:
                # Calculate confidence
                total1 = self.file_history[file1].total_changes
                total2 = self.file_history[file2].total_changes
                confidence = count / min(total1, total2)
                
                coupling = FileCoupling(
                    file1=file1,
                    file2=file2,
                    change_count=count,
                    confidence=confidence,
                    commits=pair_commits[(file1, file2)]
                )
                
                self.couplings.append(coupling)
                
        # Sort by strength
        self.couplings.sort(key=lambda c: c.strength, reverse=True)
        
    def find_clusters(self) -> List[Set[str]]:
        """Find clusters of tightly coupled files.
        
        Returns:
            List of file clusters
        """
        if not NETWORKX_AVAILABLE:
            return []
            
        import networkx as nx
        
        # Build graph
        G = nx.Graph()
        
        for coupling in self.couplings:
            if coupling.coupling_type in ["strong", "moderate"]:
                G.add_edge(
                    coupling.file1,
                    coupling.file2,
                    weight=coupling.strength
                )
                
        # Find connected components
        clusters = []
        for component in nx.connected_components(G):
            if len(component) > 1:
                clusters.append(component)
                
        return clusters
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get coupling statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.couplings:
            return {}
            
        strong_couplings = [c for c in self.couplings if c.coupling_type == "strong"]
        moderate_couplings = [c for c in self.couplings if c.coupling_type == "moderate"]
        
        # Find most coupled file
        file_coupling_counts = defaultdict(int)
        for coupling in self.couplings:
            file_coupling_counts[coupling.file1] += coupling.change_count
            file_coupling_counts[coupling.file2] += coupling.change_count
            
        most_coupled = max(file_coupling_counts.items(), key=lambda x: x[1]) if file_coupling_counts else (None, 0)
        
        # Find most active author
        author_counts = defaultdict(int)
        for commit in self.commit_data:
            author_counts[commit['author']] += len(commit['files'])
            
        most_active = max(author_counts.items(), key=lambda x: x[1]) if author_counts else (None, 0)
        
        return {
            'total_files': len(self.file_history),
            'total_commits': len(self.commit_data),
            'total_couplings': len(self.couplings),
            'strong_couplings': len(strong_couplings),
            'moderate_couplings': len(moderate_couplings),
            'avg_coupling_strength': sum(c.strength for c in self.couplings) / len(self.couplings),
            'most_coupled_file': most_coupled[0],
            'most_coupled_count': most_coupled[1],
            'most_active_author': most_active[0],
            'clusters_found': len(self.find_clusters())
        }
        
    def _render_ascii(self) -> str:
        """Render coupling as ASCII."""
        lines = []
        lines.append("=" * 80)
        lines.append(f" {self.title} ".center(80))
        lines.append("=" * 80)
        lines.append("")
        
        # Statistics
        stats = self.get_statistics()
        lines.append(f"Files tracked: {stats.get('total_files', 0)}")
        lines.append(f"Commits analyzed: {stats.get('total_commits', 0)}")
        lines.append(f"Couplings found: {stats.get('total_couplings', 0)}")
        lines.append(f"  Strong: {stats.get('strong_couplings', 0)}")
        lines.append(f"  Moderate: {stats.get('moderate_couplings', 0)}")
        lines.append("")
        
        # Top couplings
        lines.append("Strongest File Couplings:")
        lines.append("-" * 80)
        lines.append(f"{'File 1':<30} {'File 2':<30} {'Changes':<10} {'Confidence':<10}")
        lines.append("-" * 80)
        
        for coupling in self.couplings[:10]:
            file1 = Path(coupling.file1).name
            file2 = Path(coupling.file2).name
            lines.append(
                f"{truncate_text(file1, 30):<30} "
                f"{truncate_text(file2, 30):<30} "
                f"{coupling.change_count:<10d} "
                f"{coupling.confidence:<10.2f}"
            )
            
        lines.append("")
        
        # File clusters
        clusters = self.find_clusters()
        if clusters:
            lines.append("Coupled File Clusters:")
            lines.append("-" * 40)
            
            for i, cluster in enumerate(clusters[:5], 1):
                lines.append(f"\nCluster {i} ({len(cluster)} files):")
                for file_path in sorted(cluster)[:5]:
                    lines.append(f"  - {Path(file_path).name}")
                if len(cluster) > 5:
                    lines.append(f"  ... and {len(cluster) - 5} more")
                    
        # Most coupled files
        lines.append("")
        lines.append("Most Coupled Files:")
        lines.append("-" * 40)
        
        file_coupling_counts = defaultdict(int)
        for coupling in self.couplings:
            file_coupling_counts[coupling.file1] += coupling.change_count
            file_coupling_counts[coupling.file2] += coupling.change_count
            
        sorted_files = sorted(file_coupling_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for file_path, count in sorted_files:
            lines.append(f"  {Path(file_path).name:<40} {count:3d} couplings")
            
        return "\n".join(lines)
        
    def _render_html(self, width: int, height: int) -> str:
        """Render as interactive HTML graph."""
        if not PLOTLY_AVAILABLE:
            return super()._render_html(width, height)
            
        import plotly.graph_objs as go
        import plotly.offline as offline
        
        # Create nodes and edges
        nodes = list(self.file_history.keys())
        edges = [(c.file1, c.file2) for c in self.couplings if c.coupling_type in ["strong", "moderate"]]
        
        if not nodes:
            return "<div>No coupling data to visualize</div>"
            
        # Get layout
        positions = create_graph_layout(nodes, edges, "spring")
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in nodes:
            x, y = positions.get(node, (0, 0))
            node_x.append(x)
            node_y.append(y)
            
            history = self.file_history[node]
            
            # Hover text
            hover = f"{Path(node).name}<br>"
            hover += f"Changes: {history.total_changes}<br>"
            hover += f"Authors: {len(history.authors)}<br>"
            hover += f"Coupled files: {len(history.coupled_files)}"
            node_text.append(hover)
            
            # Size by change count
            node_size.append(10 + math.log(1 + history.total_changes) * 5)
            
            # Color by coupling count
            node_color.append(len(history.coupled_files))
            
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_text = []
        
        for coupling in self.couplings:
            if coupling.coupling_type in ["strong", "moderate"]:
                x0, y0 = positions.get(coupling.file1, (0, 0))
                x1, y1 = positions.get(coupling.file2, (0, 0))
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=node_size,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Couplings',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            ),
            text=[Path(n).name for n in nodes],
            textposition="top center",
            hovertext=node_text
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=self.title,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height
            )
        )
        
        # Generate HTML
        return offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        
    def _render_svg(self, width: int, height: int, dpi: int) -> str:
        """Render as SVG graph."""
        if not MATPLOTLIB_AVAILABLE:
            return super()._render_svg(width, height, dpi)
            
        import matplotlib.pyplot as plt
        from io import StringIO
        
        # Create figure
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
        
        # Create simple coupling matrix heatmap
        files = list(self.file_history.keys())[:20]  # Limit to 20 files
        matrix = [[0] * len(files) for _ in range(len(files))]
        
        # Fill matrix
        for coupling in self.couplings:
            if coupling.file1 in files and coupling.file2 in files:
                i = files.index(coupling.file1)
                j = files.index(coupling.file2)
                matrix[i][j] = coupling.change_count
                matrix[j][i] = coupling.change_count
                
        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(files)))
        ax.set_yticks(range(len(files)))
        ax.set_xticklabels([Path(f).name for f in files], rotation=45, ha='right')
        ax.set_yticklabels([Path(f).name for f in files])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Changes Together')
        
        ax.set_title(self.title)
        
        # Convert to SVG
        buffer = StringIO()
        plt.tight_layout()
        plt.savefig(buffer, format='svg', bbox_inches='tight')
        plt.close(fig)
        
        buffer.seek(0)
        return buffer.read()


def analyze_coupling_from_git(
    repo_path: Path,
    since: Optional[datetime] = None,
    min_coupling: int = 2
) -> CouplingGraph:
    """Analyze file coupling from git history.
    
    Args:
        repo_path: Path to git repository
        since: Analyze commits since this date
        min_coupling: Minimum changes together
        
    Returns:
        CouplingGraph instance
    """
    graph = CouplingGraph(min_coupling=min_coupling)
    
    # This would integrate with git module to get commit history
    # Placeholder for demonstration
    
    return graph