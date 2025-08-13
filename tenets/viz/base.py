"""Base utilities for visualization components.

This module provides shared functionality for all visualization types including
graph creation, styling, and export capabilities.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tenets.utils.logger import get_logger

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    
try:
    import plotly.graph_objs as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class VisualizationFormat(Enum):
    """Supported visualization output formats."""
    
    SVG = "svg"
    PNG = "png"
    HTML = "html"
    ASCII = "ascii"
    JSON = "json"
    AUTO = "auto"


@dataclass
class ColorScheme:
    """Color scheme for visualizations.
    
    Provides consistent colors across different visualization types.
    """
    
    primary: str = "#2E86AB"      # Blue
    secondary: str = "#A23B72"     # Purple
    success: str = "#73AB84"      # Green
    warning: str = "#F18F01"      # Orange
    danger: str = "#C73E1D"       # Red
    info: str = "#6C91BF"         # Light blue
    
    # Gradients
    gradient_low: str = "#E8F4F8"
    gradient_mid: str = "#73AB84"
    gradient_high: str = "#C73E1D"
    
    # Background/foreground
    background: str = "#FFFFFF"
    foreground: str = "#2D3436"
    muted: str = "#95A5A6"
    
    def get_gradient(self, n: int = 10) -> List[str]:
        """Get gradient colors.
        
        Args:
            n: Number of gradient steps
            
        Returns:
            List of color codes
        """
        if not MATPLOTLIB_AVAILABLE:
            # Simple gradient without matplotlib
            return [self.gradient_low, self.gradient_mid, self.gradient_high]
            
        import matplotlib.cm as cm
        cmap = cm.get_cmap('RdYlGn_r')
        return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]
        
    def get_categorical(self, n: int = 10) -> List[str]:
        """Get categorical colors.
        
        Args:
            n: Number of categories
            
        Returns:
            List of distinct colors
        """
        base_colors = [
            self.primary, self.secondary, self.success,
            self.warning, self.danger, self.info
        ]
        
        if n <= len(base_colors):
            return base_colors[:n]
            
        # Generate more colors
        if MATPLOTLIB_AVAILABLE:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab20')
            return [mcolors.to_hex(cmap(i / n)) for i in range(n)]
        else:
            # Repeat base colors
            colors = []
            for i in range(n):
                colors.append(base_colors[i % len(base_colors)])
            return colors


class VisualizationBase:
    """Base class for all visualization types.
    
    Provides common functionality for creating, styling, and
    exporting visualizations.
    """
    
    def __init__(
        self,
        title: Optional[str] = None,
        color_scheme: Optional[ColorScheme] = None,
        format: Union[str, VisualizationFormat] = VisualizationFormat.AUTO
    ):
        """Initialize visualization.
        
        Args:
            title: Visualization title
            color_scheme: Color scheme to use
            format: Output format
        """
        self.logger = get_logger(__name__)
        self.title = title
        self.color_scheme = color_scheme or ColorScheme()
        
        if isinstance(format, str):
            self.format = VisualizationFormat(format)
        else:
            self.format = format
            
        self.figure = None
        self.data = {}
        
    def render(
        self,
        output_path: Optional[Path] = None,
        width: int = 1200,
        height: int = 800,
        dpi: int = 100
    ) -> Union[str, bytes, None]:
        """Render the visualization.
        
        Args:
            output_path: Optional path to save output
            width: Width in pixels
            height: Height in pixels
            dpi: DPI for raster formats
            
        Returns:
            Rendered content or None if saved to file
        """
        # Determine format
        if self.format == VisualizationFormat.AUTO:
            if output_path:
                suffix = Path(output_path).suffix.lower()
                if suffix == '.svg':
                    format = VisualizationFormat.SVG
                elif suffix == '.png':
                    format = VisualizationFormat.PNG
                elif suffix == '.html':
                    format = VisualizationFormat.HTML
                elif suffix == '.json':
                    format = VisualizationFormat.JSON
                else:
                    format = VisualizationFormat.PNG
            else:
                format = VisualizationFormat.HTML
        else:
            format = self.format
            
        # Render based on format
        if format == VisualizationFormat.ASCII:
            content = self._render_ascii()
        elif format == VisualizationFormat.JSON:
            content = self._render_json()
        elif format == VisualizationFormat.HTML:
            content = self._render_html(width, height)
        elif format == VisualizationFormat.SVG:
            content = self._render_svg(width, height, dpi)
        elif format == VisualizationFormat.PNG:
            content = self._render_png(width, height, dpi)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(content, bytes):
                output_path.write_bytes(content)
            else:
                output_path.write_text(content)
                
            self.logger.info(f"Visualization saved to {output_path}")
            return None
        else:
            return content
            
    def _render_ascii(self) -> str:
        """Render as ASCII art.
        
        Returns:
            ASCII representation
        """
        # Override in subclasses
        return "ASCII visualization not implemented"
        
    def _render_json(self) -> str:
        """Render as JSON data.
        
        Returns:
            JSON string
        """
        return json.dumps(self.data, indent=2, default=str)
        
    def _render_html(self, width: int, height: int) -> str:
        """Render as HTML.
        
        Args:
            width: Width in pixels
            height: Height in pixels
            
        Returns:
            HTML string
        """
        # Override in subclasses for interactive HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title or 'Visualization'}</title>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .container {{ width: {width}px; height: {height}px; margin: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{self.title or 'Visualization'}</h1>
                <pre>{self._render_ascii()}</pre>
            </div>
        </body>
        </html>
        """
        
    def _render_svg(self, width: int, height: int, dpi: int) -> str:
        """Render as SVG.
        
        Args:
            width: Width in pixels
            height: Height in pixels
            dpi: DPI setting
            
        Returns:
            SVG string
        """
        if not MATPLOTLIB_AVAILABLE:
            return "<svg>Matplotlib not available</svg>"
            
        # Override in subclasses
        return "<svg>SVG not implemented</svg>"
        
    def _render_png(self, width: int, height: int, dpi: int) -> bytes:
        """Render as PNG.
        
        Args:
            width: Width in pixels
            height: Height in pixels
            dpi: DPI setting
            
        Returns:
            PNG bytes
        """
        if not MATPLOTLIB_AVAILABLE:
            return b"PNG rendering requires matplotlib"
            
        # Override in subclasses
        return b""
        
    def check_dependencies(self) -> Dict[str, bool]:
        """Check which visualization libraries are available.
        
        Returns:
            Dictionary of library availability
        """
        return {
            'matplotlib': MATPLOTLIB_AVAILABLE,
            'networkx': NETWORKX_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE,
        }


def create_graph_layout(
    nodes: List[Any],
    edges: List[Tuple[Any, Any]],
    layout_type: str = "spring"
) -> Dict[Any, Tuple[float, float]]:
    """Create graph layout positions.
    
    Args:
        nodes: List of node identifiers
        edges: List of edge tuples
        layout_type: Layout algorithm
        
    Returns:
        Dictionary mapping nodes to (x, y) positions
    """
    if not NETWORKX_AVAILABLE:
        # Simple grid layout fallback
        positions = {}
        n = len(nodes)
        cols = int(n ** 0.5) + 1
        
        for i, node in enumerate(nodes):
            x = (i % cols) / cols
            y = (i // cols) / cols
            positions[node] = (x, y)
            
        return positions
        
    # Use NetworkX for sophisticated layouts
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    if layout_type == "spring":
        pos = nx.spring_layout(G)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "spectral":
        pos = nx.spectral_layout(G)
    elif layout_type == "shell":
        pos = nx.shell_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
        
    return pos


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."