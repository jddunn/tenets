"""HTML report generator module.

This module provides HTML report generation functionality with rich
visualizations, interactive charts, and professional styling. It creates
standalone HTML reports that can be viewed in any modern web browser.

The HTML reporter generates responsive, interactive reports with embedded
JavaScript visualizations and customizable themes.
"""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger

from .generator import ReportConfig, ReportSection

# Re-export ReportGenerator for tests that patch via this module
try:  # pragma: no cover
    from .generator import ReportGenerator as ReportGenerator
except Exception:  # pragma: no cover
    ReportGenerator = None  # type: ignore


class HTMLTemplate:
    """HTML template generator for reports.

    Provides template generation for various report components including
    the main layout, charts, tables, and interactive elements.

    Attributes:
        theme: Visual theme name
        custom_css: Custom CSS styles
        include_charts: Whether to include chart libraries
    """

    def __init__(
        self, theme: str = "default", custom_css: Optional[str] = None, include_charts: bool = True
    ):
        """Initialize HTML template.

        Args:
            theme: Theme name
            custom_css: Custom CSS styles
            include_charts: Include chart libraries
        """
        self.theme = theme
        self.custom_css = custom_css
        self.include_charts = include_charts

    def get_base_template(self) -> str:
        """Get base HTML template.

        Returns:
            str: Base HTML template
        """
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    {styles}
    {scripts}
</head>
<body>
    <div class="container">
        {header}
        {navigation}
        <main class="content">
            {content}
        </main>
        {footer}
    </div>
    {chart_scripts}
</body>
</html>"""

    def get_styles(self) -> str:
        """Get CSS styles for the report.

        Returns:
            str: CSS styles
        """
        base_styles = """
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --info-color: #06b6d4;
            --background: #ffffff;
            --surface: #f8fafc;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border: #e2e8f0;
            --shadow: rgba(0, 0, 0, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: var(--background);
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary-color), #8b5cf6);
            color: white;
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px var(--shadow);
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .header .meta {
            opacity: 0.9;
            font-size: 0.95rem;
        }

        .header .score {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 8px 16px;
            border-radius: 20px;
            margin-top: 15px;
            font-weight: 600;
        }

        /* Navigation */
        .nav {
            background: var(--surface);
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            position: sticky;
            top: 20px;
            z-index: 100;
            box-shadow: 0 2px 10px var(--shadow);
        }

        .nav ul {
            list-style: none;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .nav a {
            color: var(--text-primary);
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .nav a:hover {
            background: var(--primary-color);
            color: white;
        }

        .nav a.active {
            background: var(--primary-color);
            color: white;
        }

        /* Sections */
        .section {
            background: white;
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px var(--shadow);
        }

        .section h2 {
            color: var(--text-primary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section h3 {
            color: var(--text-primary);
            margin: 20px 0 15px;
            font-size: 1.2rem;
        }

        /* Tables */
        .table-wrapper {
            overflow-x: auto;
            margin: 20px 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }

        th {
            background: var(--surface);
            color: var(--text-primary);
            font-weight: 600;
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid var(--border);
        }

        td {
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }

        tr:hover {
            background: var(--surface);
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-critical {
            background: var(--danger-color);
            color: white;
        }

        .badge-high {
            background: #f97316;
            color: white;
        }

        .badge-medium {
            background: var(--warning-color);
            color: white;
        }

        .badge-low {
            background: var(--success-color);
            color: white;
        }

        .badge-info {
            background: var(--info-color);
            color: white;
        }

        /* Charts */
        .chart-container {
            margin: 20px 0;
            padding: 20px;
            background: var(--surface);
            border-radius: 8px;
            min-height: 300px;
        }

        .chart-title {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 15px;
            text-align: center;
        }

        /* Code Snippets */
        .code-snippet {
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .code-snippet .line-number {
            display: inline-block;
            width: 40px;
            color: #64748b;
            text-align: right;
            margin-right: 15px;
            user-select: none;
        }

        .code-snippet .highlight {
            background: rgba(251, 191, 36, 0.2);
            display: block;
        }

        /* Progress Bars */
        .progress {
            height: 24px;
            background: var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin: 10px 0;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), #8b5cf6);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.85rem;
            font-weight: 600;
            transition: width 0.6s ease;
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .metric-card {
            background: var(--surface);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.3s;
            position: relative;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px var(--shadow);
        }
        
        /* Tooltip styles */
        .metric-card[data-tooltip]:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            z-index: 1000;
            margin-bottom: 10px;
            max-width: 250px;
            white-space: normal;
            text-align: left;
            line-height: 1.4;
        }
        
        .metric-card[data-tooltip]:hover::before {
            content: "";
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: #333;
            margin-bottom: 4px;
            z-index: 1000;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
            margin: 10px 0;
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Collapsible Sections */
        .collapsible {
            cursor: pointer;
            user-select: none;
        }

        .collapsible::before {
            content: '▼';
            display: inline-block;
            margin-right: 8px;
            transition: transform 0.3s;
        }

        .collapsible.collapsed::before {
            transform: rotate(-90deg);
        }

        .collapsible-content {
            max-height: 2000px;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .collapsible-content.collapsed {
            max-height: 0;
        }

        /* Alerts */
        .alert {
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .alert-success {
            background: #10b98120;
            border-left: 4px solid var(--success-color);
            color: #047857;
        }

        .alert-warning {
            background: #f59e0b20;
            border-left: 4px solid var(--warning-color);
            color: #b45309;
        }

        .alert-danger {
            background: #ef444420;
            border-left: 4px solid var(--danger-color);
            color: #b91c1c;
        }

        .alert-info {
            background: #06b6d420;
            border-left: 4px solid var(--info-color);
            color: #0e7490;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 30px;
            color: var(--text-secondary);
            border-top: 1px solid var(--border);
            margin-top: 50px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }

            .header {
                padding: 20px;
            }

            .header h1 {
                font-size: 1.8rem;
            }

            .section {
                padding: 20px;
            }

            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .nav ul {
                flex-direction: column;
                gap: 10px;
            }
        }

        /* Dark Theme */
        @media (prefers-color-scheme: dark) {
            :root {
                --background: #0f172a;
                --surface: #1e293b;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --border: #334155;
                --shadow: rgba(0, 0, 0, 0.3);
            }

            .code-snippet {
                background: #0f172a;
            }
        }

        /* Print Styles */
        @media print {
            .nav {
                display: none;
            }

            .section {
                page-break-inside: avoid;
                box-shadow: none;
                border: 1px solid var(--border);
            }

            .chart-container {
                page-break-inside: avoid;
            }
        }
    </style>
    """

        # Add custom CSS if provided
        if self.custom_css:
            base_styles += f"\n<style>\n{self.custom_css}\n</style>"

        # Add theme-specific styles
        if self.theme == "dark":
            base_styles += self._get_dark_theme_styles()
        elif self.theme == "corporate":
            base_styles += self._get_corporate_theme_styles()

        return base_styles

    def get_scripts(self) -> str:
        """Get JavaScript libraries and scripts.

        Returns:
            str: Script tags
        """
        scripts = []

        if self.include_charts:
            # Include Chart.js for charts
            scripts.append(
                '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>'
            )

            # Include Prism.js for code highlighting
            scripts.append(
                '<link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">'
            )
            scripts.append(
                '<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>'
            )
            scripts.append(
                '<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>'
            )

        return "\n    ".join(scripts)

    def get_navigation(self, sections: List[ReportSection]) -> str:
        """Generate navigation menu.

        Args:
            sections: Report sections

        Returns:
            str: Navigation HTML
        """
        nav_items = []

        for section in sections:
            if section.visible:
                icon = section.icon if section.icon else ""
                # Preserve emoji/icons as-is; HTML is written as UTF-8
                nav_items.append(f'<li><a href="#{section.id}">{icon} {section.title}</a></li>')

        return f"""
    <nav class="nav">
        <ul>
            {" ".join(nav_items)}
        </ul>
    </nav>
    """

    def _get_dark_theme_styles(self) -> str:
        """Get dark theme specific styles.

        Returns:
            str: Dark theme CSS
        """
        return """
    <style>
        :root {
            --background: #0f172a;
            --surface: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #334155;
        }
    </style>
    """

    def _get_corporate_theme_styles(self) -> str:
        """Get corporate theme specific styles.

        Returns:
            str: Corporate theme CSS
        """
        return """
    <style>
        :root {
            --primary-color: #1e40af;
            --secondary-color: #475569;
            --success-color: #059669;
            --warning-color: #d97706;
            --danger-color: #dc2626;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .header {
            background: linear-gradient(135deg, #1e40af, #1e293b);
        }
    </style>
    """


class HTMLReporter:
    """HTML report generator.

    Generates standalone HTML reports with rich visualizations and
    interactive elements from analysis results.

    Attributes:
        config: Configuration object
        logger: Logger instance
        template: HTML template generator
    """

    def __init__(self, config: TenetsConfig):
        """Initialize HTML reporter.

        Args:
            config: TenetsConfig instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.template = HTMLTemplate()

    def generate(
        self,
        sections: List[ReportSection],
        metadata: Dict[str, Any],
        output_path: Path,
        report_config: ReportConfig,
    ) -> Path:
        """Generate HTML report.

        Args:
            sections: Report sections
            metadata: Report metadata
            output_path: Output file path
            report_config: Report configuration

        Returns:
            Path: Path to generated report
        """
        self.logger.debug(f"Generating HTML report to {output_path}")

        # Set template configuration
        self.template = HTMLTemplate(
            theme=report_config.theme,
            custom_css=self._load_custom_css(report_config.custom_css),
            include_charts=report_config.include_charts,
        )

        # Generate HTML content
        html_content = self._generate_html(sections, metadata, report_config)

        # Ensure output is ASCII-safe for environments that read with
        # platform default encodings (e.g., cp1252 on Windows). Convert
        # non-ASCII characters to HTML entities to avoid decode errors
        # when tests read the file without specifying encoding.
        try:
            safe_content = html_content.encode("ascii", "xmlcharrefreplace").decode("ascii")
        except Exception:
            safe_content = html_content  # Fallback; still write as-is

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(safe_content)

        self.logger.info(f"HTML report generated: {output_path}")
        return output_path

    def _generate_html(
        self, sections: List[ReportSection], metadata: Dict[str, Any], report_config: ReportConfig
    ) -> str:
        """Generate complete HTML content.

        Args:
            sections: Report sections
            metadata: Report metadata
            report_config: Report configuration

        Returns:
            str: Complete HTML content
        """
        # Generate header
        header = self._generate_header(metadata, report_config)

        # Generate navigation
        navigation = self.template.get_navigation(sections) if report_config.include_toc else ""

        # Generate sections
        content = self._generate_sections(sections, report_config)

        # Generate footer
        footer = self._generate_footer(report_config)

        # Generate chart initialization scripts
        chart_scripts = (
            self._generate_chart_scripts(sections) if report_config.include_charts else ""
        )

        # Combine into template
        html = self.template.get_base_template().format(
            title=report_config.title,
            styles=self.template.get_styles(),
            scripts=self.template.get_scripts(),
            header=header,
            navigation=navigation,
            content=content,
            footer=footer,
            chart_scripts=chart_scripts,
        )

        return html

    def _generate_header(self, metadata: Dict[str, Any], report_config: ReportConfig) -> str:
        """Generate report header.

        Args:
            metadata: Report metadata
            report_config: Report configuration

        Returns:
            str: Header HTML
        """
        summary = metadata.get("analysis_summary", {})
        health_score = summary.get("health_score", 0)

        # Determine health status
        if health_score >= 80:
            status = "Excellent"
            status_color = "var(--success-color)"
        elif health_score >= 60:
            status = "Good"
            status_color = "var(--success-color)"
        elif health_score >= 40:
            status = "Fair"
            status_color = "var(--warning-color)"
        else:
            status = "Needs Improvement"
            status_color = "var(--danger-color)"

        # Include logo if provided
        logo_html = ""
        if report_config.custom_logo and report_config.custom_logo.exists():
            logo_data = self._encode_image(report_config.custom_logo)
            logo_html = f'<img src="data:image/png;base64,{logo_data}" alt="Logo" style="height: 50px; margin-bottom: 20px;">'

        return f"""
    <header class="header">
        {logo_html}
        <h1>{report_config.title}</h1>
        <div class="meta">
            Generated: {metadata.get("generated_at", "Unknown")} |
            Files: {summary.get("total_files", 0)} |
            Lines: {summary.get("total_lines", 0):,}
        </div>
        <div class="score" style="background: {status_color}20; color: {status_color};">
            Health Score: {health_score:.1f}/100 ({status})
        </div>
    </header>
    """

    def _generate_sections(self, sections: List[ReportSection], report_config: ReportConfig) -> str:
        """Generate all sections.

        Args:
            sections: Report sections
            report_config: Report configuration

        Returns:
            str: Sections HTML
        """
        html_parts = []

        for section in sections:
            if section.visible:
                html_parts.append(self._generate_section(section, report_config))

        return "\n".join(html_parts)

    def _generate_section(self, section: ReportSection, report_config: ReportConfig) -> str:
        """Generate a single section.

        Args:
            section: Report section
            report_config: Report configuration

        Returns:
            str: Section HTML
        """
        # Section header
        icon = section.icon if section.icon else ""
        collapsible_class = "collapsible" if section.collapsible else ""

        html = f"""
        <section id="{section.id}" class="section">
            <h{section.level} class="{collapsible_class}">
                {icon} {section.title}
            </h{section.level}>
            <div class="{"collapsible-content" if section.collapsible else ""}">
        """

        # Section content
        if section.content:
            html += self._render_content(section.content)

        # Metrics
        if hasattr(section, "metrics") and section.metrics:
            html += self._render_metrics(section.metrics)

        # Tables
        for table in section.tables:
            html += self._render_table(table)

        # Charts
        if report_config.include_charts:
            for chart in section.charts:
                html += self._render_chart(chart)

        # Code snippets
        if report_config.include_code_snippets:
            for snippet in section.code_snippets:
                html += self._render_code_snippet(snippet)

        # Subsections
        for subsection in section.subsections:
            html += self._generate_section(subsection, report_config)

        html += """
            </div>
        </section>
        """

        return html

    def _render_content(self, content: Any) -> str:
        """Render section content.

        Args:
            content: Content to render

        Returns:
            str: Rendered HTML
        """
        if isinstance(content, list):
            # Process list items - handle markdown-like content
            html_parts = []
            in_list = False

            for item in content:
                item_str = str(item)

                # Handle empty lines
                if not item_str.strip():
                    if in_list:
                        html_parts.append("</ul>")
                        in_list = False
                    html_parts.append("<br>")
                    continue

                # Handle headers (markdown style)
                if item_str.startswith("###"):
                    if in_list:
                        html_parts.append("</ul>")
                        in_list = False
                    html_parts.append(f"<h3>{item_str[3:].strip()}</h3>")
                elif item_str.startswith("##"):
                    if in_list:
                        html_parts.append("</ul>")
                        in_list = False
                    html_parts.append(f"<h2>{item_str[2:].strip()}</h2>")
                elif item_str.startswith("#"):
                    if in_list:
                        html_parts.append("</ul>")
                        in_list = False
                    html_parts.append(f"<h1>{item_str[1:].strip()}</h1>")
                # Handle list items
                elif item_str.startswith("- ") or item_str.startswith("* "):
                    if not in_list:
                        html_parts.append("<ul>")
                        in_list = True
                    # Process markdown bold
                    processed = self._process_markdown(item_str[2:])
                    html_parts.append(f"<li>{processed}</li>")
                # Handle indented content (continuation of previous item)
                elif item_str.startswith("   "):
                    processed = self._process_markdown(item_str.strip())
                    html_parts.append(
                        f"<div style='margin-left: 20px; color: var(--text-secondary);'>{processed}</div>"
                    )
                else:
                    if in_list:
                        html_parts.append("</ul>")
                        in_list = False
                    # Regular paragraph
                    processed = self._process_markdown(item_str)
                    html_parts.append(f"<p>{processed}</p>")

            if in_list:
                html_parts.append("</ul>")

            return "\n".join(html_parts)
        elif isinstance(content, dict):
            return self._render_metrics(content)
        else:
            return f"<p>{self._process_markdown(str(content))}</p>"

    def _process_markdown(self, text: str) -> str:
        """Process basic markdown formatting.

        Args:
            text: Text with markdown formatting

        Returns:
            str: HTML formatted text
        """
        # Escape HTML first
        text = self._escape_html(text)

        # Bold text
        import re

        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)

        # Italic text
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        text = re.sub(r"_(.+?)_", r"<em>\1</em>", text)

        # Inline code
        text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)

        return text

    def _render_table(self, table_data: Dict[str, Any]) -> str:
        """Render a table.

        Args:
            table_data: Table data

        Returns:
            str: Table HTML
        """
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not headers or not rows:
            return ""

        header_html = "".join(f"<th>{h}</th>" for h in headers)

        rows_html = []
        for row in rows:
            cells = "".join(f"<td>{self._format_cell(cell)}</td>" for cell in row)
            rows_html.append(f"<tr>{cells}</tr>")

        return f"""
        <div class="table-wrapper">
            <table>
                <thead>
                    <tr>{header_html}</tr>
                </thead>
                <tbody>
                    {"".join(rows_html)}
                </tbody>
            </table>
        </div>
        """

    def _render_chart(self, chart_data: Dict[str, Any]) -> str:
        """Render a chart.

        Args:
            chart_data: Chart configuration and data

        Returns:
            str: Chart HTML
        """
        chart_id = f"chart-{id(chart_data)}"
        chart_type = chart_data.get("type", "bar")
        data = chart_data.get("data", {})
        title = data.get("title", "")

        return f"""
        <div class="chart-container">
            {f'<div class="chart-title">{title}</div>' if title else ""}
            <canvas id="{chart_id}"></canvas>
        </div>
        """

    def _render_code_snippet(self, snippet: Dict[str, Any]) -> str:
        """Render a code snippet.

        Args:
            snippet: Code snippet data

        Returns:
            str: Code snippet HTML
        """
        language = snippet.get("language", "text")
        code = snippet.get("code", "")
        highlight_lines = snippet.get("highlight_lines", [])

        # Split code into lines
        lines = code.split("\n")

        # Render with line numbers
        rendered_lines = []
        for i, line in enumerate(lines, 1):
            line_class = "highlight" if i in highlight_lines else ""
            rendered_lines.append(
                f'<span class="{line_class}">'
                f'<span class="line-number">{i}</span>'
                f"{self._escape_html(line)}"
                f"</span>"
            )

        return f"""
        <div class="code-snippet" data-language="{language}">
            {"".join(rendered_lines)}
        </div>
        """

    def _render_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render metrics as cards.

        Args:
            metrics: Metrics data

        Returns:
            str: Metrics HTML
        """
        # Define tooltips for common metrics
        metric_tooltips = {
            "Total Hotspots": "Files with high risk scores based on change frequency, complexity, and size",
            "Critical": "Files requiring immediate attention (risk score > 80)",
            "High Risk": "Files that should be addressed soon (risk score 60-80)",
            "Files Analyzed": "Total number of files examined for potential issues",
            "Average Complexity": "Mean cyclomatic complexity across all functions",
            "Maximum Complexity": "Highest cyclomatic complexity found in any function",
            "Complex Functions": "Number of functions with complexity > 10",
            "Total Functions": "Total number of functions analyzed",
            "Health Score": "Overall codebase health rating (0-100)",
            "Test Coverage": "Percentage of code covered by tests",
            "Excluded Files": "Files ignored during analysis",
            "Ignored Patterns": "File patterns excluded from analysis",
            "Bus Factor": "Minimum number of contributors who could derail the project if unavailable",
        }
        
        cards = []

        for key, value in metrics.items():
            tooltip = metric_tooltips.get(key, "")
            tooltip_attr = f'data-tooltip="{tooltip}"' if tooltip else ""
            
            cards.append(
                f"""
            <div class="metric-card" {tooltip_attr}>
                <div class="metric-label">{key}</div>
                <div class="metric-value">{value}</div>
            </div>
            """
            )

        return f'<div class="metrics-grid">{"".join(cards)}</div>'

    def _generate_footer(self, report_config: ReportConfig) -> str:
        """Generate report footer.

        Args:
            report_config: Report configuration

        Returns:
            str: Footer HTML
        """
        return f"""
    <footer class="footer">
        <p>{report_config.footer_text}</p>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </footer>
    """

    def _generate_chart_scripts(self, sections: List[ReportSection]) -> str:
        """Generate Chart.js initialization scripts.

        Args:
            sections: Report sections

        Returns:
            str: JavaScript for charts
        """
        scripts = ["<script>"]
        scripts.append('document.addEventListener("DOMContentLoaded", function() {')

        # Collect all charts from sections
        all_charts = []
        for section in sections:
            all_charts.extend(section.charts)
            for subsection in section.subsections:
                all_charts.extend(subsection.charts)

        # Generate chart initialization
        for chart in all_charts:
            chart_id = f"chart-{id(chart)}"
            chart_type = chart.get("type", "bar")
            data = chart.get("data", {})

            config = self._generate_chart_config(chart_type, data)

            scripts.append(
                f"""
    var ctx_{chart_id} = document.getElementById('{chart_id}');
    if (ctx_{chart_id}) {{
        new Chart(ctx_{chart_id}.getContext('2d'), {json.dumps(config)});
    }}
            """
            )

        # Add collapsible section handlers
        scripts.append(
            """
    // Collapsible sections
    document.querySelectorAll('.collapsible').forEach(function(element) {
        element.addEventListener('click', function() {
            this.classList.toggle('collapsed');
            var content = this.nextElementSibling;
            if (content) {
                content.classList.toggle('collapsed');
            }
        });
    });

    // Active navigation
    document.querySelectorAll('.nav a').forEach(function(link) {
        link.addEventListener('click', function(e) {
            document.querySelectorAll('.nav a').forEach(function(a) {
                a.classList.remove('active');
            });
            this.classList.add('active');
        });
    });
        """
        )

        scripts.append("});")
        scripts.append("</script>")

        return "\n".join(scripts)

    def _generate_chart_config(self, chart_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Chart.js configuration.

        Args:
            chart_type: Type of chart
            data: Chart data

        Returns:
            Dict[str, Any]: Chart.js configuration
        """
        labels = data.get("labels", [])
        values = data.get("values", [])
        title = data.get("title", "")
        colors = data.get("colors")

        # Default colors
        if not colors:
            colors = [
                "#2563eb",
                "#8b5cf6",
                "#10b981",
                "#f59e0b",
                "#ef4444",
                "#06b6d4",
                "#ec4899",
                "#84cc16",
            ]

        config = {
            "type": chart_type,
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": title,
                        "data": values,
                        "backgroundColor": (
                            colors if chart_type in ["pie", "doughnut"] else colors[0]
                        ),
                        "borderColor": colors if chart_type in ["pie", "doughnut"] else colors[0],
                        "borderWidth": 1,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {"display": chart_type in ["pie", "doughnut"]},
                    "title": {"display": bool(title), "text": title},
                },
            },
        }

        # Customize for specific chart types
        if chart_type == "line":
            config["data"]["datasets"][0]["fill"] = False
            config["data"]["datasets"][0]["tension"] = 0.1
        elif chart_type == "gauge":
            # Convert gauge to doughnut
            config["type"] = "doughnut"
            config["data"]["datasets"][0]["data"] = [
                data.get("value", 0),
                data.get("max", 100) - data.get("value", 0),
            ]
            config["data"]["labels"] = ["Score", "Remaining"]
            config["options"]["circumference"] = 180
            config["options"]["rotation"] = 270

        return config

    def _format_cell(self, cell: Any) -> str:
        """Format table cell value.

        Args:
            cell: Cell value

        Returns:
            str: Formatted cell HTML
        """
        if isinstance(cell, bool):
            return "✓" if cell else "✗"
        elif isinstance(cell, (int, float)):
            if isinstance(cell, float):
                return f"{cell:.2f}"
            return str(cell)
        elif cell is None:
            return "-"
        else:
            cell_str = str(cell)

            # Check for severity badges
            if cell_str.lower() in ["critical", "high", "medium", "low", "info"]:
                return f'<span class="badge badge-{cell_str.lower()}">{cell_str}</span>'

            return self._escape_html(cell_str)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Text to escape

        Returns:
            str: Escaped text
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _load_custom_css(self, css_path: Optional[Path]) -> Optional[str]:
        """Load custom CSS from file.

        Args:
            css_path: Path to CSS file

        Returns:
            Optional[str]: CSS content or None
        """
        if css_path and css_path.exists():
            try:
                return css_path.read_text()
            except Exception as e:
                self.logger.warning(f"Failed to load custom CSS: {e}")
        return None

    def _encode_image(self, image_path: Path) -> str:
        """Encode image as base64.

        Args:
            image_path: Path to image file

        Returns:
            str: Base64 encoded image
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.logger.warning(f"Failed to encode image: {e}")
            return ""


def create_html_report(
    sections: List[ReportSection],
    output_path: Path,
    title: str = "Code Analysis Report",
    config: Optional[TenetsConfig] = None,
) -> Path:
    """Convenience function to create HTML report.

    Args:
        sections: Report sections
        output_path: Output path
        title: Report title
        config: Optional configuration

    Returns:
        Path: Path to generated report
    """
    if config is None:
        config = TenetsConfig()

    reporter = HTMLReporter(config)
    report_config = ReportConfig(title=title, format="html")
    metadata = {"title": title, "generated_at": datetime.now().isoformat()}

    return reporter.generate(sections, metadata, output_path, report_config)


def create_dashboard(
    analysis_results: Dict[str, Any], output_path: Path, config: Optional[TenetsConfig] = None
) -> Path:
    """Create an interactive dashboard.

    Args:
        analysis_results: Analysis results
        output_path: Output path
        config: Optional configuration

    Returns:
        Path: Path to dashboard
    """
    # Dashboard is a specialized HTML report
    if config is None:
        config = TenetsConfig()

    # Use module-level ReportGenerator symbol so tests can patch it via this module
    generator = ReportGenerator(config) if ReportGenerator is not None else None
    if generator is None:
        # Fallback import if re-export failed for any reason
        from .generator import ReportGenerator as _RG  # type: ignore

        generator = _RG(config)
    report_config = ReportConfig(
        title="Code Analysis Dashboard",
        format="html",
        include_charts=True,
        include_toc=False,
    )

    return generator.generate(analysis_results, output_path, report_config)
