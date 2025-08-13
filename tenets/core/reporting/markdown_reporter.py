"""Markdown report generator module.

This module provides Markdown report generation functionality for creating
plain text reports that can be viewed in any text editor, converted to
other formats, or integrated with documentation systems.

The Markdown reporter generates clean, readable reports with support for
tables, code blocks, and structured content.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenets.config import TenetsConfig
from tenets.utils.logger import get_logger

from .generator import ReportConfig, ReportSection


class MarkdownReporter:
    """Markdown report generator.

    Generates Markdown-formatted reports from analysis results, suitable
    for documentation, GitHub, and other Markdown-supporting platforms.

    Attributes:
        config: Configuration object
        logger: Logger instance
        toc_entries: Table of contents entries
    """

    def __init__(self, config: TenetsConfig):
        """Initialize Markdown reporter.

        Args:
            config: TenetsConfig instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.toc_entries: List[str] = []

    def generate(
        self,
        sections: List[ReportSection],
        metadata: Dict[str, Any],
        output_path: Path,
        report_config: ReportConfig,
    ) -> Path:
        """Generate Markdown report.

        Args:
            sections: Report sections
            metadata: Report metadata
            output_path: Output file path
            report_config: Report configuration

        Returns:
            Path: Path to generated report

        Example:
            >>> reporter = MarkdownReporter(config)
            >>> report_path = reporter.generate(
            ...     sections,
            ...     metadata,
            ...     Path("report.md")
            ... )
        """
        self.logger.debug(f"Generating Markdown report to {output_path}")

        # Reset TOC entries
        self.toc_entries = []

        # Generate Markdown content
        markdown_content = self._generate_markdown(sections, metadata, report_config)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        self.logger.info(f"Markdown report generated: {output_path}")
        return output_path

    def _generate_markdown(
        self, sections: List[ReportSection], metadata: Dict[str, Any], report_config: ReportConfig
    ) -> str:
        """Generate complete Markdown content.

        Args:
            sections: Report sections
            metadata: Report metadata
            report_config: Report configuration

        Returns:
            str: Complete Markdown content
        """
        parts = []

        # Header
        parts.append(self._generate_header(metadata, report_config))

        # Executive Summary
        if report_config.include_summary:
            parts.append(self._generate_summary(metadata))

        # Table of Contents placeholder
        toc_placeholder = "<!-- TOC -->"
        if report_config.include_toc:
            parts.append(toc_placeholder)

        # Sections
        for section in sections:
            if section.visible:
                parts.append(self._generate_section(section, report_config))

        # Footer
        parts.append(self._generate_footer(report_config))

        # Combine parts
        markdown = "\n\n".join(parts)

        # Replace TOC placeholder with actual TOC
        if report_config.include_toc and self.toc_entries:
            toc = self._generate_toc()
            markdown = markdown.replace(toc_placeholder, toc)

        return markdown

    def _generate_header(self, metadata: Dict[str, Any], report_config: ReportConfig) -> str:
        """Generate report header.

        Args:
            metadata: Report metadata
            report_config: Report configuration

        Returns:
            str: Header markdown
        """
        lines = []

        # Title
        lines.append(f"# {report_config.title}")
        lines.append("")

        # Metadata table
        lines.append("## Report Information")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Generated | {metadata.get('generated_at', 'Unknown')} |")
        lines.append(f"| Generator Version | {metadata.get('generator_version', 'Unknown')} |")

        summary = metadata.get("analysis_summary", {})
        if summary:
            lines.append(f"| Total Files | {summary.get('total_files', 0)} |")
            lines.append(f"| Total Lines | {summary.get('total_lines', 0):,} |")
            lines.append(f"| Health Score | {summary.get('health_score', 0):.1f}/100 |")
            lines.append(f"| Critical Issues | {summary.get('critical_issues', 0)} |")

            languages = summary.get("languages", [])
            if languages:
                lines.append(f"| Languages | {', '.join(languages)} |")

        return "\n".join(lines)

    def _generate_summary(self, metadata: Dict[str, Any]) -> str:
        """Generate executive summary.

        Args:
            metadata: Report metadata

        Returns:
            str: Summary markdown
        """
        lines = []
        lines.append("## Executive Summary")
        lines.append("")

        summary = metadata.get("analysis_summary", {})
        health_score = summary.get("health_score", 0)

        # Health status
        if health_score >= 80:
            status = "**excellent** condition ✅"
        elif health_score >= 60:
            status = "**good** condition ✔️"
        elif health_score >= 40:
            status = "**fair** condition ⚠️"
        else:
            status = "**needs improvement** ❌"

        lines.append(
            f"The codebase is in {status} with a health score of **{health_score:.1f}/100**."
        )
        lines.append("")

        # Key findings
        lines.append("### Key Findings")
        lines.append("")

        critical_issues = summary.get("critical_issues", 0)
        if critical_issues > 0:
            lines.append(f"- 🚨 **{critical_issues} critical issues** require immediate attention")

        total_issues = metadata.get("total_issues", 0)
        if total_issues > 0:
            lines.append(f"- ⚠️ Total of **{total_issues} issues** identified across all categories")

        lines.append(
            f"- 📊 Analyzed **{summary.get('total_files', 0)} files** with **{summary.get('total_lines', 0):,} lines** of code"
        )

        return "\n".join(lines)

    def _generate_section(
        self, section: ReportSection, report_config: ReportConfig, parent_level: int = 0
    ) -> str:
        """Generate a single section.

        Args:
            section: Report section
            report_config: Report configuration
            parent_level: Parent section level for nesting

        Returns:
            str: Section markdown
        """
        lines = []

        # Section heading
        level = section.level + parent_level
        heading = "#" * min(level, 6)  # Max heading level is 6
        icon = f"{section.icon} " if section.icon else ""
        lines.append(f"{heading} {icon}{section.title}")

        # Add to TOC
        if level <= 3:  # Only include up to h3 in TOC
            indent = "  " * (level - 1)
            anchor = self._create_anchor(section.title)
            self.toc_entries.append(f"{indent}- [{section.title}](#{anchor})")

        lines.append("")

        # Section content
        if section.content:
            lines.append(self._render_content(section.content))
            lines.append("")

        # Tables
        for table in section.tables:
            lines.append(self._render_table(table))
            lines.append("")

        # Charts (as text representation or skip)
        if report_config.include_charts:
            for chart in section.charts:
                lines.append(self._render_chart(chart))
                lines.append("")

        # Code snippets
        if report_config.include_code_snippets:
            for snippet in section.code_snippets:
                lines.append(self._render_code_snippet(snippet))
                lines.append("")

        # Subsections
        for subsection in section.subsections:
            lines.append(self._generate_section(subsection, report_config, level))

        return "\n".join(lines)

    def _render_content(self, content: Any) -> str:
        """Render section content.

        Args:
            content: Content to render

        Returns:
            str: Rendered markdown
        """
        if isinstance(content, list):
            # Render as bullet list
            items = []
            for item in content:
                items.append(f"- {item}")
            return "\n".join(items)
        elif isinstance(content, dict):
            # Render as definition list or table
            return self._render_metrics(content)
        else:
            # Render as paragraph
            return str(content)

    def _render_table(self, table_data: Dict[str, Any]) -> str:
        """Render a table in Markdown.

        Args:
            table_data: Table data

        Returns:
            str: Table markdown
        """
        headers = table_data.get("headers", [])
        rows = table_data.get("rows", [])

        if not headers or not rows:
            return ""

        lines = []

        # Headers
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("|" + "|".join("-" * (len(str(h)) + 2) for h in headers) + "|")

        # Rows
        for row in rows[:100]:  # Limit rows for readability
            formatted_cells = []
            for cell in row:
                formatted_cells.append(self._format_cell(cell))
            lines.append("| " + " | ".join(formatted_cells) + " |")

        if len(rows) > 100:
            lines.append(f"*... and {len(rows) - 100} more rows*")

        return "\n".join(lines)

    def _render_chart(self, chart_data: Dict[str, Any]) -> str:
        """Render a chart as text representation.

        Args:
            chart_data: Chart configuration and data

        Returns:
            str: Chart markdown representation
        """
        chart_type = chart_data.get("type", "bar")
        data = chart_data.get("data", {})
        title = data.get("title", "Chart")

        lines = []
        lines.append(f"### 📊 {title}")
        lines.append("")

        if chart_type in ["bar", "line"]:
            # Simple ASCII bar chart
            labels = data.get("labels", [])
            values = data.get("values", [])

            if labels and values:
                max_value = max(values) if values else 1
                max_label_len = max(len(str(label)) for label in labels)

                for label, value in zip(labels, values):
                    bar_width = int((value / max_value) * 40) if max_value > 0 else 0
                    bar = "█" * bar_width
                    lines.append(f"{str(label).ljust(max_label_len)} | {bar} {value}")

        elif chart_type == "pie":
            # Text representation of pie chart
            labels = data.get("labels", [])
            values = data.get("values", [])

            if labels and values:
                total = sum(values)
                for label, value in zip(labels, values):
                    percentage = (value / total * 100) if total > 0 else 0
                    lines.append(f"- {label}: {value} ({percentage:.1f}%)")

        return "\n".join(lines)

    def _render_code_snippet(self, snippet: Dict[str, Any]) -> str:
        """Render a code snippet.

        Args:
            snippet: Code snippet data

        Returns:
            str: Code snippet markdown
        """
        language = snippet.get("language", "")
        code = snippet.get("code", "")
        highlight_lines = snippet.get("highlight_lines", [])
        filename = snippet.get("filename", "")

        lines = []

        if filename:
            lines.append(f"**File:** `{filename}`")
            lines.append("")

        # Code block with language hint
        lines.append(f"```{language}")

        # Add code with optional line highlighting (using comments)
        code_lines = code.split("\n")
        for i, line in enumerate(code_lines, 1):
            if i in highlight_lines:
                lines.append(f"{line}  # <-- ATTENTION")
            else:
                lines.append(line)

        lines.append("```")

        return "\n".join(lines)

    def _render_metrics(self, metrics: Dict[str, Any]) -> str:
        """Render metrics as a formatted list or table.

        Args:
            metrics: Metrics data

        Returns:
            str: Metrics markdown
        """
        if not metrics:
            return ""

        lines = []

        # Determine if we should use a table or list
        if len(metrics) > 5:
            # Use table for many metrics
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, value in metrics.items():
                lines.append(f"| {key} | {self._format_value(value)} |")
        else:
            # Use list for few metrics
            for key, value in metrics.items():
                lines.append(f"- **{key}:** {self._format_value(value)}")

        return "\n".join(lines)

    def _generate_toc(self) -> str:
        """Generate table of contents.

        Returns:
            str: TOC markdown
        """
        if not self.toc_entries:
            return ""

        lines = []
        lines.append("## Table of Contents")
        lines.append("")
        lines.extend(self.toc_entries)

        return "\n".join(lines)

    def _generate_footer(self, report_config: ReportConfig) -> str:
        """Generate report footer.

        Args:
            report_config: Report configuration

        Returns:
            str: Footer markdown
        """
        lines = []
        lines.append("---")
        lines.append("")
        lines.append(f"*{report_config.footer_text}*")
        lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def _format_cell(self, cell: Any) -> str:
        """Format table cell value for Markdown.

        Args:
            cell: Cell value

        Returns:
            str: Formatted cell text
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
            # Escape pipe characters in cell content
            cell_str = str(cell).replace("|", "\\|")

            # Add formatting for special values
            if cell_str.lower() in ["critical", "high"]:
                return f"**{cell_str}**"
            elif cell_str.lower() in ["medium", "warning"]:
                return f"*{cell_str}*"

            return cell_str

    def _format_value(self, value: Any) -> str:
        """Format a metric value.

        Args:
            value: Value to format

        Returns:
            str: Formatted value
        """
        if isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, float):
            # Format floats with appropriate precision
            if value < 0.01:
                return f"{value:.4f}"
            elif value < 1:
                return f"{value:.3f}"
            elif value < 100:
                return f"{value:.2f}"
            else:
                return f"{value:.1f}"
        elif isinstance(value, int):
            # Add thousands separator for large numbers
            if value >= 1000:
                return f"{value:,}"
            return str(value)
        elif isinstance(value, list):
            return ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            # Format nested dict as sub-list
            items = []
            for k, v in value.items():
                items.append(f"{k}: {self._format_value(v)}")
            return " | ".join(items)
        else:
            return str(value)

    def _create_anchor(self, text: str) -> str:
        """Create anchor link from heading text.

        Args:
            text: Heading text

        Returns:
            str: Anchor-safe text
        """
        # Remove emoji and special characters
        import re

        text = re.sub(r"[^\w\s-]", "", text)
        # Replace spaces with hyphens
        text = re.sub(r"\s+", "-", text)
        # Convert to lowercase
        return text.lower()


def create_markdown_report(
    sections: List[ReportSection],
    output_path: Path,
    title: str = "Code Analysis Report",
    config: Optional[TenetsConfig] = None,
) -> Path:
    """Convenience function to create Markdown report.

    Args:
        sections: Report sections
        output_path: Output path
        title: Report title
        config: Optional configuration

    Returns:
        Path: Path to generated report

    Example:
        >>> from tenets.core.reporting.markdown_reporter import create_markdown_report
        >>> report_path = create_markdown_report(
        ...     sections,
        ...     Path("report.md"),
        ...     title="Analysis Report"
        ... )
    """
    if config is None:
        config = TenetsConfig()

    reporter = MarkdownReporter(config)
    report_config = ReportConfig(title=title, format="markdown")
    metadata = {"title": title, "generated_at": datetime.now().isoformat()}

    return reporter.generate(sections, metadata, output_path, report_config)


def format_markdown_table(
    headers: List[str], rows: List[List[Any]], alignment: Optional[List[str]] = None
) -> str:
    """Format data as a Markdown table.

    Args:
        headers: Table headers
        rows: Table rows
        alignment: Column alignment (left, right, center)

    Returns:
        str: Formatted Markdown table

    Example:
        >>> from tenets.core.reporting.markdown_reporter import format_markdown_table
        >>> table = format_markdown_table(
        ...     ["Name", "Value", "Status"],
        ...     [["Test", 42, "Pass"], ["Demo", 17, "Fail"]]
        ... )
        >>> print(table)
    """
    if not headers or not rows:
        return ""

    lines = []

    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Format headers
    header_parts = []
    for i, header in enumerate(headers):
        header_parts.append(str(header).ljust(widths[i]))
    lines.append("| " + " | ".join(header_parts) + " |")

    # Format separator with alignment
    separator_parts = []
    for i, width in enumerate(widths):
        sep = "-" * width
        if alignment and i < len(alignment):
            align = alignment[i].lower()
            if align == "center":
                sep = ":" + sep[1:-1] + ":"
            elif align == "right":
                sep = sep[:-1] + ":"
            elif align == "left":
                sep = ":" + sep[1:]
        separator_parts.append(sep)
    lines.append("|" + "|".join(separator_parts) + "|")

    # Format rows
    for row in rows:
        row_parts = []
        for i, cell in enumerate(row):
            if i < len(widths):
                cell_str = str(cell)
                if alignment and i < len(alignment):
                    align = alignment[i].lower()
                    if align == "right":
                        row_parts.append(cell_str.rjust(widths[i]))
                    elif align == "center":
                        row_parts.append(cell_str.center(widths[i]))
                    else:
                        row_parts.append(cell_str.ljust(widths[i]))
                else:
                    row_parts.append(cell_str.ljust(widths[i]))
        lines.append("| " + " | ".join(row_parts) + " |")

    return "\n".join(lines)


def create_markdown_summary(analysis_results: Dict[str, Any], max_length: int = 1000) -> str:
    """Create a Markdown summary of analysis results.

    Args:
        analysis_results: Analysis results
        max_length: Maximum length in characters

    Returns:
        str: Markdown summary

    Example:
        >>> from tenets.core.reporting.markdown_reporter import create_markdown_summary
        >>> summary = create_markdown_summary(analysis_results)
        >>> print(summary)
    """
    lines = []

    # Title
    lines.append("# Code Analysis Summary")
    lines.append("")

    # Quick stats
    lines.append("## Quick Stats")
    lines.append("")

    if "overview" in analysis_results:
        overview = analysis_results["overview"]
        lines.append(f"- **Files:** {overview.get('total_files', 0)}")
        lines.append(f"- **Lines:** {overview.get('total_lines', 0):,}")
        lines.append(f"- **Health Score:** {overview.get('health_score', 0):.1f}/100")

    lines.append("")

    # Top issues
    if "issues" in analysis_results:
        issues = analysis_results["issues"]
        critical = sum(1 for i in issues if i.get("severity") == "critical")
        high = sum(1 for i in issues if i.get("severity") == "high")

        if critical > 0 or high > 0:
            lines.append("## Critical Issues")
            lines.append("")
            if critical > 0:
                lines.append(f"- 🚨 **{critical} critical issues** found")
            if high > 0:
                lines.append(f"- ⚠️ **{high} high priority issues** found")
            lines.append("")

    # Top recommendations
    if "recommendations" in analysis_results:
        lines.append("## Top Recommendations")
        lines.append("")

        for i, rec in enumerate(analysis_results["recommendations"][:3], 1):
            if isinstance(rec, dict):
                lines.append(f"{i}. {rec.get('action', rec)}")
            else:
                lines.append(f"{i}. {rec}")
        lines.append("")

    # Truncate if needed
    result = "\n".join(lines)
    if len(result) > max_length:
        result = result[: max_length - 3] + "..."

    return result
