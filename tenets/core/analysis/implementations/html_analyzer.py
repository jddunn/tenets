"""HTML code analyzer with modern web framework support.

This module provides comprehensive analysis for HTML files,
including support for HTML5, accessibility features,
web components, and modern framework patterns.
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tenets.models.analysis import (
    ClassInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ImportInfo,
)
from tenets.utils.logger import get_logger

from ..base import LanguageAnalyzer


class HTMLStructureParser(HTMLParser):
    """Custom HTML parser to extract structure information."""

    def __init__(self):
        super().__init__()
        self.elements = []
        self.current_depth = 0
        self.max_depth = 0
        self.element_stack = []
        self.scripts = []
        self.styles = []
        self.links = []
        self.meta_tags = []
        self.forms = []
        self.current_form = None

    def handle_starttag(self, tag, attrs):
        """Handle opening tags."""
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.element_stack.append(tag)

        attr_dict = dict(attrs)
        element_info = {
            "tag": tag,
            "attrs": attr_dict,
            "depth": self.current_depth,
            "line": self.getpos()[0],
        }
        self.elements.append(element_info)

        # Track specific elements
        if tag == "script":
            self.scripts.append(attr_dict)
        elif tag == "style":
            self.styles.append(attr_dict)
        elif tag == "link":
            self.links.append(attr_dict)
        elif tag == "meta":
            self.meta_tags.append(attr_dict)
        elif tag == "form":
            self.current_form = {
                "attrs": attr_dict,
                "inputs": [],
                "line": self.getpos()[0],
            }
            self.forms.append(self.current_form)
        elif tag in ["input", "textarea", "select", "button"] and self.current_form:
            self.current_form["inputs"].append(
                {
                    "tag": tag,
                    "attrs": attr_dict,
                }
            )

    def handle_endtag(self, tag):
        """Handle closing tags."""
        if self.element_stack and self.element_stack[-1] == tag:
            self.element_stack.pop()
            self.current_depth -= 1
        if tag == "form":
            self.current_form = None

    def handle_data(self, data):
        """Handle text content."""
        pass  # We're mainly interested in structure


class HTMLAnalyzer(LanguageAnalyzer):
    """HTML code analyzer with modern web framework support.

    Provides comprehensive analysis for HTML files including:
    - HTML5 semantic elements
    - CSS and JavaScript imports
    - Meta tags and SEO elements
    - Forms and input validation
    - Accessibility features (ARIA, alt text, etc.)
    - Web components and custom elements
    - Framework-specific patterns (React, Vue, Angular)
    - Microdata and structured data
    - DOM complexity and nesting depth
    - Performance hints (lazy loading, async/defer scripts)
    - Security considerations (CSP, integrity checks)

    Supports HTML5 and modern web development practices.
    """

    language_name = "html"
    file_extensions = [".html", ".htm", ".xhtml", ".vue", ".jsx", ".tsx"]

    def __init__(self):
        """Initialize the HTML analyzer with logger."""
        self.logger = get_logger(__name__)

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract external resource imports from HTML.

        Handles:
        - <link> tags for CSS
        - <script> tags for JavaScript
        - <import> tags for HTML imports
        - ES6 module imports in scripts
        - Framework-specific imports

        Args:
            content: HTML source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []

        # Extract CSS links
        css_pattern = r'<link[^>]+rel=["\']stylesheet["\'][^>]*>'
        for match in re.finditer(css_pattern, content, re.IGNORECASE):
            link_tag = match.group(0)
            href_match = re.search(r'href=["\']([^"\']+)["\']', link_tag)
            if href_match:
                href = href_match.group(1)
                imports.append(
                    ImportInfo(
                        module=href,
                        line=content[: match.start()].count("\n") + 1,
                        type="stylesheet",
                        is_relative=not href.startswith(("http://", "https://", "//")),
                        category=self._categorize_css_import(href),
                        integrity=self._extract_integrity(link_tag),
                        crossorigin=self._extract_crossorigin(link_tag),
                    )
                )

        # Extract JavaScript scripts
        script_pattern = r'<script[^>]*src=["\']([^"\']+)["\'][^>]*>'
        for match in re.finditer(script_pattern, content, re.IGNORECASE):
            script_tag = match.group(0)
            src = match.group(1)

            imports.append(
                ImportInfo(
                    module=src,
                    line=content[: match.start()].count("\n") + 1,
                    type="script",
                    is_relative=not src.startswith(("http://", "https://", "//")),
                    category=self._categorize_js_import(src),
                    is_async="async" in script_tag.lower(),
                    is_defer="defer" in script_tag.lower(),
                    is_module='type="module"' in script_tag or "type='module'" in script_tag,
                    integrity=self._extract_integrity(script_tag),
                    crossorigin=self._extract_crossorigin(script_tag),
                )
            )

        # Extract ES6 imports from inline scripts
        module_script_pattern = r'<script[^>]*type=["\']module["\'][^>]*>(.*?)</script>'
        for match in re.finditer(module_script_pattern, content, re.IGNORECASE | re.DOTALL):
            script_content = match.group(1)
            es6_import_pattern = (
                r'import\s+(?:{[^}]+}|[\w*]+(?:\s+as\s+\w+)?)\s+from\s+["\']([^"\']+)["\']'
            )
            for import_match in re.finditer(es6_import_pattern, script_content):
                module_path = import_match.group(1)
                imports.append(
                    ImportInfo(
                        module=module_path,
                        line=content[: match.start()].count("\n")
                        + content[: import_match.start()].count("\n")
                        + 1,
                        type="es6_module",
                        is_relative=module_path.startswith(("./", "../")),
                        category=self._categorize_js_import(module_path),
                    )
                )

        # Extract HTML imports (deprecated but still found)
        html_import_pattern = r'<link[^>]+rel=["\']import["\'][^>]*>'
        for match in re.finditer(html_import_pattern, content, re.IGNORECASE):
            link_tag = match.group(0)
            href_match = re.search(r'href=["\']([^"\']+)["\']', link_tag)
            if href_match:
                imports.append(
                    ImportInfo(
                        module=href_match.group(1),
                        line=content[: match.start()].count("\n") + 1,
                        type="html_import",
                        is_relative=not href_match.group(1).startswith(
                            ("http://", "https://", "//")
                        ),
                    )
                )

        # Extract preload/prefetch resources
        preload_pattern = (
            r'<link[^>]+rel=["\'](?:preload|prefetch|preconnect|dns-prefetch)["\'][^>]*>'
        )
        for match in re.finditer(preload_pattern, content, re.IGNORECASE):
            link_tag = match.group(0)
            rel_match = re.search(r'rel=["\']([^"\']+)["\']', link_tag)
            href_match = re.search(r'href=["\']([^"\']+)["\']', link_tag)
            if rel_match and href_match:
                imports.append(
                    ImportInfo(
                        module=href_match.group(1),
                        line=content[: match.start()].count("\n") + 1,
                        type=rel_match.group(1),
                        is_relative=not href_match.group(1).startswith(
                            ("http://", "https://", "//")
                        ),
                        as_type=self._extract_as_type(link_tag),
                    )
                )

        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported elements from HTML.

        In HTML context, exports are:
        - Custom elements/web components
        - Globally accessible IDs
        - Data attributes for JavaScript hooks
        - Microdata/structured data
        - Open Graph and social media tags

        Args:
            content: HTML source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported elements
        """
        exports = []

        # Extract elements with IDs (globally accessible)
        id_pattern = r'<([^>]+)\s+id=["\']([^"\']+)["\']'
        for match in re.finditer(id_pattern, content, re.IGNORECASE):
            tag_content = match.group(1)
            element_id = match.group(2)
            tag_name = tag_content.split()[0] if tag_content else "unknown"

            exports.append(
                {
                    "name": element_id,
                    "type": "element_id",
                    "tag": tag_name,
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Extract custom elements (web components)
        custom_element_pattern = r"<([a-z]+-[a-z-]+)[^>]*>"
        for match in re.finditer(custom_element_pattern, content, re.IGNORECASE):
            tag_name = match.group(1)

            # Skip common non-custom elements
            if not tag_name.startswith(("ng-", "v-", "x-")):  # Common framework prefixes
                exports.append(
                    {
                        "name": tag_name,
                        "type": "custom_element",
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

        # Extract data attributes (commonly used as JS hooks)
        data_attr_pattern = r'data-([a-z-]+)=["\']([^"\']+)["\']'
        data_attrs = {}
        for match in re.finditer(data_attr_pattern, content, re.IGNORECASE):
            attr_name = f"data-{match.group(1)}"
            if attr_name not in data_attrs:
                data_attrs[attr_name] = {
                    "name": attr_name,
                    "type": "data_attribute",
                    "line": content[: match.start()].count("\n") + 1,
                    "values": set(),
                }
            data_attrs[attr_name]["values"].add(match.group(2))

        for attr_info in data_attrs.values():
            attr_info["values"] = list(attr_info["values"])
            exports.append(attr_info)

        # Extract Open Graph tags
        og_pattern = r'<meta\s+property=["\']og:([^"\']+)["\'][^>]*content=["\']([^"\']+)["\']'
        for match in re.finditer(og_pattern, content, re.IGNORECASE):
            exports.append(
                {
                    "name": f"og:{match.group(1)}",
                    "type": "open_graph",
                    "content": match.group(2),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Extract structured data (JSON-LD)
        jsonld_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
        for match in re.finditer(jsonld_pattern, content, re.IGNORECASE | re.DOTALL):
            exports.append(
                {
                    "name": "structured_data",
                    "type": "json_ld",
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        # Extract microdata
        itemscope_pattern = r'<[^>]+itemscope[^>]*itemtype=["\']([^"\']+)["\']'
        for match in re.finditer(itemscope_pattern, content, re.IGNORECASE):
            exports.append(
                {
                    "name": match.group(1).split("/")[-1],
                    "type": "microdata",
                    "schema": match.group(1),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract HTML document structure.

        Extracts:
        - Semantic HTML5 elements
        - Forms and inputs
        - Navigation structures
        - ARIA landmarks
        - Framework components
        - DOM tree depth and complexity

        Args:
            content: HTML source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()

        # Parse HTML structure
        parser = HTMLStructureParser()
        try:
            parser.feed(content)
        except Exception as e:
            self.logger.warning(f"HTML parsing error: {e}")

        # Store parsed structure
        structure.elements = parser.elements
        structure.max_dom_depth = parser.max_depth
        structure.scripts = parser.scripts
        structure.styles = parser.styles
        structure.links = parser.links
        structure.meta_tags = parser.meta_tags
        structure.forms = parser.forms

        # Detect DOCTYPE
        structure.has_doctype = bool(re.search(r"<!DOCTYPE\s+html", content, re.IGNORECASE))
        structure.is_html5 = bool(re.search(r"<!DOCTYPE\s+html>", content, re.IGNORECASE))

        # Extract language
        lang_match = re.search(r'<html[^>]*lang=["\']([^"\']+)["\']', content, re.IGNORECASE)
        if lang_match:
            structure.language = lang_match.group(1)

        # Extract charset
        charset_match = re.search(r'<meta[^>]*charset=["\']([^"\']+)["\']', content, re.IGNORECASE)
        if charset_match:
            structure.charset = charset_match.group(1)

        # Extract viewport
        viewport_match = re.search(
            r'<meta[^>]*name=["\']viewport["\'][^>]*content=["\']([^"\']+)["\']',
            content,
            re.IGNORECASE,
        )
        if viewport_match:
            structure.viewport = viewport_match.group(1)
            structure.is_responsive = True

        # Count semantic HTML5 elements
        semantic_elements = [
            "header",
            "nav",
            "main",
            "article",
            "section",
            "aside",
            "footer",
            "figure",
            "figcaption",
            "mark",
            "time",
            "details",
            "summary",
        ]
        for element in semantic_elements:
            count = len(re.findall(f"<{element}[^>]*>", content, re.IGNORECASE))
            if count > 0:
                setattr(structure, f"{element}_count", count)

        # Count forms and inputs
        structure.form_count = len(re.findall(r"<form[^>]*>", content, re.IGNORECASE))
        # Inputs are considered as <input> and <textarea> elements for test expectations
        input_tags = len(re.findall(r"<input[^>]*>", content, re.IGNORECASE))
        textarea_tags = len(re.findall(r"<textarea[^>]*>", content, re.IGNORECASE))
        structure.input_count = input_tags + textarea_tags
        structure.button_count = len(re.findall(r"<button[^>]*>", content, re.IGNORECASE))

        # Count media elements
        structure.img_count = len(re.findall(r"<img[^>]*>", content, re.IGNORECASE))
        structure.video_count = len(re.findall(r"<video[^>]*>", content, re.IGNORECASE))
        structure.audio_count = len(re.findall(r"<audio[^>]*>", content, re.IGNORECASE))
        structure.canvas_count = len(re.findall(r"<canvas[^>]*>", content, re.IGNORECASE))
        structure.svg_count = len(re.findall(r"<svg[^>]*>", content, re.IGNORECASE))

        # Count interactive elements
        structure.link_count = len(re.findall(r"<a[^>]*>", content, re.IGNORECASE))
        structure.select_count = len(re.findall(r"<select[^>]*>", content, re.IGNORECASE))
        structure.textarea_count = len(re.findall(r"<textarea[^>]*>", content, re.IGNORECASE))

        # Detect frameworks
        structure.is_react = self._detect_react(content)
        structure.is_vue = self._detect_vue(content)
        structure.is_angular = self._detect_angular(content)
        structure.is_svelte = self._detect_svelte(content)

        # Count ARIA attributes
        structure.aria_labels = len(
            re.findall(r'aria-label=["\'][^"\']+["\']', content, re.IGNORECASE)
        )
        structure.aria_labelledby = len(
            re.findall(r'aria-labelledby=["\'][^"\']+["\']', content, re.IGNORECASE)
        )
        structure.aria_describedby = len(
            re.findall(r'aria-describedby=["\'][^"\']+["\']', content, re.IGNORECASE)
        )
        structure.aria_roles = len(re.findall(r'role=["\'][^"\']+["\']', content, re.IGNORECASE))
        structure.aria_live = len(
            re.findall(r'aria-live=["\'][^"\']+["\']', content, re.IGNORECASE)
        )

        # Count accessibility features
        # Only count non-empty alt attributes
        structure.alt_texts = len(
            re.findall(r'<img[^>]*alt=["\']\s*[^"\'\s][^"\']*["\'][^>]*>', content, re.IGNORECASE)
        )
        structure.label_for = len(
            re.findall(r'<label[^>]*for=["\'][^"\']+["\']', content, re.IGNORECASE)
        )
        structure.tabindex = len(re.findall(r'tabindex=["\'][^"\']+["\']', content, re.IGNORECASE))

        # Count custom elements
        structure.custom_elements = len(
            re.findall(r"<[a-z]+-[a-z-]+[^>]*>", content, re.IGNORECASE)
        )

        # Count data attributes
        structure.data_attributes = len(
            re.findall(r'data-[a-z-]+=["\'][^"\']+["\']', content, re.IGNORECASE)
        )

        # Count inline scripts and styles
        structure.inline_scripts = len(
            re.findall(
                r"<script[^>]*>(?:(?!</script>).)+</script>", content, re.IGNORECASE | re.DOTALL
            )
        )
        structure.inline_styles = len(
            re.findall(
                r"<style[^>]*>(?:(?!</style>).)+</style>", content, re.IGNORECASE | re.DOTALL
            )
        )
        structure.inline_style_attrs = len(
            re.findall(r'style=["\'][^"\']+["\']', content, re.IGNORECASE)
        )

        # Extract JavaScript event handlers
        event_handlers = [
            "onclick",
            "onchange",
            "onsubmit",
            "onload",
            "onerror",
            "onmouseover",
            "onmouseout",
            "onkeyup",
            "onkeydown",
            "onkeypress",
            "onfocus",
            "onblur",
            "ondblclick",
            "onscroll",
        ]
        structure.inline_event_handlers = 0
        for handler in event_handlers:
            structure.inline_event_handlers += len(
                re.findall(f"{handler}=[\"'][^\"']+[\"']", content, re.IGNORECASE)
            )

        # Detect lazy loading
        structure.lazy_loading = len(re.findall(r'loading=["\']lazy["\']', content, re.IGNORECASE))

        # Detect async/defer scripts
        structure.async_scripts = len(
            re.findall(r"<script[^>]*async[^>]*>", content, re.IGNORECASE)
        )
        structure.defer_scripts = len(
            re.findall(r"<script[^>]*defer[^>]*>", content, re.IGNORECASE)
        )

        # Check for service worker registration
        structure.has_service_worker = bool(
            re.search(r"navigator\.serviceWorker\.register", content, re.IGNORECASE)
        )

        # Check for web manifest
        structure.has_manifest = bool(
            re.search(r'<link[^>]*rel=["\']manifest["\']', content, re.IGNORECASE)
        )

        # Check for PWA features
        structure.is_pwa = structure.has_service_worker and structure.has_manifest

        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for HTML document.

        Calculates:
        - DOM complexity (depth, element count)
        - Form complexity
        - JavaScript complexity
        - Accessibility score
        - SEO readiness
        - Performance indicators

        Args:
            content: HTML source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()

        # Parse structure first
        structure = self.extract_structure(content, file_path)

        # Basic metrics
        lines = content.split("\n")
        metrics.line_count = len(lines)
        metrics.code_lines = len([l for l in lines if l.strip()])
        metrics.comment_lines = len(re.findall(r"<!--.*?-->", content, re.DOTALL))

        # DOM complexity
        metrics.total_elements = len(structure.elements) if hasattr(structure, "elements") else 0
        metrics.max_depth = structure.max_dom_depth if hasattr(structure, "max_dom_depth") else 0

        # Calculate average depth
        if hasattr(structure, "elements") and structure.elements:
            total_depth = sum(e.get("depth", 0) for e in structure.elements)
            metrics.avg_depth = total_depth / len(structure.elements)
        else:
            metrics.avg_depth = 0

        # Form complexity
        metrics.form_complexity = 0
        if hasattr(structure, "forms"):
            for form in structure.forms:
                # Each input adds to complexity
                metrics.form_complexity += len(form.get("inputs", []))
                # Required fields add more complexity
                for input_elem in form.get("inputs", []):
                    if "required" in input_elem.get("attrs", {}):
                        metrics.form_complexity += 1
                    # Validation patterns add complexity
                    if "pattern" in input_elem.get("attrs", {}):
                        metrics.form_complexity += 2

        # JavaScript complexity
        metrics.script_count = len(structure.scripts) if hasattr(structure, "scripts") else 0
        metrics.inline_script_count = (
            structure.inline_scripts if hasattr(structure, "inline_scripts") else 0
        )
        metrics.event_handler_count = (
            structure.inline_event_handlers if hasattr(structure, "inline_event_handlers") else 0
        )

        # Style complexity
        metrics.stylesheet_count = (
            len([l for l in structure.links if l.get("rel") == "stylesheet"])
            if hasattr(structure, "links")
            else 0
        )
        metrics.inline_style_count = (
            structure.inline_styles if hasattr(structure, "inline_styles") else 0
        )
        metrics.style_attribute_count = (
            structure.inline_style_attrs if hasattr(structure, "inline_style_attrs") else 0
        )

        # Accessibility score (0-100)
        accessibility_score = 100

        # Deduct points for missing accessibility features
        if hasattr(structure, "img_count") and structure.img_count > 0:
            if hasattr(structure, "alt_texts"):
                missing_alts = structure.img_count - structure.alt_texts
                accessibility_score -= missing_alts * 5

        if hasattr(structure, "form_count") and structure.form_count > 0:
            if hasattr(structure, "label_for"):
                if structure.label_for < structure.input_count:
                    accessibility_score -= 10

        # Add points for ARIA usage
        if hasattr(structure, "aria_labels"):
            accessibility_score = min(100, accessibility_score + (structure.aria_labels * 2))

        if hasattr(structure, "aria_roles"):
            accessibility_score = min(100, accessibility_score + (structure.aria_roles * 1))

        # Semantic HTML bonus
        semantic_bonus = 0
        for element in ["header", "nav", "main", "footer", "article", "section"]:
            if hasattr(structure, f"{element}_count"):
                semantic_bonus += getattr(structure, f"{element}_count")
        accessibility_score = min(100, accessibility_score + semantic_bonus)

        metrics.accessibility_score = max(0, accessibility_score)

        # SEO score (0-100)
        seo_score = 100

        # Check for essential SEO elements
        if not hasattr(structure, "has_doctype") or not structure.has_doctype:
            seo_score -= 10

        # Check for title
        if not re.search(r"<title[^>]*>.*?</title>", content, re.IGNORECASE | re.DOTALL):
            seo_score -= 20

        # Check for meta description
        if not re.search(r'<meta[^>]*name=["\']description["\']', content, re.IGNORECASE):
            seo_score -= 15

        # Check for viewport (mobile-friendly)
        if not hasattr(structure, "viewport"):
            seo_score -= 10

        # Check for heading hierarchy
        h1_count = len(re.findall(r"<h1[^>]*>", content, re.IGNORECASE))
        if h1_count == 0:
            seo_score -= 15
        elif h1_count > 1:
            seo_score -= 5

        # Check for Open Graph tags
        og_tags = len(re.findall(r'property=["\']og:', content, re.IGNORECASE))
        if og_tags > 0:
            seo_score = min(100, seo_score + 10)

        metrics.seo_score = max(0, seo_score)

        # Performance indicators
        metrics.performance_score = 100

        # Deduct for blocking resources
        if hasattr(structure, "scripts"):
            blocking_scripts = len(
                [s for s in structure.scripts if not s.get("async") and not s.get("defer")]
            )
            metrics.performance_score -= blocking_scripts * 5

        # Bonus for lazy loading
        if hasattr(structure, "lazy_loading"):
            metrics.performance_score = min(
                100, metrics.performance_score + (structure.lazy_loading * 2)
            )

        # Deduct for inline styles/scripts
        if hasattr(structure, "inline_scripts"):
            metrics.performance_score -= structure.inline_scripts * 2
        if hasattr(structure, "inline_style_attrs"):
            metrics.performance_score -= structure.inline_style_attrs * 0.5

        metrics.performance_score = max(0, metrics.performance_score)

        # Security indicators
        metrics.has_csp = bool(re.search(r"Content-Security-Policy", content, re.IGNORECASE))
        metrics.has_integrity_checks = bool(
            re.search(r'integrity=["\'][^"\']+["\']', content, re.IGNORECASE)
        )
        # Count only anchor href and script src links for HTTP/HTTPS
        https_links = 0
        http_links = 0
        for href in re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', content, re.IGNORECASE):
            if href.startswith("https://"):
                https_links += 1
            elif href.startswith("http://"):
                http_links += 1
        for src in re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE):
            if src.startswith("https://"):
                https_links += 1
            elif src.startswith("http://"):
                http_links += 1
        metrics.has_https_links = https_links
        metrics.has_http_links = http_links

        # Framework-specific metrics
        if structure.is_react if hasattr(structure, "is_react") else False:
            metrics.react_components = len(re.findall(r"<[A-Z][a-zA-Z]*", content))
            metrics.jsx_expressions = len(re.findall(r"\{[^}]+\}", content))

        if structure.is_vue if hasattr(structure, "is_vue") else False:
            metrics.vue_directives = len(re.findall(r"v-[a-z]+", content, re.IGNORECASE))
            metrics.vue_interpolations = len(re.findall(r"\{\{[^}]+\}\}", content))

        if structure.is_angular if hasattr(structure, "is_angular") else False:
            metrics.angular_directives = len(re.findall(r"\*ng[A-Z][a-zA-Z]*", content))
            metrics.angular_bindings = len(re.findall(r"\[(.*?)\]|\((.*?)\)", content))

        # Calculate maintainability index
        import math

        if metrics.total_elements > 0:
            # Factors affecting HTML maintainability
            depth_factor = 1 - (metrics.max_depth * 0.02)
            semantic_factor = 1 + (semantic_bonus * 0.01)
            inline_factor = 1 - ((metrics.inline_script_count + metrics.inline_style_count) * 0.02)
            accessibility_factor = metrics.accessibility_score / 100

            mi = (
                171
                - 5.2 * math.log(max(1, metrics.total_elements))
                - 0.23 * metrics.form_complexity
                - 16.2 * math.log(max(1, metrics.code_lines))
                + 20 * depth_factor
                + 10 * semantic_factor
                + 10 * inline_factor
                + 10 * accessibility_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))
        else:
            metrics.maintainability_index = 100

        return metrics

    def _categorize_css_import(self, href: str) -> str:
        """Categorize a CSS import.

        Args:
            href: CSS file URL

        Returns:
            Category string
        """
        href_lower = href.lower()

        if "bootstrap" in href_lower:
            return "bootstrap"
        elif "tailwind" in href_lower:
            return "tailwind"
        elif "bulma" in href_lower:
            return "bulma"
        elif "foundation" in href_lower:
            return "foundation"
        elif "materialize" in href_lower:
            return "materialize"
        elif "semantic" in href_lower:
            return "semantic_ui"
        elif "font" in href_lower:
            return "fonts"
        elif "normalize" in href_lower or "reset" in href_lower:
            return "reset"
        elif href.startswith(("http://", "https://", "//")):
            return "external"
        else:
            return "local"

    def _categorize_js_import(self, src: str) -> str:
        """Categorize a JavaScript import.

        Args:
            src: JavaScript file URL

        Returns:
            Category string
        """
        src_lower = src.lower()

        if "jquery" in src_lower:
            return "jquery"
        elif "react" in src_lower:
            return "react"
        elif "vue" in src_lower:
            return "vue"
        elif "angular" in src_lower:
            return "angular"
        elif "svelte" in src_lower:
            return "svelte"
        elif "bootstrap" in src_lower:
            return "bootstrap_js"
        # Detect lodash and underscore libraries only by name, not by any underscore character
        elif "lodash" in src_lower or "underscore" in src_lower:
            return "lodash"
        elif "moment" in src_lower:
            return "moment"
        elif "axios" in src_lower:
            return "axios"
        elif "d3" in src_lower:
            return "d3"
        elif "chart" in src_lower:
            return "charts"
        elif "analytics" in src_lower or "gtag" in src_lower or "googletagmanager" in src_lower:
            return "analytics"
        elif src.startswith(("http://", "https://", "//")):
            return "external"
        else:
            return "local"

    def _extract_integrity(self, tag: str) -> Optional[str]:
        """Extract integrity attribute from tag."""
        match = re.search(r'integrity=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_crossorigin(self, tag: str) -> Optional[str]:
        """Extract crossorigin attribute from tag."""
        match = re.search(r'crossorigin=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_as_type(self, tag: str) -> Optional[str]:
        """Extract 'as' attribute from preload tag."""
        match = re.search(r'as=["\']([^"\']+)["\']', tag, re.IGNORECASE)
        return match.group(1) if match else None

    def _detect_react(self, content: str) -> bool:
        """Detect if HTML uses React."""
        react_indicators = [
            r"react\.development\.js",
            r"react\.production\.min\.js",
            r"react-dom",
            r'<div\s+id=["\']root["\']',
            r"React\.createElement",
            r"ReactDOM\.render",
            r"jsx",
            r"className=",
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in react_indicators)

    def _detect_vue(self, content: str) -> bool:
        """Detect if HTML uses Vue.js."""
        vue_indicators = [
            r"vue\.js",
            r"vue\.min\.js",
            r"v-[a-z]+",
            r"\{\{[^}]+\}\}",
            r'<div\s+id=["\']app["\']',
            r"new\s+Vue",
            r":[\w-]+=",
            r"@[\w-]+=",
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in vue_indicators)

    def _detect_angular(self, content: str) -> bool:
        """Detect if HTML uses Angular."""
        angular_indicators = [
            r"angular\.js",
            r"angular\.min\.js",
            r"ng-app",
            r"ng-controller",
            r"\*ng[A-Z]",
            r"\[(.*?)\]",
            r"\((.*?)\)",
            r"<app-root>",
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in angular_indicators)

    def _detect_svelte(self, content: str) -> bool:
        """Detect if HTML uses Svelte."""
        svelte_indicators = [
            r"svelte",
            r"__svelte",
            r"svelte-[a-z0-9]+",
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in svelte_indicators)
