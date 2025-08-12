"""CSS code analyzer with preprocessor and framework support.

This module provides comprehensive analysis for CSS files,
including support for CSS3, SCSS/Sass, Less, PostCSS,
Tailwind CSS, UnoCSS, and other modern CSS frameworks.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import json

from ..base import LanguageAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)
from tenets.utils.logger import get_logger


class CSSParser:
    """Custom CSS parser for detailed analysis."""
    
    def __init__(self, content: str, is_scss: bool = False):
        self.content = content
        self.is_scss = is_scss
        self.rules = []
        self.variables = {}
        self.mixins = []
        self.functions = []
        self.keyframes = []
        self.media_queries = []
        self.supports_rules = []
        self.custom_properties = {}
        self.nesting_depth = 0
        self.max_nesting = 0
        
    def parse(self):
        """Parse CSS/SCSS content."""
        # Remove comments for parsing
        content = self._remove_comments(self.content)
        
        # Extract different CSS elements
        self._extract_variables(content)
        self._extract_custom_properties(content)
        self._extract_rules(content)
        self._extract_media_queries(content)
        self._extract_keyframes(content)
        self._extract_supports_rules(content)
        
        if self.is_scss:
            self._extract_scss_features(content)
    
    def _remove_comments(self, content: str) -> str:
        """Remove CSS comments."""
        # Remove multi-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove single-line comments (SCSS)
        if self.is_scss:
            content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        return content
    
    def _extract_variables(self, content: str):
        """Extract CSS/SCSS variables."""
        # CSS custom properties
        css_var_pattern = r'--([a-zA-Z0-9-_]+)\s*:\s*([^;]+);'
        for match in re.finditer(css_var_pattern, content):
            self.custom_properties[f"--{match.group(1)}"] = match.group(2).strip()
        
        # SCSS variables (only top-level, not inside mixin/function parameters)
        if self.is_scss:
            # Use multiline mode and ensure variables are declared at property level
            scss_var_pattern = r'^[ \t]*\$([a-zA-Z0-9-_]+)\s*:\s*([^;]+);'
            for match in re.finditer(scss_var_pattern, content, re.MULTILINE):
                self.variables[f"${match.group(1)}"] = match.group(2).strip()
    
    def _extract_custom_properties(self, content: str):
        """Extract CSS custom properties (CSS variables)."""
        pattern = r':root\s*\{([^}]+)\}'
        root_match = re.search(pattern, content, re.DOTALL)
        if root_match:
            props_content = root_match.group(1)
            prop_pattern = r'--([a-zA-Z0-9-_]+)\s*:\s*([^;]+);'
            for match in re.finditer(prop_pattern, props_content):
                self.custom_properties[f"--{match.group(1)}"] = match.group(2).strip()
    
    def _extract_rules(self, content: str):
        """Extract CSS rules and selectors."""
        # Simplified rule extraction
        rule_pattern = r'([^{]+)\s*\{([^}]+)\}'
        for match in re.finditer(rule_pattern, content):
            selector = match.group(1).strip()
            properties = match.group(2).strip()
            
            # Skip special rules
            if any(keyword in selector for keyword in ['@media', '@keyframes', '@supports', '@import']):
                continue
            
            self.rules.append({
                'selector': selector,
                'properties': self._parse_properties(properties),
                'specificity': self._calculate_specificity(selector),
            })
    
    def _extract_media_queries(self, content: str):
        """Extract media queries."""
        pattern = r'@media\s+([^{]+)\s*\{((?:[^{}]|\{[^}]*\})*)\}'
        for match in re.finditer(pattern, content):
            self.media_queries.append({
                'condition': match.group(1).strip(),
                'content': match.group(2).strip(),
            })
    
    def _extract_keyframes(self, content: str):
        """Extract keyframe animations."""
        pattern = r'@(?:keyframes|-webkit-keyframes|-moz-keyframes)\s+([a-zA-Z0-9-_]+)\s*\{((?:[^{}]|\{[^}]*\})*)\}'
        for match in re.finditer(pattern, content):
            self.keyframes.append({
                'name': match.group(1),
                'content': match.group(2).strip(),
            })
    
    def _extract_supports_rules(self, content: str):
        """Extract @supports rules."""
        pattern = r'@supports\s+([^{]+)\s*\{((?:[^{}]|\{[^}]*\})*)\}'
        for match in re.finditer(pattern, content):
            self.supports_rules.append({
                'condition': match.group(1).strip(),
                'content': match.group(2).strip(),
            })
    
    def _extract_scss_features(self, content: str):
        """Extract SCSS-specific features."""
        # Mixins
        mixin_pattern = r'@mixin\s+([a-zA-Z0-9-_]+)\s*(?:\(([^)]*)\))?\s*\{((?:[^{}]|\{[^}]*\})*)\}'
        for match in re.finditer(mixin_pattern, content):
            self.mixins.append({
                'name': match.group(1),
                'params': match.group(2) if match.group(2) else '',
                'content': match.group(3).strip(),
            })
        
        # Functions
        function_pattern = r'@function\s+([a-zA-Z0-9-_]+)\s*\(([^)]*)\)\s*\{((?:[^{}]|\{[^}]*\})*)\}'
        for match in re.finditer(function_pattern, content):
            self.functions.append({
                'name': match.group(1),
                'params': match.group(2),
                'content': match.group(3).strip(),
            })
        
        # Calculate nesting depth
        self._calculate_nesting_depth(content)
    
    def _parse_properties(self, properties_str: str) -> List[Dict[str, str]]:
        """Parse CSS properties from a rule."""
        properties = []
        prop_pattern = r'([a-zA-Z-]+)\s*:\s*([^;]+);?'
        for match in re.finditer(prop_pattern, properties_str):
            properties.append({
                'property': match.group(1),
                'value': match.group(2).strip(),
            })
        return properties
    
    def _calculate_specificity(self, selector: str) -> Tuple[int, int, int]:
        """Calculate CSS specificity (IDs, classes, elements)."""
        # Count IDs
        ids = len(re.findall(r'#[a-zA-Z0-9-_]+', selector))
        # Count classes, attributes, pseudo-classes
        classes = len(re.findall(r'\.[a-zA-Z0-9_\\:-]+', selector))
        classes += len(re.findall(r'\[[^\]]+\]', selector))  # Attribute selectors
        classes += len(re.findall(r':(?!:)[a-zA-Z-]+(?:\([^)]*\))?', selector))  # Pseudo-classes
        # Count element/type selectors (tokens starting at start or after combinators, not following . # : [ or *)
        element_tokens = re.findall(r'(?:(?<=^)|(?<=[\s>+~,(]))([a-zA-Z][a-zA-Z0-9-]*)', selector)
        elements = len(element_tokens)
        
        return (ids, classes, elements)
    
    def _calculate_nesting_depth(self, content: str):
        """Calculate maximum nesting depth in SCSS."""
        depth = 0
        max_depth = 0
        
        for char in content:
            if char == '{':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == '}':
                depth -= 1
        
        self.max_nesting = max_depth


class CSSAnalyzer(LanguageAnalyzer):
    """CSS code analyzer with preprocessor and framework support.

    Provides comprehensive analysis for CSS files including:
    - CSS3 features and properties
    - SCSS/Sass preprocessor features
    - Less preprocessor features
    - PostCSS plugins and features
    - Tailwind CSS utility classes
    - UnoCSS atomic CSS
    - CSS-in-JS patterns
    - CSS Modules
    - BEM, OOCSS, SMACSS methodologies
    - Performance metrics
    - Browser compatibility
    - Accessibility considerations
    - Design system patterns

    Supports modern CSS development practices and frameworks.
    """

    language_name = "css"
    file_extensions = [".css", ".scss", ".sass", ".less", ".styl", ".stylus", ".pcss", ".postcss"]

    def __init__(self):
        """Initialize the CSS analyzer with logger."""
        self.logger = get_logger(__name__)
        
        # Tailwind utility patterns
        self.tailwind_patterns = self._load_tailwind_patterns()
        
        # UnoCSS patterns
        self.unocss_patterns = self._load_unocss_patterns()
        
        # CSS framework patterns
        self.framework_patterns = self._load_framework_patterns()

    def extract_imports(self, content: str, file_path: Path) -> List[ImportInfo]:
        """Extract import statements from CSS.

        Handles:
        - @import statements
        - @use (Sass)
        - @forward (Sass)
        - url() functions
        - CSS Modules composes

        Args:
            content: CSS source code
            file_path: Path to the file being analyzed

        Returns:
            List of ImportInfo objects with import details
        """
        imports = []
        
        # Determine file type
        ext = file_path.suffix.lower()
        is_scss = ext in ['.scss', '.sass']
        is_less = ext == '.less'
        
        # @import statements
        import_pattern = r'@import\s+(?:url\()?["\']([^"\']+)["\'](?:\))?(?:\s+([^;]+))?;'
        for match in re.finditer(import_pattern, content):
            import_path = match.group(1)
            media_query = match.group(2)
            
            imports.append(
                ImportInfo(
                    module=import_path,
                    line=content[:match.start()].count("\n") + 1,
                    type="import",
                    is_relative=not import_path.startswith(('http://', 'https://', '//')),
                    media_query=media_query.strip() if media_query else None,
                    category=self._categorize_css_import(import_path),
                )
            )
        
        # @use statements (Sass)
        if is_scss:
            use_pattern = r'@use\s+["\']([^"\']+)["\'](?:\s+as\s+(\w+))?(?:\s+with\s*\(([^)]+)\))?;'
            for match in re.finditer(use_pattern, content):
                module_path = match.group(1)
                namespace = match.group(2)
                config = match.group(3)
                
                imports.append(
                    ImportInfo(
                        module=module_path,
                        line=content[:match.start()].count("\n") + 1,
                        type="use",
                        is_relative=not module_path.startswith(('http://', 'https://', '//')),
                        namespace=namespace,
                        config=config,
                        category=self._categorize_css_import(module_path),
                    )
                )
            
            # @forward statements (Sass)
            forward_pattern = r'@forward\s+["\']([^"\']+)["\'](?:\s+(show|hide)\s+([^;]+))?;'
            for match in re.finditer(forward_pattern, content):
                module_path = match.group(1)
                visibility_type = match.group(2)
                visibility_items = match.group(3)
                
                # Combine visibility type and items for easier testing
                if visibility_type and visibility_items:
                    visibility = f"{visibility_type} {visibility_items.strip()}"
                else:
                    visibility = None
                
                imports.append(
                    ImportInfo(
                        module=module_path,
                        line=content[:match.start()].count("\n") + 1,
                        type="forward",
                        is_relative=not module_path.startswith(('http://', 'https://', '//')),
                        visibility=visibility,
                        category=self._categorize_css_import(module_path),
                    )
                )
        
        # url() in properties (for fonts, images, etc.)
        url_pattern = r'url\(["\']?([^"\')\s]+)["\']?\)'
        for match in re.finditer(url_pattern, content):
            url_path = match.group(1)
            
            # Skip data URLs and already imported files
            if url_path.startswith('data:') or any(imp.module == url_path for imp in imports):
                continue
            
            imports.append(
                ImportInfo(
                    module=url_path,
                    line=content[:match.start()].count("\n") + 1,
                    type="url",
                    is_relative=not url_path.startswith(('http://', 'https://', '//')),
                    category=self._categorize_url_import(url_path),
                )
            )
        
        # CSS Modules composes
        composes_pattern = r'composes:\s*([a-zA-Z0-9-_\s]+)\s+from\s+["\']([^"\']+)["\'];'
        for match in re.finditer(composes_pattern, content):
            classes = match.group(1)
            module_path = match.group(2)
            
            imports.append(
                ImportInfo(
                    module=module_path,
                    line=content[:match.start()].count("\n") + 1,
                    type="composes",
                    is_relative=not module_path.startswith(('http://', 'https://', '//')),
                    composes=classes.strip(),
                    # Alias for tests that expect composed_classes
                    visibility=None,
                    category="css_module",
                )
            )
        # Backward compatibility: also attach composed_classes attribute dynamically
        for imp in imports:
            if imp.type == "composes" and getattr(imp, "composes", None):
                # Some tests reference ImportInfo.composed_classes
                try:
                    setattr(imp, "composed_classes", imp.composes)
                except Exception:
                    pass
        
        return imports

    def extract_exports(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Extract exported elements from CSS.

        In CSS context, exports are:
        - Classes that can be used by HTML
        - IDs
        - Custom properties (CSS variables)
        - Mixins (SCSS/Less)
        - Functions (SCSS)
        - Keyframe animations
        - Utility classes (Tailwind/UnoCSS)

        Args:
            content: CSS source code
            file_path: Path to the file being analyzed

        Returns:
            List of exported elements
        """
        exports = []
        
        # Parse CSS
        ext = file_path.suffix.lower()
        is_scss = ext in ['.scss', '.sass']
        parser = CSSParser(content, is_scss)
        parser.parse()
        
        # Export CSS classes (from selectors only)
        classes: Set[str] = set()
        for rule in parser.rules:
            selector = rule.get('selector', '')
            for match in re.finditer(r'\.([a-zA-Z0-9_\\:-]+)', selector):
                class_name = match.group(1)
                if class_name not in classes:
                    classes.add(class_name)
                    pos = content.find('.' + class_name)
                    exports.append({
                        "name": class_name,
                        "type": "class",
                        "line": (content[:pos].count("\n") + 1) if pos != -1 else None,
                    })
        
        # Export IDs (from selectors only, avoid hex colors)
        ids: Set[str] = set()
        for rule in parser.rules:
            selector = rule.get('selector', '')
            for match in re.finditer(r'#([a-zA-Z0-9_-]+)', selector):
                id_name = match.group(1)
                if id_name not in ids:
                    ids.add(id_name)
                    pos = content.find('#' + id_name)
                    exports.append({
                        "name": id_name,
                        "type": "id",
                        "line": (content[:pos].count("\n") + 1) if pos != -1 else None,
                    })
        
        # Export custom properties
        for prop_name, prop_value in parser.custom_properties.items():
            exports.append({
                "name": prop_name,
                "type": "custom_property",
                "value": prop_value,
            })
        
        # Export SCSS variables, mixins, functions
        if is_scss:
            for var_name, var_value in parser.variables.items():
                exports.append({
                    "name": var_name,
                    "type": "scss_variable",
                    "value": var_value,
                })
            for mixin in parser.mixins:
                exports.append({
                    "name": mixin["name"],
                    "type": "mixin",
                    "params": mixin["params"],
                })
            for func in parser.functions:
                exports.append({
                    "name": func["name"],
                    "type": "function",
                    "params": func["params"],
                })
        
        # Export keyframes
        for keyframe in parser.keyframes:
            exports.append({
                "name": keyframe["name"],
                "type": "keyframe",
            })
        
        # Export utility classes (Tailwind/UnoCSS)
        if self._is_utility_css(content):
            utility_classes = self._extract_utility_classes(content)
            for util_class in utility_classes:
                exports.append({
                    "name": util_class,
                    "type": "utility_class",
                    "framework": self._detect_utility_framework(content),
                })
        
        return exports

    def extract_structure(self, content: str, file_path: Path) -> CodeStructure:
        """Extract CSS document structure.

        Extracts:
        - Rules and selectors
        - Media queries
        - CSS architecture patterns
        - Framework usage
        - Design tokens
        - Component structure

        Args:
            content: CSS source code
            file_path: Path to the file being analyzed

        Returns:
            CodeStructure object with extracted elements
        """
        structure = CodeStructure()
        
        # Parse CSS
        ext = file_path.suffix.lower()
        is_scss = ext in ['.scss', '.sass']
        is_less = ext == '.less'
        parser = CSSParser(content, is_scss)
        parser.parse()
        
        # Store parsed data
        structure.rules = parser.rules
        structure.variables = parser.variables
        structure.custom_properties = parser.custom_properties
        structure.mixins = parser.mixins
        structure.functions = parser.functions
        structure.keyframes = parser.keyframes
        structure.media_queries = parser.media_queries
        structure.supports_rules = parser.supports_rules
        structure.max_nesting = parser.max_nesting
        
        # Detect CSS methodology
        structure.uses_bem = self._detect_bem(content)
        structure.uses_oocss = self._detect_oocss(content)
        structure.uses_smacss = self._detect_smacss(content)
        structure.uses_atomic = self._detect_atomic_css(content)
        
        # Detect frameworks
        structure.is_tailwind = self._detect_tailwind(content, file_path)
        structure.is_unocss = self._detect_unocss(content, file_path)
        structure.is_bootstrap = self._detect_bootstrap(content)
        structure.is_bulma = self._detect_bulma(content)
        structure.is_material = self._detect_material(content)
        
        # Count selectors by type (from selectors only)
        selectors_joined = ",".join(rule.get('selector', '') for rule in parser.rules)
        structure.element_selectors = len(re.findall(r'(?:(?<=^)|(?<=[\s>+~,(]))[a-zA-Z][a-zA-Z0-9-]*', selectors_joined))
        structure.class_selectors = len(re.findall(r'\.[a-zA-Z0-9_\\:-]+', selectors_joined))
        structure.id_selectors = len(re.findall(r'#[a-zA-Z0-9_-]+', selectors_joined))
        structure.attribute_selectors = len(re.findall(r'\[[^\]]+\]', selectors_joined))
        structure.pseudo_classes = len(re.findall(r':(?!:)[a-z-]+(?:\([^)]*\))?', selectors_joined))
        structure.pseudo_elements = len(re.findall(r'::[a-z-]+', selectors_joined))
        
        # Count CSS3 features
        structure.flexbox_usage = len(re.findall(r'display\s*:\s*(?:inline-)?flex', content))
        structure.grid_usage = len(re.findall(r'display\s*:\s*grid', content))
        structure.custom_property_usage = len(re.findall(r'var\(--[^)]+\)', content))
        structure.calc_usage = len(re.findall(r'calc\([^)]+\)', content))
        structure.transform_usage = len(re.findall(r'transform\s*:', content))
        structure.transition_usage = len(re.findall(r'transition\s*:', content))
        structure.animation_usage = len(re.findall(r'animation\s*:', content))
        
        # Count responsive features
        structure.media_query_count = len(parser.media_queries)
        structure.viewport_units = len(re.findall(r'\d+(?:vw|vh|vmin|vmax)\b', content))
        structure.container_queries = len(re.findall(r'@container\s+', content))
        
        # Count modern CSS features
        structure.css_nesting = len(re.findall(r'&\s*[{:.]', content)) + self._count_nested_selectors(content)
        structure.has_layers = bool(re.search(r'@layer\s+', content))
        structure.has_cascade_layers = len(re.findall(r'@layer\s+[a-z-]+\s*[{,]', content))
        
        # Design system detection
        structure.has_design_tokens = self._detect_design_tokens(content)
        structure.color_variables = self._count_color_variables(parser.custom_properties)
        structure.spacing_variables = self._count_spacing_variables(parser.custom_properties)
        structure.typography_variables = self._count_typography_variables(parser.custom_properties)
        
        # Component-based structure
        structure.component_count = self._count_components(content)
        structure.utility_count = self._count_utilities(content)
        
        # PostCSS features
        structure.uses_postcss = self._detect_postcss(content, file_path)
        structure.postcss_plugins = self._detect_postcss_plugins(content)
        
        # CSS-in-JS patterns
        structure.is_css_modules = self._detect_css_modules(content, file_path)
        structure.is_styled_components = self._detect_styled_components(content)
        
        # Performance indicators
        structure.unused_variables = self._find_unused_variables(content, parser)
        structure.duplicate_properties = self._find_duplicate_properties(parser.rules)
        structure.vendor_prefixes = len(re.findall(r'-(?:webkit|moz|ms|o)-', content))
        
        # Accessibility
        structure.focus_styles = len(re.findall(r':focus\s*[{,]', content))
        structure.focus_visible = len(re.findall(r':focus-visible\s*[{,]', content))
        structure.reduced_motion = len(re.findall(r'prefers-reduced-motion', content))
        structure.high_contrast = len(re.findall(r'prefers-contrast', content))
        structure.color_scheme = len(re.findall(r'prefers-color-scheme', content))
        
        return structure

    def calculate_complexity(self, content: str, file_path: Path) -> ComplexityMetrics:
        """Calculate complexity metrics for CSS.

        Calculates:
        - Selector complexity
        - Specificity metrics
        - Rule complexity
        - Nesting depth
        - Framework complexity
        - Performance score
        - Maintainability index

        Args:
            content: CSS source code
            file_path: Path to the file being analyzed

        Returns:
            ComplexityMetrics object with calculated metrics
        """
        metrics = ComplexityMetrics()
        
        # Parse CSS
        ext = file_path.suffix.lower()
        is_scss = ext in ['.scss', '.sass']
        parser = CSSParser(content, is_scss)
        parser.parse()
        
        # Basic metrics
        lines = content.split("\n")
        metrics.line_count = len(lines)
        metrics.code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('//')])
        metrics.comment_lines = len(re.findall(r'/\*.*?\*/', content, re.DOTALL))
        if is_scss:
            metrics.comment_lines += len([l for l in lines if l.strip().startswith('//')])
        
        # Rule metrics
        metrics.total_rules = len(parser.rules)
        metrics.total_selectors = sum(len(rule['selector'].split(',')) for rule in parser.rules)
        
        # Calculate average specificity
        total_specificity = [0, 0, 0]
        max_specificity = [0, 0, 0]
        
        for rule in parser.rules:
            spec = rule['specificity']
            total_specificity[0] += spec[0]
            total_specificity[1] += spec[1]
            total_specificity[2] += spec[2]
            
            if spec[0] > max_specificity[0]:
                max_specificity = spec
            elif spec[0] == max_specificity[0] and spec[1] > max_specificity[1]:
                max_specificity = spec
            elif spec[0] == max_specificity[0] and spec[1] == max_specificity[1] and spec[2] > max_specificity[2]:
                max_specificity = spec
        
        if metrics.total_rules > 0:
            metrics.avg_specificity = [
                total_specificity[0] / metrics.total_rules,
                total_specificity[1] / metrics.total_rules,
                total_specificity[2] / metrics.total_rules,
            ]
        else:
            metrics.avg_specificity = [0, 0, 0]
        
        metrics.max_specificity = max_specificity
        
        # Selector complexity
        metrics.complex_selectors = 0
        metrics.overqualified_selectors = 0
        
        for rule in parser.rules:
            selector = rule['selector']
            
            # Complex selector (too many parts)
            if len(selector.split()) > 3:
                metrics.complex_selectors += 1
            
            # Overqualified (element with class/id)
            if re.search(r'[a-z]+\.[a-z-]+|[a-z]+#[a-z-]+', selector, re.IGNORECASE):
                metrics.overqualified_selectors += 1
        
        # Important usage
        metrics.important_count = len(re.findall(r'!important', content))
        
        # Media query complexity
        metrics.media_query_count = len(parser.media_queries)
        metrics.media_query_complexity = sum(
            len(mq['condition'].split('and')) 
            for mq in parser.media_queries
        )
        
        # Nesting depth (for SCSS)
        metrics.max_nesting_depth = parser.max_nesting
        
        # Color usage
        metrics.unique_colors = len(set(re.findall(
            r'#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|rgba\([^)]+\)|hsl\([^)]+\)|hsla\([^)]+\)',
            content
        )))
        
        # Font usage
        metrics.unique_fonts = len(set(re.findall(
            r'font-family\s*:\s*([^;]+);',
            content
        )))
        
        # Z-index usage
        z_indices = re.findall(r'z-index\s*:\s*(-?\d+)', content)
        metrics.z_index_count = len(z_indices)
        if z_indices:
            metrics.max_z_index = max(int(z) for z in z_indices)
        else:
            metrics.max_z_index = 0
        
        # File size metrics
        metrics.file_size = len(content.encode('utf-8'))
        metrics.gzip_ratio = self._estimate_gzip_ratio(content)
        
        # Framework-specific metrics
        if self._detect_tailwind(content, file_path):
            metrics.tailwind_classes = self._count_tailwind_classes(content)
            metrics.custom_utilities = self._count_custom_utilities(content)
        
        if self._detect_unocss(content, file_path):
            metrics.unocss_classes = self._count_unocss_classes(content)
        
        # Calculate CSS complexity score
        complexity_score = (
            metrics.total_rules * 0.1 +
            metrics.complex_selectors * 2 +
            metrics.overqualified_selectors * 1.5 +
            metrics.important_count * 3 +
            metrics.max_nesting_depth * 1 +
            (metrics.max_specificity[0] * 10) +  # IDs weighted heavily
            (metrics.max_specificity[1] * 2) +   # Classes
            (metrics.max_specificity[2] * 0.5)   # Elements
        )
        metrics.complexity_score = complexity_score
        
        # Performance score
        performance_score = 100
        
        # Deduct for complexity
        performance_score -= min(30, complexity_score / 10)
        
        # Deduct for !important
        performance_score -= min(20, metrics.important_count * 2)
        
        # Deduct for deep nesting
        performance_score -= min(10, metrics.max_nesting_depth * 2)
        
        # Deduct for excessive specificity
        performance_score -= min(10, metrics.max_specificity[0] * 5)
        
        # Bonus for CSS variables usage
        if len(parser.custom_properties) > 0:
            performance_score += min(10, len(parser.custom_properties) * 0.5)
        
        metrics.performance_score = max(0, performance_score)
        
        # Calculate maintainability index
        import math
        
        if metrics.code_lines > 0:
            # Factors affecting CSS maintainability
            specificity_factor = 1 - (sum(metrics.avg_specificity) * 0.1)
            important_factor = 1 - (metrics.important_count * 0.02)
            nesting_factor = 1 - (metrics.max_nesting_depth * 0.05)
            organization_factor = 1 if len(parser.custom_properties) > 0 else 0.8
            
            mi = (
                171
                - 5.2 * math.log(max(1, metrics.total_rules))
                - 0.23 * complexity_score
                - 16.2 * math.log(max(1, metrics.code_lines))
                + 20 * specificity_factor
                + 10 * important_factor
                + 10 * nesting_factor
                + 10 * organization_factor
            )
            metrics.maintainability_index = max(0, min(100, mi))
        else:
            metrics.maintainability_index = 100
        
        return metrics

    def _load_tailwind_patterns(self) -> Dict[str, List[str]]:
        """Load Tailwind CSS utility patterns."""
        return {
            "spacing": ["p-", "m-", "px-", "py-", "pt-", "pb-", "pl-", "pr-", "mx-", "my-", "mt-", "mb-", "ml-", "mr-", "space-x-", "space-y-"],
            "sizing": ["w-", "h-", "min-w-", "min-h-", "max-w-", "max-h-"],
            "typography": ["text-", "font-", "leading-", "tracking-", "align-"],
            "colors": ["bg-", "text-", "border-", "ring-", "divide-"],
            "flexbox": ["flex", "flex-row", "flex-col", "justify-", "items-", "self-", "order-", "gap-"],
            "grid": ["grid", "grid-cols-", "grid-rows-", "col-span-", "row-span-", "gap-"],
            "borders": ["border", "border-t", "border-b", "border-l", "border-r", "rounded"],
            "effects": ["shadow", "opacity-", "blur", "brightness-", "contrast-"],
            "transforms": ["scale-", "rotate-", "translate-x-", "translate-y-", "skew-"],
            "transitions": ["transition", "duration-", "ease-", "delay-"],
            "responsive": ["sm:", "md:", "lg:", "xl:", "2xl:"],
            "states": ["hover:", "focus:", "active:", "disabled:", "group-hover:", "dark:"],
        }

    def _load_unocss_patterns(self) -> Dict[str, List[str]]:
        """Load UnoCSS utility patterns."""
        return {
            "spacing": ["p-", "m-", "p", "m", "px", "py", "pt", "pb", "pl", "pr", "mx", "my", "mt", "mb", "ml", "mr"],
            "sizing": ["w-", "h-", "w", "h", "min-w", "min-h", "max-w", "max-h", "size-"],
            "typography": ["text-", "font-", "leading-", "tracking-", "op-"],
            "colors": ["bg-", "text-", "border-", "c-", "color-"],
            "display": ["flex", "grid", "inline", "block", "hidden", "table"],
            "positioning": ["relative", "absolute", "fixed", "sticky", "static"],
            "borders": ["border", "b-", "rounded", "rd-"],
            "shortcuts": ["flex-center", "flex-between", "flex-around"],
        }

    def _load_framework_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load patterns for various CSS frameworks."""
        return {
            "bootstrap": {
                "prefixes": ["btn-", "col-", "row", "container", "navbar-", "form-", "alert-", "modal-"],
                "utilities": ["d-", "p-", "m-", "text-", "bg-", "border-", "rounded-"],
            },
            "bulma": {
                "prefixes": ["button", "column", "columns", "container", "navbar", "hero", "section"],
                "modifiers": ["is-", "has-"],
            },
            "material": {
                "prefixes": ["mdc-", "mat-", "md-"],
                "components": ["button", "card", "toolbar", "sidenav", "dialog"],
            },
            "semantic": {
                "prefixes": ["ui", "semantic"],
                "components": ["button", "card", "menu", "sidebar", "modal"],
            },
        }

    def _categorize_css_import(self, import_path: str) -> str:
        """Categorize a CSS import."""
        path_lower = import_path.lower()
        
        if "tailwind" in path_lower:
            return "tailwind"
        elif "bootstrap" in path_lower:
            return "bootstrap"
        elif "bulma" in path_lower:
            return "bulma"
        elif "material" in path_lower:
            return "material"
        elif "normalize" in path_lower or "reset" in path_lower:
            return "reset"
        elif "variables" in path_lower or "tokens" in path_lower:
            return "design_tokens"
        elif "mixins" in path_lower:
            return "mixins"
        elif "utilities" in path_lower or "helpers" in path_lower:
            return "utilities"
        elif "components" in path_lower:
            return "components"
        elif path_lower.startswith(('http://', 'https://', '//')):
            return "external"
        else:
            return "local"

    def _categorize_url_import(self, url_path: str) -> str:
        """Categorize a URL import."""
        path_lower = url_path.lower()
        
        if any(ext in path_lower for ext in ['.woff', '.woff2', '.ttf', '.otf', '.eot']):
            return "font"
        elif any(ext in path_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp']):
            return "image"
        elif any(ext in path_lower for ext in ['.mp4', '.webm', '.ogg']):
            return "video"
        else:
            return "resource"

    def _is_utility_css(self, content: str) -> bool:
        """Check if CSS contains utility classes."""
        utility_indicators = [
            r'\.[a-z]+-\d+',  # Numbered utilities
            r'\.(?:sm|md|lg|xl):[a-z]+',  # Responsive utilities
            r'\.hover:[a-z]+',  # State utilities
            r'\.[a-z]+\/\d+',  # Fractional utilities
        ]
        return any(re.search(pattern, content) for pattern in utility_indicators)

    def _extract_utility_classes(self, content: str) -> List[str]:
        """Extract utility classes from CSS."""
        utilities = set()
        
        # Tailwind-style utilities
        tailwind_pattern = r'\.([a-z]+(?:-[a-z0-9]+)*(?:\/\d+)?)'
        for match in re.finditer(tailwind_pattern, content):
            utilities.add(match.group(1))
        
        # UnoCSS-style utilities
        uno_pattern = r'\.([a-z]+(?:-?[a-z0-9]+)*)'
        for match in re.finditer(uno_pattern, content):
            utilities.add(match.group(1))
        
        return list(utilities)

    def _detect_utility_framework(self, content: str) -> str:
        """Detect which utility framework is being used."""
        if self._detect_tailwind(content, Path("")):
            return "tailwind"
        elif self._detect_unocss(content, Path("")):
            return "unocss"
        elif "tachyons" in content.lower():
            return "tachyons"
        else:
            return "custom"

    def _detect_bem(self, content: str) -> bool:
        """Detect BEM methodology usage."""
        bem_patterns = [
            r'\.[a-z]+__[a-z]+',  # Block__Element
            r'\.[a-z]+--[a-z]+',  # Block--Modifier
            r'\.[a-z]+__[a-z]+--[a-z]+',  # Block__Element--Modifier
        ]
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in bem_patterns)

    def _detect_oocss(self, content: str) -> bool:
        """Detect OOCSS methodology usage."""
        # OOCSS often uses separate structure and skin classes
        structure_classes = re.findall(r'\.(?:wrapper|container|box|media)\b', content, re.IGNORECASE)
        skin_classes = re.findall(r'\.(?:theme-|skin-|color-)', content, re.IGNORECASE)
        return len(structure_classes) > 2 and len(skin_classes) > 2

    def _detect_smacss(self, content: str) -> bool:
        """Detect SMACSS methodology usage."""
        smacss_patterns = [
            r'\.l-[a-z]+',  # Layout
            r'\.m-[a-z]+',  # Module
            r'\.is-[a-z]+',  # State
            r'\.t-[a-z]+',  # Theme
        ]
        matches = sum(1 for pattern in smacss_patterns if re.search(pattern, content, re.IGNORECASE))
        return matches >= 2

    def _detect_atomic_css(self, content: str) -> bool:
        """Detect Atomic CSS methodology usage."""
        atomic_patterns = [
            r'\.[a-z]{1,3}-\d+',  # Short class names with numbers (p-4, m-2, etc.)
            r'\.(?:p|m|w|h|t|b|l|r)-\d+',  # Single-purpose utilities
            r'\.[a-z]{1,4}\d+',  # Short class names with numbers (without dash)
            r'\.(?:p|m|w|h|t|b|l|r)\d+',  # Single-purpose utilities (without dash)
            r'\.text-[a-z]+',  # Text utilities
            r'\.bg-[a-z]+',    # Background utilities
            r'\.flex-[a-z]+',  # Flex utilities
            r'\.hover\\:[a-z-]+',  # Hover variants
        ]
        matches = sum(len(re.findall(pattern, content)) for pattern in atomic_patterns)
        return matches > 2  # Lower threshold

    def _detect_tailwind(self, content: str, file_path: Path) -> bool:
        """Detect Tailwind CSS usage."""
        tailwind_indicators = [
            r'@tailwind\s+(?:base|components|utilities)',
            r'@apply\s+',
            # Escaped responsive/state variants used in emitted CSS
            r'\.(?:sm|md|lg|xl|2xl)\\:',
            r'\.(?:hover|focus|active|disabled|group-hover|dark)\\:',
            # Common utility prefixes
            r'\.bg-[a-z-]+-\d+',
            r'\.text-[a-z-]+-\d+',
        ]
        
        if "@tailwind" in content or "tailwindcss" in content.lower():
            return True
        
        return any(re.search(pattern, content) for pattern in tailwind_indicators)

    def _detect_unocss(self, content: str, file_path: Path) -> bool:
        """Detect UnoCSS usage."""
        unocss_indicators = [
            r'@unocss',
            r'uno:',
            r'--at-apply:',
            r'\.uno-',
        ]
        
        # Check for UnoCSS config file
        if file_path.name in ['uno.config.ts', 'uno.config.js', 'unocss.config.ts']:
            return True
        
        return any(re.search(pattern, content) for pattern in unocss_indicators)

    def _detect_bootstrap(self, content: str) -> bool:
        """Detect Bootstrap usage."""
        bootstrap_patterns = [
            r'\.(?:btn|btn-primary|btn-secondary)',
            r'\.(?:container|container-fluid)',
            r'\.(?:row|col-(?:sm|md|lg|xl)-\d+)',
            r'\.(?:navbar|nav-link)',
            r'\.(?:card|card-body)',
        ]
        matches = sum(1 for pattern in bootstrap_patterns if re.search(pattern, content))
        return matches >= 3

    def _detect_bulma(self, content: str) -> bool:
        """Detect Bulma usage."""
        bulma_patterns = [
            r'\.(?:button|buttons)',
            r'\.(?:column|columns)',
            r'\.(?:is-[a-z]+|has-[a-z]+)',
            r'\.(?:hero|hero-body)',
            r'\.(?:section|container)',
        ]
        matches = sum(1 for pattern in bulma_patterns if re.search(pattern, content))
        return matches >= 3

    def _detect_material(self, content: str) -> bool:
        """Detect Material Design CSS usage."""
        material_patterns = [
            r'\.(?:mdc-[a-z-]+)',
            r'\.(?:mat-[a-z-]+)',
            r'\.(?:md-[a-z-]+)',
            r'\.(?:material-icons)',
        ]
        return any(re.search(pattern, content) for pattern in material_patterns)

    # Additional helper methods...
    def _detect_design_tokens(self, content: str) -> bool:
        """Detect design token usage."""
        token_patterns = [
            r'--(?:color|spacing|size|font|shadow|radius)-',
            r'\$(?:color|spacing|size|font|shadow|radius)-',
        ]
        return any(re.search(pattern, content) for pattern in token_patterns)

    def _count_color_variables(self, custom_properties: Dict[str, str]) -> int:
        """Count color-related CSS variables."""
        color_keywords = ['color', 'bg', 'background', 'border', 'text', 'fill', 'stroke']
        return sum(1 for prop in custom_properties if any(keyword in prop.lower() for keyword in color_keywords))

    def _count_spacing_variables(self, custom_properties: Dict[str, str]) -> int:
        """Count spacing-related CSS variables."""
        spacing_keywords = ['spacing', 'space', 'gap', 'margin', 'padding', 'inset']
        return sum(1 for prop in custom_properties if any(keyword in prop.lower() for keyword in spacing_keywords))

    def _count_typography_variables(self, custom_properties: Dict[str, str]) -> int:
        """Count typography-related CSS variables."""
        typography_keywords = ['font', 'text', 'line', 'letter', 'type', 'heading']
        return sum(1 for prop in custom_properties if any(keyword in prop.lower() for keyword in typography_keywords))

    def _count_components(self, content: str) -> int:
        """Count component-style class definitions."""
        # Components typically have longer, descriptive names
        component_pattern = r'\.[a-z]+(?:-[a-z]+){2,}'
        return len(set(re.findall(component_pattern, content)))

    def _count_utilities(self, content: str) -> int:
        """Count utility class definitions."""
        # Utilities typically have short names with modifiers
        utility_pattern = r'\.[a-z]{1,4}-(?:\d+|[a-z]{1,6})'
        return len(set(re.findall(utility_pattern, content)))

    def _detect_postcss(self, content: str, file_path: Path) -> bool:
        """Detect PostCSS usage."""
        postcss_indicators = [
            r'@custom-media',
            r'@custom-selector',
            r'@nest',
            r'&\s*{',  # Nesting
        ]
        
        # Check for PostCSS config files
        if file_path.name in ['postcss.config.js', '.postcssrc.js', '.postcssrc.json']:
            return True
        
        return any(re.search(pattern, content) for pattern in postcss_indicators)

    def _detect_postcss_plugins(self, content: str) -> List[str]:
        """Detect PostCSS plugins being used."""
        plugins = []
        
        if re.search(r'@import', content):
            plugins.append("postcss-import")
        if re.search(r'@custom-media', content):
            plugins.append("postcss-custom-media")
        if re.search(r'@nest|&\s*{', content):
            plugins.append("postcss-nested")
        if re.search(r'@apply', content):
            plugins.append("postcss-apply")
        
        return plugins

    def _detect_css_modules(self, content: str, file_path: Path) -> bool:
        """Detect CSS Modules usage."""
        # CSS Modules often use .module.css extension
        if '.module.' in file_path.name:
            return True
        
        # Check for composes keyword
        return bool(re.search(r'\bcomposes:', content))

    def _detect_styled_components(self, content: str) -> bool:
        """Detect styled-components patterns."""
        # This would typically be in JS/TS files, but check for generated class patterns
        styled_patterns = [
            r'\.sc-[a-zA-Z0-9]+',  # styled-components generated classes
            r'\.css-[a-zA-Z0-9]+',  # emotion generated classes
        ]
        return any(re.search(pattern, content) for pattern in styled_patterns)

    def _find_unused_variables(self, content: str, parser: CSSParser) -> List[str]:
        """Find potentially unused variables."""
        unused = []
        
        for var_name in parser.variables:
            # Check if variable is used
            usage_pattern = rf'\{re.escape(var_name)}\b'
            if len(re.findall(usage_pattern, content)) <= 1:  # Only defined, not used
                unused.append(var_name)
        
        for prop_name in parser.custom_properties:
            # Check if custom property is used
            usage_pattern = rf'var\({re.escape(prop_name)}\)'
            if not re.search(usage_pattern, content):
                unused.append(prop_name)
        
        return unused

    def _find_duplicate_properties(self, rules: List[Dict]) -> int:
        """Find duplicate property definitions within rules."""
        duplicates = 0
        
        for rule in rules:
            properties = rule.get('properties', [])
            seen = set()
            
            for prop in properties:
                prop_name = prop.get('property')
                if prop_name in seen:
                    duplicates += 1
                seen.add(prop_name)
        
        return duplicates

    def _estimate_gzip_ratio(self, content: str) -> float:
        """Estimate gzip compression ratio."""
        import gzip
        
        original_size = len(content.encode('utf-8'))
        compressed_size = len(gzip.compress(content.encode('utf-8')))
        
        return compressed_size / original_size if original_size > 0 else 0

    def _count_tailwind_classes(self, content: str) -> int:
        """Count Tailwind utility classes."""
        count = 0
        for category, patterns in self.tailwind_patterns.items():
            for pattern in patterns:
                count += len(re.findall(rf'\.{re.escape(pattern)}[a-z0-9-]*', content))
        return count

    def _count_unocss_classes(self, content: str) -> int:
        """Count UnoCSS utility classes."""
        count = 0
        for category, patterns in self.unocss_patterns.items():
            for pattern in patterns:
                count += len(re.findall(rf'\.{re.escape(pattern)}[a-z0-9-]*', content))
        return count

    def _count_custom_utilities(self, content: str) -> int:
        """Count custom utility classes."""
        # Custom utilities often follow patterns like .u-* or .util-*
        custom_patterns = [
            r'\.u-[a-z0-9-]+',
            r'\.util-[a-z0-9-]+',
            r'\.helper-[a-z0-9-]+',
        ]
        return sum(len(re.findall(pattern, content)) for pattern in custom_patterns)

    def _count_nested_selectors(self, content: str) -> int:
        """Count nested selectors in CSS/SCSS (beyond & nesting)."""
        # Look for selectors nested within braces
        nested_count = 0
        in_rule = False
        brace_depth = 0
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('/*', '//')):
                continue
                
            # Count opening and closing braces
            open_braces = line.count('{')
            close_braces = line.count('}')
            
            if open_braces > 0:
                if brace_depth > 0 and not line.startswith('@'):
                    # This is a selector inside another rule
                    nested_count += open_braces
                brace_depth += open_braces
                
            brace_depth -= close_braces
            
        return nested_count