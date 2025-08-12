"""
Unit tests for the CSS code analyzer with preprocessor and framework support.

This module tests the CSS-specific code analysis functionality including
import statements, selectors, CSS frameworks (Tailwind, UnoCSS), 
preprocessors (SCSS/Sass), methodologies (BEM, OOCSS), and modern CSS features.

Test Coverage:
    - Import extraction (@import, @use, @forward, url())
    - Export detection (classes, IDs, custom properties, mixins)
    - Structure extraction (rules, media queries, keyframes)
    - Complexity metrics (specificity, nesting, performance)
    - Framework detection (Tailwind, UnoCSS, Bootstrap, etc.)
    - CSS methodologies (BEM, OOCSS, SMACSS, Atomic)
    - Preprocessor features (SCSS variables, mixins, functions)
    - Modern CSS features (custom properties, grid, flexbox)
    - PostCSS and CSS Modules
    - Error handling for malformed CSS
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.css_analyzer import CSSAnalyzer, CSSParser
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestCSSAnalyzerInitialization:
    """Test suite for CSSAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = CSSAnalyzer()

        assert analyzer.language_name == "css"
        assert ".css" in analyzer.file_extensions
        assert ".scss" in analyzer.file_extensions
        assert ".sass" in analyzer.file_extensions
        assert ".less" in analyzer.file_extensions
        assert analyzer.logger is not None
        assert analyzer.tailwind_patterns is not None
        assert analyzer.unocss_patterns is not None


class TestImportExtraction:
    """Test suite for CSS import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSSAnalyzer instance."""
        return CSSAnalyzer()

    def test_extract_css_imports(self, analyzer):
        """Test extraction of @import statements."""
        css = """
@import url("reset.css");
@import "variables.css";
@import 'components/button.css';
@import url("https://fonts.googleapis.com/css2?family=Roboto");
@import "print.css" print;
@import "mobile.css" screen and (max-width: 768px);
"""
        imports = analyzer.extract_imports(css, Path("test.css"))

        assert len(imports) == 6

        # Check local import
        reset_import = next(imp for imp in imports if "reset.css" in imp.module)
        assert reset_import.is_relative is True
        assert reset_import.category == "reset"

        # Check external import
        fonts_import = next(imp for imp in imports if "fonts.googleapis.com" in imp.module)
        assert fonts_import.is_relative is False
        assert fonts_import.category == "external"

        # Check media query import
        mobile_import = next(imp for imp in imports if "mobile.css" in imp.module)
        assert mobile_import.media_query == "screen and (max-width: 768px)"

    def test_extract_scss_imports(self, analyzer):
        """Test extraction of SCSS @use and @forward statements."""
        scss = """
@use 'sass:math';
@use 'sass:color';
@use 'variables' as vars;
@use 'mixins' with (
  $base-color: #036,
  $border-radius: 4px
);
@forward 'buttons';
@forward 'forms' show $form-padding, form-background;
@forward 'typography' hide $font-size;
"""
        imports = analyzer.extract_imports(scss, Path("test.scss"))

        use_imports = [imp for imp in imports if imp.type == "use"]
        assert len(use_imports) == 4

        # Check namespaced import
        vars_import = next(imp for imp in use_imports if imp.module == "variables")
        assert vars_import.namespace == "vars"

        # Check configured import
        mixins_import = next(imp for imp in use_imports if imp.module == "mixins")
        assert mixins_import.config is not None

        # Check forward statements
        forward_imports = [imp for imp in imports if imp.type == "forward"]
        assert len(forward_imports) == 3

        forms_forward = next(imp for imp in forward_imports if imp.module == "forms")
        assert "show" in forms_forward.visibility

    def test_extract_url_imports(self, analyzer):
        """Test extraction of url() imports."""
        css = """
@font-face {
    font-family: 'MyFont';
    src: url('fonts/myfont.woff2') format('woff2'),
         url('fonts/myfont.woff') format('woff');
}

.hero {
    background-image: url('../images/hero.jpg');
}

.icon::before {
    content: url('data:image/svg+xml;utf8,<svg>...</svg>');
}

.video {
    background: url(https://example.com/video-poster.jpg);
}
"""
        imports = analyzer.extract_imports(css, Path("test.css"))

        url_imports = [imp for imp in imports if imp.type == "url"]
        # Should not include data URLs
        assert len(url_imports) == 4

        # Check font import
        font_import = next(imp for imp in url_imports if "myfont.woff2" in imp.module)
        assert font_import.category == "font"

        # Check image import
        image_import = next(imp for imp in url_imports if "hero.jpg" in imp.module)
        assert image_import.category == "image"

    def test_extract_css_modules_composes(self, analyzer):
        """Test extraction of CSS Modules composes."""
        css = """
.button {
    composes: base from './base.css';
    composes: primary from './colors.css';
    background: blue;
}

.alert {
    composes: message from 'shared/messages.css';
}
"""
        imports = analyzer.extract_imports(css, Path("test.module.css"))

        composes_imports = [imp for imp in imports if imp.type == "composes"]
        assert len(composes_imports) == 3

        base_import = next(imp for imp in composes_imports if "base.css" in imp.module)
        assert base_import.composed_classes == "base"
        assert base_import.category == "css_module"


class TestExportExtraction:
    """Test suite for CSS export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSSAnalyzer instance."""
        return CSSAnalyzer()

    def test_extract_css_classes(self, analyzer):
        """Test extraction of CSS classes."""
        css = """
.button {
    padding: 10px 20px;
}

.btn-primary {
    background: blue;
}

.card__header {
    font-size: 18px;
}

.is-active {
    display: block;
}
"""
        exports = analyzer.extract_exports(css, Path("test.css"))

        class_exports = [exp for exp in exports if exp["type"] == "class"]
        assert len(class_exports) == 4

        assert any(exp["name"] == "button" for exp in class_exports)
        assert any(exp["name"] == "btn-primary" for exp in class_exports)
        assert any(exp["name"] == "card__header" for exp in class_exports)

    def test_extract_css_ids(self, analyzer):
        """Test extraction of CSS IDs."""
        css = """
#header {
    height: 60px;
}

#main-content {
    padding: 20px;
}

#footer {
    background: #333;
}
"""
        exports = analyzer.extract_exports(css, Path("test.css"))

        id_exports = [exp for exp in exports if exp["type"] == "id"]
        assert len(id_exports) == 3

        assert any(exp["name"] == "header" for exp in id_exports)
        assert any(exp["name"] == "main-content" for exp in id_exports)

    def test_extract_custom_properties(self, analyzer):
        """Test extraction of CSS custom properties."""
        css = """
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --spacing-unit: 8px;
    --font-size-base: 16px;
}

.component {
    --component-padding: 10px;
    padding: var(--component-padding);
}
"""
        exports = analyzer.extract_exports(css, Path("test.css"))

        custom_props = [exp for exp in exports if exp["type"] == "custom_property"]
        assert len(custom_props) >= 4

        primary_color = next((exp for exp in custom_props if exp["name"] == "--primary-color"), None)
        assert primary_color is not None
        assert primary_color["value"] == "#007bff"

    def test_extract_scss_variables_and_mixins(self, analyzer):
        """Test extraction of SCSS variables and mixins."""
        scss = """
$primary-color: #007bff;
$secondary-color: #6c757d;
$base-font-size: 16px;

@mixin button-style($bg-color: $primary-color) {
    background: $bg-color;
    padding: 10px 20px;
    border-radius: 4px;
}

@mixin flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

@function rem($px) {
    @return $px / 16px * 1rem;
}
"""
        exports = analyzer.extract_exports(scss, Path("test.scss"))

        # Check SCSS variables
        scss_vars = [exp for exp in exports if exp["type"] == "scss_variable"]
        assert len(scss_vars) == 3

        primary_var = next((exp for exp in scss_vars if exp["name"] == "$primary-color"), None)
        assert primary_var is not None
        assert primary_var["value"] == "#007bff"

        # Check mixins
        mixins = [exp for exp in exports if exp["type"] == "mixin"]
        assert len(mixins) == 2

        button_mixin = next((exp for exp in mixins if exp["name"] == "button-style"), None)
        assert button_mixin is not None
        assert button_mixin["params"] is not None

        # Check functions
        functions = [exp for exp in exports if exp["type"] == "function"]
        assert len(functions) == 1

    def test_extract_keyframes(self, analyzer):
        """Test extraction of keyframe animations."""
        css = """
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@-webkit-keyframes slideIn {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(0); }
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""
        exports = analyzer.extract_exports(css, Path("test.css"))

        keyframes = [exp for exp in exports if exp["type"] == "keyframe"]
        assert len(keyframes) == 3

        assert any(exp["name"] == "fadeIn" for exp in keyframes)
        assert any(exp["name"] == "slideIn" for exp in keyframes)
        assert any(exp["name"] == "spin" for exp in keyframes)

    def test_extract_utility_classes(self, analyzer):
        """Test extraction of utility classes (Tailwind/UnoCSS style)."""
        css = """
.p-4 { padding: 1rem; }
.m-2 { margin: 0.5rem; }
.text-center { text-align: center; }
.bg-blue-500 { background-color: rgb(59 130 246); }
.hover\\:bg-blue-600:hover { background-color: rgb(37 99 235); }
.sm\\:flex { display: flex; }
"""
        exports = analyzer.extract_exports(css, Path("tailwind.css"))

        # Should detect these as utility classes
        classes = [exp for exp in exports if exp["type"] == "class"]
        assert len(classes) >= 6


class TestStructureExtraction:
    """Test suite for CSS structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSSAnalyzer instance."""
        return CSSAnalyzer()

    def test_extract_media_queries(self, analyzer):
        """Test extraction of media queries."""
        css = """
@media screen and (min-width: 768px) {
    .container {
        max-width: 768px;
    }
}

@media (min-width: 1024px) and (max-width: 1280px) {
    .container {
        max-width: 1024px;
    }
}

@media print {
    .no-print {
        display: none;
    }
}

@media (prefers-color-scheme: dark) {
    body {
        background: #000;
        color: #fff;
    }
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.media_query_count == 4
        assert len(structure.media_queries) == 4

        # Check for dark mode support
        assert structure.color_scheme > 0

    def test_extract_css_grid_and_flexbox(self, analyzer):
        """Test extraction of CSS Grid and Flexbox usage."""
        css = """
.grid-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-gap: 20px;
}

.flex-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.inline-flex {
    display: inline-flex;
    flex-direction: column;
}

.grid-item {
    display: grid;
    place-items: center;
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.grid_usage == 2
        assert structure.flexbox_usage == 2

    def test_detect_bem_methodology(self, analyzer):
        """Test detection of BEM methodology."""
        css = """
.block {
    padding: 20px;
}

.block__element {
    margin: 10px;
}

.block__element--modifier {
    color: red;
}

.another-block__element {
    display: flex;
}

.another-block--large {
    font-size: 20px;
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.uses_bem is True

    def test_detect_tailwind_css(self, analyzer):
        """Test detection of Tailwind CSS."""
        css = """
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer components {
    .btn-primary {
        @apply py-2 px-4 bg-blue-500 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700;
    }
}

.sm\\:text-center {
    text-align: center;
}

.hover\\:bg-gray-100:hover {
    background-color: rgb(243 244 246);
}
"""
        structure = analyzer.extract_structure(css, Path("tailwind.css"))

        assert structure.is_tailwind is True

    def test_detect_unocss(self, analyzer):
        """Test detection of UnoCSS."""
        css = """
/* layer: default */
.p-4 { padding: 1rem; }
.m-2 { margin: 0.5rem; }

/* layer: shortcuts */
.flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

.text-red { color: rgb(248 113 113); }
.hover\\:text-blue:hover { color: rgb(96 165 250); }
"""
        # UnoCSS detection might need file name hints
        structure = analyzer.extract_structure(css, Path("uno.css"))

        # Should detect atomic CSS patterns
        assert structure.uses_atomic is True

    def test_detect_bootstrap(self, analyzer):
        """Test detection of Bootstrap."""
        css = """
.btn {
    display: inline-block;
    padding: 0.375rem 0.75rem;
}

.btn-primary {
    color: #fff;
    background-color: #0d6efd;
}

.container {
    width: 100%;
    max-width: 1140px;
    margin: 0 auto;
}

.row {
    display: flex;
    flex-wrap: wrap;
}

.col-md-6 {
    flex: 0 0 50%;
    max-width: 50%;
}

.navbar {
    position: relative;
    display: flex;
}

.card {
    position: relative;
    display: flex;
    flex-direction: column;
}
"""
        structure = analyzer.extract_structure(css, Path("bootstrap.css"))

        assert structure.is_bootstrap is True

    def test_extract_css3_features(self, analyzer):
        """Test extraction of modern CSS3 features."""
        css = """
.element {
    /* Custom properties */
    --custom-color: #333;
    color: var(--custom-color);
    
    /* Calc */
    width: calc(100% - 20px);
    
    /* Transforms */
    transform: rotate(45deg) scale(1.2);
    
    /* Transitions */
    transition: all 0.3s ease-in-out;
    
    /* Animations */
    animation: slide 2s infinite;
    
    /* Viewport units */
    height: 100vh;
    width: 50vw;
    font-size: 5vmin;
}

/* Container queries */
@container (min-width: 400px) {
    .card {
        display: grid;
    }
}

/* CSS Layers */
@layer utilities {
    .text-center {
        text-align: center;
    }
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.custom_property_usage > 0
        assert structure.calc_usage == 1
        assert structure.transform_usage == 1
        assert structure.transition_usage == 1
        assert structure.animation_usage == 1
        assert structure.viewport_units >= 3
        assert structure.container_queries == 1
        assert structure.has_layers is True

    def test_detect_design_system(self, analyzer):
        """Test detection of design system patterns."""
        css = """
:root {
    /* Colors */
    --color-primary: #007bff;
    --color-secondary: #6c757d;
    --color-success: #28a745;
    --color-danger: #dc3545;
    
    /* Spacing */
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
    
    /* Typography */
    --font-family-base: system-ui, -apple-system, sans-serif;
    --font-size-base: 16px;
    --font-size-lg: 20px;
    --font-weight-normal: 400;
    --font-weight-bold: 700;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
}
"""
        structure = analyzer.extract_structure(css, Path("design-tokens.css"))

        assert structure.has_design_tokens is True
        assert structure.color_variables >= 4
        assert structure.spacing_variables >= 5
        assert structure.typography_variables >= 5

    def test_extract_scss_nesting(self, analyzer):
        """Test extraction of SCSS nesting depth."""
        scss = """
.component {
    padding: 20px;
    
    .header {
        font-size: 24px;
        
        .title {
            font-weight: bold;
            
            &:hover {
                color: blue;
                
                .icon {
                    transform: rotate(180deg);
                }
            }
        }
    }
    
    &.is-active {
        background: #f0f0f0;
    }
}
"""
        structure = analyzer.extract_structure(scss, Path("test.scss"))

        assert structure.max_nesting >= 4
        assert structure.css_nesting > 0

    def test_detect_postcss(self, analyzer):
        """Test detection of PostCSS features."""
        css = """
/* PostCSS custom media */
@custom-media --small-viewport (max-width: 30em);

/* PostCSS nesting */
.component {
    & .nested {
        color: blue;
    }
}

/* PostCSS custom selectors */
@custom-selector :--heading h1, h2, h3, h4, h5, h6;

:--heading {
    font-weight: bold;
}

/* Nested at-rules */
.component {
    @nest .parent & {
        color: red;
    }
}
"""
        structure = analyzer.extract_structure(css, Path("postcss.css"))

        assert structure.uses_postcss is True
        assert "postcss-custom-media" in structure.postcss_plugins
        assert "postcss-nested" in structure.postcss_plugins

    def test_detect_css_modules(self, analyzer):
        """Test detection of CSS Modules."""
        css = """
.button {
    composes: base from './base.css';
    background: blue;
    padding: 10px;
}

.primary {
    composes: button;
    background: green;
}

.warning {
    composes: button attention from './shared.css';
    background: orange;
}
"""
        structure = analyzer.extract_structure(css, Path("component.module.css"))

        assert structure.is_css_modules is True


class TestComplexityCalculation:
    """Test suite for CSS complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSSAnalyzer instance."""
        return CSSAnalyzer()

    def test_calculate_specificity(self, analyzer):
        """Test CSS specificity calculation."""
        css = """
/* Low specificity */
div { color: black; }
.class { color: blue; }

/* Medium specificity */
div.class { color: green; }
.class1.class2 { color: yellow; }
div[data-attr] { color: purple; }

/* High specificity */
#id { color: red; }
div#id.class { color: orange; }

/* Very high specificity */
#id1#id2 { color: pink; }
body #content .article h1.title { color: brown; }
"""
        metrics = analyzer.calculate_complexity(css, Path("test.css"))

        assert metrics.total_rules > 0
        assert metrics.max_specificity[0] >= 1  # At least one ID
        assert metrics.avg_specificity[0] >= 0  # Should have average

    def test_calculate_selector_complexity(self, analyzer):
        """Test selector complexity calculation."""
        css = """
/* Simple selectors */
.button { }
#header { }

/* Complex selectors */
body > div.container main article.post > header h1.title span { }
.nav ul li a:hover::after { }

/* Overqualified selectors */
div.container { }
ul.list { }
button#submit { }
"""
        metrics = analyzer.calculate_complexity(css, Path("test.css"))

        assert metrics.complex_selectors >= 2
        assert metrics.overqualified_selectors >= 3

    def test_calculate_important_usage(self, analyzer):
        """Test !important usage metrics."""
        css = """
.normal {
    color: blue;
    padding: 10px;
}

.override {
    color: red !important;
    padding: 20px !important;
    margin: 10px !important;
}

.critical {
    display: none !important;
}
"""
        metrics = analyzer.calculate_complexity(css, Path("test.css"))

        assert metrics.important_count == 4

    def test_calculate_media_query_complexity(self, analyzer):
        """Test media query complexity calculation."""
        css = """
/* Simple media query */
@media screen {
    .container { width: 100%; }
}

/* Complex media query */
@media screen and (min-width: 768px) and (max-width: 1024px) and (orientation: landscape) {
    .container { width: 1024px; }
}

/* Multiple conditions */
@media (min-width: 768px) and (max-width: 1024px),
       (min-width: 1280px) and (max-width: 1920px) {
    .container { width: 90%; }
}
"""
        metrics = analyzer.calculate_complexity(css, Path("test.css"))

        assert metrics.media_query_count >= 3
        assert metrics.media_query_complexity >= 5  # Total 'and' conditions

    def test_calculate_color_and_font_metrics(self, analyzer):
        """Test color and font usage metrics."""
        css = """
.element1 {
    color: #333;
    background: #fff;
    border-color: rgb(200, 200, 200);
}

.element2 {
    color: #333; /* Duplicate */
    background: rgba(0, 0, 0, 0.5);
    border-color: hsl(120, 100%, 50%);
}

.text1 {
    font-family: Arial, sans-serif;
}

.text2 {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

.text3 {
    font-family: Georgia, serif;
}
"""
        metrics = analyzer.calculate_complexity(css, Path("test.css"))

        assert metrics.unique_colors >= 5  # Should not count duplicate #333
        assert metrics.unique_fonts == 3

    def test_calculate_z_index_metrics(self, analyzer):
        """Test z-index usage metrics."""
        css = """
.modal-backdrop {
    z-index: 1000;
}

.modal {
    z-index: 1001;
}

.dropdown {
    z-index: 100;
}

.tooltip {
    z-index: 9999;
}

.negative {
    z-index: -1;
}
"""
        metrics = analyzer.calculate_complexity(css, Path("test.css"))

        assert metrics.z_index_count == 5
        assert metrics.max_z_index == 9999

    def test_calculate_scss_nesting_depth(self, analyzer):
        """Test SCSS nesting depth calculation."""
        scss = """
.level1 {
    .level2 {
        .level3 {
            .level4 {
                .level5 {
                    color: red;
                }
            }
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(scss, Path("test.scss"))

        assert metrics.max_nesting_depth >= 5

    def test_calculate_performance_score(self, analyzer):
        """Test performance score calculation."""
        # Good CSS
        good_css = """
:root {
    --primary: #007bff;
    --spacing: 8px;
}

.button {
    background: var(--primary);
    padding: var(--spacing);
}

.card {
    display: flex;
}
"""
        good_metrics = analyzer.calculate_complexity(good_css, Path("good.css"))

        # Poor CSS
        poor_css = """
#id1#id2#id3 { color: red !important; }
div.class1.class2.class3.class4 { padding: 10px !important; }
body > div > div > div > div > div > span { margin: 5px !important; }

.deeply {
    .nested {
        .selector {
            .structure {
                .here {
                    color: blue !important;
                }
            }
        }
    }
}
"""
        poor_metrics = analyzer.calculate_complexity(poor_css, Path("poor.scss"))

        assert good_metrics.performance_score > poor_metrics.performance_score

    def test_calculate_tailwind_metrics(self, analyzer):
        """Test Tailwind CSS specific metrics."""
        css = """
.p-4 { padding: 1rem; }
.m-2 { margin: 0.5rem; }
.flex { display: flex; }
.items-center { align-items: center; }
.justify-between { justify-content: space-between; }
.bg-blue-500 { background-color: rgb(59 130 246); }
.text-white { color: rgb(255 255 255); }
.hover\\:bg-blue-600:hover { background-color: rgb(37 99 235); }
.sm\\:flex { display: flex; }
.md\\:hidden { display: none; }

/* Custom utilities */
.u-text-truncate {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
"""
        metrics = analyzer.calculate_complexity(css, Path("tailwind.css"))

        # Should detect Tailwind classes
        assert hasattr(metrics, 'tailwind_classes')
        if hasattr(metrics, 'tailwind_classes'):
            assert metrics.tailwind_classes > 0

        assert hasattr(metrics, 'custom_utilities')
        if hasattr(metrics, 'custom_utilities'):
            assert metrics.custom_utilities >= 1


class TestCSSParser:
    """Test suite for CSSParser."""

    def test_parse_css_rules(self):
        """Test parsing of CSS rules."""
        css = """
.class1 {
    color: red;
    padding: 10px;
}

#id1 {
    background: blue;
    margin: 20px;
}

div.complex[data-attr="value"] {
    display: flex;
}
"""
        parser = CSSParser(css)
        parser.parse()

        assert len(parser.rules) >= 3
        
        # Check first rule
        first_rule = parser.rules[0]
        assert first_rule['selector'] == '.class1'
        assert len(first_rule['properties']) == 2

    def test_parse_custom_properties(self):
        """Test parsing of CSS custom properties."""
        css = """
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --spacing: 8px;
}

.component {
    --local-padding: 10px;
}
"""
        parser = CSSParser(css)
        parser.parse()

        assert len(parser.custom_properties) >= 4
        assert parser.custom_properties.get('--primary-color') == '#007bff'
        assert parser.custom_properties.get('--spacing') == '8px'

    def test_parse_scss_features(self):
        """Test parsing of SCSS features."""
        scss = """
$primary: #007bff;
$secondary: #6c757d;

@mixin button {
    padding: 10px 20px;
    border-radius: 4px;
}

@function rem($px) {
    @return $px / 16px * 1rem;
}
"""
        parser = CSSParser(scss, is_scss=True)
        parser.parse()

        assert len(parser.variables) == 2
        assert parser.variables.get('$primary') == '#007bff'

        assert len(parser.mixins) == 1
        assert parser.mixins[0]['name'] == 'button'

        assert len(parser.functions) == 1
        assert parser.functions[0]['name'] == 'rem'

    def test_parse_media_queries(self):
        """Test parsing of media queries."""
        css = """
@media screen and (min-width: 768px) {
    .container {
        max-width: 768px;
    }
}

@media print {
    .no-print {
        display: none;
    }
}
"""
        parser = CSSParser(css)
        parser.parse()

        assert len(parser.media_queries) == 2
        assert parser.media_queries[0]['condition'] == 'screen and (min-width: 768px)'

    def test_parse_keyframes(self):
        """Test parsing of keyframe animations."""
        css = """
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@-webkit-keyframes slideIn {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(0); }
}
"""
        parser = CSSParser(css)
        parser.parse()

        assert len(parser.keyframes) == 2
        assert parser.keyframes[0]['name'] == 'fadeIn'
        assert parser.keyframes[1]['name'] == 'slideIn'

    def test_calculate_specificity(self):
        """Test specificity calculation."""
        parser = CSSParser("")
        
        # Test various selectors
        assert parser._calculate_specificity('div') == (0, 0, 1)
        assert parser._calculate_specificity('.class') == (0, 1, 0)
        assert parser._calculate_specificity('#id') == (1, 0, 0)
        assert parser._calculate_specificity('div.class') == (0, 1, 1)
        assert parser._calculate_specificity('#id.class') == (1, 1, 0)
        assert parser._calculate_specificity('div#id.class') == (1, 1, 1)
        assert parser._calculate_specificity('[data-attr]') == (0, 1, 0)
        assert parser._calculate_specificity(':hover') == (0, 1, 0)


class TestErrorHandling:
    """Test suite for error handling in CSS analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSSAnalyzer instance."""
        return CSSAnalyzer()

    def test_handle_malformed_css(self, analyzer):
        """Test handling of malformed CSS."""
        css = """
.broken {
    color: red
    padding: 10px  /* Missing semicolon */
    
.unclosed {
    margin: 20px;
    /* Missing closing brace */

@media screen {
    .test {
        color: blue;
    /* Missing closing braces */
"""
        # Should not raise exception
        imports = analyzer.extract_imports(css, Path("test.css"))
        exports = analyzer.extract_exports(css, Path("test.css"))
        structure = analyzer.extract_structure(css, Path("test.css"))
        metrics = analyzer.calculate_complexity(css, Path("test.css"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty file."""
        css = ""

        imports = analyzer.extract_imports(css, Path("test.css"))
        exports = analyzer.extract_exports(css, Path("test.css"))
        structure = analyzer.extract_structure(css, Path("test.css"))
        metrics = analyzer.calculate_complexity(css, Path("test.css"))

        assert imports == []
        assert exports == []
        assert metrics.line_count == 1


class TestEdgeCases:
    """Test suite for edge cases in CSS analysis."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSSAnalyzer instance."""
        return CSSAnalyzer()

    def test_handle_css_in_js_patterns(self, analyzer):
        """Test handling of CSS-in-JS generated classes."""
        css = """
/* Styled-components generated */
.sc-bdVaJa { color: red; }
.sc-bwzfXH { padding: 10px; }

/* Emotion generated */
.css-1a2b3c { display: flex; }
.css-4d5e6f { margin: 20px; }

/* CSS Modules */
.Component_root_3FI2x { background: blue; }
.Component_active_1a2b3 { color: white; }
"""
        structure = analyzer.extract_structure(css, Path("generated.css"))

        # Should detect styled-components patterns
        assert structure.is_styled_components is True

    def test_handle_vendor_prefixes(self, analyzer):
        """Test handling of vendor prefixes."""
        css = """
.element {
    -webkit-transform: rotate(45deg);
    -moz-transform: rotate(45deg);
    -ms-transform: rotate(45deg);
    -o-transform: rotate(45deg);
    transform: rotate(45deg);
    
    -webkit-box-shadow: 0 0 10px rgba(0,0,0,0.5);
    -moz-box-shadow: 0 0 10px rgba(0,0,0,0.5);
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.vendor_prefixes >= 6

    def test_handle_css_hacks(self, analyzer):
        """Test handling of CSS hacks."""
        css = """
/* IE6 hack */
* html .ie6 { color: red; }

/* IE7 hack */
*:first-child+html .ie7 { color: blue; }

/* IE8 hack */
.ie8 { color: green\\9; }

/* Modern CSS */
@supports (display: grid) {
    .grid { display: grid; }
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert len(structure.supports_rules) >= 1

    def test_handle_css_variables_fallback(self, analyzer):
        """Test handling of CSS variables with fallbacks."""
        css = """
.element {
    color: var(--primary-color, #007bff);
    padding: var(--spacing, 10px);
    margin: var(--unknown, var(--fallback, 20px));
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.custom_property_usage >= 3

    def test_handle_mixed_syntax(self, analyzer):
        """Test handling of mixed CSS/SCSS syntax."""
        scss = """
/* Regular CSS */
.regular {
    color: blue;
}

/* SCSS variables */
$primary: #007bff;

/* SCSS nesting */
.component {
    padding: 20px;
    
    &:hover {
        background: $primary;
    }
    
    .nested {
        margin: 10px;
    }
}

/* CSS custom properties */
:root {
    --spacing: 8px;
}

/* Using both */
.mixed {
    padding: var(--spacing);
    color: $primary;
}
"""
        structure = analyzer.extract_structure(scss, Path("test.scss"))

        assert len(structure.variables) >= 1
        assert len(structure.custom_properties) >= 1
        assert structure.css_nesting > 0

    def test_handle_calc_and_functions(self, analyzer):
        """Test handling of calc() and other CSS functions."""
        css = """
.element {
    width: calc(100% - 20px);
    height: calc(100vh - var(--header-height));
    padding: min(10px, 2vw);
    margin: max(5px, 1vh);
    font-size: clamp(14px, 2vw, 20px);
    
    /* Color functions */
    color: rgb(255, 0, 0);
    background: rgba(0, 0, 255, 0.5);
    border-color: hsl(120, 100%, 50%);
    box-shadow: 0 0 10px hsla(0, 0%, 0%, 0.5);
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.calc_usage >= 2

    def test_handle_container_queries(self, analyzer):
        """Test handling of container queries."""
        css = """
.container {
    container-type: inline-size;
    container-name: card;
}

@container (min-width: 400px) {
    .card {
        display: grid;
        grid-template-columns: 1fr 2fr;
    }
}

@container card (min-width: 600px) {
    .card {
        grid-template-columns: 1fr 1fr 1fr;
    }
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.container_queries >= 2

    def test_handle_cascade_layers(self, analyzer):
        """Test handling of CSS cascade layers."""
        css = """
@layer reset, base, components, utilities;

@layer reset {
    * {
        margin: 0;
        padding: 0;
    }
}

@layer base {
    body {
        font-family: system-ui;
    }
}

@layer components {
    .button {
        padding: 10px 20px;
    }
}

@layer utilities {
    .text-center {
        text-align: center;
    }
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.has_layers is True
        assert structure.has_cascade_layers >= 4

    def test_handle_accessibility_features(self, analyzer):
        """Test handling of accessibility-related CSS."""
        css = """
/* Focus styles */
.button:focus {
    outline: 2px solid blue;
}

.link:focus-visible {
    outline: 3px solid orange;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* High contrast */
@media (prefers-contrast: high) {
    .element {
        border: 2px solid;
    }
}

/* Color scheme */
@media (prefers-color-scheme: dark) {
    body {
        background: #000;
        color: #fff;
    }
}
"""
        structure = analyzer.extract_structure(css, Path("test.css"))

        assert structure.focus_styles >= 1
        assert structure.focus_visible >= 1
        assert structure.reduced_motion >= 1
        assert structure.high_contrast >= 1
        assert structure.color_scheme >= 1