"""
Unit tests for the HTML code analyzer with modern web framework support.

This module tests the HTML-specific code analysis functionality including
resource imports, semantic elements, accessibility features,
web components, framework detection, and modern HTML5 features.

Test Coverage:
    - Import extraction (CSS, JS, modules, preload/prefetch)
    - Export detection (IDs, custom elements, data attributes)
    - Structure extraction (semantic HTML5, forms, ARIA)
    - Complexity metrics (DOM depth, accessibility, SEO, performance)
    - Framework detection (React, Vue, Angular, Svelte)
    - Web components and custom elements
    - Security features (CSP, integrity checks)
    - Error handling for malformed HTML
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.html_analyzer import HTMLAnalyzer, HTMLStructureParser
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestHTMLAnalyzerInitialization:
    """Test suite for HTMLAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = HTMLAnalyzer()

        assert analyzer.language_name == "html"
        assert ".html" in analyzer.file_extensions
        assert ".htm" in analyzer.file_extensions
        assert ".vue" in analyzer.file_extensions
        assert ".jsx" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for HTML import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide an HTMLAnalyzer instance."""
        return HTMLAnalyzer()

    def test_extract_css_imports(self, analyzer):
        """Test extraction of CSS stylesheet imports."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="styles/main.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="//fonts.googleapis.com/css?family=Roboto">
    <link rel="stylesheet" href="/assets/theme.css">
</head>
</html>
"""
        imports = analyzer.extract_imports(html, Path("test.html"))

        css_imports = [imp for imp in imports if imp.type == "stylesheet"]
        assert len(css_imports) == 4

        # Check local CSS
        local_css = next(imp for imp in css_imports if "main.css" in imp.module)
        assert local_css.is_relative is True
        assert local_css.category == "local"

        # Check Bootstrap CDN
        bootstrap_css = next(imp for imp in css_imports if "bootstrap" in imp.module)
        assert bootstrap_css.is_relative is False
        assert bootstrap_css.category == "bootstrap"

        # Check Google Fonts
        fonts_css = next(imp for imp in css_imports if "fonts.googleapis" in imp.module)
        assert fonts_css.category == "fonts"

    def test_extract_javascript_imports(self, analyzer):
        """Test extraction of JavaScript script imports."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <script src="js/app.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js" 
            integrity="sha384-vtXRMe3mGCbOeY7l30aIg8H9p3GdeSe4IFlP6G8JMa7o7lXvnz3GFKzPxzJdPfGK" 
            crossorigin="anonymous"></script>
    <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
    <script defer src="js/deferred.js"></script>
    <script type="module" src="js/module.js"></script>
</head>
</html>
"""
        imports = analyzer.extract_imports(html, Path("test.html"))

        js_imports = [imp for imp in imports if imp.type == "script"]
        assert len(js_imports) == 5

        # Check jQuery with integrity
        jquery = next(imp for imp in js_imports if "jquery" in imp.module)
        assert jquery.category == "jquery"
        assert jquery.integrity is not None
        assert jquery.crossorigin == "anonymous"

        # Check async script
        async_script = next(imp for imp in js_imports if "googletagmanager" in imp.module)
        assert async_script.is_async is True
        assert async_script.category == "analytics"

        # Check defer script
        defer_script = next(imp for imp in js_imports if "deferred.js" in imp.module)
        assert defer_script.is_defer is True

        # Check module script
        module_script = next(imp for imp in js_imports if "module.js" in imp.module)
        assert module_script.is_module is True

    def test_extract_es6_module_imports(self, analyzer):
        """Test extraction of ES6 module imports from inline scripts."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <script type="module">
        import { createApp } from 'vue';
        import App from './App.vue';
        import router from './router/index.js';
        import * as utils from '../utils/helpers.js';
        
        createApp(App).use(router).mount('#app');
    </script>
</body>
</html>
"""
        imports = analyzer.extract_imports(html, Path("test.html"))

        es6_imports = [imp for imp in imports if imp.type == "es6_module"]
        assert len(es6_imports) == 4

        vue_import = next(imp for imp in es6_imports if imp.module == "vue")
        assert vue_import.is_relative is False
        assert vue_import.category == "vue"

        app_import = next(imp for imp in es6_imports if "App.vue" in imp.module)
        assert app_import.is_relative is True

    def test_extract_preload_prefetch(self, analyzer):
        """Test extraction of preload and prefetch resources."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <link rel="preload" href="fonts/main.woff2" as="font" type="font/woff2" crossorigin>
    <link rel="prefetch" href="images/hero.jpg">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="dns-prefetch" href="//api.example.com">
</head>
</html>
"""
        imports = analyzer.extract_imports(html, Path("test.html"))

        preload = next(imp for imp in imports if imp.type == "preload")
        assert preload.as_type == "font"

        prefetch = next(imp for imp in imports if imp.type == "prefetch")
        assert "hero.jpg" in prefetch.module

        preconnect = next(imp for imp in imports if imp.type == "preconnect")
        assert "fonts.googleapis" in preconnect.module

        dns_prefetch = next(imp for imp in imports if imp.type == "dns-prefetch")
        assert "api.example.com" in dns_prefetch.module


class TestExportExtraction:
    """Test suite for HTML export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide an HTMLAnalyzer instance."""
        return HTMLAnalyzer()

    def test_extract_element_ids(self, analyzer):
        """Test extraction of element IDs."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <div id="app">App Container</div>
    <header id="main-header">Header</header>
    <nav id="navigation">Navigation</nav>
    <form id="contact-form">Form</form>
    <button id="submit-btn">Submit</button>
</body>
</html>
"""
        exports = analyzer.extract_exports(html, Path("test.html"))

        id_exports = [exp for exp in exports if exp["type"] == "element_id"]
        assert len(id_exports) == 5

        app_id = next(exp for exp in id_exports if exp["name"] == "app")
        assert app_id["tag"] == "div"

        form_id = next(exp for exp in id_exports if exp["name"] == "contact-form")
        assert form_id["tag"] == "form"

    def test_extract_custom_elements(self, analyzer):
        """Test extraction of custom elements/web components."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <my-component></my-component>
    <user-profile name="John"></user-profile>
    <shopping-cart-item></shopping-cart-item>
    
    <!-- Not custom elements -->
    <div class="not-custom"></div>
    <ng-template></ng-template>
    <v-app></v-app>
</body>
</html>
"""
        exports = analyzer.extract_exports(html, Path("test.html"))

        custom_elements = [exp for exp in exports if exp["type"] == "custom_element"]
        assert len(custom_elements) >= 3

        assert any(exp["name"] == "my-component" for exp in custom_elements)
        assert any(exp["name"] == "user-profile" for exp in custom_elements)
        assert any(exp["name"] == "shopping-cart-item" for exp in custom_elements)

    def test_extract_data_attributes(self, analyzer):
        """Test extraction of data attributes."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <div data-component="modal" data-state="closed">Modal</div>
    <button data-action="submit" data-target="form">Submit</button>
    <section data-page="home" data-analytics="true">Content</section>
</body>
</html>
"""
        exports = analyzer.extract_exports(html, Path("test.html"))

        data_attrs = [exp for exp in exports if exp["type"] == "data_attribute"]
        assert len(data_attrs) >= 6

        component_attr = next((exp for exp in data_attrs if exp["name"] == "data-component"), None)
        assert component_attr is not None
        assert "modal" in component_attr["values"]

    def test_extract_open_graph_tags(self, analyzer):
        """Test extraction of Open Graph meta tags."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta property="og:title" content="Page Title">
    <meta property="og:description" content="Page description">
    <meta property="og:image" content="https://example.com/image.jpg">
    <meta property="og:url" content="https://example.com/page">
    <meta property="og:type" content="website">
</head>
</html>
"""
        exports = analyzer.extract_exports(html, Path("test.html"))

        og_tags = [exp for exp in exports if exp["type"] == "open_graph"]
        assert len(og_tags) == 5

        title_tag = next(exp for exp in og_tags if exp["name"] == "og:title")
        assert title_tag["content"] == "Page Title"

    def test_extract_structured_data(self, analyzer):
        """Test extraction of structured data (JSON-LD)."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <script type="application/ld+json">
    {
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": "Example Page"
    }
    </script>
</head>
</html>
"""
        exports = analyzer.extract_exports(html, Path("test.html"))

        jsonld = [exp for exp in exports if exp["type"] == "json_ld"]
        assert len(jsonld) >= 1

    def test_extract_microdata(self, analyzer):
        """Test extraction of microdata."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <div itemscope itemtype="https://schema.org/Product">
        <span itemprop="name">Product Name</span>
        <span itemprop="price">$19.99</span>
    </div>
    <article itemscope itemtype="https://schema.org/Article">
        <h1 itemprop="headline">Article Title</h1>
    </article>
</body>
</html>
"""
        exports = analyzer.extract_exports(html, Path("test.html"))

        microdata = [exp for exp in exports if exp["type"] == "microdata"]
        assert len(microdata) == 2

        product = next(exp for exp in microdata if exp["name"] == "Product")
        assert "schema.org/Product" in product["schema"]


class TestStructureExtraction:
    """Test suite for HTML structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide an HTMLAnalyzer instance."""
        return HTMLAnalyzer()

    def test_extract_html5_semantic_elements(self, analyzer):
        """Test extraction of HTML5 semantic elements."""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Page</title>
</head>
<body>
    <header>
        <nav>Navigation</nav>
    </header>
    <main>
        <article>
            <section>Section 1</section>
            <section>Section 2</section>
        </article>
        <aside>Sidebar</aside>
    </main>
    <footer>Footer</footer>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.has_doctype is True
        assert structure.is_html5 is True
        assert structure.language == "en"
        assert structure.charset == "UTF-8"
        assert structure.viewport is not None
        assert structure.is_responsive is True

        assert hasattr(structure, "header_count") and structure.header_count == 1
        assert hasattr(structure, "nav_count") and structure.nav_count == 1
        assert hasattr(structure, "main_count") and structure.main_count == 1
        assert hasattr(structure, "article_count") and structure.article_count == 1
        assert hasattr(structure, "section_count") and structure.section_count == 2
        assert hasattr(structure, "aside_count") and structure.aside_count == 1
        assert hasattr(structure, "footer_count") and structure.footer_count == 1

    def test_extract_forms_and_inputs(self, analyzer):
        """Test extraction of forms and input elements."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <form id="login-form" action="/login" method="post">
        <input type="text" name="username" required>
        <input type="password" name="password" required>
        <input type="email" name="email" pattern="[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}$">
        <select name="country">
            <option value="us">United States</option>
            <option value="uk">United Kingdom</option>
        </select>
        <textarea name="comments"></textarea>
        <button type="submit">Login</button>
    </form>
    
    <form id="search-form">
        <input type="search" name="q">
        <button type="submit">Search</button>
    </form>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.form_count == 2
        assert structure.input_count == 5
        assert structure.button_count == 2
        assert structure.select_count == 1
        assert structure.textarea_count == 1

        assert len(structure.forms) == 2
        login_form = structure.forms[0]
        assert len(login_form["inputs"]) == 6  # All form controls

    def test_extract_media_elements(self, analyzer):
        """Test extraction of media elements."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <img src="image1.jpg" alt="Description">
    <img src="image2.jpg" alt="Another image">
    <img src="image3.jpg" loading="lazy">
    
    <video controls>
        <source src="movie.mp4" type="video/mp4">
    </video>
    
    <audio controls>
        <source src="audio.mp3" type="audio/mpeg">
    </audio>
    
    <canvas id="myCanvas"></canvas>
    
    <svg width="100" height="100">
        <circle cx="50" cy="50" r="40"/>
    </svg>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.img_count == 3
        assert structure.video_count == 1
        assert structure.audio_count == 1
        assert structure.canvas_count == 1
        assert structure.svg_count == 1
        assert structure.lazy_loading == 1

    def test_detect_react_framework(self, analyzer):
        """Test detection of React framework."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <script crossorigin src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
</head>
<body>
    <div id="root"></div>
    <script>
        ReactDOM.render(
            React.createElement('h1', null, 'Hello, world!'),
            document.getElementById('root')
        );
    </script>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.is_react is True

    def test_detect_vue_framework(self, analyzer):
        """Test detection of Vue.js framework."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/vue@2.6.14"></script>
</head>
<body>
    <div id="app">
        {{ message }}
        <button v-on:click="reverseMessage">Reverse</button>
        <input v-model="message">
        <ul>
            <li v-for="item in items" :key="item.id">{{ item.text }}</li>
        </ul>
    </div>
    <script>
        new Vue({
            el: '#app',
            data: {
                message: 'Hello Vue!'
            }
        });
    </script>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.is_vue is True

    def test_detect_angular_framework(self, analyzer):
        """Test detection of Angular framework."""
        html = """
<!DOCTYPE html>
<html ng-app="myApp">
<head>
    <script src="https://ajax.googleapis.com/ajax/libs/angularjs/1.8.2/angular.min.js"></script>
</head>
<body>
    <div ng-controller="myCtrl">
        <input [(ngModel)]="name">
        <p>Hello {{name}}</p>
        <button (click)="handleClick()">Click</button>
        <div *ngIf="showContent">Content</div>
        <li *ngFor="let item of items">{{item}}</li>
    </div>
    <app-root></app-root>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.is_angular is True

    def test_extract_aria_attributes(self, analyzer):
        """Test extraction of ARIA attributes for accessibility."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <nav role="navigation" aria-label="Main navigation">
        <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/about">About</a></li>
        </ul>
    </nav>
    
    <main role="main" aria-labelledby="page-title">
        <h1 id="page-title">Page Title</h1>
        <button aria-label="Close dialog" aria-describedby="close-help">X</button>
        <div id="close-help">Press escape to close</div>
    </main>
    
    <div role="alert" aria-live="polite">
        Status message
    </div>
    
    <form>
        <label for="email">Email:</label>
        <input id="email" type="email" aria-required="true">
    </form>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.aria_labels == 2
        assert structure.aria_labelledby == 1
        assert structure.aria_describedby == 1
        assert structure.aria_roles == 3
        assert structure.aria_live == 1

    def test_extract_accessibility_features(self, analyzer):
        """Test extraction of accessibility features."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <img src="image1.jpg" alt="Descriptive text">
    <img src="image2.jpg" alt="">
    <img src="image3.jpg">
    
    <form>
        <label for="name">Name:</label>
        <input id="name" type="text">
        
        <label for="email">Email:</label>
        <input id="email" type="email">
        
        <input type="checkbox" id="agree">
        <label for="agree">I agree</label>
        
        <input type="text" placeholder="No label">
    </form>
    
    <button tabindex="0">Focusable</button>
    <div tabindex="-1">Not in tab order</div>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.alt_texts == 1  # Only non-empty alt texts
        assert structure.label_for == 3
        assert structure.tabindex == 2

    def test_extract_scripts_and_styles(self, analyzer):
        """Test extraction of scripts and styles."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; }
    </style>
    <link rel="stylesheet" href="styles.css">
</head>
<body style="background: white;">
    <div style="color: red;">Inline styled</div>
    
    <script src="external.js"></script>
    <script>
        console.log('Inline script');
    </script>
    
    <button onclick="handleClick()">Click me</button>
    <input onchange="handleChange()" onkeyup="handleKeyup()">
    <div onmouseover="handleHover()" onmouseout="handleOut()">Hover</div>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.inline_scripts == 1
        assert structure.inline_styles == 1
        assert structure.inline_style_attrs == 2
        assert structure.inline_event_handlers == 5

    def test_pwa_features(self, analyzer):
        """Test detection of PWA features."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#000000">
</head>
<body>
    <script>
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
    </script>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))

        assert structure.has_manifest is True
        assert structure.has_service_worker is True
        assert structure.is_pwa is True


class TestComplexityCalculation:
    """Test suite for HTML complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide an HTMLAnalyzer instance."""
        return HTMLAnalyzer()

    def test_calculate_dom_complexity(self, analyzer):
        """Test DOM complexity calculation."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <div>
            <div>
                <div>
                    <div>Deeply nested</div>
                </div>
            </div>
        </div>
    </div>
    <section>
        <article>
            <p>Content</p>
        </article>
    </section>
</body>
</html>
"""
        metrics = analyzer.calculate_complexity(html, Path("test.html"))

        assert metrics.total_elements > 0
        assert metrics.max_depth >= 5
        assert metrics.avg_depth > 0

    def test_calculate_form_complexity(self, analyzer):
        """Test form complexity calculation."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <form>
        <input type="text" required>
        <input type="email" required pattern="[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}$">
        <input type="password" required minlength="8">
        <input type="tel" pattern="[0-9]{3}-[0-9]{3}-[0-9]{4}">
        <select required>
            <option>Option 1</option>
            <option>Option 2</option>
        </select>
        <textarea required maxlength="500"></textarea>
        <button type="submit">Submit</button>
    </form>
</body>
</html>
"""
        metrics = analyzer.calculate_complexity(html, Path("test.html"))

        assert metrics.form_complexity > 0
        # Should account for required fields and pattern validation

    def test_calculate_accessibility_score(self, analyzer):
        """Test accessibility score calculation."""
        # Good accessibility
        good_html = """
<!DOCTYPE html>
<html lang="en">
<body>
    <header>
        <nav aria-label="Main">Navigation</nav>
    </header>
    <main>
        <h1>Title</h1>
        <img src="image.jpg" alt="Description">
        <form>
            <label for="input1">Input:</label>
            <input id="input1" type="text">
        </form>
    </main>
    <footer>Footer</footer>
</body>
</html>
"""
        good_metrics = analyzer.calculate_complexity(good_html, Path("test.html"))

        # Poor accessibility
        poor_html = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <div>Navigation</div>
    </div>
    <div>
        <img src="image.jpg">
        <form>
            <input type="text" placeholder="Name">
        </form>
    </div>
</body>
</html>
"""
        poor_metrics = analyzer.calculate_complexity(poor_html, Path("test.html"))

        assert good_metrics.accessibility_score > poor_metrics.accessibility_score

    def test_calculate_seo_score(self, analyzer):
        """Test SEO score calculation."""
        # Good SEO
        good_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Page Title</title>
    <meta name="description" content="Page description">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta property="og:title" content="Page Title">
    <meta property="og:description" content="Description">
</head>
<body>
    <h1>Main Heading</h1>
    <h2>Subheading</h2>
</body>
</html>
"""
        good_metrics = analyzer.calculate_complexity(good_html, Path("test.html"))

        # Poor SEO
        poor_html = """
<html>
<body>
    <div>Content without proper structure</div>
</body>
</html>
"""
        poor_metrics = analyzer.calculate_complexity(poor_html, Path("test.html"))

        assert good_metrics.seo_score > poor_metrics.seo_score

    def test_calculate_performance_score(self, analyzer):
        """Test performance score calculation."""
        # Good performance
        good_html = """
<!DOCTYPE html>
<html>
<head>
    <script src="script.js" async></script>
    <script src="module.js" type="module"></script>
</head>
<body>
    <img src="image1.jpg" loading="lazy">
    <img src="image2.jpg" loading="lazy">
</body>
</html>
"""
        good_metrics = analyzer.calculate_complexity(good_html, Path("test.html"))

        # Poor performance
        poor_html = """
<!DOCTYPE html>
<html>
<head>
    <script src="script1.js"></script>
    <script src="script2.js"></script>
    <script src="script3.js"></script>
    <style>body { margin: 0; }</style>
    <style>div { padding: 10px; }</style>
</head>
<body style="background: white;">
    <div style="color: red;">Content</div>
    <script>console.log('inline');</script>
    <script>console.log('another');</script>
</body>
</html>
"""
        poor_metrics = analyzer.calculate_complexity(poor_html, Path("test.html"))

        assert good_metrics.performance_score > poor_metrics.performance_score

    def test_security_indicators(self, analyzer):
        """Test security indicator detection."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="Content-Security-Policy" content="default-src 'self'">
    <link rel="stylesheet" 
          href="https://cdn.example.com/style.css"
          integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC"
          crossorigin="anonymous">
    <script src="https://secure.example.com/script.js"></script>
    <script src="http://insecure.example.com/script.js"></script>
</head>
<body>
    <a href="https://secure.com">Secure</a>
    <a href="http://insecure.com">Insecure</a>
</body>
</html>
"""
        metrics = analyzer.calculate_complexity(html, Path("test.html"))

        assert metrics.has_csp is True
        assert metrics.has_integrity_checks is True
        assert metrics.has_https_links == 2
        assert metrics.has_http_links == 2

    def test_framework_specific_metrics(self, analyzer):
        """Test framework-specific metrics calculation."""
        # React
        react_html = """
<!DOCTYPE html>
<html>
<body>
    <div id="root">
        <App>
            <Header />
            <MainContent>
                {user.name}
                {items.map(item => item)}
            </MainContent>
            <Footer />
        </App>
    </div>
</body>
</html>
"""
        react_metrics = analyzer.calculate_complexity(react_html, Path("test.html"))
        assert hasattr(react_metrics, "react_components")
        assert hasattr(react_metrics, "jsx_expressions")

        # Vue
        vue_html = """
<!DOCTYPE html>
<html>
<body>
    <div id="app">
        <div v-if="show" v-for="item in items" :key="item.id">
            {{ item.name }}
            <button @click="handleClick">{{ buttonText }}</button>
        </div>
    </div>
</body>
</html>
"""
        vue_metrics = analyzer.calculate_complexity(vue_html, Path("test.html"))
        assert hasattr(vue_metrics, "vue_directives")
        assert hasattr(vue_metrics, "vue_interpolations")

        # Angular
        angular_html = """
<!DOCTYPE html>
<html>
<body>
    <div *ngFor="let item of items">
        <span [title]="item.title" (click)="selectItem(item)">
            {{item.name}}
        </span>
        <div *ngIf="item.active">Active</div>
    </div>
</body>
</html>
"""
        angular_metrics = analyzer.calculate_complexity(angular_html, Path("test.html"))
        assert hasattr(angular_metrics, "angular_directives")
        assert hasattr(angular_metrics, "angular_bindings")


class TestHTMLParser:
    """Test suite for HTMLStructureParser."""

    def test_parse_nested_structure(self):
        """Test parsing of nested HTML structure."""
        parser = HTMLStructureParser()
        html = """
<div>
    <header>
        <nav>
            <ul>
                <li><a href="/">Home</a></li>
            </ul>
        </nav>
    </header>
</div>
"""
        parser.feed(html)

        assert parser.max_depth >= 4
        assert len(parser.elements) >= 6

    def test_parse_forms(self):
        """Test parsing of form elements."""
        parser = HTMLStructureParser()
        html = """
<form id="test-form">
    <input type="text" name="username">
    <input type="password" name="password">
    <textarea name="comments"></textarea>
    <select name="country">
        <option>USA</option>
    </select>
    <button type="submit">Submit</button>
</form>
"""
        parser.feed(html)

        assert len(parser.forms) == 1
        form = parser.forms[0]
        assert form["attrs"]["id"] == "test-form"
        assert len(form["inputs"]) == 5

    def test_parse_scripts_and_links(self):
        """Test parsing of script and link tags."""
        parser = HTMLStructureParser()
        html = """
<head>
    <script src="app.js"></script>
    <script type="module" src="module.js"></script>
    <link rel="stylesheet" href="style.css">
    <link rel="preload" href="font.woff2" as="font">
    <meta name="description" content="Test">
</head>
"""
        parser.feed(html)

        assert len(parser.scripts) == 2
        assert len(parser.links) == 2
        assert len(parser.meta_tags) == 1


class TestErrorHandling:
    """Test suite for error handling in HTML analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Provide an HTMLAnalyzer instance."""
        return HTMLAnalyzer()

    def test_handle_malformed_html(self, analyzer):
        """Test handling of malformed HTML."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <p>Unclosed paragraph
        <span>Unclosed span
    </div>
    <form>
        <input type="text"
    <!-- Incomplete tag -->
</body>
"""
        # Should not raise exception
        imports = analyzer.extract_imports(html, Path("test.html"))
        exports = analyzer.extract_exports(html, Path("test.html"))
        structure = analyzer.extract_structure(html, Path("test.html"))
        metrics = analyzer.calculate_complexity(html, Path("test.html"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty file."""
        html = ""

        imports = analyzer.extract_imports(html, Path("test.html"))
        exports = analyzer.extract_exports(html, Path("test.html"))
        structure = analyzer.extract_structure(html, Path("test.html"))
        metrics = analyzer.calculate_complexity(html, Path("test.html"))

        assert imports == []
        assert exports == []
        assert structure.has_doctype is False
        assert metrics.line_count == 1


class TestEdgeCases:
    """Test suite for edge cases in HTML analysis."""

    @pytest.fixture
    def analyzer(self):
        """Provide an HTMLAnalyzer instance."""
        return HTMLAnalyzer()

    def test_handle_inline_svg(self, analyzer):
        """Test handling of inline SVG."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
        <text x="50" y="55" text-anchor="middle">SVG</text>
    </svg>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))
        assert structure.svg_count == 1

    def test_handle_template_syntax(self, analyzer):
        """Test handling of template syntax from various frameworks."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <!-- Handlebars -->
    <div>{{#if user}}{{user.name}}{{/if}}</div>
    
    <!-- EJS -->
    <div><%= user.name %></div>
    
    <!-- Pug/Jade style (in HTML context) -->
    <div class="user" data-id="#{user.id}">Content</div>
    
    <!-- JSX style -->
    <div>{user && user.name}</div>
    
    <!-- Vue -->
    <div v-text="message"></div>
    
    <!-- Angular -->
    <div [innerHTML]="htmlContent"></div>
</body>
</html>
"""
        # Should handle various template syntaxes without errors
        structure = analyzer.extract_structure(html, Path("test.html"))
        metrics = analyzer.calculate_complexity(html, Path("test.html"))

        assert structure is not None
        assert metrics is not None

    def test_handle_web_components(self, analyzer):
        """Test handling of web components with slots."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <my-element>
        <span slot="title">Title</span>
        <p slot="content">Content</p>
    </my-element>
    
    <template id="my-template">
        <style>
            :host {
                display: block;
            }
        </style>
        <slot name="title"></slot>
        <slot name="content"></slot>
    </template>
    
    <script>
        customElements.define('my-element', class extends HTMLElement {
            constructor() {
                super();
                this.attachShadow({mode: 'open'});
            }
        });
    </script>
</body>
</html>
"""
        exports = analyzer.extract_exports(html, Path("test.html"))

        custom_elements = [exp for exp in exports if exp["type"] == "custom_element"]
        assert any(exp["name"] == "my-element" for exp in custom_elements)

    def test_handle_amp_html(self, analyzer):
        """Test handling of AMP HTML."""
        html = """
<!DOCTYPE html>
<html ⚡>
<head>
    <meta charset="utf-8">
    <script async src="https://cdn.ampproject.org/v0.js"></script>
    <link rel="canonical" href="https://example.com/article">
    <meta name="viewport" content="width=device-width,minimum-scale=1,initial-scale=1">
    <style amp-boilerplate>body{-webkit-animation:-amp-start 8s steps(1,end) 0s 1 normal both}</style>
    <noscript><style amp-boilerplate>body{-webkit-animation:none}</style></noscript>
    <style amp-custom>
        body { font-family: sans-serif; }
    </style>
</head>
<body>
    <amp-img src="image.jpg" width="300" height="200" layout="responsive"></amp-img>
    <amp-video width="640" height="360" src="video.mp4" poster="poster.jpg" layout="responsive">
        <div fallback>
            <p>Your browser doesn't support HTML5 video</p>
        </div>
    </amp-video>
</body>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))
        exports = analyzer.extract_exports(html, Path("test.html"))

        # Should recognize AMP components as custom elements
        custom_elements = [exp for exp in exports if exp["type"] == "custom_element"]
        assert any("amp-" in exp["name"] for exp in custom_elements)

    def test_handle_meta_tags_variety(self, analyzer):
        """Test handling of various meta tag types."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <!-- Basic meta tags -->
    <meta charset="UTF-8">
    <meta name="description" content="Description">
    <meta name="keywords" content="keyword1, keyword2">
    <meta name="author" content="Author Name">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Open Graph -->
    <meta property="og:title" content="Title">
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://example.com">
    <meta property="og:image" content="image.jpg">
    
    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:site" content="@username">
    
    <!-- Other -->
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="robots" content="index, follow">
    <meta name="theme-color" content="#000000">
</head>
</html>
"""
        structure = analyzer.extract_structure(html, Path("test.html"))
        exports = analyzer.extract_exports(html, Path("test.html"))

        og_tags = [exp for exp in exports if exp["type"] == "open_graph"]
        assert len(og_tags) == 4

    def test_handle_complex_nesting(self, analyzer):
        """Test handling of complex nested structures."""
        html = """
<!DOCTYPE html>
<html>
<body>
    <div class="container">
        <div class="row">
            <div class="col">
                <div class="card">
                    <div class="card-header">
                        <h3>Title</h3>
                    </div>
                    <div class="card-body">
                        <div class="content">
                            <div class="nested-1">
                                <div class="nested-2">
                                    <div class="nested-3">
                                        <div class="nested-4">
                                            <div class="nested-5">
                                                Very deeply nested
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        metrics = analyzer.calculate_complexity(html, Path("test.html"))

        assert metrics.max_depth >= 10
        assert metrics.avg_depth > 5
