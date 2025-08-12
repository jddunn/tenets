"""
Unit tests for the PHP code analyzer.

This module tests the PHP-specific code analysis functionality including
use statement extraction, class/interface/trait parsing, method extraction,
and complexity calculation. The PHP analyzer handles modern PHP features
including namespaces, traits, and typed properties.

Test Coverage:
    - Use statement and include/require extraction
    - Export detection (public members)
    - Structure extraction (classes, interfaces, traits, enums)
    - Complexity metrics (cyclomatic, cognitive)
    - PHP 7+ and PHP 8+ features
    - Framework detection (Laravel, Symfony, WordPress)
    - Error handling for invalid PHP code
    - Edge cases and PHP-specific features
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tenets.core.analysis.implementations.php_analyzer import PhpAnalyzer
from tenets.models.analysis import (
    ClassInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ImportInfo,
)


class TestPhpAnalyzerInitialization:
    """Test suite for PhpAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = PhpAnalyzer()

        assert analyzer.language_name == "php"
        assert ".php" in analyzer.file_extensions
        assert ".phtml" in analyzer.file_extensions
        assert ".inc" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for PHP import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PhpAnalyzer instance."""
        return PhpAnalyzer()

    def test_extract_use_statements(self, analyzer):
        """Test extraction of use statements."""
        code = """<?php
namespace App\\Controllers;

use App\\Models\\User;
use App\\Services\\AuthService;
use Illuminate\\Support\\Facades\\DB;
use function App\\Helpers\\format_date;
use const App\\Constants\\MAX_RETRIES;
"""
        imports = analyzer.extract_imports(code, Path("UserController.php"))

        assert len(imports) == 5

        # Check class use
        user_import = next(imp for imp in imports if "User" in imp.module)
        assert user_import.type == "use_class"
        assert user_import.namespace == "App\\Controllers"

        # Check function use
        func_import = next(imp for imp in imports if "format_date" in imp.module)
        assert func_import.type == "use_function"

        # Check const use
        const_import = next(imp for imp in imports if "MAX_RETRIES" in imp.module)
        assert const_import.type == "use_const"

    def test_extract_aliased_use_statements(self, analyzer):
        """Test extraction of aliased use statements."""
        code = """<?php
use App\\Models\\User as UserModel;
use App\\Services\\Auth as AuthService;
use function strlen as str_length;
"""
        imports = analyzer.extract_imports(code, Path("app.php"))

        user_import = next(imp for imp in imports if "User" in imp.module)
        assert user_import.alias == "UserModel"

        auth_import = next(imp for imp in imports if "Auth" in imp.module)
        assert auth_import.alias == "AuthService"

    def test_extract_group_use_statements(self, analyzer):
        """Test extraction of group use statements (PHP 7+)."""
        code = """<?php
use App\\Models\\{User, Post, Comment};
use Symfony\\Component\\HttpFoundation\\{Request, Response, JsonResponse};
use function App\\Helpers\\{format_date, sanitize_input};
"""
        imports = analyzer.extract_imports(code, Path("app.php"))

        # Should expand group use statements
        assert any("User" in imp.module for imp in imports)
        assert any("Post" in imp.module for imp in imports)
        assert any("Comment" in imp.module for imp in imports)
        assert any("Request" in imp.module for imp in imports)

    def test_extract_includes_requires(self, analyzer):
        """Test extraction of include/require statements."""
        code = """<?php
include 'config.php';
include_once 'helpers.php';
require 'bootstrap.php';
require_once 'autoload.php';

include __DIR__ . '/vendor/autoload.php';
require_once dirname(__FILE__) . '/init.php';

if ($debug) {
    include 'debug.php';
}
"""
        imports = analyzer.extract_imports(code, Path("index.php"))

        # Check different include types
        config_import = next(imp for imp in imports if imp.module == "config.php")
        assert config_import.type == "include"
        assert config_import.is_file_include is True

        helpers_import = next(imp for imp in imports if imp.module == "helpers.php")
        assert helpers_import.type == "include_once"

        bootstrap_import = next(imp for imp in imports if imp.module == "bootstrap.php")
        assert bootstrap_import.type == "require"

        autoload_import = next(imp for imp in imports if imp.module == "autoload.php")
        assert autoload_import.type == "require_once"

        # Dynamic includes
        dynamic_imports = [imp for imp in imports if imp.module == "<dynamic>"]
        assert len(dynamic_imports) > 0

    def test_extract_composer_dependencies(self, analyzer):
        """Test extraction from composer.json."""
        code = """{
    "require": {
        "php": ">=7.4",
        "laravel/framework": "^8.0",
        "guzzlehttp/guzzle": "^7.0"
    },
    "require-dev": {
        "phpunit/phpunit": "^9.0",
        "mockery/mockery": "^1.4"
    }
}"""
        imports = analyzer.extract_imports(code, Path("composer.json"))

        # Should extract non-PHP dependencies
        assert any("laravel/framework" in imp.module for imp in imports)
        assert any("guzzlehttp/guzzle" in imp.module for imp in imports)

        # Check dev dependencies
        phpunit_import = next(imp for imp in imports if "phpunit" in imp.module)
        assert phpunit_import.is_dev_dependency is True


class TestExportExtraction:
    """Test suite for PHP export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PhpAnalyzer instance."""
        return PhpAnalyzer()

    def test_extract_classes(self, analyzer):
        """Test extraction of class exports."""
        code = """<?php
namespace App;

class PublicClass {
    public function method() {}
}

abstract class AbstractService {
    abstract public function process();
}

final class FinalClass {
    // Cannot be extended
}

class ChildClass extends ParentClass implements InterfaceA, InterfaceB {
    // Implementation
}
"""
        exports = analyzer.extract_exports(code, Path("classes.php"))

        class_exports = [e for e in exports if e["type"] == "class"]
        assert len(class_exports) == 4

        # Check modifiers
        abstract_class = next(e for e in exports if e["name"] == "AbstractService")
        assert "abstract" in abstract_class["modifiers"]

        final_class = next(e for e in exports if e["name"] == "FinalClass")
        assert "final" in final_class["modifiers"]

        # Check inheritance
        child_class = next(e for e in exports if e["name"] == "ChildClass")
        assert child_class["extends"] == "ParentClass"
        assert "InterfaceA" in child_class["implements"]
        assert "InterfaceB" in child_class["implements"]

    def test_extract_interfaces(self, analyzer):
        """Test extraction of interfaces."""
        code = """<?php
interface Drawable {
    public function draw();
}

interface Resizable extends Drawable {
    public function resize($width, $height);
}

interface Serializable {
    public function serialize();
    public function unserialize($data);
}
"""
        exports = analyzer.extract_exports(code, Path("interfaces.php"))

        interface_exports = [e for e in exports if e["type"] == "interface"]
        assert len(interface_exports) == 3

        # Check interface extension
        resizable = next(e for e in exports if e["name"] == "Resizable")
        assert "Drawable" in resizable["extends"]

    def test_extract_traits(self, analyzer):
        """Test extraction of traits."""
        code = """<?php
trait TimestampTrait {
    protected $created_at;
    protected $updated_at;
    
    public function touch() {
        $this->updated_at = time();
    }
}

trait Singleton {
    private static $instance;
    
    public static function getInstance() {
        if (self::$instance === null) {
            self::$instance = new self();
        }
        return self::$instance;
    }
}
"""
        exports = analyzer.extract_exports(code, Path("traits.php"))

        trait_exports = [e for e in exports if e["type"] == "trait"]
        assert len(trait_exports) == 2

        trait_names = [t["name"] for t in trait_exports]
        assert "TimestampTrait" in trait_names
        assert "Singleton" in trait_names

    def test_extract_enums(self, analyzer):
        """Test extraction of enums (PHP 8.1+)."""
        code = """<?php
enum Status {
    case PENDING;
    case APPROVED;
    case REJECTED;
}

enum HttpStatus: int {
    case OK = 200;
    case NOT_FOUND = 404;
    case SERVER_ERROR = 500;
}
"""
        exports = analyzer.extract_exports(code, Path("enums.php"))

        enum_exports = [e for e in exports if e["type"] == "enum"]
        assert len(enum_exports) == 2

        # Check backed enum
        http_enum = next(e for e in exports if e["name"] == "HttpStatus")
        assert http_enum["backed_type"] == "int"

    def test_extract_functions(self, analyzer):
        """Test extraction of global functions."""
        code = """<?php
function helper_function($param) {
    return $param;
}

function another_function(): string {
    return "test";
}

class MyClass {
    public function method() {
        // This should not be exported as global function
    }
}

// Closure (not exported)
$closure = function($x) {
    return $x * 2;
};
"""
        exports = analyzer.extract_exports(code, Path("functions.php"))

        func_exports = [e for e in exports if e["type"] == "function"]
        assert len(func_exports) == 2

        func_names = [f["name"] for f in func_exports]
        assert "helper_function" in func_names
        assert "another_function" in func_names
        assert "method" not in func_names  # Class method, not global

    def test_extract_constants(self, analyzer):
        """Test extraction of constants."""
        code = """<?php
const APP_VERSION = '1.0.0';
const DEBUG_MODE = true;

define('BASE_PATH', __DIR__);
define('MAX_UPLOAD_SIZE', 1024 * 1024 * 10);

class Config {
    const DB_HOST = 'localhost';
    const DB_PORT = 3306;
}
"""
        exports = analyzer.extract_exports(code, Path("constants.php"))

        const_exports = [e for e in exports if e["type"] == "constant"]
        assert len(const_exports) >= 4

        const_names = [c["name"] for c in const_exports]
        assert "APP_VERSION" in const_names
        assert "DEBUG_MODE" in const_names
        assert "BASE_PATH" in const_names
        assert "MAX_UPLOAD_SIZE" in const_names


class TestStructureExtraction:
    """Test suite for PHP code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PhpAnalyzer instance."""
        return PhpAnalyzer()

    def test_extract_class_members(self, analyzer):
        """Test extraction of class members with visibility."""
        code = """<?php
class User {
    public string $name;
    private int $id;
    protected ?string $email = null;
    public static int $count = 0;
    
    public function __construct(string $name) {
        $this->name = $name;
    }
    
    public function getName(): string {
        return $this->name;
    }
    
    private function validate(): bool {
        return !empty($this->name);
    }
    
    protected function log(string $message): void {
        echo $message;
    }
    
    public static function getCount(): int {
        return self::$count;
    }
}
"""
        structure = analyzer.extract_structure(code, Path("User.php"))

        user_class = structure.classes[0]

        # Check properties
        assert len(user_class.properties) >= 4

        name_prop = next(p for p in user_class.properties if p["name"] == "name")
        assert name_prop["visibility"] == "public"
        assert name_prop["type"] == "string"

        id_prop = next(p for p in user_class.properties if p["name"] == "id")
        assert id_prop["visibility"] == "private"

        email_prop = next(p for p in user_class.properties if p["name"] == "email")
        assert email_prop["visibility"] == "protected"
        assert "?" in email_prop["type"]  # Nullable

        # Check methods
        assert len(user_class.methods) >= 5

        constructor = next(m for m in user_class.methods if m["name"] == "__construct")
        assert constructor["is_constructor"] is True

        get_name = next(m for m in user_class.methods if m["name"] == "getName")
        assert get_name["return_type"] == "string"

        get_count = next(m for m in user_class.methods if m["name"] == "getCount")
        assert "static" in get_count["modifiers"]

    def test_extract_traits_usage(self, analyzer):
        """Test extraction of trait usage in classes."""
        code = """<?php
trait LoggerTrait {
    public function log($message) {
        echo $message;
    }
}

trait TimestampTrait {
    protected $created_at;
    protected $updated_at;
}

class Product {
    use LoggerTrait;
    use TimestampTrait;
    
    use SomeTrait, AnotherTrait {
        SomeTrait::method insteadof AnotherTrait;
        AnotherTrait::method as anotherMethod;
    }
}
"""
        structure = analyzer.extract_structure(code, Path("Product.php"))

        product_class = structure.classes[0]

        assert len(product_class.traits_used) >= 4
        trait_names = [t["name"] for t in product_class.traits_used]
        assert "LoggerTrait" in trait_names
        assert "TimestampTrait" in trait_names
        assert "SomeTrait" in trait_names

        # Check trait with adaptations
        adapted_trait = next(t for t in product_class.traits_used if t["name"] == "SomeTrait")
        assert adapted_trait.get("has_adaptations") is True

    def test_extract_magic_methods(self, analyzer):
        """Test extraction of PHP magic methods."""
        code = """<?php
class MagicClass {
    public function __construct() {}
    public function __destruct() {}
    public function __get($name) {}
    public function __set($name, $value) {}
    public function __isset($name) {}
    public function __unset($name) {}
    public function __call($name, $args) {}
    public static function __callStatic($name, $args) {}
    public function __toString() {}
    public function __invoke() {}
    public function __clone() {}
}
"""
        structure = analyzer.extract_structure(code, Path("MagicClass.php"))

        magic_class = structure.classes[0]

        # Check magic methods are detected
        magic_methods = [m for m in magic_class.methods if m["is_magic"]]
        assert len(magic_methods) == 11

        constructor = next(m for m in magic_class.methods if m["name"] == "__construct")
        assert constructor["is_constructor"] is True

        destructor = next(m for m in magic_class.methods if m["name"] == "__destruct")
        assert destructor["is_destructor"] is True

    def test_detect_framework(self, analyzer):
        """Test framework detection."""
        laravel_code = """<?php
namespace App\\Http\\Controllers;

use Illuminate\\Http\\Request;
use App\\Models\\User;

class UserController extends Controller {
    public function index() {
        return User::all();
    }
    
    public function store(Request $request) {
        $validated = $request->validate([
            'name' => 'required|string',
            'email' => 'required|email'
        ]);
        
        return User::create($validated);
    }
}
"""
        laravel_structure = analyzer.extract_structure(laravel_code, Path("UserController.php"))
        assert laravel_structure.framework == "Laravel"

        symfony_code = """<?php
namespace App\\Controller;

use Symfony\\Bundle\\FrameworkBundle\\Controller\\AbstractController;
use Symfony\\Component\\HttpFoundation\\Response;
use Symfony\\Component\\Routing\\Annotation\\Route;

class DefaultController extends AbstractController {
    #[Route('/home', name: 'home')]
    public function index(): Response {
        return $this->render('home.html.twig');
    }
}
"""
        symfony_structure = analyzer.extract_structure(symfony_code, Path("DefaultController.php"))
        assert symfony_structure.framework == "Symfony"

        wordpress_code = """<?php
/*
Plugin Name: My Plugin
*/

add_action('init', 'my_init_function');
add_filter('the_content', 'my_content_filter');

function my_init_function() {
    register_post_type('custom_post', [
        'public' => true,
        'label' => 'Custom Posts'
    ]);
}

class WP_My_Widget extends WP_Widget {
    public function widget($args, $instance) {
        // Widget output
    }
}
"""
        wp_structure = analyzer.extract_structure(wordpress_code, Path("plugin.php"))
        assert wp_structure.framework == "WordPress"


class TestComplexityCalculation:
    """Test suite for PHP complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PhpAnalyzer instance."""
        return PhpAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """<?php
function complex_function($x, $y) {
    if ($x > 0) {
        if ($y > 0) {
            return $x + $y;
        } elseif ($y < 0) {
            return $x - $y;
        } else {
            return $x;
        }
    } else if ($x < 0) {
        switch ($y) {
            case 0:
                return 0;
            case 1:
                return -1;
            default:
                return $y;
        }
    } else {
        return $y ?: 0;
    }
    
    try {
        risky_operation();
    } catch (Exception $e) {
        handle_error($e);
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("complex.php"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 10

    def test_calculate_cognitive_complexity(self, analyzer):
        """Test cognitive complexity calculation."""
        code = """<?php
function nested_function($items) {
    foreach ($items as $item) {        // +1
        if ($item->isValid()) {         // +2 (nesting)
            foreach ($item->data as $data) {  // +3 (more nesting)
                if ($data > 0) {        // +4 (even more nesting)
                    process($data);
                } else {
                    log_error($data);
                }
            }
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("nested.php"))

        # Cognitive complexity considers nesting
        assert metrics.cognitive >= 10

    def test_php_specific_metrics(self, analyzer):
        """Test PHP-specific complexity metrics."""
        code = """<?php
// Global usage
function bad_function() {
    global $config;
    $GLOBALS['counter']++;
    
    // Superglobals
    $user = $_POST['user'];
    $file = $_FILES['upload'];
    $session = $_SESSION['data'];
    
    // Dynamic calls
    $func = 'dynamic_function';
    $func();
    
    // Eval (dangerous)
    eval('$x = 10;');
}

// Type hints (good)
function typed_function(string $name, int $age): ?User {
    return new User($name, $age);
}

function nullable_param(?string $value): void {
    // ...
}

// Union types (PHP 8)
function union_types(int|string $value): array|false {
    // ...
}
"""
        metrics = analyzer.calculate_complexity(code, Path("metrics.php"))

        assert metrics.global_usage > 0
        assert metrics.superglobal_usage > 0
        assert metrics.eval_usage > 0
        assert metrics.dynamic_calls > 0
        assert metrics.type_hints > 0
        assert metrics.nullable_types > 0
        assert metrics.union_types > 0

    def test_attribute_metrics(self, analyzer):
        """Test PHP 8 attribute metrics."""
        code = """<?php
use Doctrine\\ORM\\Mapping as ORM;

#[ORM\\Entity]
#[ORM\\Table(name: "users")]
class User {
    #[ORM\\Id]
    #[ORM\\GeneratedValue]
    #[ORM\\Column(type: "integer")]
    private int $id;
    
    #[ORM\\Column(type: "string", length: 255)]
    private string $name;
    
    #[Route("/users/{id}", methods: ["GET"])]
    public function show(int $id): Response {
        // ...
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("User.php"))

        assert metrics.attributes > 0


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PhpAnalyzer instance."""
        return PhpAnalyzer()

    def test_handle_syntax_errors(self, analyzer):
        """Test handling of files with syntax errors."""
        code = """<?php
class Invalid {
    public function method() {
        this is not valid PHP code
        missing semicolon here
    }
// Missing closing brace
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("invalid.php"))
        exports = analyzer.extract_exports(code, Path("invalid.php"))
        structure = analyzer.extract_structure(code, Path("invalid.php"))
        metrics = analyzer.calculate_complexity(code, Path("invalid.php"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("empty.php"))
        exports = analyzer.extract_exports(code, Path("empty.php"))
        structure = analyzer.extract_structure(code, Path("empty.php"))
        metrics = analyzer.calculate_complexity(code, Path("empty.php"))

        assert imports == []
        assert exports == []
        assert metrics.line_count == 1

    def test_handle_non_php_content(self, analyzer):
        """Test handling of files with mixed content."""
        code = """
<html>
<body>
<?php
echo "Hello World";
?>
<p>Some HTML content</p>
<?php
function test() {
    return "test";
}
?>
</body>
</html>
"""
        structure = analyzer.extract_structure(code, Path("mixed.php"))

        # Should still extract PHP elements
        assert len(structure.functions) == 1


class TestEdgeCases:
    """Test suite for edge cases and PHP-specific features."""

    @pytest.fixture
    def analyzer(self):
        """Provide a PhpAnalyzer instance."""
        return PhpAnalyzer()

    def test_heredoc_nowdoc(self, analyzer):
        """Test handling of heredoc and nowdoc strings."""
        code = """<?php
$heredoc = <<<EOT
This is a heredoc string
with multiple lines
EOT;

$nowdoc = <<<'EOD'
This is a nowdoc string
with no variable interpolation
EOD;

function with_heredoc() {
    $sql = <<<SQL
    SELECT * FROM users
    WHERE active = 1
    SQL;
    
    return $sql;
}
"""
        # Should handle heredoc/nowdoc without treating content as code
        structure = analyzer.extract_structure(code, Path("strings.php"))
        assert len(structure.functions) == 1

    def test_anonymous_classes(self, analyzer):
        """Test handling of anonymous classes."""
        code = """<?php
$object = new class {
    public function method() {
        return "anonymous";
    }
};

$extended = new class extends BaseClass implements Interface {
    use SomeTrait;
    
    private $property;
    
    public function __construct() {
        $this->property = "value";
    }
};
"""
        structure = analyzer.extract_structure(code, Path("anonymous.php"))

        # Anonymous classes are counted but not named
        assert structure.anonymous_classes_count >= 2

    def test_variadic_functions(self, analyzer):
        """Test handling of variadic functions."""
        code = """<?php
function variadic_function(...$args) {
    foreach ($args as $arg) {
        process($arg);
    }
}

function typed_variadic(string ...$strings): array {
    return $strings;
}

function mixed_params($first, $second, ...$rest) {
    // Implementation
}
"""
        structure = analyzer.extract_structure(code, Path("variadic.php"))

        assert len(structure.functions) == 3

        # Check parameter parsing includes variadics
        variadic_func = structure.functions[0]
        assert any(p.get("is_variadic") for p in variadic_func.parameters)

    def test_generators(self, analyzer):
        """Test handling of generator functions."""
        code = """<?php
function simple_generator() {
    yield 1;
    yield 2;
    yield 3;
}

function key_value_generator() {
    yield 'first' => 1;
    yield 'second' => 2;
}

function delegating_generator() {
    yield from another_generator();
}
"""
        structure = analyzer.extract_structure(code, Path("generators.php"))

        # Should detect generator functions
        assert len(structure.functions) == 3
