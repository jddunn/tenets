"""
Unit tests for the Java code analyzer.

This module tests the Java-specific code analysis functionality including
import extraction, class/interface parsing, method extraction, and complexity
calculation. The Java analyzer handles modern Java features including generics,
annotations, lambdas, and records.

Test Coverage:
    - Import extraction (standard, static, wildcard)
    - Export detection (public members)
    - Structure extraction (classes, interfaces, enums, records)
    - Complexity metrics (cyclomatic, cognitive)
    - Annotation processing
    - Error handling for invalid Java code
    - Edge cases and Java-specific features
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.java_analyzer import JavaAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestJavaAnalyzerInitialization:
    """Test suite for JavaAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = JavaAnalyzer()

        assert analyzer.language_name == "java"
        assert ".java" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for Java import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaAnalyzer instance."""
        return JavaAnalyzer()

    def test_extract_standard_imports(self, analyzer):
        """Test extraction of standard import statements."""
        code = """
package com.example.app;

import java.util.List;
import java.util.Map;
import java.io.IOException;
"""
        imports = analyzer.extract_imports(code, Path("Test.java"))

        assert len(imports) == 3
        assert any(imp.module == "java.util.List" for imp in imports)
        assert any(imp.module == "java.util.Map" for imp in imports)
        assert any(imp.module == "java.io.IOException" for imp in imports)

        # Check import details
        list_import = next(imp for imp in imports if imp.module == "java.util.List")
        assert list_import.type == "import"
        assert list_import.is_relative is False
        assert list_import.category == "jdk"
        assert list_import.line == 4

    def test_extract_static_imports(self, analyzer):
        """Test extraction of static import statements."""
        code = """
package com.example.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.*;
import static java.lang.Math.PI;
"""
        imports = analyzer.extract_imports(code, Path("Test.java"))

        # Check static imports
        assert any(imp.type == "static" for imp in imports)
        
        # Check wildcard static import
        wildcard = next(imp for imp in imports if "*" in imp.module)
        assert wildcard.is_wildcard is True
        assert wildcard.type == "static"

        # Check specific static import
        pi_import = next(imp for imp in imports if "PI" in imp.module)
        assert pi_import.type == "static"

    def test_extract_wildcard_imports(self, analyzer):
        """Test extraction of wildcard imports."""
        code = """
import java.util.*;
import com.example.utils.*;
import javax.swing.*;
"""
        imports = analyzer.extract_imports(code, Path("Test.java"))

        assert len(imports) == 3
        assert all(imp.is_wildcard for imp in imports)

        # Check categories
        util_import = next(imp for imp in imports if "java.util" in imp.module)
        assert util_import.category == "jdk"

        javax_import = next(imp for imp in imports if "javax" in imp.module)
        assert javax_import.category == "javax"

    def test_categorize_imports(self, analyzer):
        """Test import categorization."""
        code = """
import java.util.List;
import javax.servlet.http.HttpServlet;
import org.springframework.stereotype.Service;
import com.example.MyClass;
"""
        imports = analyzer.extract_imports(code, Path("Test.java"))

        categories = {imp.module: imp.category for imp in imports}
        
        assert categories["java.util.List"] == "jdk"
        assert categories["javax.servlet.http.HttpServlet"] == "javax"
        assert categories["org.springframework.stereotype.Service"] == "third_party"
        assert categories["com.example.MyClass"] == "third_party"


class TestExportExtraction:
    """Test suite for Java export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaAnalyzer instance."""
        return JavaAnalyzer()

    def test_extract_public_classes(self, analyzer):
        """Test extraction of public classes."""
        code = """
package com.example;

public class PublicClass {
    private String field;
}

class PackagePrivateClass {
    int field;
}

public abstract class AbstractClass {
    public abstract void method();
}

public final class FinalClass {
}
"""
        exports = analyzer.extract_exports(code, Path("Test.java"))

        # Should only include public classes
        public_exports = [e for e in exports if e["type"] == "class"]
        assert len(public_exports) == 3

        export_names = [exp["name"] for exp in public_exports]
        assert "PublicClass" in export_names
        assert "AbstractClass" in export_names
        assert "FinalClass" in export_names
        assert "PackagePrivateClass" not in export_names

        # Check modifiers
        abstract_export = next(e for e in exports if e["name"] == "AbstractClass")
        assert "abstract" in abstract_export["modifiers"]

        final_export = next(e for e in exports if e["name"] == "FinalClass")
        assert "final" in final_export["modifiers"]

    def test_extract_public_interfaces(self, analyzer):
        """Test extraction of public interfaces."""
        code = """
public interface Runnable {
    void run();
}

public interface GenericInterface<T> {
    T process(T input);
}

interface PackagePrivateInterface {
    void method();
}
"""
        exports = analyzer.extract_exports(code, Path("Test.java"))

        interface_exports = [e for e in exports if e["type"] == "interface"]
        assert len(interface_exports) == 2

        export_names = [exp["name"] for exp in interface_exports]
        assert "Runnable" in export_names
        assert "GenericInterface" in export_names
        assert "PackagePrivateInterface" not in export_names

    def test_extract_public_enums(self, analyzer):
        """Test extraction of public enums."""
        code = """
public enum Status {
    SUCCESS, FAILURE, PENDING
}

enum InternalStatus {
    ACTIVE, INACTIVE
}
"""
        exports = analyzer.extract_exports(code, Path("Test.java"))

        enum_exports = [e for e in exports if e["type"] == "enum"]
        assert len(enum_exports) == 1
        assert enum_exports[0]["name"] == "Status"

    def test_extract_public_records(self, analyzer):
        """Test extraction of public records (Java 14+)."""
        code = """
public record Person(String name, int age) {}

record InternalRecord(String data) {}
"""
        exports = analyzer.extract_exports(code, Path("Test.java"))

        record_exports = [e for e in exports if e["type"] == "record"]
        assert len(record_exports) == 1
        assert record_exports[0]["name"] == "Person"

    def test_extract_public_methods(self, analyzer):
        """Test extraction of public methods."""
        code = """
public class MyClass {
    public void publicMethod() {}
    
    private void privateMethod() {}
    
    protected void protectedMethod() {}
    
    void packagePrivateMethod() {}
    
    public static void staticMethod() {}
    
    public final void finalMethod() {}
    
    public synchronized void synchronizedMethod() {}
}
"""
        exports = analyzer.extract_exports(code, Path("Test.java"))

        method_exports = [e for e in exports if e["type"] == "method"]
        
        method_names = [m["name"] for m in method_exports]
        assert "publicMethod" in method_names
        assert "staticMethod" in method_names
        assert "finalMethod" in method_names
        assert "synchronizedMethod" in method_names
        assert "privateMethod" not in method_names
        assert "protectedMethod" not in method_names

        # Check method properties
        static_method = next(m for m in method_exports if m["name"] == "staticMethod")
        assert static_method["is_static"] is True

        final_method = next(m for m in method_exports if m["name"] == "finalMethod")
        assert final_method["is_final"] is True


class TestStructureExtraction:
    """Test suite for code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaAnalyzer instance."""
        return JavaAnalyzer()

    def test_extract_classes_with_inheritance(self, analyzer):
        """Test extraction of classes with inheritance."""
        code = """
public class Animal {
    protected String name;
}

public class Dog extends Animal {
    private String breed;
}

public class Cat extends Animal implements Comparable<Cat> {
    public int compareTo(Cat other) {
        return 0;
    }
}

public abstract class AbstractService implements Service, Configurable {
    public abstract void serve();
}
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))

        assert len(structure.classes) == 4

        # Check Dog inheritance
        dog = next(c for c in structure.classes if c.name == "Dog")
        assert "Animal" in dog.bases

        # Check Cat implements
        cat = next(c for c in structure.classes if c.name == "Cat")
        assert "Animal" in cat.bases
        assert "Comparable<Cat>" in cat.interfaces or "Comparable" in cat.interfaces

        # Check abstract class
        abstract_service = next(c for c in structure.classes if c.name == "AbstractService")
        assert "Service" in abstract_service.interfaces
        assert "Configurable" in abstract_service.interfaces

    def test_extract_class_members(self, analyzer):
        """Test extraction of class members."""
        code = """
public class Person {
    private String name;
    private int age;
    public static final int MAX_AGE = 150;
    
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    private void validateAge() {
        if (age < 0 || age > MAX_AGE) {
            throw new IllegalArgumentException();
        }
    }
}
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))

        person_class = structure.classes[0]
        
        # Check fields
        assert len(person_class.fields) >= 3
        field_names = [f["name"] for f in person_class.fields]
        assert "name" in field_names
        assert "age" in field_names
        assert "MAX_AGE" in field_names

        # Check methods
        assert len(person_class.methods) >= 4
        method_names = [m["name"] for m in person_class.methods]
        assert "Person" in method_names  # Constructor
        assert "getName" in method_names
        assert "setName" in method_names
        assert "validateAge" in method_names

        # Check constructor
        constructor = next(m for m in person_class.methods if m["name"] == "Person")
        assert constructor["is_constructor"] is True

    def test_extract_interfaces(self, analyzer):
        """Test extraction of interfaces."""
        code = """
public interface Drawable {
    void draw();
    default void clear() {
        System.out.println("Clearing...");
    }
}

public interface Shape extends Drawable {
    double getArea();
    double getPerimeter();
}

@FunctionalInterface
public interface Calculator {
    int calculate(int a, int b);
}
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))

        assert len(structure.interfaces) == 3

        # Check interface with default method
        drawable = next(i for i in structure.interfaces if i["name"] == "Drawable")
        assert len(drawable["methods"]) == 2
        clear_method = next(m for m in drawable["methods"] if m["name"] == "clear")
        assert clear_method["is_default"] is True

        # Check interface extension
        shape = next(i for i in structure.interfaces if i["name"] == "Shape")
        assert "Drawable" in shape["extends"]

        # Check functional interface
        calculator = next(i for i in structure.interfaces if i["name"] == "Calculator")
        assert calculator["is_functional"] is True

    def test_extract_enums(self, analyzer):
        """Test extraction of enums with values."""
        code = """
public enum DayOfWeek {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY
}

public enum HttpStatus {
    OK(200),
    NOT_FOUND(404),
    INTERNAL_ERROR(500);
    
    private final int code;
    
    HttpStatus(int code) {
        this.code = code;
    }
    
    public int getCode() {
        return code;
    }
}
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))

        assert len(structure.enums) == 2

        # Check simple enum
        day_enum = next(e for e in structure.enums if e["name"] == "DayOfWeek")
        assert len(day_enum["values"]) == 7
        assert "MONDAY" in day_enum["values"]

        # Check enum with constructor
        http_enum = next(e for e in structure.enums if e["name"] == "HttpStatus")
        assert "OK" in http_enum["values"]

    def test_detect_frameworks(self, analyzer):
        """Test framework detection through annotations and imports."""
        code = """
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class UserService {
    @Autowired
    private UserRepository repository;
    
    public User findById(Long id) {
        return repository.findById(id);
    }
}
"""
        structure = analyzer.extract_structure(code, Path("UserService.java"))

        assert structure.framework == "Spring"
        assert "Service" in structure.annotations
        assert "Autowired" in structure.annotations


class TestComplexityCalculation:
    """Test suite for complexity metrics calculation."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaAnalyzer instance."""
        return JavaAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
public class ComplexClass {
    public String process(int value) {
        if (value > 0) {
            if (value > 10) {
                return "big";
            } else {
                return "small";
            }
        } else if (value < 0) {
            return "negative";
        } else {
            return "zero";
        }
    }
    
    public void loops(int n) {
        for (int i = 0; i < n; i++) {
            while (n > 0) {
                n--;
            }
        }
        
        do {
            n++;
        } while (n < 10);
    }
    
    public void switches(int option) {
        switch (option) {
            case 1:
                break;
            case 2:
                break;
            default:
                break;
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("Test.java"))

        # Base complexity = 1
        # +1 for each decision point
        assert metrics.cyclomatic >= 10

    def test_calculate_cognitive_complexity(self, analyzer):
        """Test cognitive complexity calculation."""
        code = """
public void nested(int x) {
    if (x > 0) {  // +1
        for (int i = 0; i < x; i++) {  // +2 (1 + nesting)
            if (i % 2 == 0) {  // +3 (1 + 2*nesting)
                if (i > 5) {  // +4 (1 + 3*nesting)
                    System.out.println(i);
                }
            }
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("Test.java"))

        # Cognitive complexity considers nesting
        assert metrics.cognitive >= 10

    def test_exception_handling_complexity(self, analyzer):
        """Test complexity with exception handling."""
        code = """
public class ExceptionHandler {
    public void handleExceptions() {
        try {
            riskyOperation();
        } catch (IOException e) {
            handleIO(e);
        } catch (SQLException e) {
            handleSQL(e);
        } catch (Exception e) {
            handleGeneral(e);
        } finally {
            cleanup();
        }
        
        try {
            anotherOperation();
        } catch (RuntimeException e) {
            throw new ServiceException("Failed", e);
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("Test.java"))

        assert metrics.try_blocks == 2
        assert metrics.catch_blocks >= 4
        assert metrics.finally_blocks == 1
        assert metrics.throws_declarations >= 1

    def test_lambda_and_stream_complexity(self, analyzer):
        """Test complexity with lambdas and streams."""
        code = """
import java.util.List;
import java.util.stream.Collectors;

public class StreamProcessor {
    public List<String> process(List<Integer> numbers) {
        return numbers.stream()
            .filter(n -> n > 0)
            .map(n -> n * 2)
            .filter(n -> n < 100)
            .map(n -> "Number: " + n)
            .collect(Collectors.toList());
    }
    
    public void lambdas() {
        Runnable r1 = () -> System.out.println("Hello");
        Function<Integer, Integer> square = x -> x * x;
        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("Test.java"))

        assert metrics.lambda_count >= 5
        assert metrics.stream_operations >= 4


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaAnalyzer instance."""
        return JavaAnalyzer()

    def test_handle_syntax_error(self, analyzer):
        """Test handling of files with syntax errors."""
        code = """
public class Invalid {
    public void method() {
        this is not valid Java code
    }
}
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("Test.java"))
        exports = analyzer.extract_exports(code, Path("Test.java"))
        structure = analyzer.extract_structure(code, Path("Test.java"))
        metrics = analyzer.calculate_complexity(code, Path("Test.java"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("Test.java"))
        exports = analyzer.extract_exports(code, Path("Test.java"))
        structure = analyzer.extract_structure(code, Path("Test.java"))
        metrics = analyzer.calculate_complexity(code, Path("Test.java"))

        assert imports == []
        assert exports == []
        assert len(structure.classes) == 0
        assert metrics.line_count == 1

    def test_handle_comments_only(self, analyzer):
        """Test handling of files with only comments."""
        code = """
// This file contains only comments
/* Multi-line comment
   with no actual code */
// Just documentation
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))
        metrics = analyzer.calculate_complexity(code, Path("Test.java"))

        assert len(structure.classes) == 0
        assert metrics.comment_lines > 0
        assert metrics.code_lines == 0


class TestEdgeCases:
    """Test suite for edge cases and Java-specific features."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaAnalyzer instance."""
        return JavaAnalyzer()

    def test_inner_classes(self, analyzer):
        """Test extraction of inner classes."""
        code = """
public class Outer {
    private int field;
    
    public class Inner {
        public void method() {
            System.out.println(field);
        }
    }
    
    public static class StaticInner {
        public void staticMethod() {}
    }
    
    private class PrivateInner {}
}
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))

        outer_class = next(c for c in structure.classes if c.name == "Outer")
        assert len(outer_class.inner_classes) >= 3
        assert "Inner" in outer_class.inner_classes
        assert "StaticInner" in outer_class.inner_classes

    def test_anonymous_classes(self, analyzer):
        """Test detection of anonymous classes."""
        code = """
public class Container {
    public void createAnonymous() {
        Runnable r = new Runnable() {
            @Override
            public void run() {
                System.out.println("Running");
            }
        };
        
        Thread t = new Thread(new Runnable() {
            public void run() {}
        });
    }
}
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))

        assert structure.anonymous_classes_count == 2

    def test_generics(self, analyzer):
        """Test handling of generic types."""
        code = """
public class GenericClass<T extends Comparable<T>> {
    private T value;
    
    public <U> void genericMethod(U param) {}
    
    public List<? extends Number> getNumbers() {
        return null;
    }
    
    public void wildcards(List<? super Integer> list) {}
}

public interface GenericInterface<K, V> {
    V get(K key);
    void put(K key, V value);
}
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))

        generic_class = next(c for c in structure.classes if c.name == "GenericClass")
        assert generic_class.generics is not None
        assert "T" in generic_class.generics

    def test_annotations(self, analyzer):
        """Test handling of annotations."""
        code = """
@Entity
@Table(name = "users")
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false, length = 100)
    private String username;
    
    @Transient
    private String tempData;
    
    @PrePersist
    public void prePersist() {}
}
"""
        structure = analyzer.extract_structure(code, Path("User.java"))

        assert "Entity" in structure.annotations
        assert "Table" in structure.annotations
        assert "Id" in structure.annotations

    def test_varargs(self, analyzer):
        """Test handling of varargs methods."""
        code = """
public class VarArgsExample {
    public void method(String... args) {
        for (String arg : args) {
            System.out.println(arg);
        }
    }
    
    public void mixed(int first, String... rest) {}
}
"""
        structure = analyzer.extract_structure(code, Path("Test.java"))

        example_class = structure.classes[0]
        method = next(m for m in example_class.methods if m["name"] == "method")
        assert any("..." in str(p) for p in method["parameters"])