"""
Unit tests for the C/C++ code analyzer.

This module tests the C/C++-specific code analysis functionality including
include extraction, class/struct parsing, template handling, and complexity
calculation. The analyzer handles both C and modern C++ features.

Test Coverage:
    - Include extraction (system and local)
    - Export detection (non-static symbols)
    - Structure extraction (classes, structs, templates)
    - Complexity metrics (cyclomatic, cognitive, memory safety)
    - Template and STL detection
    - Error handling for invalid C/C++ code
    - Edge cases and C++ specific features
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.cpp_analyzer import CppAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestCppAnalyzerInitialization:
    """Test suite for CppAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = CppAnalyzer()

        assert analyzer.language_name == "cpp"
        assert ".cpp" in analyzer.file_extensions
        assert ".h" in analyzer.file_extensions
        assert ".cc" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for C/C++ include extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CppAnalyzer instance."""
        return CppAnalyzer()

    def test_extract_system_includes(self, analyzer):
        """Test extraction of system includes."""
        code = """
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdio.h>
"""
        imports = analyzer.extract_imports(code, Path("test.cpp"))

        assert len(imports) == 5
        assert any(imp.module == "iostream" for imp in imports)
        assert any(imp.module == "vector" for imp in imports)
        assert any(imp.module == "stdio.h" for imp in imports)

        # Check import types
        iostream_import = next(imp for imp in imports if imp.module == "iostream")
        assert iostream_import.type == "system"
        assert iostream_import.is_relative is False
        assert iostream_import.is_stl is True

        stdio_import = next(imp for imp in imports if imp.module == "stdio.h")
        assert stdio_import.is_stdlib is True

    def test_extract_local_includes(self, analyzer):
        """Test extraction of local includes."""
        code = """
#include "myheader.h"
#include "utils/helper.hpp"
#include "../common/base.h"
"""
        imports = analyzer.extract_imports(code, Path("test.cpp"))

        assert len(imports) == 3
        assert all(imp.type == "local" for imp in imports)
        assert all(imp.is_relative for imp in imports)
        assert all(imp.is_project_header for imp in imports)

    def test_extract_conditional_includes(self, analyzer):
        """Test extraction of conditional includes."""
        code = """
#ifdef DEBUG
    #include <debug.h>
#endif

#ifndef NDEBUG
    #include <assert.h>
#else
    #include <release.h>
#endif

#if defined(WIN32)
    #include <windows.h>
#elif defined(__linux__)
    #include <unistd.h>
#endif
"""
        imports = analyzer.extract_imports(code, Path("test.cpp"))

        # Should extract all includes regardless of conditions
        assert any(imp.module == "debug.h" for imp in imports)
        assert any(imp.module == "assert.h" for imp in imports)
        assert any(imp.module == "windows.h" for imp in imports)
        assert any(imp.module == "unistd.h" for imp in imports)

        # Check conditional flag
        assert any(imp.conditional for imp in imports)

    def test_detect_include_guards(self, analyzer):
        """Test detection of include guards."""
        code = """
#ifndef MY_HEADER_H
#define MY_HEADER_H

#include <string>

#endif // MY_HEADER_H
"""
        imports = analyzer.extract_imports(code, Path("myheader.h"))

        # Check if include guard was detected
        for imp in imports:
            assert hasattr(imp, "has_include_guard")

    def test_detect_pragma_once(self, analyzer):
        """Test detection of pragma once."""
        code = """
#pragma once

#include <vector>
#include <map>
"""
        imports = analyzer.extract_imports(code, Path("header.hpp"))

        for imp in imports:
            assert hasattr(imp, "uses_pragma_once")


class TestExportExtraction:
    """Test suite for C/C++ export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CppAnalyzer instance."""
        return CppAnalyzer()

    def test_extract_functions(self, analyzer):
        """Test extraction of function exports."""
        code = """
// Non-static function (exported)
int calculate(int a, int b) {
    return a + b;
}

// Static function (not exported)
static void helper() {
    // Internal use only
}

// Inline function
inline int square(int x) {
    return x * x;
}

// Template function
template<typename T>
T max(T a, T b) {
    return a > b ? a : b;
}
"""
        exports = analyzer.extract_exports(code, Path("test.cpp"))

        func_exports = [e for e in exports if e["type"] == "function"]

        func_names = [f["name"] for f in func_exports]
        assert "calculate" in func_names
        assert "square" in func_names
        assert "max" in func_names
        assert "helper" not in func_names  # Static, not exported

        # Check function properties
        square_func = next(f for f in func_exports if f["name"] == "square")
        assert square_func["is_inline"] is True

        max_func = next(f for f in func_exports if f["name"] == "max")
        assert max_func["is_template"] is True

    def test_extract_classes_and_structs(self, analyzer):
        """Test extraction of classes and structs."""
        code = """
class PublicClass {
public:
    void method();
};

struct PublicStruct {
    int x, y;
};

// Forward declaration
class ForwardDeclared;

// Template class
template<typename T>
class Container {
    T data;
};

// Private nested class
class Outer {
    class Inner {};  // Not directly exported
};
"""
        exports = analyzer.extract_exports(code, Path("test.cpp"))

        class_names = [e["name"] for e in exports if e["type"] in ["class", "struct"]]
        assert "PublicClass" in class_names
        assert "PublicStruct" in class_names
        assert "Container" in class_names
        assert "Outer" in class_names

        # Check struct vs class
        struct_export = next(e for e in exports if e["name"] == "PublicStruct")
        assert struct_export["type"] == "struct"
        assert struct_export["default_visibility"] == "public"

        class_export = next(e for e in exports if e["name"] == "PublicClass")
        assert class_export["type"] == "class"
        assert class_export["default_visibility"] == "private"

    def test_extract_enums_and_unions(self, analyzer):
        """Test extraction of enums and unions."""
        code = """
enum Color {
    RED, GREEN, BLUE
};

enum class Status {
    OK, ERROR, PENDING
};

union Data {
    int i;
    float f;
    char c;
};
"""
        exports = analyzer.extract_exports(code, Path("test.cpp"))

        # Check enums
        color_enum = next(e for e in exports if e["name"] == "Color")
        assert color_enum["type"] == "enum"

        status_enum = next(e for e in exports if e["name"] == "Status")
        assert status_enum["type"] == "enum_class"

        # Check union
        data_union = next(e for e in exports if e["name"] == "Data")
        assert data_union["type"] == "union"

    def test_extract_typedefs_and_using(self, analyzer):
        """Test extraction of typedefs and using declarations."""
        code = """
typedef int Integer;
typedef struct Node* NodePtr;

using String = std::string;
using IntVector = std::vector<int>;

namespace MyApp {
    using namespace std;
}
"""
        exports = analyzer.extract_exports(code, Path("test.cpp"))

        typedef_names = [e["name"] for e in exports if e["type"] in ["typedef", "using_alias"]]
        assert "Integer" in typedef_names
        assert "NodePtr" in typedef_names
        assert "String" in typedef_names
        assert "IntVector" in typedef_names


class TestStructureExtraction:
    """Test suite for C/C++ code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CppAnalyzer instance."""
        return CppAnalyzer()

    def test_extract_classes_with_inheritance(self, analyzer):
        """Test extraction of classes with inheritance."""
        code = """
class Base {
public:
    virtual void method() = 0;
};

class Derived : public Base {
public:
    void method() override { }
};

class Multiple : public Base, private Helper {
    void method() override final { }
};
"""
        structure = analyzer.extract_structure(code, Path("test.cpp"))

        assert len(structure.classes) == 3

        # Check inheritance
        derived = next(c for c in structure.classes if c.name == "Derived")
        assert "Base" in derived.bases

        multiple = next(c for c in structure.classes if c.name == "Multiple")
        assert "Base" in multiple.bases
        assert "Helper" in multiple.bases

    def test_extract_class_members(self, analyzer):
        """Test extraction of class members."""
        code = """
class MyClass {
public:
    MyClass();
    ~MyClass();
    
    void publicMethod();
    virtual void virtualMethod();
    static void staticMethod();

private:
    int privateField;
    static const int CONSTANT = 42;
    
    void privateMethod();

protected:
    void protectedMethod();
};
"""
        structure = analyzer.extract_structure(code, Path("test.cpp"))

        my_class = structure.classes[0]

        # Check methods
        assert len(my_class.methods) >= 6

        method_names = [m["name"] for m in my_class.methods]
        assert "MyClass" in method_names  # Constructor
        assert "~MyClass" in method_names  # Destructor
        assert "publicMethod" in method_names
        assert "virtualMethod" in method_names

        # Check visibility
        public_method = next(m for m in my_class.methods if m["name"] == "publicMethod")
        assert public_method["visibility"] == "public"

        private_method = next(m for m in my_class.methods if m["name"] == "privateMethod")
        assert private_method["visibility"] == "private"

        # Check modifiers
        virtual_method = next(m for m in my_class.methods if m["name"] == "virtualMethod")
        assert virtual_method["is_virtual"] is True

        static_method = next(m for m in my_class.methods if m["name"] == "staticMethod")
        assert static_method["is_static"] is True

    def test_extract_templates(self, analyzer):
        """Test extraction of templates."""
        code = """
template<typename T>
class Vector {
    T* data;
    size_t size;
};

template<typename T, typename U>
class Pair {
    T first;
    U second;
};

template<>
class Vector<bool> {
    // Specialization
};

template<typename T>
T min(T a, T b) {
    return a < b ? a : b;
}
"""
        structure = analyzer.extract_structure(code, Path("test.cpp"))

        # Check template extraction
        assert len(structure.templates) > 0

        # Check template classes
        vector_class = next(c for c in structure.classes if c.name == "Vector")
        assert vector_class.is_template is True

    def test_detect_cpp_features(self, analyzer):
        """Test detection of C++ specific features."""
        code = """
#include <memory>
#include <vector>
#include <algorithm>

namespace MyNamespace {
    class MyClass {
        std::unique_ptr<int> ptr;
        std::shared_ptr<Data> shared;
        
        void lambdaExample() {
            auto lambda = [](int x) { return x * 2; };
            
            std::vector<int> vec = {1, 2, 3};
            std::for_each(vec.begin(), vec.end(), [](int& n) { n++; });
        }
    };
}
"""
        structure = analyzer.extract_structure(code, Path("test.cpp"))

        # Check namespace detection
        assert "MyNamespace" in [ns["name"] for ns in structure.namespaces]

        # Check STL usage
        assert structure.uses_stl is True

        # Check smart pointer detection
        assert "unique_ptr" in structure.smart_pointers
        assert "shared_ptr" in structure.smart_pointers

        # Check lambda detection
        assert structure.lambda_count > 0

    def test_detect_c_vs_cpp(self, analyzer):
        """Test detection of C vs C++ files."""
        c_code = """
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Hello World\\n");
    return 0;
}
"""
        c_structure = analyzer.extract_structure(c_code, Path("test.c"))
        assert c_structure.language_variant == "C"

        cpp_code = """
#include <iostream>
using namespace std;

class MyClass {
public:
    void method() { }
};
"""
        cpp_structure = analyzer.extract_structure(cpp_code, Path("test.cpp"))
        assert cpp_structure.language_variant == "C++"


class TestComplexityCalculation:
    """Test suite for C/C++ complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CppAnalyzer instance."""
        return CppAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
int complex_function(int x, int y) {
    if (x > 0) {
        if (y > 0) {
            return x + y;
        } else {
            return x - y;
        }
    } else if (x < 0) {
        for (int i = 0; i < y; i++) {
            x++;
        }
    }
    
    switch(x) {
        case 1:
            return 1;
        case 2:
            return 2;
        default:
            return 0;
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cpp"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 7

    def test_memory_management_metrics(self, analyzer):
        """Test memory management complexity metrics."""
        code = """
#include <memory>

void manual_memory() {
    int* p = new int(42);
    delete p;
    
    int* arr = new int[100];
    delete[] arr;
    
    void* raw = malloc(1024);
    free(raw);
}

void smart_pointers() {
    auto ptr1 = std::make_unique<int>(42);
    auto ptr2 = std::make_shared<int>(100);
    std::weak_ptr<int> weak = ptr2;
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cpp"))

        # Check memory management metrics
        assert metrics.new_count == 2
        assert metrics.delete_count == 2
        assert metrics.malloc_count == 1
        assert metrics.free_count == 1

        # Check smart pointer usage
        assert metrics.unique_ptr_count >= 1
        assert metrics.shared_ptr_count >= 1
        assert metrics.weak_ptr_count >= 1

        # Memory safety score should reflect smart pointer usage
        assert 0 <= metrics.memory_safety_score <= 1

    def test_template_and_macro_metrics(self, analyzer):
        """Test template and preprocessor metrics."""
        code = """
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#ifdef DEBUG
    #define LOG(x) std::cout << x << std::endl
#else
    #define LOG(x)
#endif

template<typename T>
class Container {
    T data;
};

template<typename T, typename U>
class Pair {
    T first;
    U second;
};

template<>
class Container<bool> {
    // Specialization
};
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cpp"))

        assert metrics.macro_count >= 3
        assert metrics.ifdef_count >= 1
        assert metrics.template_count >= 2
        assert metrics.template_specializations >= 1


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CppAnalyzer instance."""
        return CppAnalyzer()

    def test_handle_syntax_errors(self, analyzer):
        """Test handling of files with syntax errors."""
        code = """
class Invalid {
    this is not valid C++ code
    void method() {
        missing semicolon
    }
};
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("test.cpp"))
        exports = analyzer.extract_exports(code, Path("test.cpp"))
        structure = analyzer.extract_structure(code, Path("test.cpp"))
        metrics = analyzer.calculate_complexity(code, Path("test.cpp"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.cpp"))
        exports = analyzer.extract_exports(code, Path("test.cpp"))
        structure = analyzer.extract_structure(code, Path("test.cpp"))
        metrics = analyzer.calculate_complexity(code, Path("test.cpp"))

        assert imports == []
        assert exports == []
        assert metrics.line_count == 1


class TestEdgeCases:
    """Test suite for edge cases and special C++ features."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CppAnalyzer instance."""
        return CppAnalyzer()

    def test_operator_overloading(self, analyzer):
        """Test detection of operator overloading."""
        code = """
class Complex {
    double real, imag;
    
public:
    Complex operator+(const Complex& other) {
        return Complex(real + other.real, imag + other.imag);
    }
    
    Complex& operator+=(const Complex& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }
    
    bool operator==(const Complex& other) const {
        return real == other.real && imag == other.imag;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Complex& c);
};
"""
        structure = analyzer.extract_structure(code, Path("test.cpp"))

        # Should detect operator overloads
        assert structure.operator_overloads > 0

    def test_modern_cpp_features(self, analyzer):
        """Test handling of modern C++ features."""
        code = """
#include <memory>
#include <vector>

class Modern {
    // C++11: auto, nullptr, lambdas
    auto getValue() -> int {
        return 42;
    }
    
    // C++14: generic lambdas
    auto genericLambda = [](auto x, auto y) { return x + y; };
    
    // C++17: structured bindings
    void structuredBindings() {
        auto [x, y] = std::make_pair(1, 2);
    }
    
    // C++11: move semantics
    Modern(Modern&& other) noexcept {
        // Move constructor
    }
    
    // C++11: deleted functions
    Modern(const Modern&) = delete;
    
    // C++11: defaulted functions
    Modern() = default;
};
"""
        structure = analyzer.extract_structure(code, Path("test.cpp"))

        # Should handle modern features without errors
        assert len(structure.classes) == 1
        modern_class = structure.classes[0]
        assert modern_class.name == "Modern"
