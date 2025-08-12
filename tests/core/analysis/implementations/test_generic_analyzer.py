"""
Unit tests for the Generic code analyzer.

This module tests the generic analyzer that provides basic analysis
for files without specific language support. It handles text-based
analysis, pattern matching, and basic complexity estimation.

Test Coverage:
    - Basic import/include pattern detection
    - Export pattern detection
    - Structure extraction for various file types
    - Basic complexity metrics
    - Configuration file parsing
    - Markup and data file handling
    - Error handling for various file formats
    - Edge cases and fallback behavior
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.generic_analyzer import GenericAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestGenericAnalyzerInitialization:
    """Test suite for GenericAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = GenericAnalyzer()

        assert analyzer.language_name == "generic"
        assert analyzer.file_extensions == []  # Accepts any extension
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for generic import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GenericAnalyzer instance."""
        return GenericAnalyzer()

    def test_extract_include_patterns(self, analyzer):
        """Test extraction of various include patterns."""
        code = """
#include <stdio.h>
#include "local.h"
include 'config.php'
import 'package:flutter/material.dart'
require 'rails/all'
use strict;
source ~/.bashrc
. /etc/environment
"""
        imports = analyzer.extract_imports(code, Path("mixed.txt"))

        assert len(imports) >= 8

        # Check different include types
        assert any(imp.module == "stdio.h" for imp in imports)
        assert any(imp.module == "local.h" for imp in imports)
        assert any(imp.module == "config.php" for imp in imports)

        # Check import types
        stdio_import = next(imp for imp in imports if imp.module == "stdio.h")
        assert stdio_import.type == "include"

    def test_extract_import_patterns(self, analyzer):
        """Test extraction of import patterns."""
        code = """
import os
from datetime import datetime
require 'json'
use Data::Dumper;
"""
        imports = analyzer.extract_imports(code, Path("script.txt"))

        assert any(imp.type == "import" for imp in imports)
        assert any(imp.type == "from" for imp in imports)
        assert any(imp.type == "require" for imp in imports)
        assert any(imp.type == "use" for imp in imports)

    def test_extract_file_references(self, analyzer):
        """Test extraction of file path references."""
        code = """
file: /path/to/file.txt
path = "/usr/local/bin/script.sh"
src="./images/logo.png"
href="../styles/main.css"
url: "https://example.com/api"
"""
        imports = analyzer.extract_imports(code, Path("config.txt"))

        reference_imports = [imp for imp in imports if imp.type == "reference"]
        assert len(reference_imports) >= 4

        # Check relative path detection
        logo_import = next(imp for imp in imports if "logo.png" in imp.module)
        assert logo_import.is_relative is True

        css_import = next(imp for imp in imports if "main.css" in imp.module)
        assert css_import.is_relative is True

    def test_extract_config_dependencies(self, analyzer):
        """Test extraction from configuration files."""
        json_code = """{
    "dependencies": {
        "express": "^4.17.0",
        "mongoose": "^5.9.0"
    },
    "import": "./config.json",
    "extends": "base-config"
}"""
        imports = analyzer.extract_imports(json_code, Path("package.json"))

        # Should extract dependencies
        assert any("express" in imp.module for imp in imports)
        assert any("mongoose" in imp.module for imp in imports)
        assert any("config.json" in imp.module for imp in imports)


class TestExportExtraction:
    """Test suite for generic export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GenericAnalyzer instance."""
        return GenericAnalyzer()

    def test_extract_function_patterns(self, analyzer):
        """Test extraction of function-like patterns."""
        code = """
function myFunction() {
    return "test";
}

def python_function(arg):
    pass

func goFunction() {
    // implementation
}

sub perl_subroutine {
    # perl code
}

proc tcl_procedure {} {
    # tcl code
}

myFunc() {
    echo "shell function"
}
"""
        exports = analyzer.extract_exports(code, Path("functions.txt"))

        func_exports = [e for e in exports if e["type"] == "function"]
        assert len(func_exports) >= 6

        func_names = [f["name"] for f in func_exports]
        assert "myFunction" in func_names
        assert "python_function" in func_names
        assert "goFunction" in func_names
        assert "perl_subroutine" in func_names
        assert "myFunc" in func_names

    def test_extract_class_patterns(self, analyzer):
        """Test extraction of class-like patterns."""
        code = """
class MyClass {
    constructor() {}
}

struct DataStruct {
    int field;
}

type UserType = {
    name: string;
}

interface IService {
    void process();
}
"""
        exports = analyzer.extract_exports(code, Path("types.txt"))

        class_exports = [e for e in exports if e["type"] == "class"]
        assert len(class_exports) >= 4

        export_names = [e["name"] for e in class_exports]
        assert "MyClass" in export_names
        assert "DataStruct" in export_names
        assert "UserType" in export_names
        assert "IService" in export_names

    def test_extract_variable_patterns(self, analyzer):
        """Test extraction of variable/constant patterns."""
        code = """
export const API_KEY = "secret";
let counter = 0;
var globalVar = true;
val immutableVal = 42;
CONSTANT_VALUE = 100;
config = {
    host: "localhost"
};
module.exports.helper = function() {};
"""
        exports = analyzer.extract_exports(code, Path("vars.txt"))

        # Check various export types
        assert any(e["name"] == "API_KEY" for e in exports)
        assert any(e["name"] == "counter" for e in exports)
        assert any(e["name"] == "CONSTANT_VALUE" for e in exports)
        assert any(e["name"] == "helper" for e in exports)

    def test_extract_config_keys(self, analyzer):
        """Test extraction of configuration file keys."""
        yaml_code = """
database:
  host: localhost
  port: 5432
  
server:
  port: 8080
  
features:
  - auth
  - logging
"""
        exports = analyzer.extract_exports(yaml_code, Path("config.yaml"))

        config_keys = [e for e in exports if e["type"] == "config_key"]
        key_names = [k["name"] for k in config_keys]

        assert "database" in key_names
        assert "server" in key_names
        assert "features" in key_names

        ini_code = """
[database]
host = localhost
port = 5432

[server]
port = 8080
"""
        exports = analyzer.extract_exports(ini_code, Path("config.ini"))

        sections = [e for e in exports if e["type"] == "config_section"]
        assert len(sections) >= 2

        keys = [e for e in exports if e["type"] == "config_key"]
        assert any(k["name"] == "host" for k in keys)
        assert any(k["name"] == "port" for k in keys)


class TestStructureExtraction:
    """Test suite for generic structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GenericAnalyzer instance."""
        return GenericAnalyzer()

    def test_detect_file_types(self, analyzer):
        """Test file type detection."""
        # Configuration file
        config_structure = analyzer.extract_structure("{}", Path("config.json"))
        assert config_structure.file_type == "configuration"

        # Markup file
        markup_structure = analyzer.extract_structure("# Header", Path("doc.md"))
        assert markup_structure.file_type == "markup"

        # Data file
        data_structure = analyzer.extract_structure("a,b,c\n1,2,3", Path("data.csv"))
        assert data_structure.file_type == "data"

        # Script file
        script_structure = analyzer.extract_structure("#!/bin/bash", Path("script.sh"))
        assert script_structure.file_type == "script"

        # Stylesheet
        style_structure = analyzer.extract_structure(".class {}", Path("style.css"))
        assert style_structure.file_type == "stylesheet"

    def test_extract_functions_and_classes(self, analyzer):
        """Test extraction of functions and classes from generic text."""
        code = """
function processData(input) {
    return transform(input);
}

class DataProcessor {
    process() {
        // implementation
    }
}

def handle_request(request):
    return response

const helper = (x) => x * 2;
const arrow = () => {
    console.log("arrow function");
};
"""
        structure = analyzer.extract_structure(code, Path("code.txt"))

        assert len(structure.functions) >= 3
        func_names = [f.name for f in structure.functions]
        assert "processData" in func_names
        assert "handle_request" in func_names

        assert len(structure.classes) >= 1
        assert structure.classes[0].name == "DataProcessor"

    def test_extract_markdown_sections(self, analyzer):
        """Test extraction of markdown sections."""
        code = """
# Main Title
Some content

## Section 1
Content for section 1

### Subsection 1.1
Detailed content

## Section 2
More content

#### Deep Section
Very detailed
"""
        structure = analyzer.extract_structure(code, Path("doc.md"))

        assert len(structure.sections) >= 5

        # Check section levels
        main_title = next(s for s in structure.sections if s["title"] == "Main Title")
        assert main_title["level"] == 1

        subsection = next(s for s in structure.sections if s["title"] == "Subsection 1.1")
        assert subsection["level"] == 3

    def test_extract_todos_and_notes(self, analyzer):
        """Test extraction of TODO/FIXME comments."""
        code = """
// TODO: Implement this function
function stub() {}

# FIXME: This is broken
def broken():
    pass

/* NOTE: Important information here */
class Important {}

// XXX: Needs review
// BUG: Known issue with parsing
// HACK: Temporary workaround
"""
        structure = analyzer.extract_structure(code, Path("todos.txt"))

        assert len(structure.todos) >= 6

        todo_types = [t["type"] for t in structure.todos]
        assert "TODO" in todo_types
        assert "FIXME" in todo_types
        assert "NOTE" in todo_types
        assert "XXX" in todo_types
        assert "BUG" in todo_types
        assert "HACK" in todo_types

    def test_analyze_indentation(self, analyzer):
        """Test indentation analysis."""
        space_code = """
def function():
    if condition:
        do_something()
        if nested:
            deep_action()
"""
        structure = analyzer.extract_structure(space_code, Path("spaces.py"))

        assert structure.indent_levels["style"] == "spaces"
        assert structure.indent_levels["max_level"] > 0

        tab_code = """
function test() {
	if (condition) {
		action();
	}
}
"""
        structure = analyzer.extract_structure(tab_code, Path("tabs.js"))

        assert structure.indent_levels["style"] == "tabs"

    def test_extract_constants_and_variables(self, analyzer):
        """Test extraction of constants and variables."""
        code = """
MAX_SIZE = 100
MIN_SIZE = 10
DEBUG_MODE = True

username = "admin"
password: "secret"
host := "localhost"

export API_KEY
export DATABASE_URL
"""
        structure = analyzer.extract_structure(code, Path("config.txt"))

        assert "MAX_SIZE" in structure.constants
        assert "MIN_SIZE" in structure.constants
        assert "DEBUG_MODE" in structure.constants

        var_names = [v["name"] for v in structure.variables]
        assert "username" in var_names
        assert "password" in var_names
        assert "API_KEY" in var_names


class TestComplexityCalculation:
    """Test suite for generic complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GenericAnalyzer instance."""
        return GenericAnalyzer()

    def test_calculate_basic_metrics(self, analyzer):
        """Test calculation of basic metrics."""
        code = """
This is a sample file
with multiple lines
and some content

# Comment line
// Another comment
"""
        metrics = analyzer.calculate_complexity(code, Path("sample.txt"))

        assert metrics.line_count == 6
        assert metrics.character_count == len(code)
        assert metrics.code_lines == 4  # Non-empty lines
        assert metrics.comment_lines == 2

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test simplified cyclomatic complexity for generic files."""
        code = """
if (condition) {
    doSomething();
} else if (otherCondition) {
    doOther();
} else {
    doDefault();
}

for (i = 0; i < 10; i++) {
    while (running) {
        process();
    }
}

switch (value) {
    case 1:
        break;
    case 2:
        break;
}

result = condition ? value1 : value2;
"""
        metrics = analyzer.calculate_complexity(code, Path("code.txt"))

        # Should detect decision keywords
        assert metrics.cyclomatic >= 8

    def test_calculate_nesting_depth(self, analyzer):
        """Test nesting depth calculation."""
        code = """
function outer() {
    if (condition) {
        for (i = 0; i < 10; i++) {
            if (nested) {
                while (true) {
                    // Deep nesting
                }
            }
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("nested.txt"))

        assert metrics.max_depth >= 4

    def test_configuration_file_metrics(self, analyzer):
        """Test metrics for configuration files."""
        code = """
[section1]
key1 = value1
key2 = value2

[section2]
key3 = value3
key4 = value4
key5 = value5
"""
        metrics = analyzer.calculate_complexity(code, Path("config.ini"))

        assert metrics.key_count >= 5
        assert metrics.section_count >= 2

    def test_markup_file_metrics(self, analyzer):
        """Test metrics for markup files."""
        code = """
<html>
<head>
    <title>Test</title>
</head>
<body>
    <h1>Header</h1>
    <p>Paragraph</p>
    <div>Content</div>
</body>
</html>

# Markdown Header
## Subheader
### Sub-subheader
"""
        metrics = analyzer.calculate_complexity(code, Path("doc.html"))

        assert metrics.tag_count >= 7
        assert metrics.header_count >= 3

    def test_data_file_metrics(self, analyzer):
        """Test metrics for data files."""
        code = """name,age,city
John,30,NYC
Jane,25,LA
Bob,35,Chicago
Alice,28,Boston"""

        metrics = analyzer.calculate_complexity(code, Path("data.csv"))

        assert metrics.column_count == 3
        assert metrics.row_count == 4  # Excluding header

    def test_maintainability_index(self, analyzer):
        """Test simplified maintainability index calculation."""
        # Simple file
        simple_code = """
function simple() {
    return 42;
}
"""
        simple_metrics = analyzer.calculate_complexity(simple_code, Path("simple.txt"))

        # Complex file
        complex_code = (
            """
if (a) { if (b) { if (c) { if (d) { if (e) {
    for (i = 0; i < 100; i++) {
        while (x) { switch(y) { case 1: case 2: case 3: break; }}
    }
}}}}}
"""
            * 50
        )  # Make it very long and complex

        complex_metrics = analyzer.calculate_complexity(complex_code, Path("complex.txt"))

        # Simple file should have better maintainability
        assert simple_metrics.maintainability_index > complex_metrics.maintainability_index


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GenericAnalyzer instance."""
        return GenericAnalyzer()

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("empty.txt"))
        exports = analyzer.extract_exports(code, Path("empty.txt"))
        structure = analyzer.extract_structure(code, Path("empty.txt"))
        metrics = analyzer.calculate_complexity(code, Path("empty.txt"))

        assert imports == []
        assert exports == []
        assert metrics.line_count == 1
        assert metrics.code_lines == 0

    def test_handle_binary_content(self, analyzer):
        """Test handling of binary-like content."""
        code = "\x00\x01\x02\x03\x04\x05"

        # Should not crash
        imports = analyzer.extract_imports(code, Path("binary.dat"))
        exports = analyzer.extract_exports(code, Path("binary.dat"))
        structure = analyzer.extract_structure(code, Path("binary.dat"))
        metrics = analyzer.calculate_complexity(code, Path("binary.dat"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_very_long_lines(self, analyzer):
        """Test handling of files with very long lines."""
        code = "x" * 10000 + "\n" + "y" * 10000

        # Should handle gracefully
        metrics = analyzer.calculate_complexity(code, Path("long.txt"))

        assert metrics.line_count == 2
        assert metrics.character_count == 20002

    def test_handle_mixed_encodings(self, analyzer):
        """Test handling of mixed character encodings."""
        code = """
ASCII text
UTF-8: 你好世界
Emoji: 😀🎉
Special: café, naïve
"""
        # Should handle various encodings
        structure = analyzer.extract_structure(code, Path("mixed.txt"))
        metrics = analyzer.calculate_complexity(code, Path("mixed.txt"))

        assert metrics.line_count == 4
        assert metrics.code_lines == 4


class TestEdgeCases:
    """Test suite for edge cases."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GenericAnalyzer instance."""
        return GenericAnalyzer()

    def test_mixed_content_file(self, analyzer):
        """Test handling of files with mixed content types."""
        code = """
<!-- HTML Comment -->
<script>
function jsFunction() {
    return "mixed";
}
</script>

<style>
.css-class {
    color: red;
}
</style>

<?php
function phpFunction() {
    echo "PHP";
}
?>

# Markdown section
Some markdown content
"""
        structure = analyzer.extract_structure(code, Path("mixed.html"))

        # Should extract various patterns
        assert len(structure.functions) >= 2
        assert len(structure.sections) >= 1

    def test_log_file_patterns(self, analyzer):
        """Test handling of log file patterns."""
        code = """
2024-01-01 10:00:00 INFO Starting application
2024-01-01 10:00:01 DEBUG Loading configuration from config.yml
2024-01-01 10:00:02 ERROR Failed to connect to database.host
2024-01-01 10:00:03 WARN Retrying connection
file=/var/log/app.log
path: /usr/local/bin
"""
        imports = analyzer.extract_imports(code, Path("app.log"))

        # Should extract file references from logs
        assert any("config.yml" in imp.module for imp in imports)

    def test_dockerfile_patterns(self, analyzer):
        """Test handling of Dockerfile patterns."""
        code = """
FROM node:14
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
"""
        structure = analyzer.extract_structure(code, Path("Dockerfile"))

        # Should detect as configuration file
        assert structure.file_type == "configuration"

    def test_makefile_patterns(self, analyzer):
        """Test handling of Makefile patterns."""
        code = """
CC = gcc
CFLAGS = -Wall -O2

all: program

program: main.o utils.o
	$(CC) $(CFLAGS) -o program main.o utils.o

clean:
	rm -f *.o program

.PHONY: all clean
"""
        structure = analyzer.extract_structure(code, Path("Makefile"))

        # Should extract targets as exports
        exports = analyzer.extract_exports(code, Path("Makefile"))
        export_names = [e["name"] for e in exports]

        assert "CC" in export_names
        assert "CFLAGS" in export_names

    def test_sql_patterns(self, analyzer):
        """Test handling of SQL patterns."""
        code = """
-- Database schema
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100)
);

CREATE INDEX idx_username ON users(username);

SELECT * FROM users WHERE active = 1;
INSERT INTO logs VALUES (1, 'test');
UPDATE settings SET value = 'new' WHERE key = 'config';
"""
        structure = analyzer.extract_structure(code, Path("schema.sql"))

        # Should detect as query file
        assert structure.file_type == "query"

    def test_extremely_nested_content(self, analyzer):
        """Test handling of extremely nested content."""
        # Generate deeply nested structure
        code = ""
        for i in range(20):
            code += "  " * i + "if (condition" + str(i) + ") {\n"
        for i in range(19, -1, -1):
            code += "  " * i + "}\n"

        metrics = analyzer.calculate_complexity(code, Path("deep.txt"))

        # Should cap max depth at reasonable level
        assert metrics.max_depth <= 10  # Capped at 10 for generic files
