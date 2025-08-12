"""
Unit tests for the JavaScript/TypeScript code analyzer.

This module tests the JavaScript and TypeScript code analysis functionality
including ES6+ features, JSX/TSX support, various module systems (ES6, CommonJS),
React components, and modern JavaScript patterns.

Test Coverage:
    - Import extraction (ES6, CommonJS, dynamic imports)
    - Export detection (default, named, re-exports)
    - Structure extraction (functions, classes, React components)
    - TypeScript-specific features (interfaces, types, enums)
    - Complexity metrics calculation
    - Framework detection (React, Vue, Angular)
    - Error handling for invalid JavaScript/TypeScript
    - Edge cases and modern JS features
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.javascript_analyzer import JavaScriptAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestJavaScriptAnalyzerInitialization:
    """Test suite for JavaScriptAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = JavaScriptAnalyzer()

        assert analyzer.language_name == "javascript"
        assert ".js" in analyzer.file_extensions
        assert ".jsx" in analyzer.file_extensions
        assert ".ts" in analyzer.file_extensions
        assert ".tsx" in analyzer.file_extensions
        assert ".mjs" in analyzer.file_extensions
        assert ".cjs" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for JavaScript/TypeScript import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaScriptAnalyzer instance."""
        return JavaScriptAnalyzer()

    def test_extract_es6_default_imports(self, analyzer):
        """Test extraction of ES6 default imports."""
        code = """
import React from 'react';
import axios from 'axios';
import MyComponent from './components/MyComponent';
"""
        imports = analyzer.extract_imports(code, Path("app.js"))

        assert len(imports) == 3
        
        # Check React import
        react_import = next(imp for imp in imports if imp.module == "react")
        assert react_import.alias == "React"
        assert react_import.type == "es6_default"
        assert react_import.is_relative is False
        assert react_import.line == 2

        # Check relative import
        component_import = next(imp for imp in imports if "./components/MyComponent" in imp.module)
        assert component_import.is_relative is True

    def test_extract_es6_named_imports(self, analyzer):
        """Test extraction of ES6 named imports."""
        code = """
import { useState, useEffect } from 'react';
import { Router, Route, Link } from 'react-router-dom';
import { Button as CustomButton, Input } from './components';
"""
        imports = analyzer.extract_imports(code, Path("app.js"))

        # Should create separate ImportInfo for each named import
        assert any(imp.alias == "useState" for imp in imports)
        assert any(imp.alias == "useEffect" for imp in imports)
        assert any(imp.alias == "Router" for imp in imports)

        # Check aliased import
        button_import = next(imp for imp in imports if imp.alias == "CustomButton")
        assert button_import.original_name == "Button"
        assert button_import.type == "es6_named"

    def test_extract_es6_namespace_imports(self, analyzer):
        """Test extraction of ES6 namespace imports."""
        code = """
import * as utils from './utils';
import * as React from 'react';
import * as styles from './styles.module.css';
"""
        imports = analyzer.extract_imports(code, Path("app.js"))

        assert len(imports) == 3
        
        utils_import = next(imp for imp in imports if imp.alias == "utils")
        assert utils_import.type == "es6_namespace"
        assert utils_import.module == "./utils"
        assert utils_import.is_relative is True

    def test_extract_es6_combined_imports(self, analyzer):
        """Test extraction of combined default and named imports."""
        code = """
import React, { Component, Fragment } from 'react';
import axios, { AxiosError, AxiosResponse } from 'axios';
"""
        imports = analyzer.extract_imports(code, Path("app.js"))

        # Should have both default and named imports
        react_default = next(imp for imp in imports if imp.alias == "React")
        assert react_default.type == "es6_default"

        component_import = next(imp for imp in imports if imp.alias == "Component")
        assert component_import.type == "es6_named"

    def test_extract_es6_side_effect_imports(self, analyzer):
        """Test extraction of side-effect imports."""
        code = """
import './styles.css';
import 'polyfill';
import './config/init';
"""
        imports = analyzer.extract_imports(code, Path("app.js"))

        assert len(imports) == 3
        
        css_import = next(imp for imp in imports if imp.module == "./styles.css")
        assert css_import.type == "es6_side_effect"
        assert css_import.is_relative is True

        polyfill_import = next(imp for imp in imports if imp.module == "polyfill")
        assert polyfill_import.type == "es6_side_effect"
        assert polyfill_import.is_relative is False

    def test_extract_commonjs_imports(self, analyzer):
        """Test extraction of CommonJS require statements."""
        code = """
const fs = require('fs');
const { readFile, writeFile } = require('fs/promises');
let path = require('path');
var express = require('express');
"""
        imports = analyzer.extract_imports(code, Path("app.js"))

        # Check regular require
        fs_import = next(imp for imp in imports if imp.module == "fs" and imp.alias == "fs")
        assert fs_import.type == "commonjs"

        # Check destructured require
        destructured = [imp for imp in imports if imp.module == "fs/promises"]
        assert len(destructured) >= 2
        assert any(imp.alias == "readFile" for imp in destructured)
        assert any(imp.alias == "writeFile" for imp in destructured)

    def test_extract_dynamic_imports(self, analyzer):
        """Test extraction of dynamic imports."""
        code = """
async function loadModule() {
    const module = await import('./module');
    import('./lazy-component').then(m => console.log(m));
    
    if (condition) {
        import('conditional-module');
    }
}
"""
        imports = analyzer.extract_imports(code, Path("app.js"))

        dynamic_imports = [imp for imp in imports if imp.type == "dynamic"]
        assert len(dynamic_imports) == 3
        
        assert any(imp.module == "./module" for imp in dynamic_imports)
        assert any(imp.module == "./lazy-component" for imp in dynamic_imports)
        assert any(imp.module == "conditional-module" for imp in dynamic_imports)

    def test_extract_typescript_type_imports(self, analyzer):
        """Test extraction of TypeScript type imports."""
        code = """
import type { User, UserProfile } from './types';
import type DefaultType from './default-type';
import { Component } from 'react';
import type { FC } from 'react';
"""
        imports = analyzer.extract_imports(code, Path("app.ts"))

        # Check type imports
        type_imports = [imp for imp in imports if imp.type == "ts_type"]
        assert len(type_imports) >= 3

        user_import = next(imp for imp in type_imports if imp.alias == "User")
        assert user_import.module == "./types"

        fc_import = next(imp for imp in type_imports if imp.alias == "FC")
        assert fc_import.module == "react"


class TestExportExtraction:
    """Test suite for JavaScript/TypeScript export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaScriptAnalyzer instance."""
        return JavaScriptAnalyzer()

    def test_extract_default_exports(self, analyzer):
        """Test extraction of default exports."""
        code = """
const MyComponent = () => {
    return <div>Hello</div>;
};

export default MyComponent;

// Direct default export
export default function App() {
    return <div>App</div>;
}

// Default class export
export default class MainClass {
    constructor() {}
}
"""
        exports = analyzer.extract_exports(code, Path("app.js"))

        default_exports = [exp for exp in exports if exp.get("is_default")]
        assert len(default_exports) >= 1

        # First default export
        first_default = default_exports[0]
        assert first_default["name"] == "default"

    def test_extract_named_exports(self, analyzer):
        """Test extraction of named exports."""
        code = """
export const API_URL = 'https://api.example.com';
export let counter = 0;
export var oldStyle = 'value';

export function helperFunction() {
    return true;
}

export class ExportedClass {
    method() {}
}

export async function asyncFunction() {
    await something();
}
"""
        exports = analyzer.extract_exports(code, Path("app.js"))

        export_names = [exp["name"] for exp in exports]
        assert "API_URL" in export_names
        assert "counter" in export_names
        assert "helperFunction" in export_names
        assert "ExportedClass" in export_names
        assert "asyncFunction" in export_names

        # Check export types
        api_export = next(exp for exp in exports if exp["name"] == "API_URL")
        assert api_export["is_const"] is True

        async_export = next(exp for exp in exports if exp["name"] == "asyncFunction")
        assert async_export["is_async"] is True

    def test_extract_export_list(self, analyzer):
        """Test extraction of export lists."""
        code = """
const var1 = 'value1';
const var2 = 'value2';
function func1() {}

export { var1, var2 };
export { func1 as exportedFunc };
"""
        exports = analyzer.extract_exports(code, Path("app.js"))

        assert any(exp["name"] == "var1" for exp in exports)
        assert any(exp["name"] == "var2" for exp in exports)
        
        # Check aliased export
        aliased = next(exp for exp in exports if exp["name"] == "exportedFunc")
        assert aliased.get("original_name") == "func1"

    def test_extract_re_exports(self, analyzer):
        """Test extraction of re-exports."""
        code = """
export { Button, Input } from './components';
export * from './utils';
export * as helpers from './helpers';
export { default as MyComponent } from './MyComponent';
"""
        exports = analyzer.extract_exports(code, Path("index.js"))

        re_exports = [exp for exp in exports if exp["type"] == "re-export" or exp["type"] == "re-export-all"]
        assert len(re_exports) >= 3

        # Check specific re-export
        button_export = next((exp for exp in exports if exp["name"] == "Button"), None)
        if button_export:
            assert button_export["from_module"] == "./components"

        # Check namespace re-export
        helpers_export = next((exp for exp in exports if exp["name"] == "helpers"), None)
        if helpers_export:
            assert helpers_export["from_module"] == "./helpers"

    def test_extract_commonjs_exports(self, analyzer):
        """Test extraction of CommonJS exports."""
        code = """
module.exports = {
    function1,
    function2,
    constant: CONSTANT_VALUE
};

exports.singleExport = function() {};
exports.anotherExport = 'value';

module.exports.directExport = () => {};
"""
        exports = analyzer.extract_exports(code, Path("app.js"))

        # Check module.exports
        module_export = next((exp for exp in exports if exp["name"] == "module.exports"), None)
        assert module_export is not None

        # Check individual exports
        assert any(exp["name"] == "singleExport" for exp in exports)
        assert any(exp["name"] == "anotherExport" for exp in exports)
        assert any(exp["name"] == "directExport" for exp in exports)

    def test_extract_typescript_exports(self, analyzer):
        """Test extraction of TypeScript-specific exports."""
        code = """
export interface UserInterface {
    id: number;
    name: string;
}

export type UserType = {
    id: number;
    name: string;
};

export enum Status {
    Active,
    Inactive
}

export type { Config } from './config';
"""
        exports = analyzer.extract_exports(code, Path("app.ts"))

        export_names = [exp["name"] for exp in exports]
        assert "UserInterface" in export_names
        assert "UserType" in export_names
        assert "Status" in export_names

        # Check types
        interface_export = next(exp for exp in exports if exp["name"] == "UserInterface")
        assert interface_export["type"] == "interface"

        type_export = next(exp for exp in exports if exp["name"] == "UserType")
        assert type_export["type"] == "type"

        enum_export = next(exp for exp in exports if exp["name"] == "Status")
        assert enum_export["type"] == "enum"


class TestStructureExtraction:
    """Test suite for code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaScriptAnalyzer instance."""
        return JavaScriptAnalyzer()

    def test_extract_functions(self, analyzer):
        """Test extraction of function declarations."""
        code = """
function regularFunction(a, b) {
    return a + b;
}

async function asyncFunction() {
    await somePromise();
}

function* generatorFunction() {
    yield 1;
    yield 2;
}

const arrowFunction = (x, y) => x + y;

const asyncArrow = async () => {
    return await fetch('/api');
};

export function exportedFunction() {}
"""
        structure = analyzer.extract_structure(code, Path("app.js"))

        assert len(structure.functions) >= 6

        # Check regular function
        regular = next(f for f in structure.functions if f.name == "regularFunction")
        assert len(regular.args) == 2
        assert regular.is_async is False

        # Check async function
        async_func = next(f for f in structure.functions if f.name == "asyncFunction")
        assert async_func.is_async is True

        # Check generator function
        generator = next(f for f in structure.functions if f.name == "generatorFunction")
        assert generator.is_generator is True

        # Check arrow function
        arrow = next(f for f in structure.functions if f.name == "arrowFunction")
        assert arrow.is_arrow is True

        # Check exported function
        exported = next(f for f in structure.functions if f.name == "exportedFunction")
        assert exported.is_exported is True

    def test_extract_classes(self, analyzer):
        """Test extraction of class declarations."""
        code = """
class BaseClass {
    constructor(name) {
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
    
    static staticMethod() {
        return 'static';
    }
    
    #privateMethod() {
        return 'private';
    }
}

class DerivedClass extends BaseClass {
    constructor(name, age) {
        super(name);
        this.age = age;
    }
}

export class ExportedClass {
    method() {}
}
"""
        structure = analyzer.extract_structure(code, Path("app.js"))

        assert len(structure.classes) == 3

        # Check base class
        base_class = next(c for c in structure.classes if c.name == "BaseClass")
        assert len(base_class.methods) >= 4
        
        constructor = next(m for m in base_class.methods if m["name"] == "constructor")
        assert constructor["is_constructor"] is True

        static_method = next(m for m in base_class.methods if m["name"] == "staticMethod")
        assert static_method["is_static"] is True

        private_method = next(m for m in base_class.methods if m["name"] == "#privateMethod")
        assert private_method["is_private"] is True

        # Check derived class
        derived_class = next(c for c in structure.classes if c.name == "DerivedClass")
        assert "BaseClass" in derived_class.bases

        # Check exported class
        exported_class = next(c for c in structure.classes if c.name == "ExportedClass")
        assert exported_class.is_exported is True

    def test_extract_react_components(self, analyzer):
        """Test extraction of React components."""
        code = """
import React from 'react';

function FunctionComponent({ name }) {
    return <div>Hello {name}</div>;
}

const ArrowComponent = ({ children }) => {
    return <div>{children}</div>;
};

class ClassComponent extends React.Component {
    render() {
        return <div>Class Component</div>;
    }
}

export default function DefaultComponent() {
    return <div>Default</div>;
}

const MemoizedComponent = React.memo(({ data }) => {
    return <div>{data}</div>;
});
"""
        structure = analyzer.extract_structure(code, Path("app.jsx"))

        # Components should be detected
        assert len(structure.components) >= 5

        component_names = [c["name"] for c in structure.components]
        assert "FunctionComponent" in component_names
        assert "ArrowComponent" in component_names
        assert "ClassComponent" in component_names
        assert "DefaultComponent" in component_names
        assert "MemoizedComponent" in component_names

        # Check component types
        func_component = next(c for c in structure.components if c["name"] == "FunctionComponent")
        assert func_component["type"] == "functional"

        class_component = next(c for c in structure.components if c["name"] == "ClassComponent")
        assert class_component["type"] == "class"

    def test_extract_typescript_structures(self, analyzer):
        """Test extraction of TypeScript-specific structures."""
        code = """
interface User {
    id: number;
    name: string;
    email?: string;
}

interface Admin extends User {
    permissions: string[];
}

type Status = 'active' | 'inactive' | 'pending';

type UserWithStatus = User & { status: Status };

enum Role {
    User = 'USER',
    Admin = 'ADMIN',
    Guest = 'GUEST'
}

const enum ConstEnum {
    One = 1,
    Two = 2
}
"""
        structure = analyzer.extract_structure(code, Path("app.ts"))

        # Check interfaces
        assert len(structure.interfaces) >= 2
        
        user_interface = next(i for i in structure.interfaces if i["name"] == "User")
        assert user_interface["is_exported"] is False

        admin_interface = next(i for i in structure.interfaces if i["name"] == "Admin")
        assert "User" in admin_interface["extends"]

        # Check types
        assert len(structure.types) >= 2
        assert any(t["name"] == "Status" for t in structure.types)
        assert any(t["name"] == "UserWithStatus" for t in structure.types)

        # Check enums
        assert len(structure.enums) >= 2
        
        role_enum = next(e for e in structure.enums if e["name"] == "Role")
        assert role_enum["is_const"] is False

        const_enum = next(e for e in structure.enums if e["name"] == "ConstEnum")
        assert const_enum["is_const"] is True

    def test_detect_framework(self, analyzer):
        """Test framework detection."""
        # React detection
        react_code = """
import React, { useState } from 'react';
import { render } from 'react-dom';

function App() {
    const [count, setCount] = useState(0);
    return <div>{count}</div>;
}
"""
        react_structure = analyzer.extract_structure(react_code, Path("app.jsx"))
        assert react_structure.framework == "React"

        # Vue detection
        vue_code = """
import { createApp, ref } from 'vue';

export default {
    data() {
        return {
            message: 'Hello Vue!'
        }
    },
    mounted() {
        console.log('Component mounted');
    }
}
"""
        vue_structure = analyzer.extract_structure(vue_code, Path("app.vue"))
        assert vue_structure.framework == "Vue"

        # Angular detection
        angular_code = """
import { Component } from '@angular/core';
import { Injectable } from '@angular/core';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html'
})
export class AppComponent {
    title = 'my-app';
}
"""
        angular_structure = analyzer.extract_structure(angular_code, Path("app.ts"))
        assert angular_structure.framework == "Angular"

        # Node.js detection
        node_code = """
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
"""
        node_structure = analyzer.extract_structure(node_code, Path("server.js"))
        assert node_structure.framework == "Node.js"


class TestComplexityCalculation:
    """Test suite for complexity metrics calculation."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaScriptAnalyzer instance."""
        return JavaScriptAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
function complexFunction(value) {
    if (value > 0) {
        if (value > 10) {
            return 'big';
        } else {
            return 'small';
        }
    } else if (value < 0) {
        return 'negative';
    } else {
        return 'zero';
    }
    
    for (let i = 0; i < 10; i++) {
        if (i % 2 === 0) {
            console.log('even');
        }
    }
    
    while (value > 0) {
        value--;
    }
    
    switch (value) {
        case 1:
            break;
        case 2:
            break;
        default:
            break;
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        # Base complexity = 1
        # +1 for each decision point
        assert metrics.cyclomatic >= 10

    def test_calculate_cognitive_complexity(self, analyzer):
        """Test cognitive complexity calculation."""
        code = """
function nested(x) {
    if (x > 0) {  // +1
        for (let i = 0; i < x; i++) {  // +2 (1 + nesting)
            if (i % 2 === 0) {  // +3 (1 + 2*nesting)
                if (i > 5) {  // +4 (1 + 3*nesting)
                    console.log(i);
                }
            }
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        # Cognitive complexity considers nesting
        assert metrics.cognitive >= 10

    def test_calculate_with_ternary_operators(self, analyzer):
        """Test complexity with ternary operators."""
        code = """
const result = condition ? value1 : value2;
const nested = a > 0 ? (b > 0 ? 'both' : 'a') : (b > 0 ? 'b' : 'none');
const chained = x ? y ? z ? 'all' : 'xy' : 'x' : 'none';
"""
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        # Ternary operators add to complexity
        assert metrics.cyclomatic >= 6

    def test_calculate_with_logical_operators(self, analyzer):
        """Test complexity with logical operators."""
        code = """
function checkConditions(a, b, c) {
    if (a && b && c) {
        return 'all';
    }
    
    if (a || b || c) {
        return 'some';
    }
    
    const result = a && (b || c) && (d || e);
    
    return a ?? b ?? c ?? 'default';
}
"""
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        # Logical operators add to complexity
        assert metrics.cyclomatic >= 8

    def test_calculate_with_try_catch(self, analyzer):
        """Test complexity with try-catch blocks."""
        code = """
async function errorHandler() {
    try {
        await riskyOperation();
    } catch (error) {
        console.error(error);
    } finally {
        cleanup();
    }
    
    try {
        JSON.parse(data);
    } catch {
        return null;
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        # Try-catch blocks add to complexity
        assert metrics.cyclomatic >= 3


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaScriptAnalyzer instance."""
        return JavaScriptAnalyzer()

    def test_handle_syntax_error(self, analyzer):
        """Test handling of files with syntax errors."""
        code = """
function broken() {
    this is not valid JavaScript
    const x = {
}
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("app.js"))
        exports = analyzer.extract_exports(code, Path("app.js"))
        structure = analyzer.extract_structure(code, Path("app.js"))
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("app.js"))
        exports = analyzer.extract_exports(code, Path("app.js"))
        structure = analyzer.extract_structure(code, Path("app.js"))
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        assert imports == []
        assert exports == []
        assert len(structure.functions) == 0
        assert metrics.line_count == 1

    def test_handle_comments_only(self, analyzer):
        """Test handling of files with only comments."""
        code = """
// This file contains only comments
/* Multi-line comment
   with no actual code */
// Just documentation
"""
        structure = analyzer.extract_structure(code, Path("app.js"))
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        assert len(structure.functions) == 0
        assert len(structure.classes) == 0
        assert metrics.comment_lines > 0
        assert metrics.code_lines == 0


class TestEdgeCases:
    """Test suite for edge cases and modern JavaScript features."""

    @pytest.fixture
    def analyzer(self):
        """Provide a JavaScriptAnalyzer instance."""
        return JavaScriptAnalyzer()

    def test_destructuring_patterns(self, analyzer):
        """Test handling of destructuring patterns."""
        code = """
const { prop1, prop2: renamed } = object;
const [first, second, ...rest] = array;

function withDestructuring({ name, age = 18 }) {
    return `${name} is ${age}`;
}

const complexDestructure = ({ a: { b: { c } } }) => c;
"""
        structure = analyzer.extract_structure(code, Path("app.js"))
        
        # Should handle destructuring in functions
        func = next(f for f in structure.functions if f.name == "withDestructuring")
        assert len(func.args) == 1

    def test_spread_and_rest_operators(self, analyzer):
        """Test handling of spread and rest operators."""
        code = """
function withRest(...args) {
    return args.length;
}

const withSpread = (a, b, ...rest) => {
    return [...rest, a, b];
};

const objectSpread = { ...baseObject, newProp: 'value' };
"""
        structure = analyzer.extract_structure(code, Path("app.js"))

        # Should detect functions with rest parameters
        assert len(structure.functions) >= 2

    def test_template_literals(self, analyzer):
        """Test handling of template literals."""
        code = """
const template = `Hello ${name}`;
const multiline = `
    Line 1
    Line 2
`;
const nested = `Outer ${`Inner ${value}`}`;
const tagged = myTag`Template ${expression}`;
"""
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        # Template literals shouldn't break parsing
        assert metrics.line_count > 0

    def test_async_await_patterns(self, analyzer):
        """Test handling of async/await patterns."""
        code = """
async function sequential() {
    const result1 = await operation1();
    const result2 = await operation2();
    return [result1, result2];
}

const parallel = async () => {
    const [r1, r2] = await Promise.all([
        operation1(),
        operation2()
    ]);
    return { r1, r2 };
};

async function* asyncGenerator() {
    yield await getValue1();
    yield await getValue2();
}

for await (const value of asyncIterable) {
    process(value);
}
"""
        structure = analyzer.extract_structure(code, Path("app.js"))

        # Check async functions
        async_funcs = [f for f in structure.functions if f.is_async]
        assert len(async_funcs) >= 2

        # Check async generator
        async_gen = next(f for f in structure.functions if f.name == "asyncGenerator")
        assert async_gen.is_async is True
        assert async_gen.is_generator is True

    def test_class_fields_and_private_methods(self, analyzer):
        """Test handling of class fields and private methods."""
        code = """
class ModernClass {
    publicField = 'public';
    #privateField = 'private';
    static staticField = 'static';
    static #privateStatic = 'private static';
    
    #privateMethod() {
        return this.#privateField;
    }
    
    static #privateStaticMethod() {
        return this.#privateStatic;
    }
    
    get #privateGetter() {
        return this.#privateField;
    }
    
    set #privateSetter(value) {
        this.#privateField = value;
    }
}
"""
        structure = analyzer.extract_structure(code, Path("app.js"))

        modern_class = structure.classes[0]
        
        # Should detect private methods
        private_methods = [m for m in modern_class.methods if m["is_private"]]
        assert len(private_methods) >= 3

    def test_jsx_and_tsx_content(self, analyzer):
        """Test handling of JSX/TSX content."""
        code = """
const Component = ({ items }: { items: string[] }) => {
    return (
        <div className="container">
            <h1>Title</h1>
            {items.map((item, index) => (
                <div key={index}>
                    {item}
                </div>
            ))}
            <button onClick={() => console.log('clicked')}>
                Click me
            </button>
        </div>
    );
};

const Fragment = () => (
    <>
        <span>First</span>
        <span>Second</span>
    </>
);
"""
        structure = analyzer.extract_structure(code, Path("app.tsx"))

        # Should detect React components
        assert len(structure.components) >= 2

    def test_module_patterns(self, analyzer):
        """Test various module patterns."""
        code = """
// IIFE module pattern
const Module = (function() {
    let private = 0;
    
    return {
        increment: () => private++,
        get: () => private
    };
})();

// Revealing module pattern
const RevealingModule = (() => {
    function privateMethod() {}
    function publicMethod() {}
    
    return {
        public: publicMethod
    };
})();

// ES6 module
export const modernModule = {
    method1() {},
    method2() {}
};
"""
        structure = analyzer.extract_structure(code, Path("app.js"))

        # Should detect module patterns
        assert len(structure.variables) >= 2

    def test_decorators(self, analyzer):
        """Test handling of decorators (experimental feature)."""
        code = """
@decorator
class DecoratedClass {
    @readonly
    property = 'value';
    
    @memoize
    expensiveMethod() {
        return compute();
    }
    
    @deprecated
    oldMethod() {}
}

@Injectable()
@Component({
    selector: 'app-component'
})
export class AngularComponent {}
"""
        structure = analyzer.extract_structure(code, Path("app.ts"))

        # Should handle decorators without breaking
        assert len(structure.classes) >= 2

    def test_optional_chaining_and_nullish_coalescing(self, analyzer):
        """Test handling of optional chaining and nullish coalescing."""
        code = """
const value = object?.property?.nested;
const method = object?.method?.();
const array = object?.[index];

const defaulted = value ?? 'default';
const combined = object?.prop ?? fallback?.value ?? 'final';

function withOptional(param?) {
    return param?.toString() ?? 'no value';
}
"""
        structure = analyzer.extract_structure(code, Path("app.js"))
        metrics = analyzer.calculate_complexity(code, Path("app.js"))

        # Should handle modern operators
        assert len(structure.variables) >= 3
        assert metrics.cyclomatic >= 1