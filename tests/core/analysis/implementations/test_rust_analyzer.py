"""
Unit tests for the Rust code analyzer.

This module tests the Rust-specific code analysis functionality including
use statement extraction, struct/enum/trait parsing, function extraction,
and complexity calculation. The Rust analyzer handles modern Rust features
including ownership, lifetimes, async/await, and generics.

Test Coverage:
    - Use statement and crate extraction
    - Export detection (pub items)
    - Structure extraction (structs, enums, traits, impls)
    - Complexity metrics (cyclomatic, cognitive, unsafe code)
    - Rust-specific features (lifetimes, generics, macros)
    - Error handling for invalid Rust code
    - Edge cases and Rust idioms
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tenets.core.analysis.implementations.rust_analyzer import RustAnalyzer
from tenets.models.analysis import (
    ClassInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ImportInfo,
)


class TestRustAnalyzerInitialization:
    """Test suite for RustAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = RustAnalyzer()

        assert analyzer.language_name == "rust"
        assert ".rs" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for Rust import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RustAnalyzer instance."""
        return RustAnalyzer()

    def test_extract_use_statements(self, analyzer):
        """Test extraction of use statements."""
        code = """
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::sync::Arc;

use crate::models::User;
use super::utils::helper;
use self::internal::process;

extern crate serde;
extern crate tokio;
"""
        imports = analyzer.extract_imports(code, Path("main.rs"))

        assert len(imports) >= 8

        # Check standard library imports
        hashmap_import = next(imp for imp in imports if "HashMap" in imp.module)
        assert hashmap_import.type == "use"
        assert hashmap_import.is_relative is False

        # Check relative imports
        crate_import = next(imp for imp in imports if imp.module.startswith("crate::"))
        assert crate_import.is_relative is True

        super_import = next(imp for imp in imports if imp.module.startswith("super::"))
        assert super_import.is_relative is True

        # Check extern crates
        serde_import = next(imp for imp in imports if imp.module == "serde")
        assert serde_import.type == "extern_crate"
        assert serde_import.is_external is True

    def test_extract_grouped_imports(self, analyzer):
        """Test extraction of grouped imports."""
        code = """
use std::{
    collections::{HashMap, HashSet},
    io::{self, Read, Write},
    sync::{Arc, Mutex},
};

use tokio::{
    io::AsyncReadExt,
    net::TcpListener,
    sync::mpsc,
};
"""
        imports = analyzer.extract_imports(code, Path("main.rs"))

        # Should expand grouped imports
        assert any("HashMap" in imp.module for imp in imports)
        assert any("HashSet" in imp.module for imp in imports)
        assert any("Arc" in imp.module for imp in imports)
        assert any("Mutex" in imp.module for imp in imports)
        assert any("AsyncReadExt" in imp.module for imp in imports)

    def test_extract_aliased_imports(self, analyzer):
        """Test extraction of aliased imports."""
        code = """
use std::collections::HashMap as Map;
use std::io::Result as IoResult;
use std::thread::spawn as create_thread;

extern crate serde_json as json;
"""
        imports = analyzer.extract_imports(code, Path("main.rs"))

        # Check aliases
        map_import = next(imp for imp in imports if "HashMap" in imp.module)
        assert map_import.alias == "Map"

        result_import = next(imp for imp in imports if "Result" in imp.module)
        assert result_import.alias == "IoResult"

        json_import = next(imp for imp in imports if "serde_json" in imp.module)
        assert json_import.alias == "json"

    def test_extract_glob_imports(self, analyzer):
        """Test extraction of glob imports."""
        code = """
use std::io::prelude::*;
use super::utils::*;
"""
        imports = analyzer.extract_imports(code, Path("main.rs"))

        glob_imports = [imp for imp in imports if imp.module.endswith("*")]
        assert len(glob_imports) == 2

        for imp in glob_imports:
            assert imp.is_glob is True
            assert imp.type == "use_glob"

    def test_extract_cargo_dependencies(self, analyzer):
        """Test extraction from Cargo.toml."""
        code = """
[package]
name = "my_project"
version = "0.1.0"

[dependencies]
serde = "1.0"
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.11", default-features = false }

[dev-dependencies]
criterion = "0.3"
mockito = "0.31"
"""
        imports = analyzer.extract_imports(code, Path("Cargo.toml"))

        # Check regular dependencies
        serde_dep = next(imp for imp in imports if imp.module == "serde")
        assert serde_dep.type == "cargo_dependency"
        assert serde_dep.version == "1.0"

        # Check dev dependencies
        criterion_dep = next(imp for imp in imports if imp.module == "criterion")
        assert criterion_dep.type == "cargo_dev_dependency"
        assert criterion_dep.is_dev_dependency is True

    def test_extract_mod_declarations(self, analyzer):
        """Test extraction of mod declarations."""
        code = """
mod utils;
pub mod models;
mod tests;

mod inline_module {
    // inline module content
}
"""
        imports = analyzer.extract_imports(code, Path("lib.rs"))

        mod_imports = [imp for imp in imports if imp.type == "mod"]
        assert len(mod_imports) >= 3

        # Check module declarations
        utils_mod = next(imp for imp in imports if imp.module == "utils")
        assert utils_mod.is_module_declaration is True


class TestExportExtraction:
    """Test suite for Rust export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RustAnalyzer instance."""
        return RustAnalyzer()

    def test_extract_public_functions(self, analyzer):
        """Test extraction of public functions."""
        code = """
pub fn public_function() -> String {
    String::from("public")
}

fn private_function() -> i32 {
    42
}

pub async fn async_function() -> Result<(), Error> {
    Ok(())
}

pub unsafe fn unsafe_function() {
    // unsafe operations
}

pub const fn const_function() -> i32 {
    42
}

pub extern "C" fn c_function(x: i32) -> i32 {
    x * 2
}
"""
        exports = analyzer.extract_exports(code, Path("lib.rs"))

        func_exports = [e for e in exports if e["type"] == "function"]

        func_names = [f["name"] for f in func_exports]
        assert "public_function" in func_names
        assert "async_function" in func_names
        assert "unsafe_function" in func_names
        assert "const_function" in func_names
        assert "c_function" in func_names
        assert "private_function" not in func_names

        # Check function properties
        async_func = next(f for f in func_exports if f["name"] == "async_function")
        assert async_func["is_async"] is True

        unsafe_func = next(f for f in func_exports if f["name"] == "unsafe_function")
        assert unsafe_func["is_unsafe"] is True

        const_func = next(f for f in func_exports if f["name"] == "const_function")
        assert const_func["is_const"] is True

    def test_extract_public_structs(self, analyzer):
        """Test extraction of public structs."""
        code = """
pub struct PublicStruct {
    pub field1: String,
    field2: i32,  // private field
}

struct PrivateStruct {
    data: Vec<u8>,
}

pub struct TupleStruct(pub i32, String);

pub struct UnitStruct;

#[derive(Debug, Clone)]
pub struct DerivedStruct {
    value: String,
}
"""
        exports = analyzer.extract_exports(code, Path("lib.rs"))

        struct_exports = [e for e in exports if e["type"] == "struct"]

        struct_names = [s["name"] for s in struct_exports]
        assert "PublicStruct" in struct_names
        assert "TupleStruct" in struct_names
        assert "UnitStruct" in struct_names
        assert "DerivedStruct" in struct_names
        assert "PrivateStruct" not in struct_names

    def test_extract_public_enums(self, analyzer):
        """Test extraction of public enums."""
        code = """
pub enum Result<T, E> {
    Ok(T),
    Err(E),
}

enum PrivateEnum {
    VariantA,
    VariantB,
}

pub enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
"""
        exports = analyzer.extract_exports(code, Path("lib.rs"))

        enum_exports = [e for e in exports if e["type"] == "enum"]

        enum_names = [e["name"] for e in enum_exports]
        assert "Result" in enum_names
        assert "Message" in enum_names
        assert "PrivateEnum" not in enum_names

    def test_extract_public_traits(self, analyzer):
        """Test extraction of public traits."""
        code = """
pub trait Display {
    fn fmt(&self) -> String;
}

trait PrivateTrait {
    fn method(&self);
}

pub unsafe trait UnsafeTrait {
    unsafe fn unsafe_method(&self);
}
"""
        exports = analyzer.extract_exports(code, Path("lib.rs"))

        trait_exports = [e for e in exports if e["type"] == "trait"]

        trait_names = [t["name"] for t in trait_exports]
        assert "Display" in trait_names
        assert "UnsafeTrait" in trait_names
        assert "PrivateTrait" not in trait_names

        # Check unsafe trait
        unsafe_trait = next(t for t in trait_exports if t["name"] == "UnsafeTrait")
        assert unsafe_trait["is_unsafe"] is True

    def test_extract_type_aliases_and_constants(self, analyzer):
        """Test extraction of type aliases and constants."""
        code = """
pub type Result<T> = std::result::Result<T, Error>;
type PrivateAlias = Vec<String>;

pub const MAX_SIZE: usize = 1024;
const PRIVATE_CONST: i32 = 42;

pub static GLOBAL_CONFIG: Config = Config::default();
pub static mut MUTABLE_STATIC: i32 = 0;
"""
        exports = analyzer.extract_exports(code, Path("lib.rs"))

        # Check type aliases
        type_exports = [e for e in exports if e["type"] == "type_alias"]
        assert any(e["name"] == "Result" for e in type_exports)
        assert not any(e["name"] == "PrivateAlias" for e in type_exports)

        # Check constants
        const_exports = [e for e in exports if e["type"] == "constant"]
        assert any(e["name"] == "MAX_SIZE" for e in const_exports)
        assert not any(e["name"] == "PRIVATE_CONST" for e in const_exports)

        # Check statics
        static_exports = [e for e in exports if e["type"] == "static"]
        assert any(e["name"] == "GLOBAL_CONFIG" for e in static_exports)

        mutable_static = next(e for e in static_exports if e["name"] == "MUTABLE_STATIC")
        assert mutable_static["is_mutable"] is True

    def test_extract_public_macros(self, analyzer):
        """Test extraction of exported macros."""
        code = """
#[macro_export]
macro_rules! println_custom {
    ($($arg:tt)*) => {
        println!("Custom: {}", $($arg)*);
    };
}

macro_rules! internal_macro {
    () => {};
}
"""
        exports = analyzer.extract_exports(code, Path("lib.rs"))

        macro_exports = [e for e in exports if e["type"] == "macro"]
        assert len(macro_exports) == 1
        assert macro_exports[0]["name"] == "println_custom"


class TestStructureExtraction:
    """Test suite for Rust code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RustAnalyzer instance."""
        return RustAnalyzer()

    def test_extract_struct_with_generics(self, analyzer):
        """Test extraction of structs with generics and lifetimes."""
        code = """
struct Point<T> {
    x: T,
    y: T,
}

struct Reference<'a, T> {
    data: &'a T,
}

struct Complex<'a, T: Clone + Debug> 
where
    T: Display,
{
    value: T,
    reference: &'a str,
}
"""
        structure = analyzer.extract_structure(code, Path("structs.rs"))

        assert len(structure.classes) == 3  # Structs stored as classes

        point_struct = next(c for c in structure.classes if c.name == "Point")
        assert point_struct.generics == "T"

        ref_struct = next(c for c in structure.classes if c.name == "Reference")
        assert "'a" in ref_struct.generics

    def test_extract_enum_variants(self, analyzer):
        """Test extraction of enum variants."""
        code = """
enum Option<T> {
    Some(T),
    None,
}

enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

enum HttpStatus {
    Ok = 200,
    NotFound = 404,
    InternalError = 500,
}
"""
        structure = analyzer.extract_structure(code, Path("enums.rs"))

        assert len(structure.enums) == 4

        # Check enum variants
        message_enum = next(e for e in structure.enums if e["name"] == "Message")
        assert len(message_enum["variants"]) == 4

        quit_variant = next(v for v in message_enum["variants"] if v["name"] == "Quit")
        assert quit_variant["type"] == "unit"

        move_variant = next(v for v in message_enum["variants"] if v["name"] == "Move")
        assert move_variant["type"] == "struct"

        write_variant = next(v for v in message_enum["variants"] if v["name"] == "Write")
        assert write_variant["type"] == "tuple"

        # Check discriminant values
        http_enum = next(e for e in structure.enums if e["name"] == "HttpStatus")
        ok_variant = next(v for v in http_enum["variants"] if v["name"] == "Ok")
        assert ok_variant["discriminant"] == 200

    def test_extract_traits_and_impls(self, analyzer):
        """Test extraction of traits and implementations."""
        code = """
trait Display {
    fn fmt(&self) -> String;
}

trait Iterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
    
    fn count(self) -> usize 
    where 
        Self: Sized,
    {
        let mut count = 0;
        // default implementation
        count
    }
}

struct MyStruct {
    value: i32,
}

impl MyStruct {
    fn new(value: i32) -> Self {
        MyStruct { value }
    }
}

impl Display for MyStruct {
    fn fmt(&self) -> String {
        format!("{}", self.value)
    }
}

impl<T> From<T> for MyWrapper<T> {
    fn from(value: T) -> Self {
        MyWrapper(value)
    }
}
"""
        structure = analyzer.extract_structure(code, Path("traits.rs"))

        assert len(structure.traits) == 2

        # Check trait methods
        display_trait = next(t for t in structure.traits if t["name"] == "Display")
        assert len(display_trait["methods"]) == 1

        iterator_trait = next(t for t in structure.traits if t["name"] == "Iterator")
        assert len(iterator_trait["methods"]) == 2

        # Check default implementation
        count_method = next(m for m in iterator_trait["methods"] if m["name"] == "count")
        assert count_method["has_default"] is True

        # Check impl blocks
        assert len(structure.impl_blocks) >= 3

        # Check inherent impl
        mystruct_impl = next(
            i for i in structure.impl_blocks if i["type"] == "MyStruct" and i["trait"] is None
        )
        assert len(mystruct_impl["methods"]) >= 1

        # Check trait impl
        display_impl = next(i for i in structure.impl_blocks if i["trait"] == "Display")
        assert display_impl["type"] == "MyStruct"

    def test_extract_async_code(self, analyzer):
        """Test extraction of async functions and blocks."""
        code = """
async fn fetch_data() -> Result<String, Error> {
    let response = client.get(url).await?;
    Ok(response.text().await?)
}

fn spawn_task() {
    tokio::spawn(async move {
        process().await;
    });
}

async fn concurrent_tasks() {
    let (result1, result2) = tokio::join!(
        fetch_data(),
        another_async_fn()
    );
}
"""
        structure = analyzer.extract_structure(code, Path("async.rs"))

        # Check async functions
        assert structure.async_functions >= 2

        # Check await points
        assert structure.await_points >= 3

        # Check async function extraction
        async_func = next(f for f in structure.functions if f.name == "fetch_data")
        assert async_func.is_async is True

    def test_detect_unsafe_code(self, analyzer):
        """Test detection of unsafe code blocks."""
        code = """
unsafe fn dangerous_function() {
    // unsafe operations
}

fn safe_wrapper() {
    unsafe {
        dangerous_function();
        
        let raw_ptr = &mut 10 as *mut i32;
        *raw_ptr = 20;
    }
}

unsafe trait UnsafeTrait {
    unsafe fn unsafe_method(&self);
}

unsafe impl UnsafeTrait for MyType {
    unsafe fn unsafe_method(&self) {
        // implementation
    }
}
"""
        structure = analyzer.extract_structure(code, Path("unsafe.rs"))

        assert structure.unsafe_blocks >= 1
        assert structure.unsafe_functions >= 1

        # Check unsafe trait
        unsafe_trait = next(t for t in structure.traits if t["name"] == "UnsafeTrait")
        assert unsafe_trait["is_unsafe"] is True

        # Check unsafe impl
        unsafe_impl = next(i for i in structure.impl_blocks if i["is_unsafe"])
        assert unsafe_impl is not None

    def test_detect_test_functions(self, analyzer):
        """Test detection of test functions and benchmarks."""
        code = """
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_addition() {
        assert_eq!(2 + 2, 4);
    }
    
    #[test]
    #[should_panic]
    fn test_panic() {
        panic!("This should panic");
    }
    
    #[bench]
    fn bench_performance(b: &mut Bencher) {
        b.iter(|| {
            // benchmark code
        });
    }
}
"""
        structure = analyzer.extract_structure(code, Path("lib.rs"))

        assert structure.test_functions >= 2
        assert structure.bench_functions >= 1


class TestComplexityCalculation:
    """Test suite for Rust complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RustAnalyzer instance."""
        return RustAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
fn complex_function(x: i32, y: i32) -> i32 {
    if x > 0 {
        if y > 0 {
            x + y
        } else {
            x - y
        }
    } else if x < 0 {
        match y {
            0 => 0,
            1..=10 => y,
            _ => -1,
        }
    } else {
        if y > 0 || x == 0 {
            1
        } else {
            0
        }
    }
}

fn error_handling() -> Result<(), Error> {
    let data = fetch_data()?;
    
    process(data).unwrap();
    
    other_operation().expect("Failed");
    
    Ok(())
}
"""
        metrics = analyzer.calculate_complexity(code, Path("complex.rs"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 10

    def test_calculate_unsafe_metrics(self, analyzer):
        """Test unsafe code metrics."""
        code = """
unsafe fn unsafe_function() {
    // unsafe operations
}

unsafe trait UnsafeTrait {
    unsafe fn method(&self);
}

struct SafeWrapper;

unsafe impl UnsafeTrait for SafeWrapper {
    unsafe fn method(&self) {
        // implementation
    }
}

fn with_unsafe_blocks() {
    unsafe {
        let ptr = 0x12345 as *const i32;
        let val = *ptr;
    }
    
    unsafe {
        another_unsafe_operation();
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("unsafe.rs"))

        assert metrics.unsafe_blocks == 2
        assert metrics.unsafe_functions == 1
        assert metrics.unsafe_traits == 1
        assert metrics.unsafe_impl == 1
        assert metrics.unsafe_score > 0

    def test_calculate_lifetime_metrics(self, analyzer):
        """Test lifetime complexity metrics."""
        code = """
struct Container<'a> {
    data: &'a str,
}

fn with_lifetimes<'a, 'b>(x: &'a str, y: &'b str) -> &'a str 
where
    'b: 'a,
{
    x
}

struct Complex<'a, 'b: 'a> {
    first: &'a str,
    second: &'b str,
}
"""
        metrics = analyzer.calculate_complexity(code, Path("lifetimes.rs"))

        assert metrics.lifetime_annotations > 0
        assert metrics.lifetime_bounds > 0

    def test_calculate_generic_metrics(self, analyzer):
        """Test generic type metrics."""
        code = """
fn generic_function<T>(x: T) -> T {
    x
}

fn bounded_generic<T: Clone + Debug>(x: T) -> T {
    x.clone()
}

struct Container<T> 
where
    T: Send + Sync + Clone,
{
    data: T,
}

trait MyTrait<T, U> {
    fn method(&self, x: T) -> U;
}
"""
        metrics = analyzer.calculate_complexity(code, Path("generics.rs"))

        assert metrics.generic_types > 0
        assert metrics.trait_bounds > 0

    def test_calculate_error_handling_metrics(self, analyzer):
        """Test error handling metrics."""
        code = """
fn with_result() -> Result<String, Error> {
    let data = fetch_data()?;
    process(data)?;
    Ok(String::from("success"))
}

fn with_option() -> Option<i32> {
    let value = get_value()?;
    Some(value * 2)
}

fn with_unwrap() {
    let must_exist = get_required().unwrap();
    let expected = get_data().expect("Data required");
}

fn safe_handling() -> Result<(), Error> {
    match dangerous_operation() {
        Ok(value) => process(value),
        Err(e) => handle_error(e),
    }
    Ok(())
}
"""
        metrics = analyzer.calculate_complexity(code, Path("errors.rs"))

        assert metrics.result_types > 0
        assert metrics.option_types > 0
        assert metrics.unwrap_calls > 0
        assert metrics.expect_calls > 0
        assert metrics.question_marks >= 3

    def test_calculate_macro_metrics(self, analyzer):
        """Test macro usage metrics."""
        code = """
println!("Hello, world!");
eprintln!("Error: {}", error);

let vec = vec![1, 2, 3, 4, 5];
assert_eq!(2 + 2, 4);

#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

#[derive(Serialize, Deserialize)]
struct Data {
    value: String,
}

macro_rules! my_macro {
    ($x:expr) => {
        println!("{}", $x);
    };
}
"""
        metrics = analyzer.calculate_complexity(code, Path("macros.rs"))

        assert metrics.macro_invocations > 0
        assert metrics.derive_macros >= 2


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RustAnalyzer instance."""
        return RustAnalyzer()

    def test_handle_syntax_errors(self, analyzer):
        """Test handling of files with syntax errors."""
        code = """
fn invalid_function() {
    this is not valid Rust code
    missing semicolons everywhere
    let x = 
}

struct Incomplete {
    field: 
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("invalid.rs"))
        exports = analyzer.extract_exports(code, Path("invalid.rs"))
        structure = analyzer.extract_structure(code, Path("invalid.rs"))
        metrics = analyzer.calculate_complexity(code, Path("invalid.rs"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("empty.rs"))
        exports = analyzer.extract_exports(code, Path("empty.rs"))
        structure = analyzer.extract_structure(code, Path("empty.rs"))
        metrics = analyzer.calculate_complexity(code, Path("empty.rs"))

        assert imports == []
        assert exports == []
        assert metrics.line_count == 1


class TestEdgeCases:
    """Test suite for edge cases and Rust-specific features."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RustAnalyzer instance."""
        return RustAnalyzer()

    def test_pattern_matching(self, analyzer):
        """Test handling of pattern matching."""
        code = """
fn pattern_matching(value: Option<i32>) -> i32 {
    match value {
        Some(x) if x > 0 => x,
        Some(x) => -x,
        None => 0,
    }
}

fn if_let_pattern(value: Option<String>) {
    if let Some(s) = value {
        println!("{}", s);
    }
}

fn while_let_pattern(mut stack: Vec<i32>) {
    while let Some(top) = stack.pop() {
        println!("{}", top);
    }
}

fn destructuring() {
    let (x, y, z) = (1, 2, 3);
    let Point { x, y } = point;
}
"""
        structure = analyzer.extract_structure(code, Path("patterns.rs"))

        # Should handle pattern matching without errors
        assert len(structure.functions) >= 4

    def test_closures_and_iterators(self, analyzer):
        """Test handling of closures and iterator chains."""
        code = """
fn with_closures() {
    let add = |x, y| x + y;
    
    let complex = |x: i32| -> i32 {
        x * 2 + 1
    };
    
    let capture = || println!("{}", value);
    
    let move_closure = move |x| x + captured_value;
}

fn iterator_chains() {
    let result: Vec<_> = vec![1, 2, 3, 4, 5]
        .iter()
        .filter(|&&x| x > 2)
        .map(|x| x * 2)
        .collect();
        
    (0..10)
        .filter(|x| x % 2 == 0)
        .fold(0, |acc, x| acc + x);
}
"""
        structure = analyzer.extract_structure(code, Path("functional.rs"))

        # Should detect closures/lambdas
        assert structure.lambda_count > 0

    def test_attribute_macros(self, analyzer):
        """Test handling of attribute macros."""
        code = """
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
struct CStruct {
    field: i32,
}

#[cfg(target_os = "linux")]
fn linux_only() {
    // Linux-specific code
}

#[cfg(not(feature = "std"))]
fn no_std_function() {
    // no_std compatible
}

#[test]
#[ignore]
fn ignored_test() {
    // Test code
}

#[allow(dead_code)]
#[warn(clippy::all)]
fn with_lints() {
    // Code with specific lint settings
}
"""
        structure = analyzer.extract_structure(code, Path("attributes.rs"))

        # Check derive detection
        assert "Debug" in structure.derives
        assert "Clone" in structure.derives

    def test_const_generics(self, analyzer):
        """Test handling of const generics (Rust 1.51+)."""
        code = """
struct Array<T, const N: usize> {
    data: [T; N],
}

fn fixed_array<const N: usize>() -> [i32; N] {
    [0; N]
}

impl<T, const N: usize> Array<T, N> {
    fn new() -> Self {
        // Implementation
    }
}
"""
        structure = analyzer.extract_structure(code, Path("const_generics.rs"))

        # Should handle const generics without errors
        assert len(structure.classes) >= 1

    def test_raw_strings_and_bytes(self, analyzer):
        """Test handling of raw strings and byte literals."""
        code = """
fn raw_strings() {
    let raw = r"This is a raw string with \n not escaped";
    let raw_hash = r#"This can contain "quotes""#;
    let raw_multi = r##"Even more "nested" quotes"##;
    
    let bytes = b"byte string";
    let raw_bytes = br"raw byte string";
}
"""
        # Should handle raw strings without treating content as code
        structure = analyzer.extract_structure(code, Path("strings.rs"))
        assert len(structure.functions) == 1
