"""
Unit tests for the Dart code analyzer with Flutter support.

This module tests the Dart-specific code analysis functionality including
import/export directives, null safety features, async programming,
Flutter widgets, mixins, extensions, and modern Dart features.

Test Coverage:
    - Import extraction (import, export, part, library, deferred, conditional)
    - Export detection (classes, functions, mixins, extensions)
    - Structure extraction (constructors, methods, Flutter widgets)
    - Complexity metrics (null safety, async, Flutter-specific)
    - Null safety features (?, !, late, ??)
    - Flutter widget detection and lifecycle methods
    - Error handling for invalid Dart code
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.dart_analyzer import DartAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestDartAnalyzerInitialization:
    """Test suite for DartAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = DartAnalyzer()

        assert analyzer.language_name == "dart"
        assert ".dart" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for Dart import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a DartAnalyzer instance."""
        return DartAnalyzer()

    def test_extract_basic_imports(self, analyzer):
        """Test extraction of basic import statements."""
        code = """
import 'dart:core';
import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/user.dart';
import './utils/helpers.dart';
"""
        imports = analyzer.extract_imports(code, Path("test.dart"))

        assert len(imports) == 7

        # Check Dart SDK imports
        core_import = next(imp for imp in imports if imp.module == "dart:core")
        assert core_import.is_dart_core is True
        assert core_import.category == "dart_core"

        async_import = next(imp for imp in imports if imp.module == "dart:async")
        assert async_import.category == "dart_async"

        # Check Flutter import
        flutter_import = next(imp for imp in imports if "flutter/material" in imp.module)
        assert flutter_import.is_package is True
        assert flutter_import.category == "flutter_material"

        # Check third-party package
        provider_import = next(imp for imp in imports if "provider/provider" in imp.module)
        assert provider_import.category == "state_management"

        # Check relative imports
        user_import = next(imp for imp in imports if "user.dart" in imp.module)
        assert user_import.is_relative is True

    def test_extract_imports_with_aliases(self, analyzer):
        """Test extraction of imports with aliases."""
        code = """
import 'dart:math' as math;
import 'package:http/http.dart' as http;
import '../services/api.dart' as api;
"""
        imports = analyzer.extract_imports(code, Path("test.dart"))

        assert len(imports) == 3

        math_import = next(imp for imp in imports if "math" in imp.module)
        assert math_import.alias == "math"

        http_import = next(imp for imp in imports if "http" in imp.module)
        assert http_import.alias == "http"
        assert http_import.category == "networking"

    def test_extract_imports_with_show_hide(self, analyzer):
        """Test extraction of imports with show/hide clauses."""
        code = """
import 'dart:math' show Random, max, min;
import 'dart:async' hide Timer;
import 'package:flutter/material.dart' show Widget, BuildContext hide Theme;
"""
        imports = analyzer.extract_imports(code, Path("test.dart"))

        assert len(imports) == 3

        math_import = next(imp for imp in imports if "math" in imp.module)
        assert math_import.show_symbols == ["Random", "max", "min"]

        async_import = next(imp for imp in imports if "async" in imp.module)
        assert async_import.hide_symbols == ["Timer"]

        flutter_import = next(imp for imp in imports if "flutter" in imp.module)
        assert "Widget" in flutter_import.show_symbols
        assert "Theme" in flutter_import.hide_symbols

    def test_extract_deferred_imports(self, analyzer):
        """Test extraction of deferred imports."""
        code = """
import 'package:heavy_library/heavy.dart' deferred as heavy;
import 'large_component.dart' deferred as large;
"""
        imports = analyzer.extract_imports(code, Path("test.dart"))

        assert len(imports) == 2

        heavy_import = next(imp for imp in imports if "heavy_library" in imp.module)
        assert heavy_import.is_deferred is True
        assert heavy_import.alias == "heavy"

    def test_extract_conditional_imports(self, analyzer):
        """Test extraction of conditional imports."""
        code = """
import 'stub.dart'
    if (dart.library.io) 'io_implementation.dart'
    if (dart.library.html) 'web_implementation.dart';
"""
        imports = analyzer.extract_imports(code, Path("test.dart"))

        # Should find the main import
        assert any("stub.dart" in imp.module for imp in imports)

    def test_extract_export_statements(self, analyzer):
        """Test extraction of export statements."""
        code = """
export 'src/widgets/button.dart';
export 'src/widgets/card.dart' show Card, CardTheme;
export 'src/utils.dart' hide internalFunction;
"""
        imports = analyzer.extract_imports(code, Path("test.dart"))

        exports = [imp for imp in imports if imp.type == "export"]
        assert len(exports) == 3

        button_export = next(imp for imp in exports if "button.dart" in imp.module)
        assert button_export.type == "export"

        card_export = next(imp for imp in exports if "card.dart" in imp.module)
        assert card_export.show_symbols == ["Card", "CardTheme"]

    def test_extract_part_directives(self, analyzer):
        """Test extraction of part and part of directives."""
        code = """
library my_library;

part 'src/implementation.dart';
part 'src/helpers.dart';

// In another file:
part of my_library;
"""
        imports = analyzer.extract_imports(code, Path("test.dart"))

        # Check library declaration
        library_import = next((imp for imp in imports if imp.type == "library"), None)
        assert library_import is not None
        assert library_import.module == "my_library"

        # Check part files
        part_imports = [imp for imp in imports if imp.type == "part"]
        assert len(part_imports) == 2

        # Check part of
        part_of_import = next((imp for imp in imports if imp.type == "part_of"), None)
        assert part_of_import is not None


class TestExportExtraction:
    """Test suite for Dart export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a DartAnalyzer instance."""
        return DartAnalyzer()

    def test_extract_public_classes(self, analyzer):
        """Test extraction of public classes."""
        code = """
class PublicClass {
  void method() {}
}

abstract class AbstractClass {
  void abstractMethod();
}

final class FinalClass {}
base class BaseClass {}
interface class InterfaceClass {}
mixin class MixinClass {}

class _PrivateClass {}  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.dart"))

        public_classes = [e for e in exports if e["type"] == "class"]
        assert len(public_classes) == 6  # Excludes _PrivateClass

        abstract_class = next(e for e in public_classes if e["name"] == "AbstractClass")
        assert "abstract" in abstract_class["modifiers"]

        final_class = next(e for e in public_classes if e["name"] == "FinalClass")
        assert "final" in final_class["modifiers"]

        # Private class should not be exported
        assert not any(e["name"] == "_PrivateClass" for e in exports)

    def test_extract_mixins(self, analyzer):
        """Test extraction of mixins."""
        code = """
mixin Loggable {
  void log(String message) {
    print(message);
  }
}

base mixin BaseMixin {}

mixin ConstrainedMixin on StatefulWidget {}

mixin _PrivateMixin {}  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.dart"))

        mixins = [e for e in exports if e["type"] == "mixin"]
        assert len(mixins) >= 2  # At least Loggable and BaseMixin

        loggable = next((e for e in mixins if e["name"] == "Loggable"), None)
        assert loggable is not None

    def test_extract_public_functions(self, analyzer):
        """Test extraction of public functions."""
        code = """
void publicFunction() {}

Future<String> asyncFunction() async {
  return "result";
}

Stream<int> streamFunction() async* {
  yield 1;
}

T genericFunction<T>(T value) => value;

_privateFunction() {}  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.dart"))

        functions = [e for e in exports if e["type"] == "function"]
        assert len(functions) == 4  # Excludes _privateFunction

        async_func = next(e for e in functions if e["name"] == "asyncFunction")
        assert async_func["is_async"] is True

    def test_extract_variables_and_constants(self, analyzer):
        """Test extraction of public variables and constants."""
        code = """
final String publicFinal = "value";
const int publicConst = 42;
late String lateVariable;
var publicVar = "test";

const _privateConst = 100;  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.dart"))

        constants = [e for e in exports if e["type"] == "constant"]
        variables = [e for e in exports if e["type"] == "variable"]

        assert len(constants) >= 1
        assert len(variables) >= 2

        # Check late variable
        late_var = next((e for e in exports if e.get("is_late")), None)
        assert late_var is not None

    def test_extract_enums(self, analyzer):
        """Test extraction of enums."""
        code = """
enum Color { red, green, blue }

enum Status {
  pending('Pending'),
  approved('Approved'),
  rejected('Rejected');

  final String label;
  const Status(this.label);
}

enum _PrivateEnum { a, b }  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.dart"))

        enums = [e for e in exports if e["type"] == "enum"]
        assert len(enums) == 2  # Excludes _PrivateEnum

        color_enum = next(e for e in enums if e["name"] == "Color")
        assert color_enum is not None

    def test_extract_typedefs(self, analyzer):
        """Test extraction of typedefs."""
        code = """
typedef IntList = List<int>;
typedef StringCallback = void Function(String);
typedef Compare<T> = int Function(T a, T b);
"""
        exports = analyzer.extract_exports(code, Path("test.dart"))

        typedefs = [e for e in exports if e["type"] == "typedef"]
        assert len(typedefs) == 3

    def test_extract_extensions(self, analyzer):
        """Test extraction of extension methods."""
        code = """
extension StringExtension on String {
  bool get isEmail => contains('@');
}

extension on List<int> {
  int get sum => fold(0, (a, b) => a + b);
}

extension DateTimeExtension on DateTime {
  String get formatted => toString();
}
"""
        exports = analyzer.extract_exports(code, Path("test.dart"))

        extensions = [e for e in exports if e["type"] == "extension"]
        assert len(extensions) >= 2

        string_ext = next((e for e in extensions if e["name"] == "StringExtension"), None)
        assert string_ext is not None
        assert string_ext["on_type"] == "String"


class TestStructureExtraction:
    """Test suite for Dart code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a DartAnalyzer instance."""
        return DartAnalyzer()

    def test_extract_class_with_inheritance(self, analyzer):
        """Test extraction of class with inheritance, mixins, and interfaces."""
        code = """
abstract class Animal {
  void makeSound();
}

mixin Swimmer {
  void swim() {}
}

mixin Flyer {
  void fly() {}
}

class Duck extends Animal with Swimmer, Flyer implements Comparable<Duck> {
  @override
  void makeSound() {
    print('Quack');
  }
  
  @override
  int compareTo(Duck other) {
    return 0;
  }
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))

        assert len(structure.classes) >= 2

        duck_class = next(c for c in structure.classes if c.name == "Duck")
        assert "Animal" in duck_class.bases
        assert "Swimmer" in duck_class.mixins
        assert "Flyer" in duck_class.mixins
        assert "Comparable<Duck>" in duck_class.interfaces

    def test_extract_constructors(self, analyzer):
        """Test extraction of various constructor types."""
        code = """
class Person {
  final String name;
  final int age;
  
  // Default constructor
  Person(this.name, this.age);
  
  // Named constructor
  Person.fromJson(Map<String, dynamic> json)
      : name = json['name'],
        age = json['age'];
  
  // Factory constructor
  factory Person.anonymous() {
    return Person('Anonymous', 0);
  }
  
  // Const constructor
  const Person.constant(this.name, this.age);
  
  // Constructor with optional parameters
  Person.withDefaults([this.name = 'Unknown', this.age = 0]);
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))

        person_class = structure.classes[0]
        constructors = person_class.constructors

        assert len(constructors) >= 5

        # Check default constructor
        default_const = next(c for c in constructors if c["type"] == "default")
        assert default_const is not None

        # Check named constructor
        named_const = next(c for c in constructors if c["type"] == "named" and c.get("name") == "fromJson")
        assert named_const is not None

        # Check factory constructor
        factory_const = next(c for c in constructors if c["type"] == "factory")
        assert factory_const is not None

        # Check const constructor
        const_const = next(c for c in constructors if c["type"] == "const")
        assert const_const is not None

    def test_extract_flutter_widgets(self, analyzer):
        """Test extraction of Flutter widget classes."""
        code = """
import 'package:flutter/material.dart';

class MyStatelessWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container();
  }
}

class MyStatefulWidget extends StatefulWidget {
  @override
  State<MyStatefulWidget> createState() => _MyStatefulWidgetState();
}

class _MyStatefulWidgetState extends State<MyStatefulWidget> {
  int _counter = 0;
  
  @override
  Widget build(BuildContext context) {
    return Text('$_counter');
  }
  
  void _incrementCounter() {
    setState(() {
      _counter++;
    });
  }
}

class MyInheritedWidget extends InheritedWidget {
  final int value;
  
  const MyInheritedWidget({
    required this.value,
    required Widget child,
  }) : super(child: child);
  
  @override
  bool updateShouldNotify(MyInheritedWidget oldWidget) {
    return value != oldWidget.value;
  }
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))

        assert structure.is_flutter is True

        stateless = next(c for c in structure.classes if c.name == "MyStatelessWidget")
        assert stateless.is_widget is True
        assert stateless.widget_type == "stateless"

        stateful = next(c for c in structure.classes if c.name == "MyStatefulWidget")
        assert stateful.is_widget is True
        assert stateful.widget_type == "stateful"

        state = next(c for c in structure.classes if c.name == "_MyStatefulWidgetState")
        assert state.is_widget is True
        assert state.widget_type == "state"

        inherited = next(c for c in structure.classes if c.name == "MyInheritedWidget")
        assert inherited.is_widget is True
        assert inherited.widget_type == "inherited"

    def test_extract_async_functions(self, analyzer):
        """Test extraction of async functions and generators."""
        code = """
// Regular async function
Future<String> fetchData() async {
  await Future.delayed(Duration(seconds: 1));
  return "data";
}

// Async generator
Stream<int> countStream() async* {
  for (int i = 0; i < 10; i++) {
    await Future.delayed(Duration(seconds: 1));
    yield i;
  }
}

// Sync generator
Iterable<int> naturals() sync* {
  int k = 0;
  while (true) yield k++;
}

// Function returning Future without async
Future<void> manualFuture() {
  return Future.value();
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))

        fetch_func = next(f for f in structure.functions if f.name == "fetchData")
        assert fetch_func.is_async is True
        assert fetch_func.return_type == "Future"

        count_func = next(f for f in structure.functions if f.name == "countStream")
        assert count_func.is_async is True
        assert count_func.is_generator is True
        assert count_func.return_type == "Stream"

    def test_extract_null_safety_features(self, analyzer):
        """Test extraction of null safety features."""
        code = """
class NullSafeClass {
  String nonNullable = "value";
  String? nullable;
  late String lateInit;
  late final String lateFinal;
  
  void processData(String? input) {
    // Null assertion
    print(input!.length);
    
    // Null-aware operators
    String result = input ?? "default";
    int? length = input?.length;
    
    // Null-aware cascade
    input
      ?..trim()
      ..toLowerCase();
  }
  
  String? maybeReturn() => null;
  
  void requireNonNull(String value) {
    // No null check needed
  }
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))

        assert structure.nullable_types > 0
        assert structure.null_assertions > 0
        assert structure.late_variables > 0
        assert structure.null_aware_operators > 0

    def test_extract_extensions_and_mixins(self, analyzer):
        """Test extraction of extensions and standalone mixins."""
        code = """
extension StringHelpers on String {
  String get reversed => split('').reversed.join();
  
  bool get isNumeric => double.tryParse(this) != null;
  
  String repeat(int times) {
    return this * times;
  }
}

extension on List<int> {
  int get sum => fold(0, (a, b) => a + b);
}

mixin LogMixin {
  void log(String message) {
    print('[LOG] $message');
  }
}

base mixin CacheMixin on StatefulWidget {
  final _cache = <String, dynamic>{};
  
  T? getCached<T>(String key) => _cache[key] as T?;
  
  void cache(String key, dynamic value) {
    _cache[key] = value;
  }
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))

        assert len(structure.extensions) >= 2
        assert len(structure.mixins) >= 2

        string_ext = next(e for e in structure.extensions if "String" in e.get("on_type", ""))
        assert string_ext is not None

    def test_extract_enums_with_methods(self, analyzer):
        """Test extraction of enhanced enums with methods."""
        code = """
enum Vehicle implements Comparable<Vehicle> {
  car(tires: 4, maxSpeed: 200),
  bicycle(tires: 2, maxSpeed: 40),
  boat(tires: 0, maxSpeed: 50);

  const Vehicle({
    required this.tires,
    required this.maxSpeed,
  });

  final int tires;
  final int maxSpeed;

  @override
  int compareTo(Vehicle other) => maxSpeed - other.maxSpeed;
  
  bool get isFast => maxSpeed > 100;
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))

        assert len(structure.enums) == 1
        vehicle_enum = structure.enums[0]
        assert vehicle_enum["has_enhanced_features"] is True
        assert len(vehicle_enum["values"]) == 3


class TestComplexityCalculation:
    """Test suite for Dart complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a DartAnalyzer instance."""
        return DartAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
int complexFunction(int x, int y) {
  if (x > 0) {
    if (y > 0) {
      return x + y;
    } else if (y < 0) {
      return x - y;
    }
  }
  
  for (int i = 0; i < 10; i++) {
    if (i % 2 == 0) {
      x += i;
    }
  }
  
  switch (x) {
    case 1:
      return 1;
    case 2:
      return 2;
    default:
      return 0;
  }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 9

    def test_null_safety_metrics(self, analyzer):
        """Test null safety complexity metrics."""
        code = """
class NullSafetyExample {
  String? nullableString;
  late String lateString;
  late final int lateFinalInt;
  
  void process(String? input, {required String name}) {
    // Null assertions
    print(input!);
    print(nullableString!.length);
    
    // Null-aware operators
    String result = input ?? "default";
    int? length = input?.length;
    List<int>? numbers;
    int first = numbers?.first ?? 0;
    
    // Null-aware cascade
    input
      ?..trim()
      ..toLowerCase();
  }
  
  T? genericNullable<T>(T? value) => value;
  
  void multipleAssertions(String? a, String? b, String? c) {
    print(a! + b! + c!);
  }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))

        assert metrics.nullable_types >= 6
        assert metrics.null_assertions >= 5
        assert metrics.late_keywords >= 2
        assert metrics.null_aware_ops >= 4
        assert metrics.required_keywords >= 1

    def test_async_complexity(self, analyzer):
        """Test async/await complexity metrics."""
        code = """
import 'dart:async';

class AsyncService {
  Future<String> fetchData() async {
    await Future.delayed(Duration(seconds: 1));
    return "data";
  }
  
  Future<void> processMultiple() async {
    final result1 = await fetchData();
    final result2 = await fetchData();
    final results = await Future.wait([
      fetchData(),
      fetchData(),
    ]);
  }
  
  Stream<int> numberStream() async* {
    for (int i = 0; i < 10; i++) {
      await Future.delayed(Duration(milliseconds: 100));
      yield i;
    }
  }
  
  Future<void> withCompleter() async {
    final completer = Completer<void>();
    Future.delayed(Duration(seconds: 1), () {
      completer.complete();
    });
    await completer.future;
  }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))

        assert metrics.async_functions >= 4
        assert metrics.await_count >= 6
        assert metrics.future_count >= 4
        assert metrics.stream_count >= 1
        assert metrics.completer_count >= 1

    def test_flutter_specific_metrics(self, analyzer):
        """Test Flutter-specific complexity metrics."""
        code = """
import 'package:flutter/material.dart';

class MyWidget extends StatefulWidget {
  const MyWidget({Key? key}) : super(key: key);
  
  @override
  State<MyWidget> createState() => _MyWidgetState();
}

class _MyWidgetState extends State<MyWidget> {
  int _counter = 0;
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey();
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      body: Column(
        children: [
          Text('Counter: $_counter'),
          ElevatedButton(
            onPressed: () {
              setState(() {
                _counter++;
              });
            },
            child: Text('Increment'),
          ),
        ],
      ),
    );
  }
  
  void _showSnackBar(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Hello')),
    );
  }
}

class AnotherStateless extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      key: ValueKey('container'),
      child: Text('Hello'),
    );
  }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))

        assert metrics.widget_count > 0
        assert metrics.build_methods >= 2
        assert metrics.setstate_calls >= 1
        assert metrics.stateful_widgets >= 1
        assert metrics.stateless_widgets >= 1
        assert metrics.keys_used >= 3  # GlobalKey, ValueKey, Key in constructor
        assert metrics.context_usage >= 3

    def test_exception_handling_metrics(self, analyzer):
        """Test exception handling metrics."""
        code = """
class ErrorHandler {
  void handleErrors() {
    try {
      riskyOperation();
    } on FormatException catch (e) {
      print('Format error: $e');
    } on IOException catch (e) {
      print('IO error: $e');
      rethrow;
    } catch (e) {
      print('Unknown error: $e');
      throw CustomException('Wrapped: $e');
    } finally {
      cleanup();
    }
    
    try {
      anotherOperation();
    } catch (e, stackTrace) {
      print('Error with stack: $e\\n$stackTrace');
    }
  }
  
  void simpleThrow() {
    throw ArgumentError('Invalid argument');
  }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))

        assert metrics.try_blocks == 2
        assert metrics.catch_blocks >= 4
        assert metrics.finally_blocks == 1
        assert metrics.throw_statements >= 2
        assert metrics.rethrow_statements == 1


class TestEdgeCases:
    """Test suite for edge cases in Dart analysis."""

    @pytest.fixture
    def analyzer(self):
        """Provide a DartAnalyzer instance."""
        return DartAnalyzer()

    def test_handle_malformed_code(self, analyzer):
        """Test handling of malformed Dart code."""
        code = """
class Broken {
  This is not valid Dart code!!!
  
  void method() {
    missing semicolon
    unclosed brace
  
  // Missing closing brace for class
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("test.dart"))
        exports = analyzer.extract_exports(code, Path("test.dart"))
        structure = analyzer.extract_structure(code, Path("test.dart"))
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty file."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.dart"))
        exports = analyzer.extract_exports(code, Path("test.dart"))
        structure = analyzer.extract_structure(code, Path("test.dart"))
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))

        assert imports == []
        assert exports == []
        assert len(structure.classes) == 0
        assert metrics.line_count == 1

    def test_handle_multiline_strings(self, analyzer):
        """Test handling of multiline strings."""
        code = '''
class StringTest {
  final String multiline = """
    This is a multiline string
    with multiple lines
    and it might contain code-like syntax:
    if (true) { print("not real code"); }
  """;
  
  final String singleQuoteMulti = \'\'\'
    Another multiline
    with single quotes
  \'\'\';
  
  void method() {
    // Real code after strings
    if (true) {
      print("This is real code");
    }
  }
}
'''
        structure = analyzer.extract_structure(code, Path("test.dart"))
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))

        # Should correctly identify the class and method
        assert len(structure.classes) == 1
        # Complexity should not count code inside strings
        assert metrics.cyclomatic < 5

    def test_handle_generics_and_type_parameters(self, analyzer):
        """Test handling of complex generics."""
        code = """
class Container<T extends Comparable<T>> {
  final List<T> items;
  
  Container(this.items);
  
  T? find<K extends T>(bool Function(K) predicate) {
    for (final item in items) {
      if (item is K && predicate(item)) {
        return item;
      }
    }
    return null;
  }
  
  Map<String, List<T>> groupBy<R>(R Function(T) keySelector) {
    return {};
  }
}

typedef Comparator<T> = int Function(T a, T b);
typedef AsyncCallback<T> = Future<void> Function(T value);

class Complex<T, U extends List<T>, V extends Map<String, U>> {
  V data;
  Complex(this.data);
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))
        
        container_class = next(c for c in structure.classes if c.name == "Container")
        assert container_class.generics == "T extends Comparable<T>"

        complex_class = next(c for c in structure.classes if c.name == "Complex")
        assert complex_class is not None

    def test_handle_cascade_notation(self, analyzer):
        """Test handling of cascade notation."""
        code = """
class Builder {
  String? name;
  int? age;
  
  void build() {
    final person = Person()
      ..name = "John"
      ..age = 30
      ..address = (Address()
        ..street = "Main St"
        ..city = "New York")
      ..validate();
    
    // Null-aware cascade
    person
      ?..updateName("Jane")
      ..updateAge(31);
  }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))
        
        # Should handle cascades without errors
        assert metrics.line_count > 0

    def test_handle_records_and_patterns(self, analyzer):
        """Test handling of records and pattern matching (Dart 3.0+)."""
        code = """
// Records
(int, String) getRecord() => (42, "answer");

class PatternTest {
  void patterns() {
    var (a, b) = getRecord();
    
    final obj = switch (a) {
      1 => "one",
      2 => "two",
      _ => "other",
    };
    
    if (obj case String s when s.length > 5) {
      print(s);
    }
  }
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))
        metrics = analyzer.calculate_complexity(code, Path("test.dart"))
        
        # Should handle new syntax without errors
        assert len(structure.functions) >= 1
        assert metrics.cyclomatic > 1

    def test_handle_sealed_classes(self, analyzer):
        """Test handling of sealed classes and exhaustive switching."""
        code = """
sealed class Shape {}

class Circle extends Shape {
  final double radius;
  Circle(this.radius);
}

class Square extends Shape {
  final double side;
  Square(this.side);
}

double calculateArea(Shape shape) {
  return switch (shape) {
    Circle(:final radius) => 3.14 * radius * radius,
    Square(:final side) => side * side,
  };
}
"""
        structure = analyzer.extract_structure(code, Path("test.dart"))
        
        # Should identify sealed class pattern
        shape_class = next((c for c in structure.classes if c.name == "Shape"), None)
        assert shape_class is not None

    def test_main_function_detection(self, analyzer):
        """Test detection of main function."""
        code_with_main = """
void main() {
  print('Hello, World!');
}
"""
        structure = analyzer.extract_structure(code_with_main, Path("test.dart"))
        assert structure.has_main is True

        code_without_main = """
void notMain() {
  print('Hello, World!');
}
"""
        structure = analyzer.extract_structure(code_without_main, Path("test.dart"))
        assert structure.has_main is False

    def test_test_file_detection(self, analyzer):
        """Test detection of test files."""
        code = """
import 'package:test/test.dart';

void main() {
  test('simple test', () {
    expect(1 + 1, equals(2));
  });
}
"""
        structure = analyzer.extract_structure(code, Path("my_test.dart"))
        assert structure.is_test_file is True

        structure = analyzer.extract_structure(code, Path("test/unit/my_test.dart"))
        assert structure.is_test_file is True

        structure = analyzer.extract_structure(code, Path("lib/src/my_class.dart"))
        assert structure.is_test_file is False