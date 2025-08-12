"""
Unit tests for the Scala code analyzer with functional programming support.

This module tests the Scala-specific code analysis functionality including
import statements, case classes, traits, pattern matching, implicits,
for comprehensions, and both Scala 2.x and 3.x syntax.

Test Coverage:
    - Import extraction (wildcards, renames, multiple imports)
    - Export detection (classes, traits, objects, implicits)
    - Structure extraction (case classes, companion objects, ADTs)
    - Complexity metrics (pattern matching, functional programming)
    - Implicit definitions and conversions
    - Scala 3 features (given/using, extension methods, enums)
    - Error handling for invalid Scala code
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.scala_analyzer import ScalaAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestScalaAnalyzerInitialization:
    """Test suite for ScalaAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = ScalaAnalyzer()

        assert analyzer.language_name == "scala"
        assert ".scala" in analyzer.file_extensions
        assert ".sc" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for Scala import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a ScalaAnalyzer instance."""
        return ScalaAnalyzer()

    def test_extract_basic_imports(self, analyzer):
        """Test extraction of basic import statements."""
        code = """
package com.example.app

import scala.collection.mutable
import scala.concurrent.Future
import java.util.Date
import akka.actor.ActorSystem
"""
        imports = analyzer.extract_imports(code, Path("test.scala"))

        assert len(imports) == 5  # Including package declaration

        # Check package
        package_import = next(imp for imp in imports if imp.type == "package")
        assert package_import.module == "com.example.app"

        # Check Scala imports
        mutable_import = next(imp for imp in imports if "mutable" in imp.module)
        assert mutable_import.category == "scala_collections"

        future_import = next(imp for imp in imports if "Future" in imp.module)
        assert future_import.category == "scala_concurrent"

        # Check Java import
        date_import = next(imp for imp in imports if "Date" in imp.module)
        assert date_import.category == "java"

        # Check Akka import
        akka_import = next(imp for imp in imports if "ActorSystem" in imp.module)
        assert akka_import.category == "akka"

    def test_extract_wildcard_imports(self, analyzer):
        """Test extraction of wildcard imports."""
        code = """
import scala.collection._
import java.util._
import scala.concurrent.ExecutionContext.Implicits._
"""
        imports = analyzer.extract_imports(code, Path("test.scala"))

        assert len(imports) == 3

        collection_import = next(imp for imp in imports if "collection._" in imp.module)
        assert collection_import.is_wildcard is True

        implicits_import = next(imp for imp in imports if "Implicits._" in imp.module)
        assert implicits_import.is_wildcard is True

    def test_extract_multiple_imports(self, analyzer):
        """Test extraction of multiple imports with braces."""
        code = """
import scala.collection.{List, Map, Set}
import java.util.{ArrayList, HashMap, HashSet}
import scala.concurrent.{Future, Promise, duration}
"""
        imports = analyzer.extract_imports(code, Path("test.scala"))

        # Should expand to individual imports
        assert any("List" in imp.module for imp in imports)
        assert any("Map" in imp.module for imp in imports)
        assert any("Set" in imp.module for imp in imports)
        assert any("ArrayList" in imp.module for imp in imports)

    def test_extract_renamed_imports(self, analyzer):
        """Test extraction of renamed imports."""
        code = """
import java.util.{List => JList, Map => JMap}
import scala.collection.mutable.{Map => MutableMap}
import java.util.{Date => JDate, _}
"""
        imports = analyzer.extract_imports(code, Path("test.scala"))

        # Check renamed imports
        jlist_import = next(imp for imp in imports if imp.alias == "JList")
        assert "List" in jlist_import.module
        assert jlist_import.is_renamed is True

        mutable_map_import = next(imp for imp in imports if imp.alias == "MutableMap")
        assert "Map" in mutable_map_import.module

    def test_extract_hidden_imports(self, analyzer):
        """Test extraction of imports with hiding."""
        code = """
import java.util.{List => _, Map => _, _}
import scala.collection.{Set => _, _}
"""
        imports = analyzer.extract_imports(code, Path("test.scala"))

        # Hidden imports should not appear, but wildcard should
        assert not any(imp.alias == "_" and "List" in imp.module for imp in imports)
        assert any(imp.is_wildcard for imp in imports)

    def test_extract_given_imports_scala3(self, analyzer):
        """Test extraction of given imports (Scala 3)."""
        code = """
import cats.implicits.given
import cats.{Show, given}
import scala.math.Ordering.Double.given
"""
        imports = analyzer.extract_imports(code, Path("test.scala"))

        given_imports = [imp for imp in imports if imp.type == "given_import" or imp.is_given]
        assert len(given_imports) >= 1


class TestExportExtraction:
    """Test suite for Scala export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a ScalaAnalyzer instance."""
        return ScalaAnalyzer()

    def test_extract_classes(self, analyzer):
        """Test extraction of class exports."""
        code = """
class PublicClass
abstract class AbstractClass
final class FinalClass
sealed class SealedClass
case class CaseClass(name: String, age: Int)

private class PrivateClass  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.scala"))

        public_classes = [e for e in exports if e["type"] in ["class", "case_class"]]
        assert len(public_classes) == 5  # Excludes PrivateClass

        # Check modifiers
        abstract_class = next(e for e in public_classes if e["name"] == "AbstractClass")
        assert "abstract" in abstract_class["modifiers"]

        case_class = next(e for e in public_classes if e["name"] == "CaseClass")
        assert case_class["type"] == "case_class"

    def test_extract_traits(self, analyzer):
        """Test extraction of traits."""
        code = """
trait PublicTrait
sealed trait SealedTrait
trait GenericTrait[T]

private trait PrivateTrait  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.scala"))

        traits = [e for e in exports if e["type"] == "trait"]
        assert len(traits) == 3  # Excludes PrivateTrait

        sealed_trait = next(e for e in traits if e["name"] == "SealedTrait")
        assert sealed_trait["is_sealed"] is True

    def test_extract_objects(self, analyzer):
        """Test extraction of objects."""
        code = """
object Singleton
case object CaseObject
object CompanionObject

private object PrivateObject  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.scala"))

        objects = [e for e in exports if "object" in e["type"]]
        assert len(objects) == 3  # Excludes PrivateObject

        case_object = next(e for e in objects if e["name"] == "CaseObject")
        assert case_object["type"] == "case_object"

    def test_extract_functions(self, analyzer):
        """Test extraction of functions/methods."""
        code = """
def publicFunction(): Unit = {}
override def overrideMethod(): String = ""
implicit def implicitConversion(x: Int): String = x.toString
protected def protectedMethod(): Unit = {}
private def privateMethod(): Unit = {}  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.scala"))

        functions = [e for e in exports if e["type"] == "function"]
        assert len(functions) == 4  # Excludes privateMethod

        implicit_func = next(e for e in functions if e["name"] == "implicitConversion")
        assert implicit_func["is_implicit"] is True

        override_func = next(e for e in functions if e["name"] == "overrideMethod")
        assert override_func["is_override"] is True

    def test_extract_values_and_variables(self, analyzer):
        """Test extraction of vals and vars."""
        code = """
val immutableValue = 42
var mutableVariable = "test"
lazy val lazyValue = expensiveComputation()
implicit val implicitValue = "implicit"
private val privateValue = 100  // Should not be exported
"""
        exports = analyzer.extract_exports(code, Path("test.scala"))

        values = [e for e in exports if e["type"] in ["value", "variable"]]
        assert len(values) == 4  # Excludes privateValue

        mutable_var = next(e for e in values if e["name"] == "mutableVariable")
        assert mutable_var["is_mutable"] is True

        lazy_val = next(e for e in values if e["name"] == "lazyValue")
        assert lazy_val["is_lazy"] is True

        implicit_val = next(e for e in values if e["name"] == "implicitValue")
        assert implicit_val["is_implicit"] is True

    def test_extract_type_aliases(self, analyzer):
        """Test extraction of type aliases."""
        code = """
type StringMap = Map[String, String]
type Result[T] = Either[Error, T]
opaque type UserId = String  // Scala 3
"""
        exports = analyzer.extract_exports(code, Path("test.scala"))

        type_aliases = [e for e in exports if e["type"] == "type_alias"]
        assert len(type_aliases) >= 2

        opaque_type = next((e for e in type_aliases if e.get("is_opaque")), None)
        if opaque_type:
            assert opaque_type["name"] == "UserId"

    def test_extract_enums_scala3(self, analyzer):
        """Test extraction of enums (Scala 3)."""
        code = """
enum Color:
  case Red, Green, Blue

enum Result[+T]:
  case Success(value: T)
  case Failure(error: String)
"""
        exports = analyzer.extract_exports(code, Path("test.scala"))

        enums = [e for e in exports if e["type"] == "enum"]
        if enums:  # Only if Scala 3 syntax is detected
            assert len(enums) == 2
            assert all(e["scala_version"] == 3 for e in enums)

    def test_extract_given_instances_scala3(self, analyzer):
        """Test extraction of given instances (Scala 3)."""
        code = """
given intOrdering: Ordering[Int] = Ordering.Int
given Conversion[String, Int] = _.toInt
given listMonoid[T]: Monoid[List[T]] with
  def empty = Nil
  def combine(x: List[T], y: List[T]) = x ++ y
"""
        exports = analyzer.extract_exports(code, Path("test.scala"))

        givens = [e for e in exports if e["type"] == "given"]
        if givens:  # Only if Scala 3 syntax is detected
            assert len(givens) >= 2
            assert all(e["scala_version"] == 3 for e in givens)


class TestStructureExtraction:
    """Test suite for Scala code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a ScalaAnalyzer instance."""
        return ScalaAnalyzer()

    def test_extract_class_with_inheritance(self, analyzer):
        """Test extraction of class with inheritance and mixins."""
        code = """
abstract class Animal {
  def makeSound(): String
}

trait Swimmer {
  def swim(): Unit = println("Swimming")
}

trait Flyer {
  def fly(): Unit = println("Flying")
}

class Duck extends Animal with Swimmer with Flyer {
  override def makeSound(): String = "Quack"
}
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        assert len(structure.classes) >= 2

        duck_class = next(c for c in structure.classes if c.name == "Duck")
        assert "Animal" in duck_class.bases
        assert "Swimmer" in duck_class.mixins
        assert "Flyer" in duck_class.mixins

    def test_extract_case_class_with_companion(self, analyzer):
        """Test extraction of case class with companion object."""
        code = """
case class Person(name: String, age: Int) {
  def greet(): String = s"Hello, I'm $name"
}

object Person {
  def apply(name: String): Person = Person(name, 0)
  
  implicit val ordering: Ordering[Person] = Ordering.by(_.age)
}
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        person_class = next(c for c in structure.classes if c.name == "Person")
        assert person_class.is_case_class is True
        assert person_class.has_companion is True

        person_object = next(o for o in structure.objects if o["name"] == "Person")
        assert person_object["is_companion"] is True

    def test_extract_sealed_trait_hierarchy(self, analyzer):
        """Test extraction of sealed trait ADT hierarchy."""
        code = """
sealed trait Tree[+T]
case class Node[T](value: T, left: Tree[T], right: Tree[T]) extends Tree[T]
case object Empty extends Tree[Nothing]

sealed abstract class Result[+T]
case class Success[T](value: T) extends Result[T]
case class Failure(error: String) extends Result[Nothing]
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        # Check sealed trait
        tree_trait = next(t for t in structure.traits if t["name"] == "Tree")
        assert tree_trait["is_sealed"] is True

        # Check case classes extending trait
        node_class = next(c for c in structure.classes if c.name == "Node")
        assert node_class.is_case_class is True
        assert "Tree[T]" in node_class.bases

        # Check case object
        empty_object = next(o for o in structure.objects if o["name"] == "Empty")
        assert empty_object["is_case_object"] is True

    def test_extract_curried_functions(self, analyzer):
        """Test extraction of curried functions."""
        code = """
def add(x: Int)(y: Int): Int = x + y

def fold[A, B](list: List[A])(init: B)(f: (B, A) => B): B = 
  list.foldLeft(init)(f)

def configure(timeout: Int)(retries: Int)(implicit ec: ExecutionContext): Unit = {
  // configuration logic
}
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        add_func = next(f for f in structure.functions if f.name == "add")
        assert add_func.is_curried is True
        assert len(add_func.parameters) == 2  # Two parameter groups

        fold_func = next(f for f in structure.functions if f.name == "fold")
        assert fold_func.is_curried is True
        assert fold_func.type_parameters == "A, B"

    def test_extract_implicit_definitions(self, analyzer):
        """Test extraction of implicit definitions."""
        code = """
implicit val defaultTimeout: Int = 5000

implicit def stringToInt(s: String): Int = s.toInt

implicit class RichString(val s: String) extends AnyVal {
  def isPalindrome: Boolean = s == s.reverse
}

class Service(implicit val ec: ExecutionContext) {
  def doAsync()(implicit timeout: Duration): Future[String] = ???
}
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        assert structure.implicit_defs >= 3
        assert structure.implicit_params >= 2

    def test_extract_pattern_matching(self, analyzer):
        """Test extraction of pattern matching constructs."""
        code = """
def describe(x: Any): String = x match {
  case 0 => "zero"
  case i: Int if i > 0 => "positive int"
  case s: String => s"string: $s"
  case list: List[_] => s"list of size ${list.size}"
  case _ => "unknown"
}

val result = someValue match {
  case Success(value) => value
  case Failure(error) => throw error
}

for {
  x <- List(1, 2, 3)
  if x > 1
  y <- List("a", "b")
} yield (x, y)
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        assert structure.match_expressions >= 2
        assert structure.case_statements >= 7
        assert structure.for_comprehensions >= 1

    def test_extract_type_parameters_with_variance(self, analyzer):
        """Test extraction of type parameters with variance annotations."""
        code = """
class Container[+T](val value: T)
trait Function[-A, +B] {
  def apply(a: A): B
}
class Invariant[T]
type F[+A] = Option[A]
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        container_class = next(c for c in structure.classes if c.name == "Container")
        assert container_class.type_parameters == "+T"

        function_trait = next(t for t in structure.traits if t["name"] == "Function")
        assert function_trait["type_parameters"] == "-A, +B"

    def test_extract_for_comprehensions(self, analyzer):
        """Test extraction of for comprehensions and generators."""
        code = """
val result = for {
  x <- Future(1)
  y <- Future(2)
  z <- Future(3)
} yield x + y + z

val filtered = for {
  i <- 1 to 10
  if i % 2 == 0
  j <- 1 to i
  if i + j > 10
} yield (i, j)
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        assert structure.for_comprehensions >= 2
        assert structure.yield_expressions >= 2

    def test_extract_scala3_features(self, analyzer):
        """Test extraction of Scala 3 specific features."""
        code = """
// Extension methods
extension (s: String)
  def greet: String = s"Hello, $s"
  def shout: String = s.toUpperCase

// Given instances
given intOrdering: Ordering[Int] = Ordering.Int

given listMonoid[T]: Monoid[List[T]] with
  def empty = Nil
  def combine(x: List[T], y: List[T]) = x ++ y

// Using clauses
def sort[T](list: List[T])(using ord: Ordering[T]): List[T] =
  list.sorted

// Enum
enum Color:
  case Red, Green, Blue
  
  def rgb: Int = this match
    case Red => 0xFF0000
    case Green => 0x00FF00
    case Blue => 0x0000FF
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        if structure.scala_version == 3:
            assert structure.extension_methods > 0
            assert structure.given_instances > 0
            assert structure.using_clauses > 0
            assert len(structure.enums) > 0


class TestComplexityCalculation:
    """Test suite for Scala complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a ScalaAnalyzer instance."""
        return ScalaAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
def complexFunction(x: Int, y: Int): Int = {
  if (x > 0) {
    if (y > 0) {
      x + y
    } else if (y < 0) {
      x - y
    } else {
      x
    }
  } else {
    for (i <- 1 to 10) {
      if (i % 2 == 0) {
        println(i)
      }
    }
    
    x match {
      case 1 => 1
      case 2 => 2
      case _ => 0
    }
  }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 10

    def test_pattern_matching_complexity(self, analyzer):
        """Test pattern matching complexity metrics."""
        code = """
def process(value: Any): String = value match {
  case i: Int if i > 0 => "positive"
  case i: Int if i < 0 => "negative"
  case 0 => "zero"
  case s: String => s
  case list: List[_] => list.mkString(",")
  case Some(x) => x.toString
  case None => "none"
  case _ => "unknown"
}

sealed trait Result[+T]
case class Success[T](value: T) extends Result[T]
case class Failure(error: String) extends Result[Nothing]

def handle[T](result: Result[T]): T = result match {
  case Success(value) => value
  case Failure(error) => throw new Exception(error)
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        assert metrics.match_expressions >= 2
        assert metrics.case_clauses >= 10
        assert metrics.pattern_guards >= 2

    def test_functional_programming_metrics(self, analyzer):
        """Test functional programming complexity metrics."""
        code = """
val doubled = List(1, 2, 3).map(_ * 2)
val filtered = doubled.filter(_ > 2)
val sum = filtered.fold(0)(_ + _)

val result = for {
  x <- Option(1)
  y <- Option(2)
  z <- Option(3)
} yield x + y + z

val partial: PartialFunction[Int, String] = {
  case 1 => "one"
  case 2 => "two"
}

val add = (x: Int) => (y: Int) => x + y
val addFive = add(5)

List(1, 2, 3).flatMap(x => List(x, x * 2))
  .filter(_ % 2 == 0)
  .map(_.toString)
  .reduce(_ + _)
"""
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        assert metrics.lambda_count >= 6
        assert metrics.higher_order_functions >= 7
        assert metrics.for_comprehensions >= 1
        assert metrics.partial_functions >= 1

    def test_implicit_complexity(self, analyzer):
        """Test implicit-related complexity metrics."""
        code = """
implicit val timeout: Duration = 5.seconds

implicit def stringToInt(s: String): Int = s.toInt

implicit class RichInt(val x: Int) extends AnyVal {
  def isEven: Boolean = x % 2 == 0
}

def process(data: String)(implicit ec: ExecutionContext, timeout: Duration): Future[Int] = {
  Future {
    data.toInt
  }
}

def sort[T](list: List[T])(implicit ord: Ordering[T]): List[T] = 
  list.sorted
"""
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        assert metrics.implicit_defs >= 3
        assert metrics.implicit_params >= 3
        assert metrics.implicit_conversions >= 1

    def test_type_system_metrics(self, analyzer):
        """Test type system complexity metrics."""
        code = """
class Container[+T](val value: T)
trait Function[-A, +B]
type F[+A] = Option[A]
type G = Map[String, List[Int]]

def process[T <: Comparable[T]](x: T, y: T): T = 
  if (x.compareTo(y) > 0) x else y

def existential(list: List[_]): Int = list.size

type Aux[T] = Service { type Output = T }

trait Higher[F[_]] {
  def map[A, B](fa: F[A])(f: A => B): F[B]
}

val wildcard: List[_ <: AnyRef] = List("test")
"""
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        assert metrics.type_parameters >= 5
        assert metrics.variance_annotations >= 3
        assert metrics.type_aliases >= 3

    def test_collection_metrics(self, analyzer):
        """Test collection usage metrics."""
        code = """
import scala.collection.mutable

val immutableList = List(1, 2, 3)
val immutableSet = Set("a", "b", "c")
val immutableMap = Map("key" -> "value")
val immutableVector = Vector(1, 2, 3)
val immutableSeq = Seq(1, 2, 3)

val mutableList = mutable.ListBuffer(1, 2, 3)
val mutableSet = mutable.Set("a", "b")
val mutableMap = mutable.Map("key" -> "value")
val mutableArray = mutable.ArrayBuffer(1, 2, 3)
"""
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        assert metrics.immutable_collections >= 5
        assert metrics.mutable_collections >= 4

    def test_error_handling_metrics(self, analyzer):
        """Test error handling and monad usage metrics."""
        code = """
import scala.util.{Try, Success, Failure}

def divide(x: Int, y: Int): Option[Int] = 
  if (y != 0) Some(x / y) else None

def safeOperation(): Try[String] = Try {
  riskyOperation()
} match {
  case Success(value) => Success(value.toUpperCase)
  case Failure(e) => Failure(new Exception("Wrapped", e))
}

def process(): Either[String, Int] = 
  for {
    x <- Right(10)
    y <- Right(20)
  } yield x + y

try {
  dangerousOp()
} catch {
  case e: IOException => handleIO(e)
  case e: SQLException => handleSQL(e)
  case _: Exception => handleGeneric()
} finally {
  cleanup()
}

throw new IllegalArgumentException("Invalid")
"""
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        assert metrics.option_usage >= 2
        assert metrics.try_usage >= 3
        assert metrics.either_usage >= 2
        assert metrics.try_blocks >= 1
        assert metrics.catch_blocks >= 1
        assert metrics.finally_blocks >= 1
        assert metrics.throw_statements >= 1


class TestErrorHandling:
    """Test suite for error handling in Scala analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Provide a ScalaAnalyzer instance."""
        return ScalaAnalyzer()

    def test_handle_malformed_code(self, analyzer):
        """Test handling of malformed Scala code."""
        code = """
class Broken {
  This is not valid Scala code!!!
  
  def method() = {
    missing closing brace
  
  // Missing closing brace for class
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("test.scala"))
        exports = analyzer.extract_exports(code, Path("test.scala"))
        structure = analyzer.extract_structure(code, Path("test.scala"))
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty file."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.scala"))
        exports = analyzer.extract_exports(code, Path("test.scala"))
        structure = analyzer.extract_structure(code, Path("test.scala"))
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        assert imports == []
        assert exports == []
        assert len(structure.classes) == 0
        assert metrics.line_count == 1


class TestEdgeCases:
    """Test suite for edge cases in Scala analysis."""

    @pytest.fixture
    def analyzer(self):
        """Provide a ScalaAnalyzer instance."""
        return ScalaAnalyzer()

    def test_handle_string_interpolation(self, analyzer):
        """Test handling of string interpolation."""
        code = '''
val name = "World"
val greeting = s"Hello, $name"
val complex = s"""
  |This is a multiline string
  |with interpolation: ${name.toUpperCase}
  |and it might contain code-like syntax:
  |if (true) { "not real code" }
""".stripMargin

class Test {
  def format(x: Int): String = f"Value: $x%04d"
  def raw(): String = raw"No escape: \n\t"
}
'''
        structure = analyzer.extract_structure(code, Path("test.scala"))
        metrics = analyzer.calculate_complexity(code, Path("test.scala"))

        # Should correctly identify the class
        assert len(structure.classes) == 1
        # Complexity should not count code inside strings
        assert metrics.cyclomatic < 5

    def test_handle_xml_literals(self, analyzer):
        """Test handling of XML literals in Scala."""
        code = """
val xml = <div class="container">
  <h1>Title</h1>
  <p>Paragraph with {expression}</p>
</div>

def renderHtml(title: String): scala.xml.Elem = 
  <html>
    <head><title>{title}</title></head>
    <body>
      <h1>{title}</h1>
    </body>
  </html>
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        # Should handle XML without errors
        assert len(structure.functions) >= 1

    def test_handle_symbolic_operators(self, analyzer):
        """Test handling of symbolic method names."""
        code = """
class Vector(val x: Double, val y: Double) {
  def +(other: Vector): Vector = 
    new Vector(x + other.x, y + other.y)
  
  def *(scalar: Double): Vector = 
    new Vector(x * scalar, y * scalar)
  
  def unary_- : Vector = new Vector(-x, -y)
  
  def ::(elem: Double): List[Double] = List(elem, x, y)
}

object :: {
  def unapply[T](list: List[T]): Option[(T, List[T])] = 
    list.headOption.map((_, list.tail))
}
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        vector_class = next(c for c in structure.classes if c.name == "Vector")
        assert len(vector_class.methods) >= 4

    def test_handle_package_objects(self, analyzer):
        """Test handling of package objects."""
        code = """
package com.example

package object utils {
  type StringMap = Map[String, String]
  
  implicit class RichString(val s: String) extends AnyVal {
    def words: Array[String] = s.split("\\s+")
  }
  
  def helper(x: Int): String = x.toString
}

package subpackage {
  class InnerClass
}
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        # Should find the package object
        assert any(o["name"] == "utils" for o in structure.objects)

    def test_handle_by_name_parameters(self, analyzer):
        """Test handling of by-name parameters."""
        code = """
def retry[T](n: Int)(fn: => T): T = {
  try {
    fn
  } catch {
    case e: Exception if n > 1 => 
      retry(n - 1)(fn)
  }
}

class Logger {
  def debug(message: => String): Unit = 
    if (isDebugEnabled) println(message)
  
  def isDebugEnabled: Boolean = true
}

def assert(condition: => Boolean, message: => String): Unit = 
  if (!condition) throw new AssertionError(message)
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        retry_func = next(f for f in structure.functions if f.name == "retry")
        assert retry_func is not None

    def test_handle_self_types(self, analyzer):
        """Test handling of self types."""
        code = """
trait Database {
  def connect(): Unit
}

trait UserRepository {
  self: Database =>
  
  def findUser(id: Int): Option[User] = {
    connect()
    // query logic
    None
  }
}

trait Logging {
  this: Service =>
  
  def log(message: String): Unit = println(s"[$serviceName] $message")
}

trait Service {
  def serviceName: String
}
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        assert len(structure.traits) >= 4

    def test_handle_extractors(self, analyzer):
        """Test handling of extractors and unapply methods."""
        code = """
object Email {
  def apply(user: String, domain: String): String = s"$user@$domain"
  
  def unapply(email: String): Option[(String, String)] = {
    val parts = email.split("@")
    if (parts.length == 2) Some((parts(0), parts(1)))
    else None
  }
}

object Twice {
  def unapply(x: Int): Option[Int] = 
    if (x % 2 == 0) Some(x / 2) else None
}

val email = "test@example.com" match {
  case Email(user, domain) => s"User: $user, Domain: $domain"
  case _ => "Invalid email"
}
"""
        structure = analyzer.extract_structure(code, Path("test.scala"))

        email_object = next(o for o in structure.objects if o["name"] == "Email")
        assert any(m["name"] == "unapply" for m in email_object["methods"])

    def test_main_detection(self, analyzer):
        """Test detection of main method and App trait."""
        code_with_main = """
object Main {
  def main(args: Array[String]): Unit = {
    println("Hello, World!")
  }
}
"""
        structure = analyzer.extract_structure(code_with_main, Path("test.scala"))
        assert structure.has_main is True

        code_with_app = """
object MyApp extends App {
  println("Hello from App")
}
"""
        structure = analyzer.extract_structure(code_with_app, Path("test.scala"))
        assert structure.has_main is True

        code_without_main = """
class NotMain {
  def notMain(): Unit = println("Not main")
}
"""
        structure = analyzer.extract_structure(code_without_main, Path("test.scala"))
        assert structure.has_main is False

    def test_scala_version_detection(self, analyzer):
        """Test detection of Scala 2 vs Scala 3."""
        scala2_code = """
implicit val ordering: Ordering[Int] = Ordering.Int
implicit def conversion(x: Int): String = x.toString
"""
        structure = analyzer.extract_structure(scala2_code, Path("test.scala"))
        assert structure.scala_version == 2

        scala3_code = """
given Ordering[Int] = Ordering.Int
extension (x: Int)
  def toText: String = x.toString
"""
        structure = analyzer.extract_structure(scala3_code, Path("test.scala"))
        assert structure.scala_version == 3
