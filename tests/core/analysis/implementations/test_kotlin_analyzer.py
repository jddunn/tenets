"""
Unit tests for the Kotlin code analyzer with Android and multiplatform support.

This module tests the Kotlin-specific code analysis functionality including
import statements, data classes, sealed classes, coroutines, null safety,
extension functions, Android components, and modern Kotlin features.

Test Coverage:
    - Import extraction (wildcards, aliases)
    - Export detection (classes, interfaces, objects, extensions)
    - Structure extraction (data classes, sealed hierarchies, companions)
    - Complexity metrics (null safety, coroutines, Android-specific)
    - Extension functions and properties
    - Delegation patterns
    - Android components (Activities, Fragments, ViewModels)
    - Error handling for invalid Kotlin code
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.kotlin_analyzer import KotlinAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestKotlinAnalyzerInitialization:
    """Test suite for KotlinAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = KotlinAnalyzer()

        assert analyzer.language_name == "kotlin"
        assert ".kt" in analyzer.file_extensions
        assert ".kts" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for Kotlin import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a KotlinAnalyzer instance."""
        return KotlinAnalyzer()

    def test_extract_basic_imports(self, analyzer):
        """Test extraction of basic import statements."""
        code = """
package com.example.app

import kotlin.collections.List
import kotlin.coroutines.CoroutineContext
import java.util.Date
import kotlinx.coroutines.launch
import androidx.appcompat.app.AppCompatActivity
"""
        imports = analyzer.extract_imports(code, Path("test.kt"))

        assert len(imports) == 6  # Including package declaration

        # Check package
        package_import = next(imp for imp in imports if imp.type == "package")
        assert package_import.module == "com.example.app"

        # Check Kotlin imports
        list_import = next(imp for imp in imports if "List" in imp.module)
        assert list_import.category == "kotlin_collections"

        coroutine_import = next(imp for imp in imports if "CoroutineContext" in imp.module)
        assert coroutine_import.category == "kotlin_coroutines"

        # Check kotlinx import
        launch_import = next(imp for imp in imports if "launch" in imp.module)
        assert launch_import.category == "kotlinx_coroutines"

        # Check Android import
        activity_import = next(imp for imp in imports if "AppCompatActivity" in imp.module)
        assert activity_import.category == "android"
        assert activity_import.is_android is True

    def test_extract_wildcard_imports(self, analyzer):
        """Test extraction of wildcard imports."""
        code = """
import kotlin.collections.*
import java.util.*
import kotlinx.coroutines.*
"""
        imports = analyzer.extract_imports(code, Path("test.kt"))

        assert len(imports) == 3
        assert all(imp.is_wildcard for imp in imports)

        collections_import = next(imp for imp in imports if "collections" in imp.module)
        assert collections_import.category == "kotlin_collections"

    def test_extract_aliased_imports(self, analyzer):
        """Test extraction of aliased imports."""
        code = """
import java.util.List as JList
import kotlin.collections.List as KList
import androidx.lifecycle.ViewModel as VM
"""
        imports = analyzer.extract_imports(code, Path("test.kt"))

        assert len(imports) == 3

        jlist_import = next(imp for imp in imports if imp.alias == "JList")
        assert "java.util.List" in jlist_import.module

        klist_import = next(imp for imp in imports if imp.alias == "KList")
        assert "kotlin.collections.List" in klist_import.module

        vm_import = next(imp for imp in imports if imp.alias == "VM")
        assert "ViewModel" in vm_import.module
        assert vm_import.is_android is True


class TestExportExtraction:
    """Test suite for Kotlin export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a KotlinAnalyzer instance."""
        return KotlinAnalyzer()

    def test_extract_classes(self, analyzer):
        """Test extraction of class exports."""
        code = """
class PublicClass
internal class InternalClass
private class PrivateClass
open class OpenClass
abstract class AbstractClass
sealed class SealedClass
data class DataClass(val name: String, val age: Int)
enum class EnumClass { A, B, C }
annotation class AnnotationClass
inline class InlineClass(val value: String)
value class ValueClass(val value: Int)
"""
        exports = analyzer.extract_exports(code, Path("test.kt"))

        public_exports = [e for e in exports if e["visibility"] != "private"]
        assert len(public_exports) == 10  # Excludes PrivateClass

        # Check class types
        data_class = next(e for e in exports if e["name"] == "DataClass")
        assert data_class["type"] == "data_class"

        enum_class = next(e for e in exports if e["name"] == "EnumClass")
        assert enum_class["type"] == "enum_class"

        sealed_class = next(e for e in exports if e["name"] == "SealedClass")
        assert sealed_class["type"] == "sealed_class"

        annotation_class = next(e for e in exports if e["name"] == "AnnotationClass")
        assert annotation_class["type"] == "annotation_class"

        value_class = next((e for e in exports if e["name"] == "ValueClass"), None)
        if value_class:
            assert value_class["type"] == "value_class"

    def test_extract_interfaces(self, analyzer):
        """Test extraction of interfaces."""
        code = """
interface PublicInterface
internal interface InternalInterface
private interface PrivateInterface
sealed interface SealedInterface
fun interface FunctionalInterface {
    fun invoke(x: Int): String
}
"""
        exports = analyzer.extract_exports(code, Path("test.kt"))

        interfaces = [e for e in exports if "interface" in e["type"]]
        public_interfaces = [i for i in interfaces if i["visibility"] != "private"]
        assert len(public_interfaces) == 4  # Excludes PrivateInterface

        sealed_interface = next(i for i in interfaces if i["name"] == "SealedInterface")
        assert sealed_interface["type"] == "sealed_interface"

        fun_interface = next(i for i in interfaces if i["name"] == "FunctionalInterface")
        assert fun_interface["is_fun_interface"] is True

    def test_extract_objects(self, analyzer):
        """Test extraction of objects."""
        code = """
object Singleton
internal object InternalObject
private object PrivateObject

class MyClass {
    companion object {
        const val CONSTANT = 42
    }
    
    companion object Named {
        fun helper() = "help"
    }
}
"""
        exports = analyzer.extract_exports(code, Path("test.kt"))

        objects = [e for e in exports if "object" in e["type"]]
        public_objects = [o for o in objects if o.get("visibility") != "private"]

        singleton = next((o for o in objects if o["name"] == "Singleton"), None)
        assert singleton is not None
        assert singleton["type"] == "object"

        # Companion objects
        companion_objects = [o for o in objects if o["type"] == "companion_object"]
        assert len(companion_objects) >= 1

    def test_extract_functions(self, analyzer):
        """Test extraction of functions."""
        code = """
fun publicFunction() {}
internal fun internalFunction() {}
private fun privateFunction() {}
suspend fun suspendFunction() {}
inline fun inlineFunction(block: () -> Unit) {}
tailrec fun tailrecFunction(n: Int): Int = if (n <= 1) n else tailrecFunction(n - 1)
operator fun plus(other: Int) = this + other
infix fun and(other: Boolean) = this && other
"""
        exports = analyzer.extract_exports(code, Path("test.kt"))

        functions = [e for e in exports if e["type"] == "function"]
        public_functions = [f for f in functions if f["visibility"] != "private"]
        assert len(public_functions) >= 7  # Excludes privateFunction

        suspend_func = next(f for f in functions if f["name"] == "suspendFunction")
        assert suspend_func["is_suspend"] is True

        inline_func = next(f for f in functions if f["name"] == "inlineFunction")
        assert inline_func["is_inline"] is True

        operator_func = next(f for f in functions if f["name"] == "plus")
        assert operator_func["is_operator"] is True

    def test_extract_extension_functions(self, analyzer):
        """Test extraction of extension functions."""
        code = """
fun String.isPalindrome(): Boolean = this == this.reversed()
suspend fun List<Int>.processAsync() = coroutineScope { }
inline fun <T> T.apply(block: T.() -> Unit): T { block(); return this }
operator fun Int.times(str: String) = str.repeat(this)
"""
        exports = analyzer.extract_exports(code, Path("test.kt"))

        extension_funcs = [e for e in exports if e["type"] == "extension_function"]
        assert len(extension_funcs) >= 4

        palindrome = next(f for f in extension_funcs if f["name"] == "isPalindrome")
        assert palindrome["receiver"] == "String"

        process_async = next(f for f in extension_funcs if f["name"] == "processAsync")
        assert process_async["is_suspend"] is True

    def test_extract_properties(self, analyzer):
        """Test extraction of properties."""
        code = """
val publicVal = 42
var publicVar = "mutable"
internal val internalVal = true
private val privateVal = 3.14
const val CONSTANT = "constant"
lateinit var lateInitVar: String

val String.lastChar: Char
    get() = this[length - 1]

var StringBuilder.lastChar: Char
    get() = this[length - 1]
    set(value) { setCharAt(length - 1, value) }
"""
        exports = analyzer.extract_exports(code, Path("test.kt"))

        properties = [e for e in exports if "property" in e["type"]]
        public_properties = [p for p in properties if p.get("visibility") != "private"]
        assert len(public_properties) >= 7  # Excludes privateVal

        const_prop = next((p for p in properties if p["name"] == "CONSTANT"), None)
        if const_prop:
            assert const_prop["is_const"] is True

        lateinit_prop = next((p for p in properties if p["name"] == "lateInitVar"), None)
        if lateinit_prop:
            assert lateinit_prop["is_lateinit"] is True

        # Extension properties
        ext_properties = [p for p in properties if p["type"] == "extension_property"]
        assert len(ext_properties) >= 2

    def test_extract_type_aliases(self, analyzer):
        """Test extraction of type aliases."""
        code = """
typealias StringMap = Map<String, String>
internal typealias Handler = (Int) -> Unit
private typealias PrivateAlias = List<String>
"""
        exports = analyzer.extract_exports(code, Path("test.kt"))

        type_aliases = [e for e in exports if e["type"] == "typealias"]
        public_aliases = [t for t in type_aliases if t["visibility"] != "private"]
        assert len(public_aliases) == 2  # Excludes PrivateAlias


class TestStructureExtraction:
    """Test suite for Kotlin code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a KotlinAnalyzer instance."""
        return KotlinAnalyzer()

    def test_extract_data_class(self, analyzer):
        """Test extraction of data classes."""
        code = """
data class Person(
    val name: String,
    val age: Int,
    var email: String? = null
) {
    fun greet() = "Hello, I'm $name"
}

data class Point(val x: Int, val y: Int)

data class Response<T>(
    val data: T?,
    val error: String? = null
)
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        assert len(structure.classes) == 3

        person_class = next(c for c in structure.classes if c.name == "Person")
        assert person_class.is_data_class is True
        assert len(person_class.constructor_params) == 3

        # Check constructor parameters
        email_param = next(p for p in person_class.constructor_params if p["name"] == "email")
        assert email_param.get("default") == "null"
        assert "?" in email_param.get("type", "")

    def test_extract_sealed_class_hierarchy(self, analyzer):
        """Test extraction of sealed class hierarchies."""
        code = """
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val message: String) : Result<Nothing>()
    object Loading : Result<Nothing>()
}

sealed interface State
class Idle : State
class Running : State
data class Finished(val result: Int) : State
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        result_class = next(c for c in structure.classes if c.name == "Result")
        assert result_class.is_sealed is True

        # Check sealed interface
        state_interface = next(i for i in structure.interfaces if i["name"] == "State")
        assert state_interface["is_sealed"] is True

    def test_extract_class_with_delegation(self, analyzer):
        """Test extraction of classes using delegation."""
        code = """
interface Base {
    fun print()
}

class BaseImpl(val x: Int) : Base {
    override fun print() { println(x) }
}

class Derived(b: Base) : Base by b {
    override fun print() {
        println("Derived")
    }
}

class DelegatedProperties {
    val lazyValue: String by lazy { "computed" }
    var observed: String by Delegates.observable("initial") { _, old, new ->
        println("$old -> $new")
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        derived_class = next(c for c in structure.classes if c.name == "Derived")
        assert "Base" in derived_class.interfaces
        assert derived_class.delegates.get("Base") == "b"

    def test_extract_companion_object(self, analyzer):
        """Test extraction of companion objects."""
        code = """
class MyClass {
    companion object {
        const val CONSTANT = 42
        
        @JvmStatic
        fun staticMethod() = "static"
        
        @JvmField
        val field = "field"
    }
}

class Factory {
    companion object Named {
        fun create(): Factory = Factory()
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        my_class = next(c for c in structure.classes if c.name == "MyClass")
        assert my_class.companion_object is not None

        factory_class = next(c for c in structure.classes if c.name == "Factory")
        assert factory_class.companion_object is not None

    def test_extract_android_components(self, analyzer):
        """Test extraction of Android components."""
        code = """
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModel

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }
}

class HomeFragment : Fragment() {
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_home, container, false)
    }
}

class MainViewModel : ViewModel() {
    val data = MutableLiveData<String>()
    
    fun loadData() {
        viewModelScope.launch {
            // Load data
        }
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        assert structure.is_android is True

        main_activity = next(c for c in structure.classes if c.name == "MainActivity")
        assert main_activity.android_type == "activity"

        home_fragment = next(c for c in structure.classes if c.name == "HomeFragment")
        assert home_fragment.android_type == "fragment"

        main_viewmodel = next(c for c in structure.classes if c.name == "MainViewModel")
        assert main_viewmodel.android_type == "viewmodel"

    def test_extract_coroutine_functions(self, analyzer):
        """Test extraction of coroutine and suspend functions."""
        code = """
suspend fun fetchData(): String {
    return withContext(Dispatchers.IO) {
        // Fetch data
        "data"
    }
}

fun processAsync() = GlobalScope.launch {
    val data = async { fetchData() }
    println(data.await())
}

suspend fun flowExample() = flow {
    for (i in 1..3) {
        delay(100)
        emit(i)
    }
}

class CoroutineClass {
    suspend fun suspendMethod() = coroutineScope {
        launch { doWork() }
    }
    
    private suspend fun doWork() {
        delay(1000)
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        assert structure.suspend_functions >= 3
        assert structure.coroutine_launches >= 2
        assert structure.flow_usage >= 1

    def test_extract_extension_functions_and_properties(self, analyzer):
        """Test extraction of extension functions and properties."""
        code = """
fun String.removeSpaces() = this.replace(" ", "")

val String.wordCount: Int
    get() = this.split("\\s+".toRegex()).size

var StringBuilder.lastChar: Char
    get() = this[length - 1]
    set(value) {
        this.setCharAt(length - 1, value)
    }

inline fun <T> T.apply(block: T.() -> Unit): T {
    block()
    return this
}

fun List<Int>.sum(): Int {
    var result = 0
    for (element in this) {
        result += element
    }
    return result
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        assert structure.extension_functions >= 3
        assert structure.extension_properties >= 2

    def test_extract_lambda_and_scope_functions(self, analyzer):
        """Test extraction of lambda expressions and scope functions."""
        code = """
fun example() {
    val numbers = listOf(1, 2, 3, 4, 5)
    
    // Lambda expressions
    val doubled = numbers.map { it * 2 }
    val filtered = numbers.filter { x -> x > 2 }
    val sum = numbers.fold(0) { acc, n -> acc + n }
    
    // Scope functions
    val result = "Hello".let {
        println(it)
        it.length
    }
    
    val sb = StringBuilder().apply {
        append("Hello")
        append(" ")
        append("World")
    }
    
    with(numbers) {
        println(size)
        println(first())
    }
    
    numbers.also { list ->
        println("List has ${list.size} elements")
    }.run {
        filter { it > 2 }
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        assert structure.lambda_expressions >= 3
        assert structure.scope_functions >= 4


class TestComplexityCalculation:
    """Test suite for Kotlin complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a KotlinAnalyzer instance."""
        return KotlinAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
fun complexFunction(x: Int, y: Int): Int {
    return when {
        x > 0 && y > 0 -> {
            if (x > y) x else y
        }
        x < 0 || y < 0 -> {
            for (i in 1..10) {
                if (i % 2 == 0) {
                    println(i)
                }
            }
            -1
        }
        else -> {
            try {
                x / y
            } catch (e: ArithmeticException) {
                0
            }
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 8

    def test_null_safety_metrics(self, analyzer):
        """Test null safety complexity metrics."""
        code = """
class NullSafetyExample {
    var nullable: String? = null
    lateinit var lateInit: String
    
    fun process(input: String?) {
        // Null assertion
        println(input!!)
        
        // Safe call
        println(input?.length)
        
        // Elvis operator
        val length = input?.length ?: 0
        
        // Safe call chain
        val result = input?.trim()?.uppercase()?.reversed()
        
        // Let with null check
        input?.let {
            println("Not null: $it")
        }
        
        // Multiple null assertions
        val x: String? = null
        val y: String? = null
        println(x!! + y!!)
    }
    
    fun multipleElvis(a: String?, b: String?, c: String?) {
        val result = a ?: b ?: c ?: "default"
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        assert metrics.nullable_types >= 5
        assert metrics.null_assertions >= 3
        assert metrics.safe_calls >= 4
        assert metrics.elvis_operators >= 4
        assert metrics.lateinit_count >= 1
        assert metrics.let_calls >= 1

    def test_coroutine_metrics(self, analyzer):
        """Test coroutine complexity metrics."""
        code = """
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

class CoroutineExample {
    fun launchExample() = GlobalScope.launch {
        val deferred = async {
            delay(1000)
            "Result"
        }
        println(deferred.await())
    }
    
    suspend fun suspendFunction() = coroutineScope {
        launch {
            delay(100)
        }
        
        async {
            delay(200)
        }.await()
    }
    
    fun flowExample() = flow {
        for (i in 1..5) {
            emit(i)
        }
    }.map { it * 2 }
     .filter { it > 5 }
    
    fun channelExample() = GlobalScope.launch {
        val channel = Channel<Int>()
        launch {
            for (x in 1..5) channel.send(x)
            channel.close()
        }
    }
    
    fun blockingExample() = runBlocking {
        delay(100)
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        assert metrics.suspend_functions >= 1
        assert metrics.coroutine_launches >= 4
        assert metrics.await_calls >= 2
        assert metrics.flow_usage >= 1
        assert metrics.channel_usage >= 1
        assert metrics.runblocking_usage >= 1

    def test_when_expression_metrics(self, analyzer):
        """Test when expression complexity metrics."""
        code = """
fun processValue(x: Any): String {
    return when (x) {
        is String -> "String: $x"
        is Int -> when {
            x > 0 -> "Positive"
            x < 0 -> "Negative"
            else -> "Zero"
        }
        is List<*> -> "List of size ${x.size}"
        is Map<*, *> -> "Map with ${x.size} entries"
        else -> "Unknown"
    }
}

sealed class State {
    object Loading : State()
    data class Success(val data: String) : State()
    data class Error(val message: String) : State()
}

fun handleState(state: State) = when (state) {
    is State.Loading -> "Loading..."
    is State.Success -> "Success: ${state.data}"
    is State.Error -> "Error: ${state.message}"
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        assert metrics.when_expressions >= 3
        assert metrics.when_branches >= 11

    def test_android_specific_metrics(self, analyzer):
        """Test Android-specific complexity metrics."""
        code = """
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.fragment.app.Fragment

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
    }
}

class HomeFragment : Fragment() {
    private var _binding: FragmentHomeBinding? = null
    private val binding get() = _binding!!
}

class MainViewModel : ViewModel() {
    private val _data = MutableLiveData<String>()
    val data: LiveData<String> = _data
    
    private val _list = MutableLiveData<List<Item>>()
    val list: LiveData<List<Item>> = _list
    
    init {
        data.observe(viewLifecycleOwner) { value ->
            updateUI(value)
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        assert metrics.activity_count >= 1
        assert metrics.fragment_count >= 1
        assert metrics.viewmodel_count >= 1
        assert metrics.livedata_usage >= 4
        assert metrics.observer_usage >= 1
        assert metrics.binding_usage >= 3

    def test_delegation_metrics(self, analyzer):
        """Test delegation pattern metrics."""
        code = """
import kotlin.properties.Delegates

class DelegationExample {
    // Lazy property
    val lazyValue: String by lazy {
        println("Computing")
        "Result"
    }
    
    val anotherLazy by lazy { computeValue() }
    
    // Observable property
    var observed: String by Delegates.observable("initial") { prop, old, new ->
        println("$old -> $new")
    }
    
    // Vetoable property
    var max: Int by Delegates.vetoable(0) { prop, old, new ->
        new > old
    }
    
    // Custom delegation
    val custom by CustomDelegate()
    
    // Map delegation
    class User(val map: Map<String, Any?>) {
        val name: String by map
        val age: Int by map
    }
}

interface Base {
    fun print()
}

class Derived(b: Base) : Base by b
"""
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        assert metrics.delegation_count >= 6
        assert metrics.lazy_properties >= 2
        assert metrics.observable_properties >= 2


class TestErrorHandling:
    """Test suite for error handling in Kotlin analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Provide a KotlinAnalyzer instance."""
        return KotlinAnalyzer()

    def test_handle_malformed_code(self, analyzer):
        """Test handling of malformed Kotlin code."""
        code = """
class Broken {
    This is not valid Kotlin code!!!
    
    fun method() {
        missing closing brace
    
    // Missing closing brace for class
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("test.kt"))
        exports = analyzer.extract_exports(code, Path("test.kt"))
        structure = analyzer.extract_structure(code, Path("test.kt"))
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty file."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.kt"))
        exports = analyzer.extract_exports(code, Path("test.kt"))
        structure = analyzer.extract_structure(code, Path("test.kt"))
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        assert imports == []
        assert exports == []
        assert len(structure.classes) == 0
        assert metrics.line_count == 1


class TestEdgeCases:
    """Test suite for edge cases in Kotlin analysis."""

    @pytest.fixture
    def analyzer(self):
        """Provide a KotlinAnalyzer instance."""
        return KotlinAnalyzer()

    def test_handle_string_templates(self, analyzer):
        """Test handling of string templates and raw strings."""
        code = '''
fun stringTemplates() {
    val name = "World"
    val greeting = "Hello, $name"
    val complex = "Result: ${computeValue()}"
    
    val multiline = """
        |This is a multiline string
        |with template: $name
        |and expression: ${name.length}
        |It might contain code-like syntax:
        |if (true) { "not real code" }
    """.trimMargin()
    
    val rawString = """
        No escape: \n\t
        Dollar sign: ${'$'}
    """
}
'''
        structure = analyzer.extract_structure(code, Path("test.kt"))
        metrics = analyzer.calculate_complexity(code, Path("test.kt"))

        # Should correctly identify the function
        assert len(structure.functions) >= 1
        # Complexity should not count code inside strings
        assert metrics.cyclomatic < 5

    def test_handle_inline_and_value_classes(self, analyzer):
        """Test handling of inline and value classes."""
        code = """
inline class Password(val value: String)

@JvmInline
value class UserId(val id: Long) {
    init {
        require(id > 0) { "Id must be positive" }
    }
    
    fun isValid(): Boolean = id > 0
}

// Regular inline function
inline fun <reified T> isType(value: Any): Boolean = value is T

// Inline with non-local returns
inline fun operation(block: () -> Unit) {
    println("Before")
    block()
    println("After")
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        password_class = next((c for c in structure.classes if c.name == "Password"), None)
        if password_class:
            assert password_class.is_value_class is True

        userid_class = next((c for c in structure.classes if c.name == "UserId"), None)
        if userid_class:
            assert userid_class.is_value_class is True

    def test_handle_multiplatform_declarations(self, analyzer):
        """Test handling of Kotlin Multiplatform declarations."""
        code = """
expect class Platform {
    val name: String
}

actual class Platform {
    actual val name: String = "Android"
}

@JvmStatic
fun jvmStaticMethod() {}

@JvmOverloads
fun overloadedMethod(x: Int = 0, y: Int = 0) {}

@JvmName("customName")
fun renamedForJvm() {}

@JsName("jsCustomName")
fun renamedForJs() {}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        assert structure.is_multiplatform is True

    def test_handle_operator_overloading(self, analyzer):
        """Test handling of operator overloading."""
        code = """
data class Point(val x: Int, val y: Int) {
    operator fun plus(other: Point) = Point(x + other.x, y + other.y)
    operator fun minus(other: Point) = Point(x - other.x, y - other.y)
    operator fun unaryMinus() = Point(-x, -y)
    operator fun inc() = Point(x + 1, y + 1)
    
    operator fun get(index: Int) = when (index) {
        0 -> x
        1 -> y
        else -> throw IndexOutOfBoundsException()
    }
    
    operator fun invoke() = "$x, $y"
}

operator fun Int.times(str: String) = str.repeat(this)
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        point_class = next(c for c in structure.classes if c.name == "Point")
        operator_methods = [m for m in point_class.methods if "operator" in m.get("modifiers", [])]
        assert len(operator_methods) >= 5

    def test_handle_type_parameters_and_variance(self, analyzer):
        """Test handling of generic type parameters with variance."""
        code = """
class Box<T>(val value: T)

class Container<out T>(val value: T) {
    fun get(): T = value
}

class MutableContainer<in T> {
    fun set(value: T) { }
}

interface Producer<out T> {
    fun produce(): T
}

interface Consumer<in T> {
    fun consume(item: T)
}

class Complex<T : Number, U : Comparable<U>>(val first: T, val second: U)

inline fun <reified T> genericFunction(value: Any): T? {
    return value as? T
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        box_class = next(c for c in structure.classes if c.name == "Box")
        assert box_class.type_parameters == "T"

        container_class = next(c for c in structure.classes if c.name == "Container")
        assert "out T" in str(container_class.type_parameters)

    def test_handle_destructuring_declarations(self, analyzer):
        """Test handling of destructuring declarations."""
        code = """
fun destructuringExample() {
    val (name, age) = Person("Alice", 30)
    
    val map = mapOf("key1" to "value1", "key2" to "value2")
    for ((key, value) in map) {
        println("$key: $value")
    }
    
    val list = listOf(1, 2, 3)
    val (first, second, third) = list
    
    data class Result(val success: Boolean, val data: String?, val error: String?)
    val (success, data, error) = Result(true, "data", null)
}
"""
        structure = analyzer.extract_structure(code, Path("test.kt"))

        # Should handle destructuring without errors
        assert len(structure.functions) >= 1

    def test_main_function_detection(self, analyzer):
        """Test detection of main function."""
        code_with_main = """
fun main() {
    println("Hello, World!")
}
"""
        structure = analyzer.extract_structure(code_with_main, Path("test.kt"))
        assert structure.has_main is True

        code_with_args_main = """
fun main(args: Array<String>) {
    println("Args: ${args.joinToString()}")
}
"""
        structure = analyzer.extract_structure(code_with_args_main, Path("test.kt"))
        assert structure.has_main is True

        code_without_main = """
fun notMain() {
    println("Not main")
}
"""
        structure = analyzer.extract_structure(code_without_main, Path("test.kt"))
        assert structure.has_main is False

    def test_test_file_detection(self, analyzer):
        """Test detection of test files."""
        test_code = """
import org.junit.Test
import kotlin.test.assertEquals

class MyClassTest {
    @Test
    fun testSomething() {
        assertEquals(4, 2 + 2)
    }
}
"""
        structure = analyzer.extract_structure(test_code, Path("MyClassTest.kt"))
        assert structure.is_test_file is True

        structure = analyzer.extract_structure(test_code, Path("test/MyClass.kt"))
        assert structure.is_test_file is True

        structure = analyzer.extract_structure(test_code, Path("src/main/MyClass.kt"))
        assert structure.is_test_file is False
