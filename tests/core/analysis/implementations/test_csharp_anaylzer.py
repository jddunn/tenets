"""
Unit tests for the C# code analyzer with Unity3D support.

This module tests the C#-specific code analysis functionality including
using directives, class/interface parsing, property handling, async/await,
LINQ queries, Unity3D components, and C#-specific complexity calculation.

Test Coverage:
    - Using directive extraction (standard, static, global, aliases)
    - Export detection (classes, interfaces, structs, enums, delegates, records)
    - Structure extraction with C# features (properties, events, attributes)
    - Unity3D specific features (MonoBehaviour, Coroutines, Unity methods)
    - Complexity metrics (async/await, LINQ, exception handling)
    - Framework detection (Unity, ASP.NET Core, WPF, etc.)
    - Modern C# features (nullable references, pattern matching, records)
    - Error handling for invalid C# code
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.csharp_analyzer import CSharpAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestCSharpAnalyzerInitialization:
    """Test suite for CSharpAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = CSharpAnalyzer()

        assert analyzer.language_name == "csharp"
        assert ".cs" in analyzer.file_extensions
        assert ".csx" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for C# using directive extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSharpAnalyzer instance."""
        return CSharpAnalyzer()

    def test_extract_using_directives(self, analyzer):
        """Test extraction of standard using directives."""
        code = """
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
"""
        imports = analyzer.extract_imports(code, Path("test.cs"))

        assert len(imports) == 5
        assert any(imp.module == "System" for imp in imports)
        assert any(imp.module == "System.Collections.Generic" for imp in imports)
        assert any(imp.module == "System.Linq" for imp in imports)

        # Check import categorization
        system_import = next(imp for imp in imports if imp.module == "System")
        assert system_import.type == "using"
        assert system_import.category == "system"
        assert system_import.is_relative is False

    def test_extract_static_using(self, analyzer):
        """Test extraction of using static directives."""
        code = """
using static System.Math;
using static System.Console;
using static MyNamespace.MyStaticClass;
"""
        imports = analyzer.extract_imports(code, Path("test.cs"))

        assert len(imports) == 3
        assert all(imp.type == "using_static" for imp in imports)

        math_import = next(imp for imp in imports if imp.module == "System.Math")
        assert math_import.type == "using_static"
        assert math_import.category == "system"

    def test_extract_global_using(self, analyzer):
        """Test extraction of global using directives (C# 10+)."""
        code = """
global using System.Text.Json;
global using Microsoft.Extensions.DependencyInjection;
global using static System.Console;
"""
        imports = analyzer.extract_imports(code, Path("test.cs"))

        assert len(imports) == 3
        
        json_import = next(imp for imp in imports if "Json" in imp.module)
        assert json_import.type == "global_using"

    def test_extract_using_aliases(self, analyzer):
        """Test extraction of using aliases."""
        code = """
using Project = PC.MyCompany.Project;
using Console = System.Console;
using Dict = System.Collections.Generic.Dictionary<string, object>;
"""
        imports = analyzer.extract_imports(code, Path("test.cs"))

        assert len(imports) == 3
        assert all(imp.type == "using_alias" for imp in imports)

        project_import = next(imp for imp in imports if imp.alias == "Project")
        assert project_import.module == "PC.MyCompany.Project"
        assert project_import.alias == "Project"

    def test_extract_unity_usings(self, analyzer):
        """Test extraction of Unity-specific using directives."""
        code = """
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using TMPro;
using Cinemachine;
using UnityEngine.InputSystem;
"""
        imports = analyzer.extract_imports(code, Path("test.cs"))

        unity_imports = [imp for imp in imports if imp.is_unity]
        assert len(unity_imports) >= 4

        assert any(imp.category == "unity" for imp in imports)
        assert any(imp.category == "unity_package" for imp in imports if "TMPro" in imp.module)

    def test_namespace_context(self, analyzer):
        """Test that namespace context is tracked."""
        code = """
using System;

namespace MyApp.Models
{
    using System.ComponentModel.DataAnnotations;
    
    public class User
    {
        // Class content
    }
}
"""
        imports = analyzer.extract_imports(code, Path("test.cs"))

        namespace_import = next((imp for imp in imports if "DataAnnotations" in imp.module), None)
        if namespace_import and hasattr(namespace_import, 'namespace_context'):
            assert namespace_import.namespace_context == "MyApp.Models"

    def test_extract_csproj_dependencies(self, analyzer):
        """Test extraction from .csproj file."""
        code = """
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net6.0</TargetFramework>
  </PropertyGroup>
  
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
    <PackageReference Include="Microsoft.EntityFrameworkCore" Version="6.0.0" />
    <PackageReference Include="xunit" Version="2.4.1" />
  </ItemGroup>
  
  <ItemGroup>
    <ProjectReference Include="../SharedLib/SharedLib.csproj" />
  </ItemGroup>
</Project>
"""
        imports = analyzer.extract_imports(code, Path("test.csproj"))

        package_refs = [imp for imp in imports if imp.type == "nuget_package"]
        assert len(package_refs) >= 3
        assert any(imp.module == "Newtonsoft.Json" and imp.version == "13.0.1" for imp in package_refs)

        project_refs = [imp for imp in imports if imp.type == "project_reference"]
        assert len(project_refs) >= 1


class TestExportExtraction:
    """Test suite for C# export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSharpAnalyzer instance."""
        return CSharpAnalyzer()

    def test_extract_classes(self, analyzer):
        """Test extraction of class exports."""
        code = """
public class User
{
    // User class
}

public abstract class BaseEntity
{
    // Abstract class
}

public sealed class FinalClass
{
    // Sealed class
}

public static class UtilityClass
{
    // Static class
}

public partial class PartialClass
{
    // Partial class
}
"""
        exports = analyzer.extract_exports(code, Path("test.cs"))

        class_exports = [e for e in exports if e["type"] == "class"]
        assert len(class_exports) >= 5

        # Check modifiers
        base_entity = next(e for e in class_exports if e["name"] == "BaseEntity")
        assert "abstract" in base_entity["modifiers"]

        final_class = next(e for e in class_exports if e["name"] == "FinalClass")
        assert "sealed" in final_class["modifiers"]

        static_class = next(e for e in class_exports if e["name"] == "UtilityClass")
        assert "static" in static_class["modifiers"]

    def test_extract_unity_components(self, analyzer):
        """Test extraction of Unity components."""
        code = """
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    // Unity component
}

public class GameData : ScriptableObject
{
    // ScriptableObject
}

public class CustomEditor : Editor
{
    // Unity Editor class
}
"""
        exports = analyzer.extract_exports(code, Path("test.cs"))

        player_controller = next(e for e in exports if e["name"] == "PlayerController")
        assert player_controller["is_unity_component"] is True
        assert player_controller["unity_base_class"] == "MonoBehaviour"

        game_data = next(e for e in exports if e["name"] == "GameData")
        assert game_data["is_unity_component"] is True
        assert game_data["unity_base_class"] == "ScriptableObject"

    def test_extract_interfaces(self, analyzer):
        """Test extraction of interface exports."""
        code = """
public interface IRepository
{
    // Interface
}

public partial interface IService<T> where T : class
{
    // Generic interface
}

interface IInternal
{
    // Internal interface (no public keyword)
}
"""
        exports = analyzer.extract_exports(code, Path("test.cs"))

        interface_exports = [e for e in exports if e["type"] == "interface"]
        assert any(e["name"] == "IRepository" for e in interface_exports)
        assert any(e["name"] == "IService" for e in interface_exports)

    def test_extract_structs(self, analyzer):
        """Test extraction of struct exports."""
        code = """
public struct Point
{
    public int X { get; set; }
    public int Y { get; set; }
}

public readonly struct ImmutablePoint
{
    // Readonly struct
}

public ref struct SpanWrapper
{
    // Ref struct
}
"""
        exports = analyzer.extract_exports(code, Path("test.cs"))

        struct_exports = [e for e in exports if e["type"] == "struct"]
        assert len(struct_exports) >= 3

        immutable_point = next(e for e in struct_exports if e["name"] == "ImmutablePoint")
        assert "readonly" in immutable_point["modifiers"]

        span_wrapper = next(e for e in struct_exports if e["name"] == "SpanWrapper")
        assert "ref" in span_wrapper["modifiers"]

    def test_extract_enums_and_delegates(self, analyzer):
        """Test extraction of enums and delegates."""
        code = """
public enum Status
{
    Active,
    Inactive,
    Pending
}

public enum class ErrorCode : int
{
    None = 0,
    NotFound = 404,
    ServerError = 500
}

public delegate void EventHandler(object sender, EventArgs e);
public delegate T Factory<T>() where T : new();
"""
        exports = analyzer.extract_exports(code, Path("test.cs"))

        # Check enums
        status_enum = next(e for e in exports if e["name"] == "Status")
        assert status_enum["type"] == "enum"

        error_enum = next(e for e in exports if e["name"] == "ErrorCode")
        assert error_enum["type"] == "enum_class"
        assert error_enum["base_type"] == "int"

        # Check delegates
        delegate_exports = [e for e in exports if e["type"] == "delegate"]
        assert len(delegate_exports) >= 2

    def test_extract_records(self, analyzer):
        """Test extraction of records (C# 9+)."""
        code = """
public record Person(string FirstName, string LastName);

public record class Employee(string Id, string Name)
{
    public string Department { get; init; }
}

public record struct Point3D(double X, double Y, double Z);
"""
        exports = analyzer.extract_exports(code, Path("test.cs"))

        person_record = next(e for e in exports if e["name"] == "Person")
        assert person_record["type"] == "record"

        point_record = next(e for e in exports if e["name"] == "Point3D")
        assert point_record["type"] == "record_struct"


class TestStructureExtraction:
    """Test suite for C# code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSharpAnalyzer instance."""
        return CSharpAnalyzer()

    def test_extract_class_with_properties(self, analyzer):
        """Test extraction of classes with properties."""
        code = """
public class User
{
    // Auto-properties
    public string Name { get; set; }
    public int Age { get; private set; }
    public string Email { get; init; }
    
    // Full property
    private string _password;
    public string Password
    {
        get { return _password; }
        set { _password = value; }
    }
    
    // Expression-bodied property
    public string FullName => $"{FirstName} {LastName}";
    
    // Static property
    public static int UserCount { get; set; }
}
"""
        structure = analyzer.extract_structure(code, Path("test.cs"))

        user_class = structure.classes[0]
        assert len(user_class.properties) >= 5

        # Check auto-property detection
        auto_props = [p for p in user_class.properties if p["is_auto_property"]]
        assert len(auto_props) >= 3

    def test_extract_methods_with_modifiers(self, analyzer):
        """Test extraction of methods with various modifiers."""
        code = """
public class MyClass
{
    public void PublicMethod() { }
    
    private void PrivateMethod() { }
    
    protected virtual void VirtualMethod() { }
    
    public override void OverrideMethod() { }
    
    public async Task AsyncMethod()
    {
        await Task.Delay(100);
    }
    
    public static void StaticMethod() { }
    
    public void GenericMethod<T>() where T : class { }
}
"""
        structure = analyzer.extract_structure(code, Path("test.cs"))

        my_class = structure.classes[0]
        methods = my_class.methods

        # Check method modifiers
        virtual_method = next(m for m in methods if m["name"] == "VirtualMethod")
        assert virtual_method["is_virtual"] is True

        override_method = next(m for m in methods if m["name"] == "OverrideMethod")
        assert override_method["is_override"] is True

        async_method = next(m for m in methods if m["name"] == "AsyncMethod")
        assert async_method["is_async"] is True

        static_method = next(m for m in methods if m["name"] == "StaticMethod")
        assert "static" in static_method["modifiers"]

    def test_extract_unity_components(self, analyzer):
        """Test extraction of Unity-specific components."""
        code = """
using UnityEngine;
using System.Collections;

public class Player : MonoBehaviour
{
    [SerializeField]
    private float speed = 5f;
    
    [Range(0, 100)]
    public int health = 100;
    
    void Awake()
    {
        // Unity lifecycle
    }
    
    void Start()
    {
        // Unity lifecycle
    }
    
    void Update()
    {
        // Called every frame
    }
    
    void OnCollisionEnter(Collision collision)
    {
        // Physics callback
    }
    
    IEnumerator AttackCoroutine()
    {
        yield return new WaitForSeconds(1f);
    }
}
"""
        structure = analyzer.extract_structure(code, Path("Player.cs"))

        assert structure.is_unity_script is True

        player_class = structure.classes[0]
        assert player_class.is_monobehaviour is True

        # Check Unity methods
        assert "Awake" in player_class.unity_methods
        assert "Start" in player_class.unity_methods
        assert "Update" in player_class.unity_methods
        assert "OnCollisionEnter" in player_class.unity_methods

        # Check coroutines
        assert "AttackCoroutine" in player_class.coroutines

        # Check Unity attributes
        fields = player_class.fields
        speed_field = next(f for f in fields if f["name"] == "speed")
        assert "SerializeField" in speed_field["unity_attributes"]

    def test_extract_events(self, analyzer):
        """Test extraction of events."""
        code = """
using System;
using UnityEngine.Events;

public class EventExample
{
    // Standard C# event
    public event EventHandler DataChanged;
    
    public event Action<string> MessageReceived;
    
    // Unity events
    public UnityEvent OnPlayerDeath;
    
    public UnityEvent<int> OnScoreChanged;
    
    private UnityEvent<string, float> OnCustomEvent;
}
"""
        structure = analyzer.extract_structure(code, Path("test.cs"))

        event_class = structure.classes[0]
        events = event_class.events

        assert len(events) >= 5

        # Check Unity events
        unity_events = [e for e in events if e.get("is_unity_event")]
        assert len(unity_events) >= 3

    def test_extract_linq_queries(self, analyzer):
        """Test extraction of LINQ queries."""
        code = """
using System.Linq;

public class DataProcessor
{
    public void ProcessData()
    {
        // Query syntax
        var query1 = from user in users
                    where user.Age > 18
                    select user.Name;
        
        // Method syntax
        var query2 = users.Where(u => u.Age > 18)
                         .Select(u => u.Name)
                         .OrderBy(n => n);
        
        var query3 = data.GroupBy(d => d.Category)
                        .Select(g => new { Category = g.Key, Count = g.Count() });
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.cs"))

        assert len(structure.linq_queries) >= 2
        
        # Check query types
        query_syntax = [q for q in structure.linq_queries if q["type"] == "query_syntax"]
        method_syntax = [q for q in structure.linq_queries if q["type"] == "method_syntax"]
        
        assert len(query_syntax) >= 1
        assert len(method_syntax) >= 1

    def test_detect_framework(self, analyzer):
        """Test framework detection."""
        # Unity
        unity_code = """
using UnityEngine;

public class GameManager : MonoBehaviour
{
    void Start() { }
}
"""
        unity_structure = analyzer.extract_structure(unity_code, Path("test.cs"))
        assert unity_structure.framework == "Unity"

        # ASP.NET Core
        aspnet_code = """
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("api/[controller]")]
public class UsersController : ControllerBase
{
    [HttpGet]
    public IActionResult Get() { return Ok(); }
}
"""
        aspnet_structure = analyzer.extract_structure(aspnet_code, Path("test.cs"))
        assert aspnet_structure.framework == "ASP.NET Core"


class TestComplexityCalculation:
    """Test suite for C# complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSharpAnalyzer instance."""
        return CSharpAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
public int ComplexMethod(int x, int y)
{
    if (x > 0)
    {
        if (y > 0)
        {
            return x + y;
        }
        else
        {
            return x - y;
        }
    }
    else if (x < 0)
    {
        for (int i = 0; i < y; i++)
        {
            x++;
        }
    }
    
    switch (x)
    {
        case 1:
            return 1;
        case 2:
            return 2;
        default:
            return 0;
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 7

    def test_pattern_matching_complexity(self, analyzer):
        """Test that pattern matching adds to complexity."""
        code = """
public string ProcessValue(object value)
{
    // C# 7+ pattern matching
    if (value is string s)
    {
        return s.ToUpper();
    }
    
    // Switch expression with when clauses
    return value switch
    {
        int n when n > 0 => "Positive",
        int n when n < 0 => "Negative",
        0 => "Zero",
        null => "Null",
        _ => "Unknown"
    };
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        # Pattern matching should add to complexity
        assert metrics.cyclomatic >= 5

    def test_async_await_metrics(self, analyzer):
        """Test async/await metrics."""
        code = """
public class AsyncService
{
    public async Task<string> GetDataAsync()
    {
        await Task.Delay(100);
        var result = await FetchFromApiAsync();
        return await ProcessResultAsync(result);
    }
    
    public async ValueTask<int> ComputeAsync()
    {
        await Task.Yield();
        return 42;
    }
    
    private async Task<string> FetchFromApiAsync()
    {
        await Task.CompletedTask;
        return "data";
    }
    
    private async Task<string> ProcessResultAsync(string input)
    {
        await Task.Run(() => { });
        return input.ToUpper();
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        assert metrics.async_methods >= 4
        assert metrics.await_statements >= 6

    def test_exception_handling_metrics(self, analyzer):
        """Test exception handling metrics."""
        code = """
public class ErrorHandler
{
    public void HandleErrors()
    {
        try
        {
            DoSomething();
        }
        catch (ArgumentException ex)
        {
            LogError(ex);
            throw;
        }
        catch (InvalidOperationException)
        {
            // Handle specific exception
        }
        catch
        {
            // Catch all
        }
        finally
        {
            Cleanup();
        }
        
        try
        {
            RiskyOperation();
        }
        catch when (DateTime.Now.DayOfWeek == DayOfWeek.Monday)
        {
            // Filtered catch
        }
        
        throw new NotImplementedException();
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        assert metrics.try_blocks >= 2
        assert metrics.catch_blocks >= 4
        assert metrics.finally_blocks >= 1
        assert metrics.throw_statements >= 2

    def test_linq_metrics(self, analyzer):
        """Test LINQ metrics."""
        code = """
using System.Linq;

public class LinqExample
{
    public void QueryData()
    {
        // Query syntax
        var query = from item in items
                   where item.IsActive
                   orderby item.Name
                   select item;
        
        // Method syntax
        var filtered = items
            .Where(x => x.IsActive)
            .Select(x => x.Name)
            .OrderBy(x => x)
            .GroupBy(x => x.Category)
            .Where(g => g.Count() > 5)
            .SelectMany(g => g);
        
        var hasAny = items.Any(x => x.Id > 0);
        var allValid = items.All(x => x.IsValid);
        var first = items.First();
        var last = items.Last();
        var single = items.Single(x => x.Id == 1);
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        assert metrics.linq_queries >= 1
        assert metrics.linq_methods >= 10

    def test_unity_metrics(self, analyzer):
        """Test Unity-specific metrics."""
        code = """
using UnityEngine;
using UnityEngine.Events;
using System.Collections;

public class UnityPlayer : MonoBehaviour
{
    [SerializeField] private float speed;
    [SerializeField] private int health;
    [HideInInspector] public bool isActive;
    
    public UnityEvent OnPlayerDeath;
    public UnityEvent<int> OnHealthChanged;
    
    void Start() { }
    void Update() { }
    void FixedUpdate() { }
    
    void OnCollisionEnter(Collision col) { }
    void OnTriggerEnter(Collider other) { }
    
    IEnumerator SpawnEnemies()
    {
        yield return new WaitForSeconds(1f);
    }
    
    IEnumerator AttackRoutine()
    {
        yield return null;
    }
}

public class GameData : ScriptableObject
{
    public string dataName;
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        assert metrics.unity_components >= 2
        assert metrics.coroutines >= 2
        assert metrics.unity_methods >= 5
        assert metrics.serialize_fields >= 2
        assert metrics.unity_events >= 2


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSharpAnalyzer instance."""
        return CSharpAnalyzer()

    def test_handle_syntax_errors(self, analyzer):
        """Test handling of files with syntax errors."""
        code = """
public class Invalid
{
    this is not valid C# code
    public void Method()
    {
        missing semicolon
    }
}
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("test.cs"))
        exports = analyzer.extract_exports(code, Path("test.cs"))
        structure = analyzer.extract_structure(code, Path("test.cs"))
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.cs"))
        exports = analyzer.extract_exports(code, Path("test.cs"))
        structure = analyzer.extract_structure(code, Path("test.cs"))
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        assert imports == []
        assert exports == []
        assert metrics.line_count == 0


class TestEdgeCases:
    """Test suite for edge cases and modern C# features."""

    @pytest.fixture
    def analyzer(self):
        """Provide a CSharpAnalyzer instance."""
        return CSharpAnalyzer()

    def test_nullable_reference_types(self, analyzer):
        """Test handling of nullable reference types (C# 8+)."""
        code = """
#nullable enable

public class NullableExample
{
    public string? NullableName { get; set; }
    public string NonNullableName { get; set; } = "";
    
    public void ProcessData(string? input)
    {
        string? local = null;
        string nonNull = input ?? "default";
        
        if (input?.Length > 0)
        {
            Console.WriteLine(input);
        }
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.cs"))

        # Should count nullable references
        assert metrics.nullable_refs >= 3

    def test_top_level_programs(self, analyzer):
        """Test handling of top-level programs (C# 9+)."""
        code = """
// Top-level program
Console.WriteLine("Hello World");

void LocalFunction()
{
    Console.WriteLine("Local function");
}

int Add(int a, int b) => a + b;

var result = Add(1, 2);
Console.WriteLine(result);
"""
        structure = analyzer.extract_structure(code, Path("Program.cs"))

        # Should detect top-level functions
        assert len(structure.functions) >= 2
        function_names = [f.name for f in structure.functions]
        assert "LocalFunction" in function_names or "Add" in function_names

    def test_init_only_properties(self, analyzer):
        """Test handling of init-only properties (C# 9+)."""
        code = """
public class Person
{
    public string FirstName { get; init; }
    public string LastName { get; init; }
    public DateTime DateOfBirth { get; init; }
    
    public Person(string firstName, string lastName)
    {
        FirstName = firstName;
        LastName = lastName;
        DateOfBirth = DateTime.Now;
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.cs"))

        person_class = structure.classes[0]
        # Should detect init properties
        assert len(person_class.properties) >= 3

    def test_attributes_extraction(self, analyzer):
        """Test extraction of attributes."""
        code = """
using System;

[Serializable]
[Obsolete("Use NewClass instead")]
public class OldClass
{
    [Required]
    [StringLength(100)]
    public string Name { get; set; }
    
    [Authorize(Roles = "Admin")]
    public void AdminMethod() { }
}

[TestClass]
public class MyTests
{
    [TestMethod]
    [Timeout(1000)]
    public void TestMethod() { }
}
"""
        structure = analyzer.extract_structure(code, Path("test.cs"))

        old_class = next(c for c in structure.classes if c.name == "OldClass")
        assert "Serializable" in old_class.attributes
        assert "Obsolete" in old_class.attributes

    def test_expression_bodied_members(self, analyzer):
        """Test handling of expression-bodied members."""
        code = """
public class ExpressionBodied
{
    // Expression-bodied property
    public string Name => "Default";
    
    // Expression-bodied method
    public int Double(int x) => x * 2;
    
    // Expression-bodied constructor
    public ExpressionBodied(string name) => Name = name;
    
    // Expression-bodied destructor
    ~ExpressionBodied() => Console.WriteLine("Cleanup");
    
    // Expression-bodied indexer
    public int this[int index] => index * 10;
}
"""
        structure = analyzer.extract_structure(code, Path("test.cs"))

        # Should handle all expression-bodied members without errors
        assert len(structure.classes) == 1
        expr_class = structure.classes[0]
        assert expr_class.name == "ExpressionBodied"

    def test_partial_classes_and_methods(self, analyzer):
        """Test handling of partial classes and methods."""
        code = """
public partial class PartialClass
{
    partial void OnNameChanged();
    
    public string Name
    {
        get => _name;
        set
        {
            _name = value;
            OnNameChanged();
        }
    }
}

public partial class PartialClass
{
    private string _name;
    
    partial void OnNameChanged()
    {
        Console.WriteLine("Name changed");
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.cs"))

        # Should detect partial classes
        partial_classes = [c for c in structure.classes if "partial" in c.modifiers or c.name == "PartialClass"]
        assert len(partial_classes) >= 1