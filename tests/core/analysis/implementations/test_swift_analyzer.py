"""
Unit tests for the Swift code analyzer with iOS/macOS and SwiftUI support.

This module tests the Swift-specific code analysis functionality including
import statements, structs, enums, protocols, actors, optionals,
async/await, SwiftUI views, UIKit components, and modern Swift features.

Test Coverage:
    - Import extraction (frameworks, conditional imports)
    - Export detection (classes, structs, enums, protocols, actors)
    - Structure extraction (SwiftUI views, UIKit controllers, extensions)
    - Complexity metrics (optionals, async/await, property wrappers)
    - Optional handling (force unwraps, chaining, nil coalescing)
    - SwiftUI and UIKit patterns
    - Protocol-oriented programming
    - Error handling for invalid Swift code
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tenets.core.analysis.implementations.swift_analyzer import SwiftAnalyzer
from tenets.models.analysis import (
    ClassInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ImportInfo,
)


class TestSwiftAnalyzerInitialization:
    """Test suite for SwiftAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = SwiftAnalyzer()

        assert analyzer.language_name == "swift"
        assert ".swift" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for Swift import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a SwiftAnalyzer instance."""
        return SwiftAnalyzer()

    def test_extract_basic_imports(self, analyzer):
        """Test extraction of basic import statements."""
        code = """
import Foundation
import UIKit
import SwiftUI
import Combine
import CoreData
"""
        imports = analyzer.extract_imports(code, Path("test.swift"))

        assert len(imports) == 5

        # Check Foundation import
        foundation_import = next(imp for imp in imports if imp.module == "Foundation")
        assert foundation_import.category == "foundation"
        assert foundation_import.is_apple_framework is True

        # Check UIKit import
        uikit_import = next(imp for imp in imports if imp.module == "UIKit")
        assert uikit_import.category == "ui_framework"

        # Check SwiftUI import
        swiftui_import = next(imp for imp in imports if imp.module == "SwiftUI")
        assert swiftui_import.category == "swiftui"

        # Check Combine import
        combine_import = next(imp for imp in imports if imp.module == "Combine")
        assert combine_import.category == "combine"

    def test_extract_targeted_imports(self, analyzer):
        """Test extraction of targeted imports."""
        code = """
import struct Swift.Array
import class UIKit.UIViewController
import enum Foundation.ComparisonResult
import protocol SwiftUI.View
import func Darwin.sqrt
"""
        imports = analyzer.extract_imports(code, Path("test.swift"))

        assert len(imports) == 5

        array_import = next(imp for imp in imports if "Array" in imp.module)
        assert array_import.import_kind == "struct"

        viewcontroller_import = next(imp for imp in imports if "UIViewController" in imp.module)
        assert viewcontroller_import.import_kind == "class"

    def test_extract_conditional_imports(self, analyzer):
        """Test extraction of conditional imports."""
        code = """
@testable import MyModule
@_exported import CoreFoundation
import XCTest
"""
        imports = analyzer.extract_imports(code, Path("test.swift"))

        assert len(imports) == 3

        testable_import = next(imp for imp in imports if imp.module == "MyModule")
        assert testable_import.is_testable is True
        assert testable_import.type == "testable_import"

        exported_import = next(imp for imp in imports if imp.module == "CoreFoundation")
        assert exported_import.is_exported is True
        assert exported_import.type == "exported_import"


class TestExportExtraction:
    """Test suite for Swift export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a SwiftAnalyzer instance."""
        return SwiftAnalyzer()

    def test_extract_classes(self, analyzer):
        """Test extraction of class exports."""
        code = """
public class PublicClass {}
open class OpenClass {}
internal class InternalClass {}
private class PrivateClass {}
final class FinalClass {}
public final class PublicFinalClass {}
"""
        exports = analyzer.extract_exports(code, Path("test.swift"))

        public_exports = [e for e in exports if e["access_level"] in ["public", "open"]]
        assert len(public_exports) == 3  # PublicClass, OpenClass, PublicFinalClass

        open_class = next(e for e in exports if e["name"] == "OpenClass")
        assert open_class["is_open"] is True

        final_class = next(e for e in exports if e["name"] == "PublicFinalClass")
        assert final_class["is_final"] is True

    def test_extract_structs(self, analyzer):
        """Test extraction of struct exports."""
        code = """
public struct PublicStruct {}
internal struct InternalStruct {}
private struct PrivateStruct {}

public struct Point {
    public var x: Double
    public var y: Double
}
"""
        exports = analyzer.extract_exports(code, Path("test.swift"))

        struct_exports = [e for e in exports if e["type"] == "struct"]
        public_structs = [s for s in struct_exports if s["access_level"] == "public"]
        assert len(public_structs) == 2  # PublicStruct and Point

    def test_extract_enums(self, analyzer):
        """Test extraction of enum exports."""
        code = """
public enum PublicEnum {
    case one, two, three
}

internal enum InternalEnum {}
private enum PrivateEnum {}

public indirect enum BinaryTree {
    case empty
    indirect case node(value: Int, left: BinaryTree, right: BinaryTree)
}

public enum Result<T> {
    case success(T)
    case failure(Error)
}
"""
        exports = analyzer.extract_exports(code, Path("test.swift"))

        enum_exports = [e for e in exports if e["type"] == "enum"]
        public_enums = [e for e in enum_exports if e["access_level"] == "public"]
        assert len(public_enums) == 3

        indirect_enum = next(e for e in public_enums if e["name"] == "BinaryTree")
        assert indirect_enum["is_indirect"] is True

    def test_extract_protocols(self, analyzer):
        """Test extraction of protocol exports."""
        code = """
public protocol PublicProtocol {
    func method()
}

internal protocol InternalProtocol {}
private protocol PrivateProtocol {}

public protocol Drawable {
    func draw()
}
"""
        exports = analyzer.extract_exports(code, Path("test.swift"))

        protocol_exports = [e for e in exports if e["type"] == "protocol"]
        public_protocols = [p for p in protocol_exports if p["access_level"] == "public"]
        assert len(public_protocols) == 2  # PublicProtocol and Drawable

    def test_extract_actors(self, analyzer):
        """Test extraction of actor exports."""
        code = """
public actor PublicActor {
    private var counter = 0
    
    public func increment() {
        counter += 1
    }
}

internal actor InternalActor {}
private actor PrivateActor {}
"""
        exports = analyzer.extract_exports(code, Path("test.swift"))

        actor_exports = [e for e in exports if e["type"] == "actor"]
        public_actors = [a for a in actor_exports if a["access_level"] == "public"]
        assert len(public_actors) == 1

    def test_extract_functions(self, analyzer):
        """Test extraction of function exports."""
        code = """
public func publicFunction() {}
internal func internalFunction() {}
private func privateFunction() {}

public async func asyncFunction() async throws -> String {
    return "result"
}

public func throwingFunction() throws {}
"""
        exports = analyzer.extract_exports(code, Path("test.swift"))

        func_exports = [e for e in exports if e["type"] == "function"]
        public_funcs = [f for f in func_exports if f["access_level"] == "public"]
        assert len(public_funcs) == 3

        async_func = next(f for f in public_funcs if f["name"] == "asyncFunction")
        assert async_func["is_async"] is True
        assert async_func["is_throwing"] is True

    def test_extract_properties(self, analyzer):
        """Test extraction of property exports."""
        code = """
public let publicConstant = 42
public var publicVariable = "test"
internal let internalConstant = true
private var privateVariable = 3.14

public class MyClass {
    public static let shared = MyClass()
    public lazy var lazyProperty = computeValue()
    public weak var delegate: MyDelegate?
}
"""
        exports = analyzer.extract_exports(code, Path("test.swift"))

        property_exports = [e for e in exports if e["type"] == "property"]
        public_props = [p for p in property_exports if p["access_level"] == "public"]
        assert len(public_props) >= 2

        constant = next((p for p in public_props if p["name"] == "publicConstant"), None)
        if constant:
            assert constant["is_constant"] is True

        variable = next((p for p in public_props if p["name"] == "publicVariable"), None)
        if variable:
            assert variable["is_variable"] is True

    def test_extract_extensions(self, analyzer):
        """Test extraction of extension exports."""
        code = """
public extension String {
    func customMethod() -> Bool {
        return true
    }
}

internal extension Array {}
private extension Dictionary {}
"""
        exports = analyzer.extract_exports(code, Path("test.swift"))

        extension_exports = [e for e in exports if e["type"] == "extension"]
        public_extensions = [e for e in extension_exports if e["access_level"] == "public"]
        assert len(public_extensions) == 1
        assert public_extensions[0]["extended_type"] == "String"


class TestStructureExtraction:
    """Test suite for Swift code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a SwiftAnalyzer instance."""
        return SwiftAnalyzer()

    def test_extract_swiftui_view(self, analyzer):
        """Test extraction of SwiftUI View structures."""
        code = """
import SwiftUI

struct ContentView: View {
    @State private var counter = 0
    @Binding var username: String
    @StateObject private var viewModel = ViewModel()
    @ObservedObject var settings: Settings
    @EnvironmentObject var appState: AppState
    @Environment(\\.colorScheme) var colorScheme
    
    var body: some View {
        VStack {
            Text("Counter: \\(counter)")
                .font(.title)
                .foregroundColor(.blue)
            
            Button("Increment") {
                counter += 1
            }
            .padding()
            .background(Color.gray)
            .cornerRadius(8)
        }
        .animation(.easeInOut, value: counter)
    }
}

struct DetailView: View {
    var body: some View {
        Text("Detail")
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        assert structure.is_swiftui is True
        assert len(structure.structs) >= 2

        content_view = next(s for s in structure.structs if s["name"] == "ContentView")
        assert content_view["is_swiftui_view"] is True
        assert "View" in content_view["protocols"]

        assert structure.property_wrappers > 0
        assert structure.view_modifiers > 0
        assert structure.body_count >= 2

    def test_extract_uikit_viewcontroller(self, analyzer):
        """Test extraction of UIKit ViewController."""
        code = """
import UIKit

class MainViewController: UIViewController {
    @IBOutlet weak var titleLabel: UILabel!
    @IBOutlet weak var submitButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
    }
    
    @IBAction func buttonTapped(_ sender: UIButton) {
        print("Button tapped")
    }
    
    private func setupUI() {
        titleLabel.text = "Welcome"
    }
}

class CustomView: UIView {
    override func draw(_ rect: CGRect) {
        // Custom drawing
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        assert structure.is_ios is True
        assert structure.is_uikit is True

        main_vc = next(c for c in structure.classes if c.name == "MainViewController")
        assert main_vc.superclass == "UIViewController"
        assert main_vc.ui_type == "view_controller"

        custom_view = next(c for c in structure.classes if c.name == "CustomView")
        assert custom_view.ui_type == "uiview"

    def test_extract_protocol_conformance(self, analyzer):
        """Test extraction of protocol conformance."""
        code = """
protocol Drawable {
    func draw()
}

protocol Resizable {
    func resize(to size: CGSize)
}

class Shape: Drawable, Resizable {
    func draw() {
        // Drawing implementation
    }
    
    func resize(to size: CGSize) {
        // Resize implementation
    }
}

struct Point: Equatable, Hashable, Codable {
    let x: Int
    let y: Int
}

extension String: Drawable {
    func draw() {
        print(self)
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        shape_class = next(c for c in structure.classes if c.name == "Shape")
        assert "Drawable" in shape_class.protocols
        assert "Resizable" in shape_class.protocols

        point_struct = next(s for s in structure.structs if s["name"] == "Point")
        assert "Equatable" in point_struct["protocols"]
        assert "Hashable" in point_struct["protocols"]
        assert "Codable" in point_struct["protocols"]

        string_extension = next(e for e in structure.extensions if e["extended_type"] == "String")
        assert "Drawable" in string_extension["protocols"]

    def test_extract_async_await_patterns(self, analyzer):
        """Test extraction of async/await patterns."""
        code = """
import Foundation

actor DataManager {
    private var cache: [String: Data] = [:]
    
    func fetchData(for key: String) async throws -> Data {
        if let cached = cache[key] {
            return cached
        }
        
        let data = try await downloadData(key: key)
        cache[key] = data
        return data
    }
    
    private func downloadData(key: String) async throws -> Data {
        // Simulated async download
        try await Task.sleep(nanoseconds: 1_000_000_000)
        return Data()
    }
}

@MainActor
class ViewModel: ObservableObject {
    @Published var isLoading = false
    
    func loadData() async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            async let data1 = fetchData(id: 1)
            async let data2 = fetchData(id: 2)
            
            let results = try await [data1, data2]
            process(results)
        } catch {
            print("Error: \\(error)")
        }
    }
    
    private func fetchData(id: Int) async throws -> Data {
        try await Task.sleep(nanoseconds: 1_000_000_000)
        return Data()
    }
    
    private func process(_ data: [Data]) {
        // Process data
    }
}

func performTasks() async {
    await withTaskGroup(of: Int.self) { group in
        for i in 1...10 {
            group.addTask {
                await computeValue(i)
            }
        }
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        assert len(structure.actors) >= 1
        assert structure.async_functions >= 4
        assert structure.await_count >= 5
        assert structure.task_count >= 1

        data_manager = next(a for a in structure.actors if a["name"] == "DataManager")
        assert data_manager is not None

    def test_extract_enum_with_associated_values(self, analyzer):
        """Test extraction of enums with associated values."""
        code = """
enum Result<Success, Failure: Error> {
    case success(Success)
    case failure(Failure)
}

enum NetworkError: Error {
    case badURL
    case timeout(seconds: Int)
    case httpError(code: Int, message: String)
}

enum Color: String, CaseIterable {
    case red = "FF0000"
    case green = "00FF00"
    case blue = "0000FF"
}

indirect enum ArithmeticExpression {
    case number(Int)
    case addition(ArithmeticExpression, ArithmeticExpression)
    case multiplication(ArithmeticExpression, ArithmeticExpression)
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        assert len(structure.enums) >= 4

        result_enum = next(e for e in structure.enums if e["name"] == "Result")
        assert result_enum["generics"] is not None

        network_error = next(e for e in structure.enums if e["name"] == "NetworkError")
        assert "Error" in network_error["protocols"]

        color_enum = next(e for e in structure.enums if e["name"] == "Color")
        assert color_enum["raw_type"] == "String"
        assert "CaseIterable" in color_enum["protocols"]

        arithmetic_enum = next(e for e in structure.enums if e["name"] == "ArithmeticExpression")
        assert arithmetic_enum["is_indirect"] is True

    def test_extract_generics_and_constraints(self, analyzer):
        """Test extraction of generic types with constraints."""
        code = """
class Container<T> {
    private var items: [T] = []
    
    func add(_ item: T) {
        items.append(item)
    }
}

struct Stack<Element: Equatable> {
    private var storage: [Element] = []
    
    mutating func push(_ element: Element) {
        storage.append(element)
    }
    
    mutating func pop() -> Element? {
        storage.popLast()
    }
}

func swap<T>(_ a: inout T, _ b: inout T) {
    let temp = a
    a = b
    b = temp
}

func findIndex<T: Equatable>(of value: T, in array: [T]) -> Int? {
    for (index, item) in array.enumerated() {
        if item == value {
            return index
        }
    }
    return nil
}

extension Array where Element: Comparable {
    func isSorted() -> Bool {
        for i in 1..<count {
            if self[i-1] > self[i] {
                return false
            }
        }
        return true
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        container_class = next(c for c in structure.classes if c.name == "Container")
        assert container_class.generics == "T"

        stack_struct = next(s for s in structure.structs if s["name"] == "Stack")
        assert "Element: Equatable" in str(stack_struct["generics"])

        array_extension = next(e for e in structure.extensions if e["extended_type"] == "Array")
        assert array_extension["where_clause"] is not None

    def test_extract_property_wrappers(self, analyzer):
        """Test extraction of property wrappers."""
        code = """
import SwiftUI
import Combine

class ViewModel: ObservableObject {
    @Published var username = ""
    @Published var isLoggedIn = false
    
    init() {}
}

struct SettingsView: View {
    @AppStorage("theme") var theme = "light"
    @SceneStorage("selectedTab") var selectedTab = 0
    @FocusState private var isInputFocused: Bool
    @GestureState private var dragOffset = CGSize.zero
    
    var body: some View {
        Form {
            TextField("Username", text: .constant(""))
                .focused($isInputFocused)
        }
    }
}

@propertyWrapper
struct Clamped<Value: Comparable> {
    private var value: Value
    let range: ClosedRange<Value>
    
    init(wrappedValue: Value, _ range: ClosedRange<Value>) {
        self.range = range
        self.value = min(max(wrappedValue, range.lowerBound), range.upperBound)
    }
    
    var wrappedValue: Value {
        get { value }
        set { value = min(max(newValue, range.lowerBound), range.upperBound) }
    }
}

struct Game {
    @Clamped(0...100) var health = 100
    @Clamped(0...10) var level = 1
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        assert structure.property_wrappers > 0
        assert structure.combine_publishers > 0


class TestComplexityCalculation:
    """Test suite for Swift complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a SwiftAnalyzer instance."""
        return SwiftAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
func complexFunction(_ value: Any) -> String {
    switch value {
    case let int as Int:
        if int > 0 {
            return "positive"
        } else if int < 0 {
            return "negative"
        } else {
            return "zero"
        }
    case let string as String:
        guard !string.isEmpty else {
            return "empty"
        }
        return string
    case let array as [Any]:
        for item in array {
            if item is Int {
                continue
            }
        }
        return "array"
    default:
        return "unknown"
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 10

    def test_optional_handling_metrics(self, analyzer):
        """Test optional handling complexity metrics."""
        code = """
class OptionalExample {
    var optional: String? = nil
    var implicitlyUnwrapped: String! = "test"
    
    func processOptional(_ input: String?) -> Int {
        // Force unwrap
        let forced = input!
        
        // Optional chaining
        let length = input?.count
        
        // Nil coalescing
        let value = input ?? "default"
        let chained = optional ?? input ?? "fallback"
        
        // Guard let
        guard let unwrapped = input else {
            return 0
        }
        
        // If let
        if let value = optional {
            print(value)
        }
        
        // Multiple optional binding
        if let first = optional,
           let second = input,
           first == second {
            return 1
        }
        
        // Guard with multiple conditions
        guard let result = optional,
              !result.isEmpty,
              result.count > 5 else {
            return -1
        }
        
        // Optional chaining with method calls
        let processed = input?.trimmingCharacters(in: .whitespaces)
                             ?.lowercased()
                             ?.replacingOccurrences(of: " ", with: "_")
        
        return unwrapped.count
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        assert metrics.optional_types >= 3
        assert metrics.force_unwraps >= 1
        assert metrics.optional_chaining >= 4
        assert metrics.nil_coalescing >= 3
        assert metrics.guard_statements >= 2
        assert metrics.if_let_bindings >= 2
        assert metrics.guard_let_bindings >= 2

    def test_async_await_metrics(self, analyzer):
        """Test async/await complexity metrics."""
        code = """
import Foundation

class AsyncService {
    func fetchData() async throws -> String {
        try await Task.sleep(nanoseconds: 1_000_000_000)
        return "data"
    }
    
    func processMultiple() async throws {
        // Sequential awaits
        let result1 = try await fetchData()
        let result2 = try await fetchData()
        
        // Concurrent awaits
        async let concurrent1 = fetchData()
        async let concurrent2 = fetchData()
        let results = try await [concurrent1, concurrent2]
        
        // Task group
        await withTaskGroup(of: String.self) { group in
            for i in 1...5 {
                group.addTask {
                    try! await self.fetchData()
                }
            }
        }
        
        // Detached task
        Task.detached {
            await self.backgroundWork()
        }
    }
    
    @MainActor
    func updateUI() async {
        // UI update on main actor
    }
    
    private func backgroundWork() async {
        // Background work
    }
}

actor Counter {
    private var value = 0
    
    func increment() {
        value += 1
    }
    
    func getValue() -> Int {
        value
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        assert metrics.async_functions >= 4
        assert metrics.await_calls >= 6
        assert metrics.task_groups >= 2
        assert metrics.main_actor >= 1
        assert metrics.actor_count >= 1

    def test_swiftui_metrics(self, analyzer):
        """Test SwiftUI-specific complexity metrics."""
        code = """
import SwiftUI

struct ContentView: View {
    @State private var counter = 0
    @StateObject private var viewModel = ViewModel()
    @ObservedObject var settings: Settings
    @Binding var isPresented: Bool
    @Environment(\\.colorScheme) var colorScheme
    @EnvironmentObject var appState: AppState
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Counter: \\(counter)")
                    .font(.title)
                    .foregroundColor(.blue)
                    .padding()
                    .background(Color.gray.opacity(0.2))
                    .cornerRadius(10)
                    .shadow(radius: 5)
                
                HStack {
                    Button("Increment") {
                        withAnimation(.spring()) {
                            counter += 1
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    
                    Button("Reset") {
                        counter = 0
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.horizontal)
                
                List(viewModel.items) { item in
                    Text(item.title)
                        .font(.body)
                }
                
                ForEach(0..<5) { index in
                    Text("Item \\(index)")
                }
                
                GeometryReader { geometry in
                    Text("Width: \\(geometry.size.width)")
                        .frame(width: geometry.size.width)
                }
            }
            .navigationTitle("Home")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Settings") { }
                }
            }
        }
        .animation(.easeInOut, value: counter)
        .transition(.slide)
    }
}

@ViewBuilder
func conditionalView(_ condition: Bool) -> some View {
    if condition {
        Text("True")
    } else {
        Text("False")
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        assert metrics.state_wrappers >= 1
        assert metrics.stateobject_wrappers >= 1
        assert metrics.observedobject_wrappers >= 1
        assert metrics.binding_wrappers >= 1
        assert metrics.environment_wrappers >= 2
        assert metrics.swiftui_views >= 1
        assert metrics.view_body_count >= 1
        assert metrics.view_modifiers >= 10
        assert metrics.foreach_usage >= 1
        assert metrics.geometryreader_usage >= 1

    def test_combine_metrics(self, analyzer):
        """Test Combine framework metrics."""
        code = """
import Combine
import Foundation

class NetworkService: ObservableObject {
    @Published var data: String = ""
    @Published var isLoading = false
    
    private let subject = PassthroughSubject<String, Never>()
    private let currentValue = CurrentValueSubject<Int, Never>(0)
    private var cancellables = Set<AnyCancellable>()
    
    func setupSubscriptions() {
        subject
            .debounce(for: .seconds(0.5), scheduler: RunLoop.main)
            .removeDuplicates()
            .sink { value in
                print(value)
            }
            .store(in: &cancellables)
        
        currentValue
            .map { $0 * 2 }
            .filter { $0 > 10 }
            .flatMap { value in
                Just(value)
                    .delay(for: .seconds(1), scheduler: RunLoop.main)
            }
            .sink { value in
                print(value)
            }
            .store(in: &cancellables)
        
        Publishers.CombineLatest($data, $isLoading)
            .sink { data, loading in
                print("Data: \\(data), Loading: \\(loading)")
            }
            .store(in: &cancellables)
    }
    
    func publisher() -> AnyPublisher<String, Error> {
        Future { promise in
            promise(.success("result"))
        }
        .eraseToAnyPublisher()
    }
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        assert metrics.combine_publishers >= 4
        assert metrics.published_wrappers >= 2
        assert metrics.combine_subscriptions >= 3
        assert metrics.combine_operators >= 6  # debounce, map, filter, flatMap, etc.

    def test_error_handling_metrics(self, analyzer):
        """Test error handling complexity metrics."""
        code = """
enum NetworkError: Error {
    case invalidURL
    case timeout
    case serverError(Int)
}

func riskyOperation() throws -> String {
    throw NetworkError.timeout
}

func handleErrors() {
    do {
        let result = try riskyOperation()
        print(result)
    } catch NetworkError.timeout {
        print("Timeout occurred")
    } catch NetworkError.serverError(let code) {
        print("Server error: \\(code)")
    } catch {
        print("Unknown error: \\(error)")
    }
    
    // Try? and try!
    let optional = try? riskyOperation()
    let forced = try! safeOperation()
    
    defer {
        cleanup()
    }
    
    defer {
        finalCleanup()
    }
}

func throwingFunction() throws {
    guard condition else {
        throw NetworkError.invalidURL
    }
}

func rethrowingFunction<T>(_ closure: () throws -> T) rethrows -> T {
    try closure()
}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        assert metrics.do_blocks >= 1
        assert metrics.try_statements >= 4  # try, try?, try!, and in rethrows
        assert metrics.catch_blocks >= 3
        assert metrics.throw_statements >= 2
        assert metrics.defer_statements >= 2


class TestErrorHandling:
    """Test suite for error handling in Swift analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Provide a SwiftAnalyzer instance."""
        return SwiftAnalyzer()

    def test_handle_malformed_code(self, analyzer):
        """Test handling of malformed Swift code."""
        code = """
class Broken {
    This is not valid Swift code!!!
    
    func method() {
        missing closing brace
    
    // Missing closing brace for class
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("test.swift"))
        exports = analyzer.extract_exports(code, Path("test.swift"))
        structure = analyzer.extract_structure(code, Path("test.swift"))
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty file."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.swift"))
        exports = analyzer.extract_exports(code, Path("test.swift"))
        structure = analyzer.extract_structure(code, Path("test.swift"))
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        assert imports == []
        assert exports == []
        assert len(structure.classes) == 0
        assert metrics.line_count == 1


class TestEdgeCases:
    """Test suite for edge cases in Swift analysis."""

    @pytest.fixture
    def analyzer(self):
        """Provide a SwiftAnalyzer instance."""
        return SwiftAnalyzer()

    def test_handle_string_interpolation(self, analyzer):
        """Test handling of string interpolation and multiline strings."""
        code = '''
func stringExamples() {
    let name = "World"
    let greeting = "Hello, \\(name)"
    let complex = "Result: \\(computeValue())"
    
    let multiline = """
        This is a multiline string
        with interpolation: \\(name)
        and expression: \\(2 + 2)
        It might contain code-like syntax:
        if true { "not real code" }
        """
    
    let rawString = #"This is a raw string \\n with "quotes""#
    let rawMultiline = #"""
        Raw multiline
        With \\(no) interpolation
        """#
}
'''
        structure = analyzer.extract_structure(code, Path("test.swift"))
        metrics = analyzer.calculate_complexity(code, Path("test.swift"))

        # Should correctly identify the function
        assert len(structure.functions) >= 1
        # Complexity should not count code inside strings
        assert metrics.cyclomatic < 5

    def test_handle_result_builders(self, analyzer):
        """Test handling of result builders."""
        code = """
import SwiftUI

@ViewBuilder
func makeView(showDetails: Bool) -> some View {
    if showDetails {
        DetailView()
    } else {
        SummaryView()
    }
}

@ViewBuilder
var conditionalContent: some View {
    Text("Header")
    
    if condition {
        Text("Condition met")
    }
    
    ForEach(items) { item in
        ItemView(item: item)
    }
    
    Text("Footer")
}

@resultBuilder
struct StringBuilder {
    static func buildBlock(_ components: String...) -> String {
        components.joined()
    }
}

@StringBuilder
func buildString() -> String {
    "Hello"
    " "
    "World"
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        assert structure.result_builders >= 2

    def test_handle_protocol_with_associated_types(self, analyzer):
        """Test handling of protocols with associated types."""
        code = """
protocol Container {
    associatedtype Item
    var count: Int { get }
    mutating func append(_ item: Item)
    subscript(i: Int) -> Item { get }
}

protocol IteratorProtocol {
    associatedtype Element
    mutating func next() -> Element?
}

struct Stack<Element>: Container {
    var items: [Element] = []
    
    var count: Int {
        items.count
    }
    
    mutating func append(_ item: Element) {
        items.append(item)
    }
    
    subscript(i: Int) -> Element {
        items[i]
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        container_protocol = next(p for p in structure.protocols if p["name"] == "Container")
        assert container_protocol is not None

        stack_struct = next(s for s in structure.structs if s["name"] == "Stack")
        assert "Container" in stack_struct["protocols"]

    def test_handle_property_observers(self, analyzer):
        """Test handling of property observers."""
        code = """
class StepCounter {
    var totalSteps: Int = 0 {
        willSet(newTotalSteps) {
            print("About to set totalSteps to \\(newTotalSteps)")
        }
        didSet {
            if totalSteps > oldValue {
                print("Added \\(totalSteps - oldValue) steps")
            }
        }
    }
}

struct Size {
    var width: Double
    var height: Double
    
    var area: Double {
        get {
            width * height
        }
        set {
            width = sqrt(newValue)
            height = sqrt(newValue)
        }
    }
    
    var perimeter: Double {
        2 * (width + height)
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        # Should handle property observers without errors
        assert len(structure.classes) >= 1
        assert len(structure.structs) >= 1

    def test_handle_custom_operators(self, analyzer):
        """Test handling of custom operators."""
        code = """
infix operator **: MultiplicationPrecedence

func ** (base: Int, power: Int) -> Int {
    Int(pow(Double(base), Double(power)))
}

prefix operator +++

prefix func +++ (value: inout Int) -> Int {
    value += 2
    return value
}

postfix operator ---

postfix func --- (value: inout Int) -> Int {
    let current = value
    value -= 2
    return current
}

struct Vector2D {
    var x: Double
    var y: Double
    
    static func + (left: Vector2D, right: Vector2D) -> Vector2D {
        Vector2D(x: left.x + right.x, y: left.y + right.y)
    }
    
    static func - (left: Vector2D, right: Vector2D) -> Vector2D {
        Vector2D(x: left.x - right.x, y: left.y - right.y)
    }
    
    static prefix func - (vector: Vector2D) -> Vector2D {
        Vector2D(x: -vector.x, y: -vector.y)
    }
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        # Should handle custom operators without errors
        assert len(structure.functions) >= 3
        assert len(structure.structs) >= 1

    def test_handle_key_paths(self, analyzer):
        """Test handling of key paths."""
        code = """
struct Person {
    var name: String
    var age: Int
    var address: Address
}

struct Address {
    var street: String
    var city: String
}

let people = [
    Person(name: "Alice", age: 30, address: Address(street: "123 Main", city: "NYC")),
    Person(name: "Bob", age: 25, address: Address(street: "456 Oak", city: "LA"))
]

let names = people.map(\\.name)
let ages = people.map(\\.age)
let cities = people.map(\\.address.city)

let namePath = \\Person.name
let cityPath = \\Person.address.city

func getValue<T, V>(from object: T, at keyPath: KeyPath<T, V>) -> V {
    object[keyPath: keyPath]
}
"""
        structure = analyzer.extract_structure(code, Path("test.swift"))

        # Should handle key paths without errors
        assert len(structure.structs) >= 2

    def test_main_app_detection(self, analyzer):
        """Test detection of main app entry point."""
        code_with_main = """
import SwiftUI

@main
struct MyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
"""
        structure = analyzer.extract_structure(code_with_main, Path("test.swift"))
        assert structure.has_main is True

        code_with_uiapplicationmain = """
import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
}
"""
        structure = analyzer.extract_structure(code_with_uiapplicationmain, Path("test.swift"))
        assert structure.has_main is True

        code_without_main = """
struct RegularStruct {
    var value: Int
}
"""
        structure = analyzer.extract_structure(code_without_main, Path("test.swift"))
        assert structure.has_main is False

    def test_test_file_detection(self, analyzer):
        """Test detection of test files."""
        test_code = """
import XCTest

class MyClassTests: XCTestCase {
    func testExample() {
        XCTAssertEqual(2 + 2, 4)
    }
}
"""
        structure = analyzer.extract_structure(test_code, Path("MyClassTests.swift"))
        assert structure.is_test_file is True

        structure = analyzer.extract_structure(test_code, Path("Tests/MyClass.swift"))
        assert structure.is_test_file is True

        structure = analyzer.extract_structure(test_code, Path("Sources/MyClass.swift"))
        assert structure.is_test_file is False
