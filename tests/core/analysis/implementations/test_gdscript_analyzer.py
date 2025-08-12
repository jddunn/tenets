"""
Unit tests for the GDScript code analyzer.

This module tests the GDScript-specific code analysis functionality including
preload/load statements, signal declarations, export variables, node references,
Godot lifecycle methods, and both Godot 3.x and 4.x syntax.

Test Coverage:
    - Import extraction (preload, load, extends, class_name)
    - Export detection (signals, export vars, public functions)
    - Structure extraction (functions, signals, onready vars, node refs)
    - Complexity metrics (Godot-specific complexity)
    - Lifecycle method detection
    - Tool script detection
    - Both Godot 3.x and 4.x syntax
    - Error handling for invalid GDScript code
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.gdscript_analyzer import GDScriptAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestGDScriptAnalyzerInitialization:
    """Test suite for GDScriptAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = GDScriptAnalyzer()

        assert analyzer.language_name == "gdscript"
        assert ".gd" in analyzer.file_extensions
        assert ".tres" in analyzer.file_extensions
        assert ".tscn" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for GDScript import extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GDScriptAnalyzer instance."""
        return GDScriptAnalyzer()

    def test_extract_preload_statements(self, analyzer):
        """Test extraction of preload statements."""
        code = """
const Player = preload("res://scripts/player.gd")
const Bullet = preload("res://scenes/bullet.tscn")
var enemy_scene = preload("res://enemies/enemy.tscn")
const Icon = preload("res://icon.png")
"""
        imports = analyzer.extract_imports(code, Path("test.gd"))

        assert len(imports) == 4
        
        player_import = next(imp for imp in imports if "player.gd" in imp.module)
        assert player_import.type == "preload"
        assert player_import.alias == "Player"
        assert player_import.is_relative is True
        assert player_import.resource_type == "script"

        bullet_import = next(imp for imp in imports if "bullet.tscn" in imp.module)
        assert bullet_import.resource_type == "scene"

        icon_import = next(imp for imp in imports if "icon.png" in imp.module)
        assert icon_import.resource_type == "texture"

    def test_extract_load_statements(self, analyzer):
        """Test extraction of load statements."""
        code = """
func _ready():
    var texture = load("res://assets/sprite.png")
    var sound = load("res://audio/effect.ogg")
    var resource = load("user://save_data.tres")
"""
        imports = analyzer.extract_imports(code, Path("test.gd"))

        load_imports = [imp for imp in imports if imp.type == "load"]
        assert len(load_imports) == 3
        
        assert all(imp.is_runtime_load for imp in load_imports)
        
        texture_import = next(imp for imp in load_imports if "sprite.png" in imp.module)
        assert texture_import.resource_type == "texture"

        sound_import = next(imp for imp in load_imports if "effect.ogg" in imp.module)
        assert sound_import.resource_type == "audio"

    def test_extract_extends_statements(self, analyzer):
        """Test extraction of extends statements."""
        code = """
extends Node2D

class InnerClass extends Reference:
    pass

# Another file
extends "res://base/base_enemy.gd"
"""
        imports = analyzer.extract_imports(code, Path("test.gd"))

        extends_imports = [imp for imp in imports if imp.type == "extends"]
        assert len(extends_imports) == 2

        node2d_import = next(imp for imp in extends_imports if imp.module == "Node2D")
        assert node2d_import.is_inheritance is True
        assert node2d_import.parent_type == "class"

        base_enemy_import = next(imp for imp in extends_imports if "base_enemy.gd" in imp.module)
        assert base_enemy_import.parent_type == "script"
        assert base_enemy_import.is_relative is True

    def test_extract_class_name_declaration(self, analyzer):
        """Test extraction of class_name declarations."""
        code = """
class_name Player
extends KinematicBody2D

# With icon
class_name Enemy, "res://icons/enemy.svg"
"""
        imports = analyzer.extract_imports(code, Path("test.gd"))

        # Should find the icon as an import
        icon_imports = [imp for imp in imports if imp.type == "icon"]
        assert len(icon_imports) == 1
        assert icon_imports[0].module == "res://icons/enemy.svg"
        assert icon_imports[0].associated_class == "Enemy"

    def test_extract_tool_script_declaration(self, analyzer):
        """Test extraction of tool script declaration."""
        code_v3 = """
tool
extends EditorPlugin

func _ready():
    pass
"""
        imports_v3 = analyzer.extract_imports(code_v3, Path("test.gd"))
        
        tool_import = next((imp for imp in imports_v3 if imp.type == "tool_mode"), None)
        assert tool_import is not None
        assert tool_import.is_editor_script is True

        # Godot 4.x syntax
        code_v4 = """
@tool
extends EditorPlugin

func _ready():
    pass
"""
        imports_v4 = analyzer.extract_imports(code_v4, Path("test.gd"))
        
        tool_annotation = next((imp for imp in imports_v4 if imp.type == "annotation"), None)
        assert tool_annotation is not None
        assert tool_annotation.is_editor_script is True


class TestExportExtraction:
    """Test suite for GDScript export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GDScriptAnalyzer instance."""
        return GDScriptAnalyzer()

    def test_extract_class_name_export(self, analyzer):
        """Test extraction of class_name as global export."""
        code = """
class_name PlayerCharacter, "res://icons/player.png"
extends CharacterBody2D
"""
        exports = analyzer.extract_exports(code, Path("test.gd"))

        class_export = next(e for e in exports if e["type"] == "global_class")
        assert class_export["name"] == "PlayerCharacter"
        assert class_export["icon"] == "res://icons/player.png"
        assert class_export["is_autoload_candidate"] is True

    def test_extract_export_variables_v3(self, analyzer):
        """Test extraction of export variables (Godot 3.x syntax)."""
        code = """
export var health = 100
export(int) var max_health = 100
export(float, 0.0, 1.0) var speed_multiplier = 1.0
export(String, FILE, "*.json") var config_file = ""
export(NodePath) var target_path
export(Array, Resource) var items = []
"""
        exports = analyzer.extract_exports(code, Path("test.gd"))

        export_vars = [e for e in exports if e["type"] == "export_var"]
        assert len(export_vars) == 6

        health_export = next(e for e in export_vars if e["name"] == "health")
        assert health_export["inspector_visible"] is True

        max_health_export = next(e for e in export_vars if e["name"] == "max_health")
        assert max_health_export["export_type"] == "int"

        speed_export = next(e for e in export_vars if e["name"] == "speed_multiplier")
        assert "float" in speed_export["export_type"]

    def test_extract_export_variables_v4(self, analyzer):
        """Test extraction of export variables (Godot 4.x syntax)."""
        code = """
@export var health: int = 100
@export_range(0.0, 1.0) var speed_multiplier: float = 1.0
@export_file("*.json") var config_file: String = ""
@export_node_path("Area2D") var detection_area: NodePath
@export_category("Items")
@export var items: Array[Resource] = []
"""
        exports = analyzer.extract_exports(code, Path("test.gd"))

        export_vars = [e for e in exports if e["type"] == "export_var" and e.get("godot_version") == 4]
        assert len(export_vars) >= 4

        speed_export = next(e for e in export_vars if e["name"] == "speed_multiplier")
        assert speed_export["export_modifier"] == "range"

        config_export = next(e for e in export_vars if e["name"] == "config_file")
        assert config_export["export_modifier"] == "file"

    def test_extract_signals(self, analyzer):
        """Test extraction of signals."""
        code = """
signal health_changed(new_health)
signal player_died
signal item_collected(item_type, amount)
signal boss_defeated()
"""
        exports = analyzer.extract_exports(code, Path("test.gd"))

        signals = [e for e in exports if e["type"] == "signal"]
        assert len(signals) == 4

        health_signal = next(s for s in signals if s["name"] == "health_changed")
        assert health_signal["is_event"] is True
        assert "new_health" in health_signal["parameters"]

        item_signal = next(s for s in signals if s["name"] == "item_collected")
        assert len(item_signal["parameters"]) == 2

    def test_extract_public_functions(self, analyzer):
        """Test extraction of public functions."""
        code = """
func take_damage(amount: int) -> void:
    health -= amount

func get_health() -> int:
    return health

func _ready():  # Private (starts with underscore)
    pass

static func create_instance():
    return new()
"""
        exports = analyzer.extract_exports(code, Path("test.gd"))

        functions = [e for e in exports if e["type"] == "function"]
        assert len(functions) == 3  # Excludes _ready

        take_damage = next(f for f in functions if f["name"] == "take_damage")
        assert take_damage["is_public"] is True

        create_func = next(f for f in functions if f["name"] == "create_instance")
        assert create_func["is_static"] is True

    def test_extract_enums_and_constants(self, analyzer):
        """Test extraction of enums and constants."""
        code = """
enum State {
    IDLE,
    WALKING,
    RUNNING,
    JUMPING
}

enum Direction { LEFT = -1, RIGHT = 1 }

const MAX_SPEED = 500.0
const GRAVITY = 980.0
const JUMP_FORCE = -400.0
"""
        exports = analyzer.extract_exports(code, Path("test.gd"))

        enums = [e for e in exports if e["type"] == "enum"]
        assert len(enums) == 2

        state_enum = next(e for e in enums if e["name"] == "State")
        assert state_enum is not None

        constants = [e for e in exports if e["type"] == "constant"]
        assert len(constants) == 3
        assert all(c["is_public"] for c in constants)


class TestStructureExtraction:
    """Test suite for GDScript structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GDScriptAnalyzer instance."""
        return GDScriptAnalyzer()

    def test_extract_basic_structure(self, analyzer):
        """Test extraction of basic GDScript structure."""
        code = """
class_name Player
extends CharacterBody2D

signal health_changed(value)

export var max_health = 100
var health = 100

func _ready():
    health = max_health

func take_damage(amount: int) -> void:
    health -= amount
    emit_signal("health_changed", health)
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        assert structure.class_name == "Player"
        assert structure.parent_class == "CharacterBody2D"
        assert len(structure.signals) == 1
        assert len(structure.export_vars) == 1
        assert len(structure.functions) == 2

        ready_func = next(f for f in structure.functions if f.name == "_ready")
        assert ready_func.is_lifecycle is True
        assert ready_func.is_virtual is True

    def test_extract_inner_classes(self, analyzer):
        """Test extraction of inner classes."""
        code = """
extends Node

class Weapon extends Resource:
    var damage: int
    var range: float
    
    func fire():
        pass

class Inventory:
    var items = []
    
    func add_item(item):
        items.append(item)
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        # Should have main class + 2 inner classes
        assert len(structure.classes) == 3

        weapon_class = next(c for c in structure.classes if c.name == "Weapon")
        assert weapon_class.is_inner_class is True
        assert "Resource" in weapon_class.bases

        inventory_class = next(c for c in structure.classes if c.name == "Inventory")
        assert inventory_class.is_inner_class is True

    def test_extract_typed_gdscript(self, analyzer):
        """Test extraction of typed GDScript elements."""
        code = """
extends Node

var player_name: String = "Player"
var level: int = 1
var position: Vector2 = Vector2.ZERO
var inventory: Array[Item] = []

func calculate_damage(base_damage: float, multiplier: float = 1.0) -> float:
    return base_damage * multiplier

func get_player() -> Player:
    return $Player as Player

func process_items(items: Array[Item]) -> void:
    for item in items:
        print(item.name)
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        # Check typed variables
        player_var = next(v for v in structure.variables if v["name"] == "player_name")
        assert player_var["type"] == "String"

        inventory_var = next(v for v in structure.variables if v["name"] == "inventory")
        assert "Array" in inventory_var["type"]

        # Check typed functions
        calc_func = next(f for f in structure.functions if f.name == "calculate_damage")
        assert calc_func.return_type == "float"
        assert len(calc_func.parameters) == 2
        assert calc_func.parameters[0]["type"] == "float"
        assert calc_func.parameters[1].get("default") == "1.0"

    def test_extract_onready_variables_v3(self, analyzer):
        """Test extraction of onready variables (Godot 3.x)."""
        code = """
extends Node2D

onready var sprite = $Sprite
onready var animation_player: AnimationPlayer = $AnimationPlayer
onready var collision_shape = get_node("CollisionShape2D")
onready var health_bar = $"../UI/HealthBar"
onready var timer: Timer = Timer.new()
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        assert len(structure.onready_vars) == 5

        sprite_var = next(v for v in structure.onready_vars if v["name"] == "sprite")
        assert sprite_var["is_node_ref"] is True
        assert sprite_var["node_path"] == "Sprite"

        health_bar_var = next(v for v in structure.onready_vars if v["name"] == "health_bar")
        assert health_bar_var["node_path"] == "../UI/HealthBar"

        timer_var = next(v for v in structure.onready_vars if v["name"] == "timer")
        assert timer_var["is_node_ref"] is False  # Using new() instead

    def test_extract_onready_variables_v4(self, analyzer):
        """Test extraction of onready variables (Godot 4.x)."""
        code = """
extends Node2D

@onready var sprite: Sprite2D = $Sprite2D
@onready var animation_player: AnimationPlayer = $AnimationPlayer
@onready var collision_shape = get_node("CollisionShape2D")
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        onready_v4 = [v for v in structure.onready_vars if v.get("godot_4")]
        assert len(onready_v4) == 3

        sprite_var = next(v for v in onready_v4 if v["name"] == "sprite")
        assert sprite_var["type"] == "Sprite2D"
        assert sprite_var["is_node_ref"] is True

    def test_extract_lifecycle_methods(self, analyzer):
        """Test extraction of Godot lifecycle methods."""
        code = """
extends Node2D

func _ready():
    print("Ready")

func _enter_tree():
    pass

func _exit_tree():
    pass

func _process(delta):
    pass

func _physics_process(delta):
    pass

func _input(event):
    pass

func _unhandled_input(event):
    pass

func _draw():
    pass

func _notification(what):
    pass
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        lifecycle_funcs = [f for f in structure.functions if f.is_lifecycle]
        assert len(lifecycle_funcs) == 9

        process_func = next(f for f in lifecycle_funcs if f.name == "_process")
        assert process_func.is_virtual is True
        assert process_func.is_private is True

    def test_extract_signal_connections(self, analyzer):
        """Test extraction of signal-related patterns."""
        code = """
extends Node

signal custom_signal(data)

func _ready():
    $Button.connect("pressed", self, "_on_button_pressed")
    connect("custom_signal", self, "_on_custom_signal")
    
    # Godot 4.x syntax
    $Button.pressed.connect(_on_button_pressed)

func _on_button_pressed():
    emit_signal("custom_signal", {"value": 10})

func _on_custom_signal(data):
    print(data)

func _on_area_2d_body_entered(body):
    pass
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        assert structure.connect_calls == 3
        assert structure.emit_signal_calls == 1

        # Check callback detection
        callbacks = [f for f in structure.functions if f.name.startswith("_on_")]
        assert len(callbacks) == 3

    def test_extract_setget_properties(self, analyzer):
        """Test extraction of setget properties."""
        code = """
extends Node

var health = 100 setget set_health, get_health
var mana = 50 setget set_mana
var level = 1 setget , get_level  # Getter only

func set_health(value):
    health = clamp(value, 0, 100)

func get_health():
    return health

func set_mana(value):
    mana = max(0, value)

func get_level():
    return level * 2
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        assert len(structure.setget_properties) == 3

        health_prop = next(p for p in structure.setget_properties if p["name"] == "health")
        assert health_prop["setter"] == "set_health"
        assert health_prop["getter"] == "get_health"

        level_prop = next(p for p in structure.setget_properties if p["name"] == "level")
        assert level_prop["setter"] is None
        assert level_prop["getter"] == "get_level"

    def test_detect_tool_and_resource_scripts(self, analyzer):
        """Test detection of tool scripts and custom resources."""
        tool_code = """
@tool
extends EditorPlugin

func _enter_tree():
    pass
"""
        tool_structure = analyzer.extract_structure(tool_code, Path("plugin.gd"))
        assert tool_structure.is_tool_script is True
        assert tool_structure.is_editor_plugin is True

        resource_code = """
class_name ItemData
extends Resource

@export var name: String = ""
@export var value: int = 0
"""
        resource_structure = analyzer.extract_structure(resource_code, Path("item_data.gd"))
        assert resource_structure.is_custom_resource is True


class TestComplexityCalculation:
    """Test suite for GDScript complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GDScriptAnalyzer instance."""
        return GDScriptAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
func complex_function(value):
    if value > 0:
        if value > 10:
            return "high"
        elif value > 5:
            return "medium"
        else:
            return "low"
    
    for i in range(10):
        if i % 2 == 0:
            print(i)
    
    match value:
        1:
            return "one"
        2:
            return "two"
        _:
            return "other"
"""
        metrics = analyzer.calculate_complexity(code, Path("test.gd"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 9

    def test_godot_specific_metrics(self, analyzer):
        """Test Godot-specific complexity metrics."""
        code = """
extends Node2D

signal health_changed
signal mana_changed
signal level_up

@export var max_health = 100
@export var max_mana = 50
@export_range(1, 99) var level = 1

@onready var sprite = $Sprite2D
@onready var collision = $CollisionShape2D
@onready var health_bar = $"../UI/HealthBar"

func _ready():
    $Timer.connect("timeout", self, "_on_timer_timeout")
    connect("health_changed", $"../UI", "update_health")
    
func take_damage(amount):
    emit_signal("health_changed")
    
func use_skill():
    emit_signal("mana_changed")
    
func gain_experience(xp):
    emit_signal("level_up")
"""
        metrics = analyzer.calculate_complexity(code, Path("test.gd"))

        assert metrics.signal_count == 3
        assert metrics.export_count >= 3
        assert metrics.onready_count == 3
        assert metrics.node_ref_count >= 4
        assert metrics.connect_count >= 2
        assert metrics.emit_count >= 3

    def test_type_hint_metrics(self, analyzer):
        """Test metrics for typed GDScript."""
        code = """
extends Node

var untyped_var = 10
var typed_var: int = 10
var typed_float: float = 1.0
const TYPED_CONST: String = "constant"

func untyped_func(param):
    return param

func typed_func(param: int) -> int:
    return param * 2

func mixed_func(a, b: float) -> String:
    return str(a + b)

func fully_typed(name: String, age: int, height: float) -> Dictionary:
    return {"name": name, "age": age, "height": height}
"""
        metrics = analyzer.calculate_complexity(code, Path("test.gd"))

        assert metrics.typed_vars >= 3
        assert metrics.typed_funcs >= 2
        assert metrics.return_types >= 3

    def test_networking_metrics(self, analyzer):
        """Test RPC/networking metrics."""
        code = """
extends Node

@rpc("any_peer")
func update_position(pos):
    position = pos

@rpc("authority", "reliable")
func spawn_player(id, pos):
    pass

func _ready():
    rpc("update_position", position)
    rpc_unreliable("update_rotation", rotation)
    
remotesync func sync_state(state):
    pass

master func master_function():
    pass

puppet func puppet_function():
    pass
"""
        metrics = analyzer.calculate_complexity(code, Path("test.gd"))

        assert metrics.rpc_count >= 5

    def test_indentation_based_nesting(self, analyzer):
        """Test nesting depth calculation for indentation-based syntax."""
        code = """
func nested_function():
    if true:
        for i in range(10):
            while i > 0:
                if i % 2 == 0:
                    print("even")
                    if i > 5:
                        print("greater than 5")
                        return
"""
        metrics = analyzer.calculate_complexity(code, Path("test.gd"))

        assert metrics.max_depth >= 5


class TestGodotVersionDetection:
    """Test suite for Godot version detection."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GDScriptAnalyzer instance."""
        return GDScriptAnalyzer()

    def test_detect_godot_3(self, analyzer):
        """Test detection of Godot 3.x syntax."""
        code = """
extends Node2D

export var speed = 100
export(int, 0, 100) var health = 100
onready var sprite = $Sprite

func _ready():
    $Button.connect("pressed", self, "_on_button_pressed")
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        assert structure.godot_version == 3

    def test_detect_godot_4(self, analyzer):
        """Test detection of Godot 4.x syntax."""
        code = """
extends Node2D

@export var speed = 100
@export_range(0, 100) var health: int = 100
@onready var sprite = $Sprite2D

func _ready():
    $Button.pressed.connect(_on_button_pressed)
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))

        assert structure.godot_version == 4


class TestErrorHandling:
    """Test suite for error handling in GDScript analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GDScriptAnalyzer instance."""
        return GDScriptAnalyzer()

    def test_handle_malformed_code(self, analyzer):
        """Test handling of malformed GDScript code."""
        code = """
extends Node2D

func broken_function()
    This is not valid GDScript
    missing colons and indentation

signal incomplete_signal(

var no_value = 

func another_func():
    if true
        print("missing colon")
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("test.gd"))
        exports = analyzer.extract_exports(code, Path("test.gd"))
        structure = analyzer.extract_structure(code, Path("test.gd"))
        metrics = analyzer.calculate_complexity(code, Path("test.gd"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty file."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.gd"))
        exports = analyzer.extract_exports(code, Path("test.gd"))
        structure = analyzer.extract_structure(code, Path("test.gd"))
        metrics = analyzer.calculate_complexity(code, Path("test.gd"))

        assert imports == []
        assert exports == []
        assert len(structure.functions) == 0
        assert metrics.line_count == 1

    def test_handle_scene_file_with_inline_script(self, analyzer):
        """Test handling of .tscn files with inline GDScript."""
        code = """
[gd_scene load_steps=2 format=3]

[sub_resource type="GDScript" id="1"]
script/source = "
extends Node2D

func _ready():
    print('Hello from inline script')
"

[node name="Root" type="Node2D"]
script = SubResource("1")
"""
        # Should handle gracefully even though it's not pure GDScript
        structure = analyzer.extract_structure(code, Path("test.tscn"))
        assert isinstance(structure, CodeStructure)


class TestEdgeCases:
    """Test suite for edge cases in GDScript analysis."""

    @pytest.fixture
    def analyzer(self):
        """Provide a GDScriptAnalyzer instance."""
        return GDScriptAnalyzer()

    def test_multiline_expressions(self, analyzer):
        """Test handling of multiline expressions."""
        code = """
var long_array = [
    "item1",
    "item2",
    "item3"
]

var dict = {
    "key1": "value1",
    "key2": "value2",
    "nested": {
        "inner": "value"
    }
}

func multiline_call():
    some_function(
        parameter1,
        parameter2,
        parameter3
    )
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))
        
        assert len(structure.variables) >= 2
        assert len(structure.functions) == 1

    def test_string_interpolation_and_multiline(self, analyzer):
        """Test handling of GDScript string features."""
        code = '''
extends Node

var player_name = "Player"
var message = "Hello, %s!" % player_name
var interpolated = "Score: %d" % [score]

var multiline_string = """
This is a multiline string
with multiple lines
"""

func format_text():
    return "Player: {name}, Level: {level}".format({
        "name": player_name,
        "level": 10
    })
'''
        structure = analyzer.extract_structure(code, Path("test.gd"))
        
        # Should handle without errors
        assert len(structure.variables) >= 3
        assert len(structure.functions) == 1

    def test_lambda_and_callables(self, analyzer):
        """Test handling of lambda functions and callables."""
        code = """
extends Node

func _ready():
    var my_lambda = func(x): return x * 2
    
    var callable = Callable(self, "_on_timer_timeout")
    
    $Button.pressed.connect(func(): print("Inline lambda"))
    
    var array = [1, 2, 3, 4, 5]
    var filtered = array.filter(func(x): return x > 2)
"""
        metrics = analyzer.calculate_complexity(code, Path("test.gd"))
        
        # Should detect complexity from lambdas
        assert metrics.cyclomatic >= 2

    def test_await_and_coroutines(self, analyzer):
        """Test handling of await and coroutines (Godot 4.x)."""
        code = """
extends Node

func async_function():
    await get_tree().create_timer(1.0).timeout
    print("Timer finished")
    
    var result = await make_http_request()
    return result

func another_async():
    await async_function()
    
signal my_signal

func wait_for_signal():
    await my_signal
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))
        
        assert len(structure.functions) >= 3

    def test_annotations_and_decorators(self, analyzer):
        """Test handling of various Godot 4.x annotations."""
        code = """
@tool
@icon("res://icon.svg")
extends EditorPlugin

@export_category("Player Settings")
@export var player_name: String = "Player"
@export_group("Stats")
@export var health: int = 100
@export var mana: int = 50

@export_subgroup("Movement")
@export var speed: float = 5.0
@export_exp_easing var acceleration: float = 1.0

@warning_ignore("unused_parameter")
func unused_param_func(param):
    pass

@deprecated("Use new_function instead")
func old_function():
    pass
"""
        structure = analyzer.extract_structure(code, Path("test.gd"))
        
        assert structure.is_tool_script is True
        assert len(structure.export_vars) >= 5