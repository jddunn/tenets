"""
Unit tests for the Ruby code analyzer.

This module tests the Ruby-specific code analysis functionality including
require/gem extraction, class/module parsing, method visibility handling,
metaprogramming detection, and Ruby-specific complexity calculation.

Test Coverage:
    - Require and gem dependency extraction
    - Export detection (classes, modules, methods, constants)
    - Structure extraction with Ruby-specific features
    - Complexity metrics including ABC and metaprogramming
    - Block, proc, and lambda detection
    - Framework detection (Rails, Sinatra, RSpec)
    - Ruby-specific patterns (symbols, instance variables, DSLs)
    - Error handling for invalid Ruby code
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import pytest

from tenets.core.analysis.implementations.ruby_analyzer import RubyAnalyzer
from tenets.models.analysis import (
    ImportInfo,
    CodeStructure,
    ComplexityMetrics,
    FunctionInfo,
    ClassInfo,
)


class TestRubyAnalyzerInitialization:
    """Test suite for RubyAnalyzer initialization."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = RubyAnalyzer()

        assert analyzer.language_name == "ruby"
        assert ".rb" in analyzer.file_extensions
        assert ".rake" in analyzer.file_extensions
        assert ".gemspec" in analyzer.file_extensions
        assert analyzer.logger is not None


class TestImportExtraction:
    """Test suite for Ruby require/gem extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RubyAnalyzer instance."""
        return RubyAnalyzer()

    def test_extract_requires(self, analyzer):
        """Test extraction of require statements."""
        code = """
require 'json'
require 'net/http'
require "active_record"
require 'rails/all'
"""
        imports = analyzer.extract_imports(code, Path("test.rb"))

        assert len(imports) == 4
        assert any(imp.module == "json" for imp in imports)
        assert any(imp.module == "net/http" for imp in imports)
        assert any(imp.module == "active_record" for imp in imports)

        # Check import types
        json_import = next(imp for imp in imports if imp.module == "json")
        assert json_import.type == "require"
        assert json_import.is_relative is False
        assert json_import.is_stdlib is True

        rails_import = next(imp for imp in imports if imp.module == "rails/all")
        assert rails_import.is_gem is True

    def test_extract_require_relative(self, analyzer):
        """Test extraction of require_relative statements."""
        code = """
require_relative 'helper'
require_relative '../lib/utils'
require_relative './models/user'
"""
        imports = analyzer.extract_imports(code, Path("test.rb"))

        assert len(imports) == 3
        assert all(imp.type == "require_relative" for imp in imports)
        assert all(imp.is_relative for imp in imports)
        assert all(imp.is_project_file for imp in imports)

    def test_extract_gems(self, analyzer):
        """Test extraction of gem statements."""
        code = """
gem 'rails', '~> 7.0'
gem 'pg'
gem 'sidekiq', '>= 6.0'
gem "rspec", "3.12.0"
"""
        imports = analyzer.extract_imports(code, Path("test.rb"))

        assert len(imports) == 4
        assert all(imp.type == "gem" for imp in imports)
        assert all(imp.is_gem for imp in imports)

        rails_gem = next(imp for imp in imports if imp.module == "rails")
        assert rails_gem.version == "~> 7.0"

        pg_gem = next(imp for imp in imports if imp.module == "pg")
        assert pg_gem.version is None

    def test_extract_load(self, analyzer):
        """Test extraction of load statements."""
        code = """
load 'config.rb'
load './initializers/constants.rb'
load File.join(Rails.root, 'lib', 'tasks', 'custom.rake')
"""
        imports = analyzer.extract_imports(code, Path("test.rb"))

        assert any(imp.module == "config.rb" and imp.type == "load" for imp in imports)
        assert any(imp.reloads is True for imp in imports)  # load can reload files

    def test_extract_autoload(self, analyzer):
        """Test extraction of autoload statements."""
        code = """
autoload :MyModule, 'my_module'
autoload :Helper, './lib/helper'
Module.autoload :SubModule, 'sub_module'
"""
        imports = analyzer.extract_imports(code, Path("test.rb"))

        autoloads = [imp for imp in imports if imp.type == "autoload"]
        assert len(autoloads) >= 2
        assert any(imp.alias == "MyModule" for imp in autoloads)
        assert all(imp.lazy_load for imp in autoloads)

    def test_conditional_requires(self, analyzer):
        """Test extraction of conditional requires."""
        code = """
require 'debug' if ENV['DEBUG']
require 'pry' unless Rails.env.production?
require 'simplecov' if ENV['COVERAGE']
"""
        imports = analyzer.extract_imports(code, Path("test.rb"))

        assert any(imp.module == "debug" and imp.conditional for imp in imports)
        assert all(imp.type == "conditional_require" for imp in imports if imp.conditional)

    def test_bundler_require(self, analyzer):
        """Test detection of Bundler.require."""
        code = """
require 'bundler/setup'
Bundler.require(:default, Rails.env)
"""
        imports = analyzer.extract_imports(code, Path("test.rb"))

        bundler_imports = [imp for imp in imports if "Bundler" in imp.module or imp.type == "bundler_require"]
        assert len(bundler_imports) > 0
        assert any(imp.loads_all_gems for imp in bundler_imports if hasattr(imp, 'loads_all_gems'))

    def test_gemfile_dependencies(self, analyzer):
        """Test extraction from Gemfile."""
        code = """
source 'https://rubygems.org'

gem 'rails', '~> 7.0.0'
gem 'pg', '~> 1.1'

group :development, :test do
  gem 'rspec-rails'
  gem 'factory_bot_rails'
end

group :development do
  gem 'spring'
end
"""
        imports = analyzer.extract_imports(code, Path("Gemfile"))

        gem_deps = [imp for imp in imports if imp.type in ["gem", "gemfile_dependency"]]
        assert len(gem_deps) >= 5
        assert any(imp.module == "rails" for imp in gem_deps)
        assert any(imp.module == "rspec-rails" for imp in gem_deps)


class TestExportExtraction:
    """Test suite for Ruby export extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RubyAnalyzer instance."""
        return RubyAnalyzer()

    def test_extract_classes(self, analyzer):
        """Test extraction of class exports."""
        code = """
class User < ApplicationRecord
  # User model
end

class Admin < User
  # Admin inherits from User
end

class Service::PaymentProcessor
  # Namespaced class
end

class << self
  # Singleton class
end
"""
        exports = analyzer.extract_exports(code, Path("test.rb"))

        class_exports = [e for e in exports if e["type"] == "class"]
        assert any(e["name"] == "User" for e in class_exports)
        assert any(e["name"] == "Admin" for e in class_exports)

        user_class = next(e for e in class_exports if e["name"] == "User")
        assert user_class["superclass"] == "ApplicationRecord"

        admin_class = next(e for e in class_exports if e["name"] == "Admin")
        assert admin_class["superclass"] == "User"

    def test_extract_modules(self, analyzer):
        """Test extraction of module exports."""
        code = """
module Helpers
  # Helper module
end

module Concerns::Trackable
  # Concern module
end

module API
  module V1
    # Nested module
  end
end
"""
        exports = analyzer.extract_exports(code, Path("test.rb"))

        module_exports = [e for e in exports if e["type"] == "module"]
        assert any(e["name"] == "Helpers" for e in module_exports)
        assert any(e["name"] == "API" for e in module_exports)

    def test_extract_methods_with_visibility(self, analyzer):
        """Test extraction of methods with visibility modifiers."""
        code = """
class MyClass
  def public_method
    # This is public
  end

  private

  def private_method
    # This is private
  end

  protected

  def protected_method
    # This is protected
  end

  public

  def another_public_method
    # Back to public
  end
end

def top_level_method
  # Top-level method
end
"""
        exports = analyzer.extract_exports(code, Path("test.rb"))

        method_exports = [e for e in exports if e["type"] == "method"]
        
        # Only public methods should be exported
        assert any(e["name"] == "public_method" for e in method_exports)
        assert any(e["name"] == "another_public_method" for e in method_exports)
        assert any(e["name"] == "top_level_method" for e in method_exports)
        assert not any(e["name"] == "private_method" for e in method_exports)
        assert not any(e["name"] == "protected_method" for e in method_exports)

    def test_extract_constants(self, analyzer):
        """Test extraction of constants."""
        code = """
VERSION = "1.0.0"
MAX_RETRIES = 3
API_KEY = ENV['API_KEY']
DEFAULT_OPTIONS = { timeout: 30, retries: 3 }
"""
        exports = analyzer.extract_exports(code, Path("test.rb"))

        constant_exports = [e for e in exports if e["type"] == "constant"]
        assert any(e["name"] == "VERSION" for e in constant_exports)
        assert any(e["name"] == "MAX_RETRIES" for e in constant_exports)
        assert any(e["name"] == "DEFAULT_OPTIONS" for e in constant_exports)

    def test_extract_special_methods(self, analyzer):
        """Test extraction of special Ruby methods."""
        code = """
class MyClass
  def self.class_method
    # Class method
  end

  def predicate?
    # Predicate method
  end

  def dangerous!
    # Bang method
  end

  def setter=(value)
    # Setter method
  end
end
"""
        exports = analyzer.extract_exports(code, Path("test.rb"))

        method_exports = [e for e in exports if e["type"] == "method"]
        
        class_method = next((e for e in method_exports if e["name"] == "class_method"), None)
        if class_method:
            assert class_method["is_class_method"] is True

        predicate = next((e for e in method_exports if e["name"] == "predicate?"), None)
        if predicate:
            assert predicate["is_predicate"] is True

        bang = next((e for e in method_exports if e["name"] == "dangerous!"), None)
        if bang:
            assert bang["is_bang_method"] is True

        setter = next((e for e in method_exports if e["name"] == "setter="), None)
        if setter:
            assert setter["is_setter"] is True


class TestStructureExtraction:
    """Test suite for Ruby code structure extraction."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RubyAnalyzer instance."""
        return RubyAnalyzer()

    def test_extract_class_with_methods(self, analyzer):
        """Test extraction of classes with methods and visibility."""
        code = """
class User < ActiveRecord::Base
  attr_reader :name
  attr_writer :email
  attr_accessor :age

  def initialize(name)
    @name = name
  end

  def public_method
    puts "Public"
  end

  private

  def private_method
    puts "Private"
  end

  protected

  def protected_method
    puts "Protected"
  end
end
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))

        assert len(structure.classes) == 1
        user_class = structure.classes[0]
        
        assert user_class.name == "User"
        assert "ActiveRecord::Base" in user_class.bases

        # Check attributes
        assert len(user_class.attributes) >= 3
        attr_names = [a["name"] for a in user_class.attributes]
        assert "name" in attr_names
        assert "email" in attr_names
        assert "age" in attr_names

        # Check methods
        assert len(user_class.methods) >= 4
        method_names = [m["name"] for m in user_class.methods]
        assert "initialize" in method_names
        assert "public_method" in method_names
        assert "private_method" in method_names

        # Check visibility
        public_method = next(m for m in user_class.methods if m["name"] == "public_method")
        assert public_method["visibility"] == "public"

        private_method = next(m for m in user_class.methods if m["name"] == "private_method")
        assert private_method["visibility"] == "private"

    def test_extract_modules_with_mixins(self, analyzer):
        """Test extraction of modules with include/extend."""
        code = """
module Trackable
  def track
    # tracking logic
  end
end

module Searchable
  def self.included(base)
    base.extend(ClassMethods)
  end

  module ClassMethods
    def search(query)
      # search logic
    end
  end
end

class Product
  include Trackable
  include Searchable
  extend ActiveSupport::Concern
end
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))

        # Check modules
        assert any(m["name"] == "Trackable" for m in structure.modules)
        assert any(m["name"] == "Searchable" for m in structure.modules)

        # Check class with includes/extends
        product_class = next(c for c in structure.classes if c.name == "Product")
        assert "Trackable" in product_class.included_modules
        assert "Searchable" in product_class.included_modules
        assert "ActiveSupport::Concern" in product_class.extended_modules

    def test_detect_blocks_procs_lambdas(self, analyzer):
        """Test detection of blocks, procs, and lambdas."""
        code = """
# Blocks
[1, 2, 3].each do |n|
  puts n
end

[1, 2, 3].map { |n| n * 2 }

# Procs
my_proc = Proc.new { |x| x * 2 }
another_proc = proc { |x| x + 1 }

# Lambdas
my_lambda = lambda { |x| x * 2 }
arrow_lambda = ->(x) { x * 2 }
unicode_lambda = λ { |x| x * 2 }
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))

        assert structure.block_count >= 2  # do...end and {}
        assert structure.proc_count >= 2  # Proc.new and proc
        assert structure.lambda_count >= 3  # lambda, ->, and λ

    def test_extract_instance_and_class_variables(self, analyzer):
        """Test extraction of instance and class variables."""
        code = """
class MyClass
  @@class_var = 0
  @class_instance_var = []

  def initialize
    @instance_var = 1
    @another_var = "hello"
  end

  def method
    @dynamic_var = true
    @@class_var += 1
  end
end

$global_var = "global"
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))

        assert "@instance_var" in structure.instance_variables
        assert "@another_var" in structure.instance_variables
        assert "@dynamic_var" in structure.instance_variables
        
        assert "@@class_var" in structure.class_variables
        
        assert "$global_var" in structure.global_variables

    def test_extract_aliases(self, analyzer):
        """Test extraction of method aliases."""
        code = """
class MyClass
  def original_method
    puts "Original"
  end

  alias new_method original_method
  alias_method :another_alias, :original_method
end
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))

        assert len(structure.aliases) >= 2
        
        alias_names = [(a["new_name"], a["original_name"]) for a in structure.aliases]
        assert ("new_method", "original_method") in alias_names
        assert ("another_alias", "original_method") in alias_names

    def test_detect_rails_framework(self, analyzer):
        """Test detection of Rails framework."""
        code = """
class UsersController < ApplicationController
  before_action :authenticate_user!
  
  def index
    @users = User.all
  end
end

class User < ActiveRecord::Base
  has_many :posts
  belongs_to :organization
  validates :email, presence: true
end
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))

        assert structure.framework == "Rails"

    def test_detect_rspec(self, analyzer):
        """Test detection of RSpec framework."""
        code = """
require 'spec_helper'

describe User do
  context 'when valid' do
    it 'saves successfully' do
      user = User.new(name: 'Test')
      expect(user.save).to be true
    end
  end
end
"""
        structure = analyzer.extract_structure(code, Path("user_spec.rb"))

        assert structure.framework == "RSpec"
        assert structure.is_test_file is True


class TestComplexityCalculation:
    """Test suite for Ruby complexity metrics."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RubyAnalyzer instance."""
        return RubyAnalyzer()

    def test_calculate_cyclomatic_complexity(self, analyzer):
        """Test cyclomatic complexity calculation."""
        code = """
def complex_method(x, y)
  if x > 0
    if y > 0
      return x + y
    else
      return x - y
    end
  elsif x < 0
    y.times do |i|
      x += i
    end
  end
  
  case x
  when 1
    return 1
  when 2
    return 2
  else
    return 0
  end
end
"""
        metrics = analyzer.calculate_complexity(code, Path("test.rb"))

        # Should have significant complexity
        assert metrics.cyclomatic >= 7

    def test_calculate_abc_metrics(self, analyzer):
        """Test ABC (Assignment, Branch, Condition) metrics calculation."""
        code = """
def abc_example(data)
  # Assignments
  x = 10
  y = 20
  z = x + y
  result ||= []
  
  # Branches (method calls)
  puts "Hello"
  data.map { |d| d * 2 }
  super
  yield if block_given?
  
  # Conditions
  if x > y && z == 30
    return true
  elsif x < y || z != 30
    return false
  end
  
  x > 0 ? "positive" : "negative"
end
"""
        metrics = analyzer.calculate_complexity(code, Path("test.rb"))

        assert metrics.assignments > 0
        assert metrics.branches > 0
        assert metrics.conditions > 0
        assert metrics.abc_score > 0

    def test_metaprogramming_complexity(self, analyzer):
        """Test metaprogramming complexity metrics."""
        code = """
class DynamicClass
  define_method :dynamic_method do |arg|
    puts arg
  end

  def method_missing(method_name, *args)
    if method_name.to_s.start_with?('find_by_')
      # Handle dynamic finders
    else
      super
    end
  end

  class_eval do
    attr_accessor :dynamic_attr
  end

  instance_eval do
    @class_var = "value"
  end

  send(:define_method, :another_method) { "dynamic" }
end
"""
        metrics = analyzer.calculate_complexity(code, Path("test.rb"))

        assert metrics.metaprogramming_score >= 5  # Multiple metaprogramming methods used

    def test_iterator_complexity(self, analyzer):
        """Test that iterators add to complexity."""
        code = """
def iterator_example(items)
  items.each do |item|
    puts item
  end
  
  items.map { |i| i * 2 }
  items.select { |i| i > 0 }
  items.reject { |i| i.nil? }
  
  5.times do
    puts "Hello"
  end
  
  1.upto(10) do |n|
    puts n
  end
  
  10.downto(1) do |n|
    puts n
  end
end
"""
        metrics = analyzer.calculate_complexity(code, Path("test.rb"))

        # Each iterator should add to complexity
        assert metrics.cyclomatic >= 7

    def test_test_file_metrics(self, analyzer):
        """Test metrics specific to test files."""
        code = """
require 'spec_helper'

describe User do
  describe '#full_name' do
    it 'returns the full name' do
      user = User.new(first: 'John', last: 'Doe')
      expect(user.full_name).to eq('John Doe')
    end
  end
  
  context 'validations' do
    it 'requires an email' do
      user = User.new
      expect(user).not_to be_valid
      assert user.errors[:email].present?
    end
  end
end
"""
        metrics = analyzer.calculate_complexity(code, Path("user_spec.rb"))

        assert metrics.test_count >= 2
        assert metrics.expectation_count >= 3  # expect statements
        assert metrics.assertion_count >= 1  # assert statement


class TestErrorHandling:
    """Test suite for error handling."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RubyAnalyzer instance."""
        return RubyAnalyzer()

    def test_handle_syntax_errors(self, analyzer):
        """Test handling of files with syntax errors."""
        code = """
class Invalid
  this is not valid Ruby code
  def method
    missing end
  # Missing end
"""
        # Should not raise exception
        imports = analyzer.extract_imports(code, Path("test.rb"))
        exports = analyzer.extract_exports(code, Path("test.rb"))
        structure = analyzer.extract_structure(code, Path("test.rb"))
        metrics = analyzer.calculate_complexity(code, Path("test.rb"))

        assert isinstance(imports, list)
        assert isinstance(exports, list)
        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_handle_empty_file(self, analyzer):
        """Test handling of empty files."""
        code = ""

        imports = analyzer.extract_imports(code, Path("test.rb"))
        exports = analyzer.extract_exports(code, Path("test.rb"))
        structure = analyzer.extract_structure(code, Path("test.rb"))
        metrics = analyzer.calculate_complexity(code, Path("test.rb"))

        assert imports == []
        assert exports == []
        assert metrics.line_count == 0


class TestEdgeCases:
    """Test suite for edge cases and special Ruby features."""

    @pytest.fixture
    def analyzer(self):
        """Provide a RubyAnalyzer instance."""
        return RubyAnalyzer()

    def test_operator_methods(self, analyzer):
        """Test detection of operator methods."""
        code = """
class Vector
  def +(other)
    # Vector addition
  end
  
  def -(other)
    # Vector subtraction
  end
  
  def [](index)
    # Array access
  end
  
  def []=(index, value)
    # Array assignment
  end
  
  def <<(value)
    # Append operator
  end
  
  def ==(other)
    # Equality
  end
  
  def <=>(other)
    # Spaceship operator
  end
end
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))

        vector_class = structure.classes[0]
        operator_methods = [m for m in vector_class.methods if m.get("is_operator")]
        assert len(operator_methods) >= 5

    def test_singleton_class(self, analyzer):
        """Test handling of singleton classes."""
        code = """
class MyClass
  class << self
    def class_method
      "class method"
    end
    
    private
    
    def private_class_method
      "private class method"
    end
  end
end

obj = Object.new
class << obj
  def singleton_method
    "singleton"
  end
end
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))

        # Should detect singleton class patterns
        assert any(c.is_singleton for c in structure.classes if hasattr(c, "is_singleton"))

    def test_heredoc_strings(self, analyzer):
        """Test handling of heredoc strings."""
        code = """
sql = <<-SQL
  SELECT * FROM users
  WHERE active = true
SQL

html = <<~HTML
  <div>
    <p>Hello</p>
  </div>
HTML

def method_with_heredoc
  <<-RUBY
    puts "This is Ruby code in a heredoc"
  RUBY
end
"""
        # Should handle heredocs without issues
        structure = analyzer.extract_structure(code, Path("test.rb"))
        metrics = analyzer.calculate_complexity(code, Path("test.rb"))

        assert isinstance(structure, CodeStructure)
        assert isinstance(metrics, ComplexityMetrics)

    def test_dsl_patterns(self, analyzer):
        """Test handling of Ruby DSL patterns."""
        code = """
# Rails routes DSL
Rails.application.routes.draw do
  root 'home#index'
  
  resources :users do
    member do
      get 'profile'
    end
    
    collection do
      get 'search'
    end
  end
  
  namespace :api do
    namespace :v1 do
      resources :posts
    end
  end
end

# RSpec DSL
RSpec.describe User do
  let(:user) { User.new }
  
  before do
    # setup
  end
  
  after do
    # teardown
  end
end
"""
        structure = analyzer.extract_structure(code, Path("test.rb"))
        
        # Should handle DSL patterns
        assert structure.block_count > 0
        assert isinstance(structure, CodeStructure)