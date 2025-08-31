"""Tests for content transformation utilities."""

from tenets.core.distiller.transform import (
    apply_transformations,
    condense_whitespace,
    detect_language_from_extension,
    strip_comments,
)


class TestLanguageDetection:
    """Test language detection from file extensions."""

    def test_detect_python(self):
        """Test Python detection."""
        assert detect_language_from_extension("test.py") == "python"
        assert detect_language_from_extension("app.pyw") == "python"
        assert detect_language_from_extension("/path/to/file.py") == "python"

    def test_detect_javascript(self):
        """Test JavaScript detection."""
        assert detect_language_from_extension("app.js") == "javascript"
        assert detect_language_from_extension("component.jsx") == "javascript"

    def test_detect_typescript(self):
        """Test TypeScript detection."""
        assert detect_language_from_extension("app.ts") == "typescript"
        assert detect_language_from_extension("component.tsx") == "typescript"

    def test_detect_java(self):
        """Test Java detection."""
        assert detect_language_from_extension("Main.java") == "java"

    def test_detect_c_cpp(self):
        """Test C/C++ detection."""
        assert detect_language_from_extension("main.c") == "c"
        assert detect_language_from_extension("main.cc") == "cpp"
        assert detect_language_from_extension("main.cpp") == "cpp"

    def test_detect_other_languages(self):
        """Test detection of other languages."""
        assert detect_language_from_extension("app.cs") == "csharp"
        assert detect_language_from_extension("main.go") == "go"
        assert detect_language_from_extension("lib.rs") == "rust"
        assert detect_language_from_extension("index.php") == "php"
        assert detect_language_from_extension("app.rb") == "ruby"
        assert detect_language_from_extension("script.sh") == "shell"
        assert detect_language_from_extension("script.bash") == "bash"
        assert detect_language_from_extension("query.sql") == "sql"
        assert detect_language_from_extension("App.kt") == "kotlin"
        assert detect_language_from_extension("Main.scala") == "scala"
        assert detect_language_from_extension("App.swift") == "swift"
        assert detect_language_from_extension("Main.hs") == "haskell"
        assert detect_language_from_extension("script.lua") == "lua"

    def test_detect_unknown(self):
        """Test unknown extensions."""
        assert detect_language_from_extension("file.unknown") == ""
        assert detect_language_from_extension("noextension") == ""
        assert detect_language_from_extension("") == ""

    def test_case_insensitive(self):
        """Test case-insensitive detection."""
        assert detect_language_from_extension("TEST.PY") == "python"
        assert detect_language_from_extension("App.JS") == "javascript"


class TestStripComments:
    """Test comment stripping functionality."""

    def test_strip_python_comments(self):
        """Test stripping Python comments."""
        code = '''# This is a comment
def hello():
    """Docstring should be removed."""
    print("hello")  # inline comment
    # Another comment
    return'''

        result = strip_comments(code, "python")

        assert "# This is a comment" not in result
        assert "# inline comment" not in result
        assert "# Another comment" not in result
        assert 'print("hello")' in result
        assert "def hello():" in result

    def test_strip_javascript_comments(self):
        """Test stripping JavaScript comments."""
        code = """// Single line comment
function hello() {
    /* Multi-line
       comment */
    console.log("hello"); // inline
    return;
}"""

        result = strip_comments(code, "javascript")

        assert "// Single line comment" not in result
        assert "/* Multi-line" not in result
        assert "// inline" not in result
        assert 'console.log("hello")' in result
        assert "function hello()" in result

    def test_strip_multiline_comments(self):
        """Test stripping multi-line comments."""
        code = """function test() {
    /* This is a
       multi-line comment
       that spans several lines */
    return true;
}"""

        result = strip_comments(code, "javascript")

        assert "/*" not in result
        assert "*/" not in result
        assert "multi-line comment" not in result
        assert "return true;" in result

    def test_preserve_code_after_comment_marker(self):
        """Test that code after comment marker on same line is preserved."""
        code = """def test():
    x = 5  # Set x to 5
    y = 10 # Set y to 10
    return x + y"""

        result = strip_comments(code, "python")

        assert "x = 5" in result
        assert "y = 10" in result
        assert "# Set x to 5" not in result

    def test_strip_comments_empty_input(self):
        """Test stripping comments from empty input."""
        assert strip_comments("", "python") == ""
        assert strip_comments("", "javascript") == ""

    def test_strip_comments_no_language(self):
        """Test stripping with no language."""
        code = "# Comment\ncode here"
        assert strip_comments(code, "") == code

    def test_strip_comments_unknown_language(self):
        """Test stripping with unknown language."""
        code = "// Comment\ncode here"
        assert strip_comments(code, "unknown_lang") == code

    def test_strip_comments_preserves_strings(self):
        """Test that comment markers in strings are preserved."""
        code = """def test():
    url = "http://example.com"  # This comment should be removed
    path = "C://Users//file.txt"
    return url"""

        result = strip_comments(code, "python")

        assert "http://example.com" in result
        assert "C://Users//file.txt" in result
        assert "# This comment should be removed" not in result

    def test_strip_comments_safeguard(self):
        """Test safeguard against removing too much content."""
        # Create code where most lines are comments
        code = """# Comment 1
# Comment 2
# Comment 3
# Comment 4
# Comment 5
# Comment 6
# Comment 7
# Comment 8
actual_code = 1
# Comment 9"""

        # Should return original if >60% would be removed
        result = strip_comments(code, "python")
        assert result == code  # Safeguard triggered

    def test_strip_sql_comments(self):
        """Test stripping SQL comments."""
        code = """-- This is a SQL comment
SELECT * FROM users
/* Multi-line comment
   in SQL */
WHERE id = 1; -- inline comment"""

        result = strip_comments(code, "sql")

        assert "-- This is a SQL comment" not in result
        assert "SELECT * FROM users" in result
        assert "/* Multi-line comment" not in result
        assert "WHERE id = 1;" in result


class TestCondenseWhitespace:
    """Test whitespace condensing functionality."""

    def test_condense_multiple_blank_lines(self):
        """Test condensing multiple blank lines."""
        text = "line1\n\n\n\nline2\n\n\n\n\nline3"
        result = condense_whitespace(text)

        assert result == "line1\n\nline2\n\nline3\n"

    def test_trim_trailing_spaces(self):
        """Test trimming trailing spaces."""
        text = "line1   \nline2  \t\nline3\t  "
        result = condense_whitespace(text)

        assert "line1   " not in result
        assert "line2  \t" not in result
        assert result == "line1\nline2\nline3\n"

    def test_ensure_final_newline(self):
        """Test ensuring final newline."""
        text = "line1\nline2"
        result = condense_whitespace(text)

        assert result.endswith("\n")
        assert result == "line1\nline2\n"

    def test_condense_empty_input(self):
        """Test condensing empty input."""
        assert condense_whitespace("") == ""

    def test_condense_preserves_single_blank_lines(self):
        """Test that single blank lines are preserved."""
        text = "line1\n\nline2\n\nline3"
        result = condense_whitespace(text)

        assert result == "line1\n\nline2\n\nline3\n"

    def test_condense_complex_whitespace(self):
        """Test condensing complex whitespace patterns."""
        text = """def function():



    x = 1



    y = 2

    return x + y"""

        result = condense_whitespace(text)

        # Should have at most 2 consecutive newlines
        assert "\n\n\n" not in result
        assert "x = 1" in result
        assert "y = 2" in result


class TestApplyTransformations:
    """Test combined transformations."""

    def test_apply_no_transformations(self):
        """Test with no transformations enabled."""
        code = "# Comment\ndef test():\n\n\n    pass"
        result, stats = apply_transformations(code, "python", remove_comments=False, condense=False)

        assert result == code
        assert stats["changed"] == False
        assert stats["removed_comment_lines"] == 0
        assert stats["condensed_blank_runs"] == 0

    def test_apply_remove_comments_only(self):
        """Test removing comments only."""
        code = "# Comment\ndef test():\n    pass  # inline"
        result, stats = apply_transformations(code, "python", remove_comments=True, condense=False)

        assert "# Comment" not in result
        assert "# inline" not in result
        assert "def test():" in result
        assert stats["changed"] == True
        assert stats["removed_comment_lines"] > 0

    def test_apply_condense_only(self):
        """Test condensing only."""
        code = "def test():\n\n\n\n    pass"
        result, stats = apply_transformations(code, "python", remove_comments=False, condense=True)

        assert "\n\n\n" not in result
        assert "def test():" in result
        assert stats["changed"] == True
        assert stats["condensed_blank_runs"] > 0

    def test_apply_both_transformations(self):
        """Test applying both transformations."""
        code = """# Header comment


def test():
    \"\"\"Docstring.\"\"\"


    x = 1  # Set x


    return x"""

        result, stats = apply_transformations(code, "python", remove_comments=True, condense=True)

        assert "# Header comment" not in result
        assert "# Set x" not in result
        assert "\n\n\n" not in result
        assert "x = 1" in result
        assert stats["changed"] == True
        assert stats["removed_comment_lines"] > 0
        assert stats["condensed_blank_runs"] > 0

    def test_apply_transformations_order(self):
        """Test that transformations are applied in correct order."""
        code = """# Comment line


    # Another comment


def test():
    pass"""

        # Comments removed first, then whitespace condensed
        result, stats = apply_transformations(code, "python", remove_comments=True, condense=True)

        # After removing comments, blank lines should be condensed
        assert result.strip() == "def test():\n    pass"
        assert stats["changed"] == True

    def test_apply_transformations_with_unknown_language(self):
        """Test transformations with unknown language."""
        code = "// Comment\n\n\ncode here"
        result, stats = apply_transformations(code, "unknown", remove_comments=True, condense=True)

        # Comments not removed (unknown language), but whitespace condensed
        assert "// Comment" in result
        assert "\n\n\n" not in result
        assert stats["removed_comment_lines"] == 0
        assert stats["condensed_blank_runs"] > 0

    def test_apply_transformations_empty_input(self):
        """Test transformations on empty input."""
        result, stats = apply_transformations("", "python", remove_comments=True, condense=True)

        assert result == ""
        assert stats["changed"] == False

    def test_apply_transformations_preserves_functionality(self):
        """Test that transformations preserve code functionality."""
        code = '''def calculate(x, y):
    """Calculate sum of x and y."""
    # Validate inputs
    if x < 0:  # Check x
        x = 0


    # Calculate result
    result = x + y  # Add them



    return result'''

        result, stats = apply_transformations(code, "python", remove_comments=True, condense=True)

        # Check that essential code structure is preserved
        assert "def calculate(x, y):" in result
        assert "if x < 0:" in result
        assert "x = 0" in result
        assert "result = x + y" in result
        assert "return result" in result

        # Check that transformations were applied
        assert '"""Calculate sum' not in result
        assert "# Validate inputs" not in result
        assert "\n\n\n" not in result
