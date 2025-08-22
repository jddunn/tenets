"""Tests for enhanced report generation features.

Tests cover:
- README detection and summarization
- Root path display
- Excluded files section
- Automatic report naming
- Enhanced data rendering
"""

from unittest.mock import MagicMock

from tenets.core.reporting.generator import ReportConfig, ReportGenerator


class TestReadmeDetection:
    """Test README detection and summarization."""

    def test_find_readme_basic(self, tmp_path):
        """Test basic README detection."""
        # Create a README file
        readme_path = tmp_path / "README.md"
        readme_content = "# Test Project\n\nThis is a test project."
        readme_path.write_text(readme_content)

        generator = ReportGenerator(MagicMock())
        data = {"root_path": str(tmp_path)}

        readme_info = generator._find_readme(data)

        assert readme_info is not None
        assert readme_info["name"] == "README.md"
        assert readme_info["content"] == readme_content
        assert readme_info["original_length"] == len(readme_content)
        assert readme_info["condensed_by"] == 0

    def test_find_readme_long_content(self, tmp_path):
        """Test README truncation for long content."""
        # Create a long README
        readme_path = tmp_path / "README.md"
        readme_content = "# Test Project\n\n" + ("This is a line.\n" * 200)
        readme_path.write_text(readme_content)

        generator = ReportGenerator(MagicMock())
        data = {"root_path": str(tmp_path)}

        readme_info = generator._find_readme(data)

        assert readme_info is not None
        assert readme_info["condensed_by"] > 0
        assert "... [README truncated for brevity]" in readme_info["content"]
        assert readme_info["displayed_length"] < readme_info["original_length"]

    def test_find_readme_different_formats(self, tmp_path):
        """Test detection of different README formats."""
        formats = ["README.md", "README.rst", "README.txt", "README"]

        for format_name in formats:
            test_path = tmp_path / f"test_{format_name}"
            test_path.mkdir()

            readme_path = test_path / format_name
            readme_path.write_text(f"Content for {format_name}")

            generator = ReportGenerator(MagicMock())
            data = {"root_path": str(test_path)}

            readme_info = generator._find_readme(data)

            assert readme_info is not None
            assert readme_info["name"] == format_name

    def test_find_readme_missing(self, tmp_path):
        """Test when no README exists."""
        generator = ReportGenerator(MagicMock())
        data = {"root_path": str(tmp_path)}

        readme_info = generator._find_readme(data)

        assert readme_info is None

    def test_create_readme_section(self):
        """Test README section creation."""
        generator = ReportGenerator(MagicMock())

        readme_info = {
            "name": "README.md",
            "content": "# Test Project",
            "lines": 10,
            "original_length": 500,
            "displayed_length": 500,
            "condensed_by": 0,
        }

        section = generator._create_readme_section(readme_info)

        assert section.id == "readme"
        assert section.title == "Project README"
        assert section.collapsible is True
        assert section.collapsed is True
        assert "README.md" in section.content[0]
        assert "10 lines" in section.content[0]

    def test_create_readme_section_condensed(self):
        """Test README section with condensed content."""
        generator = ReportGenerator(MagicMock())

        readme_info = {
            "name": "README.md",
            "content": "# Truncated content...",
            "lines": 100,
            "original_length": 5000,
            "displayed_length": 2000,
            "condensed_by": 60.0,
        }

        section = generator._create_readme_section(readme_info)

        assert "Condensed by 60.0%" in section.content[1]
        assert "5,000 characters" in section.content[0]


class TestExcludedFilesSection:
    """Test excluded files section generation."""

    def test_create_excluded_files_section_empty(self):
        """Test excluded files section with no data."""
        generator = ReportGenerator(MagicMock())
        data = {}
        config = ReportConfig()

        section = generator._create_excluded_files_section(data, config)

        assert section.id == "excluded_files"
        assert section.title == "Excluded Files"
        assert section.collapsible is True
        assert section.collapsed is True
        assert section.metrics["Excluded Files"] == 0
        assert section.metrics["Ignored Patterns"] == 0

    def test_create_excluded_files_section_with_patterns(self):
        """Test excluded files section with ignore patterns."""
        generator = ReportGenerator(MagicMock())
        data = {"ignored_patterns": ["*.pyc", "__pycache__", ".git/*", "node_modules/*"]}
        config = ReportConfig()

        section = generator._create_excluded_files_section(data, config)

        assert section.metrics["Ignored Patterns"] == 4
        assert "### Ignored Patterns" in section.content
        assert "*.pyc" in str(section.content)
        assert "__pycache__" in str(section.content)

    def test_create_excluded_files_section_with_files(self):
        """Test excluded files section with file list."""
        generator = ReportGenerator(MagicMock())

        # Create a list of excluded files with different extensions
        excluded_files = (
            [f"test{i}.pyc" for i in range(15)]
            + [f"cache{i}.tmp" for i in range(8)]
            + ["file_without_ext", "another_file"]
        )

        data = {"excluded_files": excluded_files}
        config = ReportConfig()

        section = generator._create_excluded_files_section(data, config)

        assert section.metrics["Excluded Files"] == len(excluded_files)
        # Check for the header with file count
        assert f"### Excluded Files ({len(excluded_files)} files)" in section.content
        assert ".pyc" in str(section.content)
        assert ".tmp" in str(section.content)
        assert "no extension" in str(section.content)

        # Check for truncation message
        content_str = str(section.content)
        assert "... and 5 more" in content_str  # For .pyc files (15 total, showing 10)

    def test_create_excluded_files_section_large_list(self):
        """Test excluded files section with large file list."""
        generator = ReportGenerator(MagicMock())

        # Create a large list of files
        excluded_files = [f"file{i}.tmp" for i in range(200)]

        data = {"excluded_files": excluded_files}
        config = ReportConfig()

        section = generator._create_excluded_files_section(data, config)

        assert section.metrics["Excluded Files"] == 200
        assert "... and 100 more excluded files" in str(section.content)


class TestRootPathDisplay:
    """Test root path display in reports."""

    def test_summary_section_with_root_path(self):
        """Test that root path is displayed in summary."""
        generator = ReportGenerator(MagicMock())
        generator.metadata = {"analysis_summary": {"health_score": 85}}

        data = {"root_path": "/home/user/projects/myproject", "metrics": {}}

        section = generator._create_summary_section(data)

        assert section.id == "summary"
        # Check that root path is in content
        content_str = str(section.content)
        assert "Project Path:" in content_str
        assert "myproject" in content_str

    def test_summary_section_no_root_path(self):
        """Test summary when no root path provided."""
        generator = ReportGenerator(MagicMock())
        generator.metadata = {"analysis_summary": {"health_score": 85}}

        data = {"metrics": {}}

        section = generator._create_summary_section(data)

        content_str = str(section.content)
        assert "Project Path:" not in content_str

    def test_summary_section_current_directory(self):
        """Test summary with current directory as root."""
        generator = ReportGenerator(MagicMock())
        generator.metadata = {"analysis_summary": {"health_score": 85}}

        data = {"root_path": ".", "metrics": {}}

        section = generator._create_summary_section(data)

        content_str = str(section.content)
        assert "Project Path:" not in content_str  # Don't show for current dir


class TestEnhancedDataRendering:
    """Test enhanced data rendering in reports."""

    def test_complexity_section_with_real_data(self):
        """Test complexity section with actual data structure."""
        generator = ReportGenerator(MagicMock())

        complexity_data = {
            "total_files": 318,
            "total_functions": 781,
            "total_classes": 129,
            "avg_complexity": 1.0,
            "max_complexity": 1,
            "complex_functions": 0,
            "complexity_distribution": {
                "simple (1-5)": 781,
                "moderate (6-10)": 0,
                "complex (11-20)": 0,
                "very complex (21+)": 0,
            },
        }

        config = ReportConfig(include_charts=False)
        section = generator._create_complexity_section(complexity_data, config)

        assert section.metrics["Average Complexity"] == "1.00"
        assert section.metrics["Maximum Complexity"] == 1
        assert section.metrics["Complex Functions"] == 0
        assert section.metrics["Total Functions"] == 781

    def test_hotspot_section_with_empty_data(self):
        """Test hotspot section handles empty data gracefully."""
        generator = ReportGenerator(MagicMock())

        hotspot_data = {
            "total_hotspots": 242,
            "critical_count": 0,
            "high_count": 0,
            "files_analyzed": 0,
        }

        config = ReportConfig(include_charts=False)
        section = generator._create_hotspots_section(hotspot_data, config)

        assert section.metrics["Total Hotspots"] == 242
        assert section.metrics["Critical"] == 0
        assert section.metrics["High Risk"] == 0
        assert section.metrics["Files Analyzed"] == 0

    def test_file_overview_with_language_data(self):
        """Test file overview section with language breakdown."""
        generator = ReportGenerator(MagicMock())

        data = {
            "metrics": {
                "languages": {
                    "python": {
                        "files": 199,
                        "lines": 117689,
                        "avg_file_size": 591,
                        "functions": 149,
                        "classes": 129,
                    },
                    "html": {
                        "files": 26,
                        "lines": 59505,
                        "avg_file_size": 2289,
                        "functions": 0,
                        "classes": 0,
                    },
                },
                "largest_files": [
                    {"name": "index.html", "lines": 54699, "language": "html"},
                    {"name": "hotspots.py", "lines": 1861, "language": "python"},
                ],
            }
        }

        config = ReportConfig(include_charts=False)
        section = generator._create_file_overview_section(data, config)

        assert section.id == "file_overview"
        assert len(section.tables) > 0

        # Check language table
        lang_table = section.tables[0]
        assert "Language" in lang_table["headers"]
        assert "Files" in lang_table["headers"]
        assert "Lines" in lang_table["headers"]


class TestReportGeneration:
    """Test complete report generation with new features."""

    def test_generate_full_report_with_enhancements(self, tmp_path):
        """Test generating a complete report with all enhancements."""
        # Create a README
        readme_path = tmp_path / "README.md"
        readme_path.write_text("# Test Project\n\nA test project for unit tests.")

        # Setup generator
        config = MagicMock()
        generator = ReportGenerator(config)

        # Prepare test data
        data = {
            "root_path": str(tmp_path),
            "total_files": 100,
            "total_lines": 10000,
            "metrics": {
                "total_files": 100,
                "total_lines": 10000,
                "languages": {"python": {"files": 80, "lines": 8000, "avg_file_size": 100}},
            },
            "complexity": {
                "avg_complexity": 3.5,
                "max_complexity": 15,
                "complex_functions": 5,
                "total_functions": 100,
            },
            "excluded_files": ["test.pyc", "cache.tmp"],
            "ignored_patterns": ["*.pyc", "__pycache__"],
        }

        report_config = ReportConfig(
            title="Test Report", format="json", include_charts=False, include_summary=True
        )

        # Generate report
        output_path = tmp_path / "report.json"
        generator.generate(data, output_path, report_config)

        # Verify sections were created
        assert len(generator.sections) > 0

        # Check for specific sections
        section_ids = [s.id for s in generator.sections]
        assert "summary" in section_ids
        assert "file_overview" in section_ids
        assert "readme" in section_ids
        assert "excluded_files" in section_ids
        assert "complexity" in section_ids

        # Verify root path is in summary
        summary_section = next(s for s in generator.sections if s.id == "summary")
        content_str = str(summary_section.content)
        assert tmp_path.name in content_str or str(tmp_path) in content_str
