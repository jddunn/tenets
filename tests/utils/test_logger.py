"""Tests for logging utilities."""

import logging
import os
from unittest.mock import Mock, patch

from tenets.utils.logger import _configure_root, get_logger


class TestLogger:
    """Test suite for logging utilities."""

    def setup_method(self):
        """Reset logging configuration before each test."""
        # Reset the configured flag
        import tenets.utils.logger

        tenets.utils.logger._CONFIGURED = False

        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def teardown_method(self):
        """Clean up after each test."""
        # Reset environment variable
        if "TENETS_LOG_LEVEL" in os.environ:
            del os.environ["TENETS_LOG_LEVEL"]

        # Reset configured flag
        import tenets.utils.logger

        tenets.utils.logger._CONFIGURED = False

    def test_get_logger_default(self):
        """Test getting logger with defaults."""
        logger = get_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "tenets"
        assert logger.level == logging.INFO

    def test_get_logger_with_name(self):
        """Test getting logger with specific name."""
        logger = get_logger("tenets.test.module")

        assert logger.name == "tenets.test.module"

    def test_get_logger_with_level(self):
        """Test getting logger with specific level."""
        logger = get_logger(level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_environment_variable_debug(self):
        """Test LOG_LEVEL environment variable set to DEBUG."""
        os.environ["TENETS_LOG_LEVEL"] = "DEBUG"

        logger = get_logger()

        assert logger.level == logging.DEBUG

    def test_environment_variable_warning(self):
        """Test LOG_LEVEL environment variable set to WARNING."""
        os.environ["TENETS_LOG_LEVEL"] = "WARNING"

        logger = get_logger()

        assert logger.level == logging.WARNING

    def test_environment_variable_error(self):
        """Test LOG_LEVEL environment variable set to ERROR."""
        os.environ["TENETS_LOG_LEVEL"] = "ERROR"

        logger = get_logger()

        assert logger.level == logging.ERROR

    def test_environment_variable_critical(self):
        """Test LOG_LEVEL environment variable set to CRITICAL."""
        os.environ["TENETS_LOG_LEVEL"] = "CRITICAL"

        logger = get_logger()

        assert logger.level == logging.CRITICAL

    def test_environment_variable_invalid(self):
        """Test invalid LOG_LEVEL environment variable."""
        os.environ["TENETS_LOG_LEVEL"] = "INVALID"

        logger = get_logger()

        # Should default to INFO
        assert logger.level == logging.INFO

    def test_environment_variable_case_insensitive(self):
        """Test that environment variable is case insensitive."""
        os.environ["TENETS_LOG_LEVEL"] = "dEbUg"

        logger = get_logger()

        assert logger.level == logging.DEBUG

    def test_explicit_level_overrides_env(self):
        """Test that explicit level overrides environment variable."""
        os.environ["TENETS_LOG_LEVEL"] = "DEBUG"

        logger = get_logger(level=logging.ERROR)

        assert logger.level == logging.ERROR

    @patch("tenets.utils.logger._RICH_INSTALLED", True)
    @patch("tenets.utils.logger.RichHandler")
    def test_rich_handler_when_available(self, mock_rich_handler):
        """Test that RichHandler is used when available."""
        mock_handler = Mock()
        mock_rich_handler.return_value = mock_handler

        logger = get_logger()

        # Check that RichHandler was called
        mock_rich_handler.assert_called_once()
        # Verify essential parameters are present
        _, kwargs = mock_rich_handler.call_args
        assert kwargs.get("rich_tracebacks") is True
        assert kwargs.get("show_time") is True
        assert kwargs.get("show_path") is False

    @patch("tenets.utils.logger._RICH_INSTALLED", False)
    def test_standard_handler_when_rich_unavailable(self):
        """Test that standard StreamHandler is used when Rich is unavailable."""
        logger = get_logger()

        root_logger = logging.getLogger()
        handlers = root_logger.handlers

        # Should have at least one handler
        assert len(handlers) > 0

        # Should be StreamHandler
        assert any(isinstance(h, logging.StreamHandler) for h in handlers)

    def test_configure_root_idempotent(self):
        """Test that root configuration is idempotent."""
        import tenets.utils.logger

        # First configuration
        _configure_root(logging.INFO)
        assert tenets.utils.logger._CONFIGURED == True

        # Get handler count
        root_logger = logging.getLogger()
        handler_count = len(root_logger.handlers)

        # Second configuration should not add more handlers
        _configure_root(logging.DEBUG)

        assert len(root_logger.handlers) == handler_count

    def test_multiple_loggers_share_config(self):
        """Test that multiple loggers share configuration."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        # Both should be configured
        assert logger1.level == logging.INFO
        assert logger2.level == logging.INFO

        # Change environment and get new logger
        os.environ["TENETS_LOG_LEVEL"] = "DEBUG"
        logger3 = get_logger("module3")

        assert logger3.level == logging.DEBUG

    def test_logger_propagation(self):
        """Test that logger propagation is enabled."""
        logger = get_logger("tenets.sub.module")

        assert logger.propagate == True

    def test_logging_output(self, caplog):
        """Test actual logging output."""
        logger = get_logger("test", level=logging.DEBUG)

        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

        assert "Debug message" in caplog.text
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text
        assert "Critical message" in caplog.text

    def test_logger_with_exception(self, caplog):
        """Test logging with exception information."""
        with caplog.at_level(logging.ERROR):
            logger = get_logger("test", level=logging.ERROR)

            try:
                raise ValueError("Test exception")
            except ValueError:
                logger.error("Error occurred", exc_info=True)

            assert "Error occurred" in caplog.text
            assert "ValueError: Test exception" in caplog.text
            assert "Traceback" in caplog.text

    @patch("tenets.utils.logger._RICH_INSTALLED", False)
    def test_format_string_without_rich(self):
        """Test format string when Rich is not available."""
        import tenets.utils.logger

        tenets.utils.logger._CONFIGURED = False

        logger = get_logger("test.module")

        # Check that format includes necessary fields
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]
        formatter = handler.formatter

        # Format string should include time, level, name, message
        assert formatter._fmt is not None
        assert "asctime" in formatter._fmt
        assert "levelname" in formatter._fmt
        assert "name" in formatter._fmt
        assert "message" in formatter._fmt

    @patch("tenets.utils.logger._RICH_INSTALLED", True)
    def test_format_string_with_rich(self):
        """Test format string when Rich is available."""
        import tenets.utils.logger

        tenets.utils.logger._CONFIGURED = False

        with patch("tenets.utils.logger.RichHandler"):
            logger = get_logger("test.module")

            root_logger = logging.getLogger()
            handler = root_logger.handlers[0]
            formatter = handler.formatter

            # Rich uses simpler format
            if formatter and hasattr(formatter, "_fmt"):
                assert "message" in formatter._fmt

    def test_child_logger_inheritance(self):
        """Test that child loggers inherit from parent."""
        # Create parent logger
        parent = get_logger("tenets", level=logging.WARNING)

        # Create child logger without explicit level
        child = get_logger("tenets.child")

        # Child should inherit parent's effective level
        assert child.getEffectiveLevel() == logging.WARNING

    def test_logger_name_none(self):
        """Test logger with None name."""
        logger = get_logger(None)

        assert logger.name == "tenets"

    def test_concurrent_logger_creation(self):
        """Test concurrent logger creation."""
        import threading

        loggers = []

        def create_logger(name):
            logger = get_logger(f"test.{name}")
            loggers.append(logger)

        threads = []
        for i in range(10):
            t = threading.Thread(target=create_logger, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All loggers should be created successfully
        assert len(loggers) == 10

        # All should have correct names
        for i, logger in enumerate(loggers):
            assert "test." in logger.name

    def test_logger_levels_enum(self):
        """Test that all logging levels work correctly."""
        levels = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]

        for level_int, level_str in levels:
            os.environ["TENETS_LOG_LEVEL"] = level_str

            # Reset configuration
            import tenets.utils.logger

            tenets.utils.logger._CONFIGURED = False

            logger = get_logger(f"test_{level_str}")
            assert logger.level == level_int

    def test_logger_with_custom_handler(self):
        """Test adding custom handler to logger."""
        logger = get_logger("custom")

        # Add custom handler
        custom_handler = logging.StreamHandler()
        custom_handler.setLevel(logging.CRITICAL)
        logger.addHandler(custom_handler)

        # Logger should have the custom handler
        assert custom_handler in logger.handlers

    def test_effective_level_calculation(self):
        """Test effective level calculation for hierarchical loggers."""
        # Set root logger to WARNING
        root = logging.getLogger()
        root.setLevel(logging.WARNING)

        # Create tenets logger with INFO
        tenets_logger = get_logger("tenets", level=logging.INFO)

        # Create child without explicit level
        child = logging.getLogger("tenets.child")

        # Child's effective level should be INFO (from parent)
        assert child.getEffectiveLevel() == logging.INFO

    @patch.dict(os.environ, {}, clear=True)
    def test_no_environment_variable(self):
        """Test behavior when environment variable is not set."""
        # Ensure TENETS_LOG_LEVEL is not set
        assert "TENETS_LOG_LEVEL" not in os.environ

        logger = get_logger()

        # Should default to INFO
        assert logger.level == logging.INFO
