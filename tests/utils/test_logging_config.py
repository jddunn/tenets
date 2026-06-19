"""configure_logging: keep MCP stderr quiet, route detailed logs to a file."""
import logging

from tenets.utils import logger as tlog


def test_min_level_filter_drops_below_threshold():
    f = tlog._MinLevelFilter(logging.WARNING)
    info = logging.LogRecord("n", logging.INFO, "", 0, "i", None, None)
    warn = logging.LogRecord("n", logging.WARNING, "", 0, "w", None, None)
    assert f.filter(info) is False
    assert f.filter(warn) is True


def test_configure_logging_routes_info_to_file(tmp_path):
    log_file = str(tmp_path / "tenets.log")
    tlog.configure_logging(stderr_level=logging.WARNING, file_level=logging.INFO, log_file=log_file)
    logging.getLogger("tenets.x").info("hello-info-line")
    logging.getLogger("tenets.x").warning("hello-warn-line")
    for h in logging.getLogger().handlers:
        h.flush()
    content = open(log_file).read()
    assert "hello-info-line" in content
    assert "hello-warn-line" in content


def test_stderr_handler_blocks_info_even_after_level_lowering(tmp_path):
    log_file = str(tmp_path / "tenets.log")
    tlog.configure_logging(stderr_level=logging.WARNING, file_level=logging.INFO, log_file=log_file)
    # a later get_logger lowers root/handler levels; the filter must still hold the WARNING floor
    tlog.get_logger("tenets.something", level=logging.DEBUG)
    root = logging.getLogger()
    stderr_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert stderr_handlers, "expected a stderr StreamHandler"
    sh = stderr_handlers[0]
    info_rec = logging.LogRecord("n", logging.INFO, "", 0, "x", None, None)
    warn_rec = logging.LogRecord("n", logging.WARNING, "", 0, "y", None, None)
    assert not sh.filter(info_rec)  # INFO still blocked from stderr
    assert sh.filter(warn_rec)  # WARNING still allowed through
