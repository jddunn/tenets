"""CLI entrypoint tests for tenets-mcp."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def run_main_with_args(args: list[str]):
    """Helper to invoke main() with patched sys.argv."""
    from tenets.mcp import server as server_module

    with patch.object(sys, "argv", ["tenets-mcp"] + args):
        server_module.main()


def fake_server():
    """Create a fake server object with a recording run() method."""
    called = {"args": None}

    class _Fake:
        def run(self, **kwargs):
            called["args"] = kwargs

    return _Fake(), called


def test_cli_version_prints_and_exits(capsys):
    """tenets-mcp --version prints version and exits."""
    from tenets import __version__
    from tenets.mcp import server as server_module

    with patch.object(sys, "argv", ["tenets-mcp", "--version"]):
        with pytest.raises(SystemExit):
            server_module.main()

    out = capsys.readouterr().out.strip()
    assert f"tenets-mcp v{__version__}" in out


@pytest.mark.parametrize(
    "transport,host,port", [("stdio", "127.0.0.1", 8080), ("sse", "0.0.0.0", 8081), ("http", "0.0.0.0", 9090)]
)
def test_cli_transports_parsed(transport, host, port):
    """CLI passes transport/host/port to server.run()."""
    # Patch create_server to avoid starting a real server
    from tenets.mcp import server as server_module

    fake, called = fake_server()
    with patch.object(server_module, "create_server", return_value=fake):
        with patch.object(sys, "argv", ["tenets-mcp", "--transport", transport, "--host", host, "--port", str(port)]):
            server_module.main()

    assert called["args"] is not None
    assert called["args"]["transport"] == transport
    assert called["args"]["host"] == host
    assert called["args"]["port"] == port


