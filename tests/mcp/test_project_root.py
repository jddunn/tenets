"""tenets-mcp pins cwd to the project root so .tenets.yml is discovered."""
import os

from tenets.mcp.server import _chdir_to_project_root


def _real(p):
    return os.path.realpath(str(p))


def test_chdir_honors_tenets_project_root(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.setenv("TENETS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    _chdir_to_project_root()
    assert _real(os.getcwd()) == _real(tmp_path)


def test_chdir_honors_claude_project_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("TENETS_PROJECT_ROOT", raising=False)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    _chdir_to_project_root()
    assert _real(os.getcwd()) == _real(tmp_path)


def test_tenets_project_root_takes_precedence(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    monkeypatch.setenv("TENETS_PROJECT_ROOT", str(a))
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(b))
    monkeypatch.chdir(tmp_path)
    _chdir_to_project_root()
    assert _real(os.getcwd()) == _real(a)


def test_chdir_noop_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("TENETS_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    _chdir_to_project_root()
    assert _real(os.getcwd()) == _real(tmp_path)


def test_chdir_ignores_invalid_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.setenv("TENETS_PROJECT_ROOT", str(tmp_path / "does-not-exist"))
    monkeypatch.chdir(tmp_path)
    _chdir_to_project_root()
    assert _real(os.getcwd()) == _real(tmp_path)
