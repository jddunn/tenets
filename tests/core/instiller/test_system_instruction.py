"""Tests for system instruction injection behavior.

Covers:
- Enable/disable and empty guard
- Placement options and formatting
- Session once-per-session vs no-session behavior
- Distill integration path metadata
"""

from pathlib import Path
import tempfile

from tenets.config import TenetsConfig
from tenets.core.instiller.instiller import Instiller
from tenets.models.context import ContextResult
from tenets import Tenets


def make_config(tmpdir: str) -> TenetsConfig:
    cfg = TenetsConfig()
    cfg.cache.directory = Path(tmpdir)
    # Minimal tenet config defaults are okay; we focus on system instruction
    cfg.tenet.system_instruction_enabled = True
    cfg.tenet.system_instruction = "Be concise. Prefer code examples."
    cfg.tenet.system_instruction_position = "top"
    cfg.tenet.system_instruction_format = "markdown"
    cfg.tenet.system_instruction_once_per_session = True
    return cfg


def test_injection_disabled_guard():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp)
        cfg.tenet.system_instruction_enabled = False
        inst = Instiller(cfg)
        text, meta = inst.inject_system_instruction("Hello", session="s1")
        assert text == "Hello"
        assert meta["system_instruction_injected"] is False
        assert meta.get("reason") == "disabled_or_empty"


def test_injection_empty_guard():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp)
        cfg.tenet.system_instruction = None
        inst = Instiller(cfg)
        text, meta = inst.inject_system_instruction("Hello", session="s1")
        assert text == "Hello"
        assert meta["system_instruction_injected"] is False
        assert meta.get("reason") == "disabled_or_empty"


def test_once_per_session_behavior():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp)
        inst = Instiller(cfg)
        original = "# Title\nBody"
        # First inject in session -> should inject
        t1, m1 = inst.inject_system_instruction(original, session="abc")
        assert m1["system_instruction_injected"] is True
        # Second inject in same session -> should skip
        t2, m2 = inst.inject_system_instruction(original, session="abc")
        assert m2.get("reason") == "already_injected_in_session"
        assert t2 == original
        # New session -> inject again
        t3, m3 = inst.inject_system_instruction(original, session="xyz")
        assert m3["system_instruction_injected"] is True


def test_every_distill_without_session():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp)
        cfg.tenet.system_instruction_once_per_session = True
        inst = Instiller(cfg)
        original = "hello"
        # No session: should inject every time
        a1, m1 = inst.inject_system_instruction(original, session=None)
        a2, m2 = inst.inject_system_instruction(original, session=None)
        assert m1["system_instruction_injected"] is True
        assert m2["system_instruction_injected"] is True


def test_placement_after_header():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp)
        cfg.tenet.system_instruction_position = "after_header"
        inst = Instiller(cfg)
        content = "# Header\n\nSome text\n"
        modified, meta = inst.inject_system_instruction(content, session="s")
        assert meta["system_instruction_injected"] is True
        assert "Header" in modified
        # Ensure instruction appears after first header line
        assert modified.splitlines()[1].strip() != "# Header"


def test_format_comment_plain_xml():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp)
        inst = Instiller(cfg)
        base = "content"
        # comment
        cfg.tenet.system_instruction_format = "comment"
        t1, _ = inst.inject_system_instruction(base)
        assert "<!--" in t1
        # plain
        cfg.tenet.system_instruction_format = "plain"
        t2, _ = inst.inject_system_instruction(base)
        assert "<!--" not in t2 and "<system-instruction>" not in t2
        # xml
        cfg.tenet.system_instruction_format = "xml"
        t3, _ = inst.inject_system_instruction(base)
        assert "<system-instruction>" in t3


def test_distill_integration_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config(tmp)
        ten = Tenets(cfg)
        # Create a simple ContextResult by bypassing heavy distiller: call inject directly and wrap
        result = ContextResult(files=[], context="content", format="markdown", metadata={})
        # Manually simulate distill’s injection hook
        modified, meta = ten.instiller.inject_system_instruction(result.context, session="s")
        if meta.get("system_instruction_injected"):
            result = ContextResult(
                files=result.files,
                context=modified,
                format=result.format,
                metadata={**result.metadata, "system_instruction": meta},
            )
        assert "system_instruction" in result.metadata


def test_persistence_once_per_session_across_instances():
    """System instruction once-per-session should persist across Instiller instances."""
    with tempfile.TemporaryDirectory() as tmp:
        # First instance injects for the session
        cfg1 = make_config(tmp)
        inst1 = Instiller(cfg1)
        base = "# Header\n\nBody"
        t1, m1 = inst1.inject_system_instruction(base, session="sess")
        assert m1["system_instruction_injected"] is True

        # New instance with same cache directory should see it's already injected
        cfg2 = make_config(tmp)
        inst2 = Instiller(cfg2)
        t2, m2 = inst2.inject_system_instruction(base, session="sess")
        assert m2.get("reason") == "already_injected_in_session"
        assert t2 == base
