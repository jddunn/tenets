"""_analyze_corpus is prompt-independent and O(N^2) (import graph) — memoize it so a
long-lived ranker (the MCP server) doesn't recompute it on every query."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tenets.config import TenetsConfig
from tenets.core.ranking.ranker import RelevanceRanker
from tenets.models.analysis import FileAnalysis


def _fa(path, content, h, imports=()):
    return FileAnalysis(
        path=path, content=content, language="python", size=len(content),
        imports=list(imports), hash=h,
    )


def _ranker():
    cfg = TenetsConfig()
    cfg.cache.index_enabled = False  # isolate the memo from the disk corpus index
    return RelevanceRanker(cfg)


def test_analyze_corpus_is_memoized_for_unchanged_files():
    r = _ranker()
    files = [_fa("a.py", "import os\nclass FileHandler: pass", "h1"),
             _fa("b.py", "def write_file(): pass", "h2")]
    pc = MagicMock()
    with patch.object(r, "_build_dependency_tree", wraps=r._build_dependency_tree) as spy:
        s1 = r._analyze_corpus(files, pc)
        s2 = r._analyze_corpus(files, pc)
    assert s2 is s1  # memoized: same stats object returned
    assert spy.call_count == 1  # heavy (O(N^2)) work ran exactly once


def test_memo_invalidates_when_a_file_changes():
    r = _ranker()
    files = [_fa("a.py", "alpha", "h1"), _fa("b.py", "bravo", "h2")]
    pc = MagicMock()
    with patch.object(r, "_build_dependency_tree", wraps=r._build_dependency_tree) as spy:
        r._analyze_corpus(files, pc)
        changed = [_fa("a.py", "alpha changed", "h1-NEW"), files[1]]
        r._analyze_corpus(changed, pc)
    assert spy.call_count == 2  # content hash changed -> recompute, no stale memo


def test_memo_invalidates_when_analysis_depth_changes():
    # Same path/content/hash, but a DEEPER analysis (thorough vs fast mode) resolved
    # more imports -> the import graph differs, so the memo must NOT serve the shallow
    # corpus stats for the deep query.
    r = _ranker()
    pc = MagicMock()
    shallow = [_fa("a.py", "import os\nimport sys", "h1")]  # imports = ()
    deep = [_fa("a.py", "import os\nimport sys", "h1",
                imports=[SimpleNamespace(module="os"), SimpleNamespace(module="sys")])]
    with patch.object(r, "_build_dependency_tree", wraps=r._build_dependency_tree) as spy:
        r._analyze_corpus(shallow, pc)
        r._analyze_corpus(deep, pc)
    assert spy.call_count == 2  # deeper imports -> recompute, not a stale shallow memo
