"""The import-graph build memoizes per-module resolution to kill the O(N^2)
_resolve_import scan — WITHOUT changing the resulting graph (so import_centrality,
a 0.10-weight ranking factor, stays byte-identical)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tenets.config import TenetsConfig
from tenets.core.ranking.ranker import RelevanceRanker
from tenets.models.analysis import FileAnalysis


def _mod(m):
    return SimpleNamespace(module=m)


def _fa(path, imports):
    return FileAnalysis(
        path=path, content="x", language="python", size=1,
        imports=[_mod(m) for m in imports], hash=path,
    )


def _ranker():
    cfg = TenetsConfig()
    cfg.cache.index_enabled = False
    return RelevanceRanker(cfg)


# 6 import occurrences across 3 files; distinct resolution keys =
# "os", "utils", (".a","pkg/b.py") -> 3
FILES = [
    _fa("pkg/a.py", ["os", "utils"]),
    _fa("pkg/utils.py", ["os"]),
    _fa("pkg/b.py", ["utils", ".a"]),
]


def test_import_graph_byte_identical_to_direct_resolution():
    """The memoized build must equal resolving each import directly (un-memoized)."""
    r = _ranker()
    stats = r._analyze_corpus(FILES, MagicMock())
    got = {k: set(v) for k, v in stats["import_graph"].items()}
    ref = {}
    for f in FILES:
        for imp in f.imports:
            res = r._resolve_import(imp.module, f.path, FILES)
            if res:
                ref.setdefault(res, set()).add(f.path)
    assert got == ref


def test_resolution_memoized_per_distinct_module():
    """6 import occurrences but only 3 distinct keys -> _resolve_import runs 3x, not 6x."""
    r = _ranker()
    with patch.object(r, "_resolve_import", wraps=r._resolve_import) as spy:
        r._analyze_corpus(FILES, MagicMock())
    assert spy.call_count == 3
