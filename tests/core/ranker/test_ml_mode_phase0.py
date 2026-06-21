"""Phase 0 correctness: ml mode routing, weights, query-embed-once, no eager load.

These regression tests pin the four Phase 0 bug fixes. They must run without the
optional ``ml`` extra installed (no sentence-transformers / torch) — semantic
behavior is exercised via fakes / monkeypatching, never a real model.
"""

from tenets.cli.commands.rank import resolve_mode

# --- Task 1: --ml routes to ml mode ---------------------------------------


def test_ml_flag_implies_ml_mode_when_mode_unset():
    assert resolve_mode(mode=None, ml=True) == "ml"


def test_explicit_mode_wins_over_ml_flag():
    assert resolve_mode(mode="fast", ml=True) == "fast"


def test_no_ml_flag_keeps_default():
    assert resolve_mode(mode=None, ml=False) == "balanced"


# --- Task 2: ML weights stable once a backend is configured ----------------


def test_ml_weights_stable_before_model_loads():
    from tenets.core.ranking.strategies import MLRankingStrategy

    s = MLRankingStrategy()
    s._backend_configured = True  # backend configured, model NOT yet loaded
    s._model = None
    w = s.get_weights()
    assert w.get("semantic_similarity", 0) > 0, "semantic must be weighted before lazy load"


def test_ml_weights_unchanged_when_no_backend():
    """With no backend configured and no model loaded (today's default),
    ml weights must fall back to thorough weights — no semantic weighting."""
    from tenets.core.ranking.strategies import MLRankingStrategy, ThoroughRankingStrategy

    s = MLRankingStrategy()
    assert s._backend_configured is False
    assert s._model is None
    assert s.get_weights() == ThoroughRankingStrategy().get_weights()


# --- Task 3: query embedded once per ranking pass --------------------------


def test_query_embedded_once_for_many_files():
    from tenets.core.ranking.strategies import MLRankingStrategy

    calls = {"n": 0}

    class FakeModel:
        def encode(self, texts, **kw):
            # one call may batch; count *every* encode invocation
            calls["n"] += 1
            import numpy as np

            n = len(texts) if isinstance(texts, list) else 1
            # non-zero so cosine is well-defined (norm != 0)
            return np.ones((n, 8), dtype="float32")

    s = MLRankingStrategy()
    s._model = FakeModel()
    s._model_loaded = True
    s.embed_query_for_pass("find the retry logic")  # embed query once
    for _ in range(5):
        s.semantic_for_file("def f(): ...")  # reuses cached query vec
    # 1 query encode + up to 5 file encodes; the query must NOT be re-embedded
    # per file (that would be 10). Assert the query side stays single.
    assert calls["n"] <= 6, "query must be embedded once, not re-embedded per file"
    assert s._query_vec is not None


def test_semantic_for_file_zero_without_query_vec():
    """semantic_for_file returns 0.0 if no query was embedded for the pass."""
    from tenets.core.ranking.strategies import MLRankingStrategy

    s = MLRankingStrategy()

    class FakeModel:
        def encode(self, texts, **kw):
            import numpy as np

            n = len(texts) if isinstance(texts, list) else 1
            return np.ones((n, 8), dtype="float32")

    s._model = FakeModel()
    s._model_loaded = True
    # no embed_query_for_pass() call -> no cached query vec
    assert s.semantic_for_file("def f(): ...") == 0.0


# --- Task 4: no embedding model loads at construction / MCP warmup ---------


def test_constructing_thorough_strategy_loads_no_embedding_model(monkeypatch):
    """ThoroughRankingStrategy() must not instantiate an embedding model.

    The model is resolved lazily only when semantic similarity is actually
    needed. We patch the ranker-module SentenceTransformer symbol (the one the
    strategy resolves first) with a tripwire and assert it is never called
    during construction.
    """
    import tenets.core.ranking.ranker as ranker_mod
    from tenets.core.ranking.strategies import ThoroughRankingStrategy

    loaded = {"n": 0}

    class Tripwire:
        def __init__(self, *a, **k):
            loaded["n"] += 1

    monkeypatch.setattr(ranker_mod, "SentenceTransformer", Tripwire, raising=False)

    ThoroughRankingStrategy()  # construction must not load a model
    assert loaded["n"] == 0


def test_constructing_ranker_loads_no_embedding_model(monkeypatch):
    """Constructing the RelevanceRanker (which pre-populates the THOROUGH
    strategy and is touched by MCP warmup) must not load an embedding model."""
    import tenets.core.ranking.ranker as ranker_mod

    loaded = {"n": 0}

    class Tripwire:
        def __init__(self, *a, **k):
            loaded["n"] += 1

    monkeypatch.setattr(ranker_mod, "SentenceTransformer", Tripwire, raising=False)

    from tenets.config import TenetsConfig
    from tenets.core.ranking.ranker import RelevanceRanker

    RelevanceRanker(TenetsConfig())  # construction must not load a model
    assert loaded["n"] == 0
