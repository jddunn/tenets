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
