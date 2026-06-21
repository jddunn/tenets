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
