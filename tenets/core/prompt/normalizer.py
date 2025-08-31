"""Entity and keyword normalization utilities.

Provides lightweight normalization (case-folding, punctuation removal,
singularization, lemmatization when available) and tracks variant mappings
for explainability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:  # Optional lemmatization via nltk if available
    from nltk.stem import WordNetLemmatizer  # type: ignore

    _LEM = WordNetLemmatizer()
    _HAS_NLTK = True
except Exception:  # pragma: no cover - optional
    _LEM = None
    _HAS_NLTK = False


_PUNCT_RE = re.compile(r"[\u2000-\u206F\u2E00-\u2E7F'\-\.,;:!?()\[\]{}<>\"`~@#$%^&*_+=|/\\]")


def _simple_singularize(token: str) -> str:
    """Very basic English singularization as a fallback.

    Not perfect, but good enough for grouping common plurals without heavy deps.
    """
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 2 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 1 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@dataclass
class NormalizationResult:
    canonical: str
    steps: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)


class EntityNormalizer:
    """Normalize entities/keywords and record variant mappings."""

    def normalize(self, token: str) -> NormalizationResult:
        original = token
        steps: List[str] = []
        variants: List[str] = [original]

        # Lowercase
        t = token.lower()
        if t != token:
            steps.append("lowercase")
            variants.append(t)

        # Strip punctuation
        t2 = _PUNCT_RE.sub(" ", t).strip()
        t2 = re.sub(r"\s+", " ", t2)
        if t2 != t:
            steps.append("strip_punct")
            variants.append(t2)

        # Lemmatize (if available) else simple singularization
        if _HAS_NLTK:
            try:
                base = _LEM.lemmatize(t2)
                if base != t2:
                    steps.append("lemmatize")
                    variants.append(base)
            except Exception:
                base = _simple_singularize(t2)
                if base != t2:
                    steps.append("singularize")
                    variants.append(base)
        else:
            base = _simple_singularize(t2)
            if base != t2:
                steps.append("singularize")
                variants.append(base)

        canonical = base or t2
        return NormalizationResult(
            canonical=canonical, steps=steps, variants=list(dict.fromkeys(variants))
        )


def normalize_list(items: List[str]) -> Tuple[List[str], Dict[str, Dict[str, List[str]]]]:
    """Normalize a list and return unique canonicals + per-item metadata.

    Returns:
        (canonicals, meta_by_original) where meta contains steps and variants.
    """
    norm = EntityNormalizer()
    canonicals: List[str] = []
    meta: Dict[str, Dict[str, List[str]]] = {}

    seen = set()
    for item in items:
        res = norm.normalize(item)
        if res.canonical not in seen:
            canonicals.append(res.canonical)
            seen.add(res.canonical)
        meta[item] = {"steps": res.steps, "variants": res.variants}

    return canonicals, meta
