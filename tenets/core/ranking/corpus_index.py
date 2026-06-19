"""Persistent, incremental BM25 + TF-IDF corpus index.

The ranker rebuilds the entire lexical corpus on every query (re-tokenizing every
file). This index builds it once and reuses it: unchanged files are never
re-tokenized across calls — an in-memory warm layer serves repeated queries in
the same process, and a DiskCache (SQLite + pickle) survives restarts. Only
changed/new files are tokenized; deleted files are removed.

Parity: for an unchanged corpus the produced BM25 calculator is byte-identical to
a fresh ``build_corpus`` (lazy IDF → order-independent). TF-IDF bakes insertion-
time IDF into its vectors, so it is byte-identical on unchanged-corpus reload and
functionally-equivalent (negligible drift, 0.10 secondary weight) under edits.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from tenets.core.nlp.bm25 import BM25Calculator
from tenets.core.nlp.keyword_extractor import TFIDFCalculator
from tenets.storage.cache import DiskCache
from tenets.utils.logger import get_logger


def corpus_root_key(documents: List[Tuple[str, str]]) -> str:
    """A stable key identifying a scan (so the same tree reuses its warm index).

    Uses the common directory prefix of the document paths (stable as files within
    the tree are edited/added/removed); falls back to a hash of the path set.
    """
    paths = sorted(d for d, _ in documents)
    if not paths:
        return "empty"
    try:
        cp = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
        if cp:
            return cp
    except Exception:
        pass
    return "set:" + hashlib.md5("\n".join(paths).encode("utf-8", "replace")).hexdigest()[:16]


def content_signature(content: str) -> str:
    """Change signature for a document (content hash — the ranker already holds
    the content in memory, so this is accurate and needs no filesystem stat)."""
    if not content:
        return "0:0"
    data = content.encode("utf-8", "replace")
    return f"{len(data)}:{hashlib.md5(data).hexdigest()}"


class CorpusIndex:
    """Build-once / incrementally-update BM25 + TF-IDF calculators."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self._disk: Optional[DiskCache] = (
            DiskCache(Path(cache_dir), name="corpus_index") if cache_dir is not None else None
        )
        self._mem: Dict[str, dict] = {}

    @staticmethod
    def _key(root_key: str) -> str:
        return f"corpus_index::{root_key}"

    def build(
        self,
        root_key: str,
        documents: List[Tuple[str, str]],
        *,
        make_bm25: Callable[[], BM25Calculator],
        make_tfidf: Callable[[], TFIDFCalculator],
        config_sig: str = "",
    ) -> Tuple[BM25Calculator, TFIDFCalculator]:
        """Return (bm25, tfidf) populated for ``documents`` [(doc_id, content)].

        ``config_sig`` fingerprints the calculator configuration (k1/b/stopwords);
        a mismatch discards the persisted index and rebuilds from scratch.
        """
        incoming: Dict[str, str] = {doc_id: text for doc_id, text in documents}
        incoming_sigs = {d: content_signature(c) for d, c in incoming.items()}

        warm = self._mem.get(root_key)
        if warm is not None and warm.get("config_sig") == config_sig:
            bm25, tfidf, prior_sigs = warm["bm25"], warm["tfidf"], warm["sigs"]
        else:
            bm25, tfidf = make_bm25(), make_tfidf()
            prior_sigs: Dict[str, str] = {}
            blob = self._disk.get(self._key(root_key)) if self._disk is not None else None
            if blob is not None and blob.get("config_sig") == config_sig:
                try:
                    bm25.load_state(blob["bm25"])
                    tfidf.load_state(blob["tfidf"])
                    prior_sigs = dict(blob["sigs"])
                except Exception as e:  # corrupt/stale → rebuild from scratch
                    self.logger.warning(f"corpus index reload failed, rebuilding: {e}")
                    bm25, tfidf, prior_sigs = make_bm25(), make_tfidf(), {}

        deleted = [d for d in prior_sigs if d not in incoming_sigs]
        changed = [d for d, s in incoming_sigs.items() if prior_sigs.get(d) != s]

        for d in deleted:
            bm25._remove_document(d)
            tfidf.remove_document(d)
        for d in changed:
            bm25.add_document(d, incoming[d])  # BM25 add_document updates in place
            tfidf.remove_document(d)  # TF-IDF add doesn't auto-remove on update
            tfidf.add_document(d, incoming[d])

        self._mem[root_key] = {
            "bm25": bm25,
            "tfidf": tfidf,
            "sigs": incoming_sigs,
            "config_sig": config_sig,
        }
        # Persist only when the corpus actually changed — re-pickling the whole
        # blob on every (unchanged) query would cost as much as it saves.
        if self._disk is not None and (changed or deleted):
            try:
                self._disk.put(
                    self._key(root_key),
                    {
                        "version": 1,
                        "config_sig": config_sig,
                        "bm25": bm25.to_state(),
                        "tfidf": tfidf.to_state(),
                        "sigs": incoming_sigs,
                    },
                )
            except Exception as e:
                self.logger.warning(f"corpus index persist failed: {e}")

        self.logger.debug(
            f"corpus index '{root_key}': {len(incoming)} docs "
            f"({len(changed)} tokenized, {len(deleted)} removed)"
        )
        return bm25, tfidf
