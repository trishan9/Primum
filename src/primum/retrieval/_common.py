"""Helpers shared by the index classes: corpus resolution, cache fingerprints, exact top-k."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from primum.types import Passage


def passage_text(passage: Passage) -> str:
    """The string that represents a passage to every retriever (title, then body)."""
    return f"{passage.title} {passage.text}".strip()


def resolve_corpus(corpus: str | Sequence[Passage]) -> tuple[list[Passage], Path | None]:
    """Return the passages and, for a named corpus, the directory its indexes are cached in.

    A corpus given as an explicit passage list is never cached: it has no stable identity.
    """
    if not isinstance(corpus, str):
        return list(corpus), None
    # Imported lazily so that the retrieval engine can be used (and tested) on explicit
    # passage lists without the data package or any processed corpus being present.
    from primum.config import get_settings
    from primum.data import load_passages

    return load_passages(corpus), get_settings().index_dir / corpus


def corpus_fingerprint(passages: Sequence[Passage]) -> str:
    """Hash of everything an index depends on, in order.

    An index stores row numbers, not passage ids, so a cached index is only valid for exactly
    the passage list it was built from. Loading compares this fingerprint and rebuilds on any
    difference; a stale index would otherwise return the wrong passages without any error.
    """
    digest = hashlib.sha256()
    for passage in passages:
        for field in (passage.passage_id, passage.title, passage.text):
            digest.update(field.encode("utf-8"))
            digest.update(b"\x1f")
    return digest.hexdigest()


def read_json(path: Path) -> dict | None:
    """Parsed JSON object at `path`, or None if the file does not exist."""
    return json.loads(path.read_text()) if path.exists() else None


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Row indices of the k largest scores, best first, in O(n + k log k).

    Ties are broken by the lower row index, including ties that straddle the cut-off, so a
    ranking is a deterministic function of the scores (`np.argpartition` alone is not).
    """
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k < n:
        kth = np.partition(scores, n - k)[n - k]  # the k-th largest value
        above = np.flatnonzero(scores > kth)
        tied = np.flatnonzero(scores == kth)[: k - above.size]
        candidates = np.concatenate([above, tied])
    else:
        candidates = np.arange(n)
    order = np.lexsort((candidates, -scores[candidates]))  # primary key: score, descending
    return candidates[order]


def select_device() -> str:
    """Best available torch device: CUDA, then Apple-silicon MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
