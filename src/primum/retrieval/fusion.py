"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    score(d) = sum over rankings r that contain d of   1 / (rrf_k + rank_r(d))

with 1-based ranks. A document missing from a ranking simply receives nothing from it.

RRF uses ranks only, never scores. That is the point: BM25 scores are unbounded and depend
on query length, cosine similarities live in [-1, 1], and no normalisation makes the two
commensurable across queries. Ranks are. The constant rrf_k (60 in the original paper) damps
the head of each list: with rrf_k = 60 a rank-1 document earns 1/61 and a rank-10 document
1/70, so agreement between rankers outweighs a single ranker's top position.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def rrf_fuse[T: Hashable](
    rankings: Sequence[Sequence[T]], *, rrf_k: int = 60
) -> list[tuple[T, float]]:
    """Fuse ranked lists of document ids (best first) into one list of (id, RRF score).

    The result is sorted by score, descending. Equal scores keep the order in which the
    documents were first seen (earlier ranking first, then better rank), which makes the
    fused ranking deterministic.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)
    # `sorted` is stable and dicts preserve insertion order, hence the tie-break above.
    return sorted(scores.items(), key=lambda item: -item[1])
