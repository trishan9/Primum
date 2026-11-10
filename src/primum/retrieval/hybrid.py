"""Retrievers composed from the two first-stage indexes: fusion, reranking, query expansion."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from primum.retrieval._common import passage_text, top_k
from primum.retrieval.bm25 import BM25Index
from primum.retrieval.dense import DenseIndex
from primum.retrieval.fusion import rrf_fuse
from primum.retrieval.rerank import CrossEncoderReranker
from primum.types import Hit


class Retriever(Protocol):
    """Anything that maps a query to a ranked list of passages."""

    name: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Hit]]: ...


class HybridRetriever:
    """BM25 and a dense index fused by Reciprocal Rank Fusion.

    Lexical and dense retrieval fail differently. BM25 misses paraphrases ("heart attack"
    against "myocardial infarction"); a dense encoder blurs rare exact tokens (gene symbols,
    drug names, dosages). Each retriever contributes its top `candidate_k` passages and RRF
    rewards the passages both of them rank highly.

    The ranking does not depend on the requested k: `search(q, k)` is always the first k
    entries of one fixed ranking, so an evaluation that asks for 100 hits scores the same
    top 10 that the deployed system (which asks for 5) serves. That ranking is
      1. the head: RRF over each retriever's top `candidate_k`, always at exactly that depth.
         Fusing deeper lists would change the head itself, since a passage at rank 31 in one
         list and 70 in the other (1/91 + 1/130 = 0.0187) outranks a passage that only one
         list placed first (1/61 = 0.0164), but only once both lists reach depth 70;
      2. the tail, read only when k exceeds the head (deep metrics such as Recall@100): the
         remaining passages ordered by RRF over the two *full* rankings, which is also
         independent of k. Tail scores live on that full-depth RRF scale, so they need not
         be below the head's scores; ranks, not scores, order the list.

    Every hit carries both first-stage scores in `components` ("bm25", "dense"), whether or
    not that retriever nominated it, because the cosine is the scale-free relevance signal
    the confidence model consumes. The ranks ("bm25_rank", "dense_rank") are present only
    when the retriever placed the passage inside its own candidate list, i.e. in the head.
    """

    _CHUNK = 64  # queries per pair of score matrices; bounds memory at 2 * 64 * n_passages

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        *,
        name: str,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        # Fusion joins the two rankings on row numbers, so both indexes must have been
        # built over the same passages in the same order.
        if [p.passage_id for p in bm25.passages] != [p.passage_id for p in dense.passages]:
            raise ValueError("BM25 and dense indexes were built over different passage lists")
        self.name = name
        self._bm25 = bm25
        self._dense = dense
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Hit]]:
        results: list[list[Hit]] = []
        for start in range(0, len(queries), self._CHUNK):
            chunk = queries[start : start + self._CHUNK]
            lexical = self._bm25.score_all(chunk)
            semantic = self._dense.score_all(chunk)
            for bm25_scores, dense_scores in zip(lexical, semantic, strict=True):
                results.append(self._rank(bm25_scores, dense_scores, k))
        return results

    def _rank(self, bm25_scores: np.ndarray, dense_scores: np.ndarray, k: int) -> list[Hit]:
        bm25_top = top_k(bm25_scores, self._candidate_k)
        bm25_top = bm25_top[bm25_scores[bm25_top] > 0]  # zero score = not a lexical match
        dense_top = top_k(dense_scores, self._candidate_k)
        bm25_rank = {row: rank for rank, row in enumerate(bm25_top.tolist(), start=1)}
        dense_rank = {row: rank for rank, row in enumerate(dense_top.tolist(), start=1)}

        fused = rrf_fuse([list(bm25_rank), list(dense_rank)], rrf_k=self._rrf_k)[:k]
        if len(fused) < k:
            fused += self._full_depth_tail(bm25_scores, dense_scores, [r for r, _ in fused], k)

        hits = []
        for rank, (row, rrf_score) in enumerate(fused, start=1):
            components = {
                "bm25": float(bm25_scores[row]),
                "dense": float(dense_scores[row]),
                "rrf": rrf_score,
            }
            if row in bm25_rank:
                components["bm25_rank"] = float(bm25_rank[row])
            if row in dense_rank:
                components["dense_rank"] = float(dense_rank[row])
            hits.append(
                Hit(
                    passage=self._bm25.passages[row],
                    score=rrf_score,
                    rank=rank,
                    retriever=self.name,
                    components=components,
                )
            )
        return hits

    def _full_depth_tail(
        self, bm25_scores: np.ndarray, dense_scores: np.ndarray, head: list[int], k: int
    ) -> list[tuple[int, float]]:
        """The passages ranked after `head`, best first, by RRF over the two complete rankings.

        Vectorised rather than `rrf_fuse` because it scores every passage in the corpus.
        """
        n = dense_scores.shape[0]
        positions = np.arange(1, n + 1, dtype=np.float64)
        fused = np.zeros(n)
        # Stable sorts break ties by row index, as `top_k` does for the head.
        bm25_order = np.argsort(-bm25_scores, kind="stable")
        bm25_order = bm25_order[bm25_scores[bm25_order] > 0]
        fused[bm25_order] += 1.0 / (self._rrf_k + positions[: bm25_order.size])
        fused[np.argsort(-dense_scores, kind="stable")] += 1.0 / (self._rrf_k + positions)
        fused[head] = -np.inf
        rows = top_k(fused, min(k, n) - len(head))
        return [(int(row), float(fused[row])) for row in rows]


class RerankedRetriever:
    """A base retriever whose top `candidate_k` candidates are re-ordered by a cross-encoder.

    The pool is always exactly the base's top `candidate_k`, whatever k is asked for, so the
    reranked top 10 is the same in an evaluation that requests 100 hits as in deployment.
    Asking for more than `candidate_k` hits appends the base's later hits in base order; the
    cross-encoder never saw them, so they carry no "rerank" component and keep the base score.
    Reranking thus permutes the top `candidate_k` and leaves Recall@k for k >= candidate_k
    equal to the base retriever's.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: CrossEncoderReranker,
        *,
        name: str,
        candidate_k: int = 50,
    ) -> None:
        self.name = name
        self._base = base
        self._reranker = reranker
        self._candidate_k = candidate_k

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Hit]]:
        # The base ranking is itself independent of k, so its first candidate_k hits are the
        # same pool whether we fetch candidate_k or more.
        ranked = self._base.search_batch(queries, max(self._candidate_k, k))
        pools = [hits[: self._candidate_k] for hits in ranked]
        pairs = [
            (query, passage_text(hit.passage))
            for query, pool in zip(queries, pools, strict=True)
            for hit in pool
        ]
        scores = self._reranker.score(pairs)

        results: list[list[Hit]] = []
        offset = 0
        for hits, pool in zip(ranked, pools, strict=True):
            own = scores[offset : offset + len(pool)]
            offset += len(pool)
            # Stable sort: passages the cross-encoder scores equally keep their fused order.
            order = np.argsort(-own, kind="stable").tolist()
            reranked = [
                pool[i].model_copy(
                    update={
                        "score": float(own[i]),
                        "components": {**pool[i].components, "rerank": float(own[i])},
                    }
                )
                for i in order
            ]
            results.append(
                [
                    hit.model_copy(update={"rank": rank, "retriever": self.name})
                    for rank, hit in enumerate((reranked + hits[len(pool) :])[:k], start=1)
                ]
            )
        return results


class ExpandedQueryRetriever:
    """A base retriever queried with a rewritten query (e.g. concept synonyms appended)."""

    def __init__(self, base: Retriever, expand: Callable[[str], str], *, name: str) -> None:
        self.name = name
        self._base = base
        self._expand = expand

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Hit]]:
        batches = self._base.search_batch([self._expand(query) for query in queries], k)
        return [
            [hit.model_copy(update={"retriever": self.name}) for hit in hits] for hits in batches
        ]
