"""Ranking metrics that return one value per query.

Each function returns an array aligned with the iteration order of `qrels`, not a mean:
the statistics module bootstraps and permutes these per-query values to obtain confidence
intervals and paired significance tests, and a mean cannot be resampled.

Conventions, chosen to reproduce `pytrec_eval` (and therefore published BEIR numbers):

- A document is relevant when its judged grade is > 0; unjudged documents have grade 0.
- A query with no relevant document scores 0 on every metric.
- A document id repeated in a ranking counts at its first position only. This matters when
  several retrieved passages are chunks of the same document.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

Run = Mapping[str, Sequence[str]]  # query id -> ranked document ids, best first
Qrels = Mapping[str, Mapping[str, int]]  # query id -> {document id: relevance grade}


def ndcg_at_k(run: Run, qrels: Qrels, k: int = 10) -> np.ndarray:
    """Normalised discounted cumulative gain at k, with graded relevance.

        DCG@k  = sum_{i=1..k} rel_i / log2(i + 1)
        nDCG@k = DCG@k / IDCG@k

    where rel_i is the grade of the document at rank i and IDCG@k is the DCG@k of the ideal
    ranking (all judged documents sorted by grade, descending). The gain is linear in the
    grade, as in trec_eval; the exponential gain 2^rel - 1 used in some learning-to-rank
    papers gives different numbers and is not comparable to BEIR.
    """

    def ndcg(ranking: list[str], grades: Mapping[str, int]) -> float:
        ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
        if not ideal:
            return 0.0
        gains = [max(grades.get(doc, 0), 0) for doc in ranking[:k]]
        return _dcg(gains) / _dcg(ideal)

    return _per_query(run, qrels, ndcg)


def recall_at_k(run: Run, qrels: Qrels, k: int) -> np.ndarray:
    """Recall@k = |relevant documents in the top k| / |relevant documents|."""

    def recall(ranking: list[str], grades: Mapping[str, int]) -> float:
        relevant = {doc for doc, grade in grades.items() if grade > 0}
        if not relevant:
            return 0.0
        return len(relevant.intersection(ranking[:k])) / len(relevant)

    return _per_query(run, qrels, recall)


def mrr_at_k(run: Run, qrels: Qrels, k: int = 10) -> np.ndarray:
    """Reciprocal rank of the first relevant document, or 0 if none is in the top k.

    The mean of this array over queries is MRR@k.
    """

    def reciprocal_rank(ranking: list[str], grades: Mapping[str, int]) -> float:
        for rank, doc in enumerate(ranking[:k], start=1):
            if grades.get(doc, 0) > 0:
                return 1.0 / rank
        return 0.0

    return _per_query(run, qrels, reciprocal_rank)


def average_precision(run: Run, qrels: Qrels, k: int | None = None) -> np.ndarray:
    """Average precision; its mean over queries is MAP.

        AP = (1 / R) * sum over ranks i holding a relevant document of  Precision@i

    with R the total number of relevant documents for the query, so relevant documents that
    are never retrieved count as precision 0. With `k`, only the top k ranks are summed
    (trec_eval's `map_cut_k`); R is not truncated.
    """

    def ap(ranking: list[str], grades: Mapping[str, int]) -> float:
        n_relevant = sum(grade > 0 for grade in grades.values())
        if n_relevant == 0:
            return 0.0
        found, total = 0, 0.0
        for rank, doc in enumerate(ranking[:k], start=1):
            if grades.get(doc, 0) > 0:
                found += 1
                total += found / rank
        return total / n_relevant

    return _per_query(run, qrels, ap)


def _dcg(gains: Sequence[float]) -> float:
    return float(sum(gain / np.log2(rank + 1) for rank, gain in enumerate(gains, start=1)))


def _per_query(
    run: Run, qrels: Qrels, metric: Callable[[list[str], Mapping[str, int]], float]
) -> np.ndarray:
    missing = [qid for qid in qrels if qid not in run]
    if missing:
        # Scoring an absent query as 0 would hide a bug in the caller; a retriever that
        # found nothing is represented by an empty ranking, which is scored (as 0).
        raise ValueError(
            f"run has no ranking for {len(missing)} judged queries, e.g. {missing[0]!r}"
        )
    return np.array([metric(_unique(run[qid]), grades) for qid, grades in qrels.items()])


def _unique(ranking: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(ranking))
