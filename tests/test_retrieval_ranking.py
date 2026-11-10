"""Reciprocal Rank Fusion, exact top-k and the ranking metrics, against hand-computed values."""

from __future__ import annotations

from math import log2

import numpy as np
import pytest

from primum.retrieval._common import top_k
from primum.retrieval.fusion import rrf_fuse
from primum.retrieval.metrics import average_precision, mrr_at_k, ndcg_at_k, recall_at_k

# --- RRF ---------------------------------------------------------------------------------


def test_rrf_matches_hand_computation():
    # Ranking A: a, b, c.  Ranking B: b, c, d.  rrf_k = 60.
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], rrf_k=60)
    expected = {
        "b": 1 / 62 + 1 / 61,  # rank 2 in A, rank 1 in B
        "c": 1 / 63 + 1 / 62,  # rank 3 in A, rank 2 in B
        "a": 1 / 61,  # rank 1 in A only
        "d": 1 / 63,  # rank 3 in B only
    }
    assert [doc for doc, _ in fused] == ["b", "c", "a", "d"]
    for doc, score in fused:
        assert score == pytest.approx(expected[doc])


def test_rrf_rewards_agreement_over_a_single_top_rank():
    # "x" is first for one ranker and absent from the other; "y" is third for both.
    fused = dict(rrf_fuse([["x", "p", "y"], ["q", "r", "y"]], rrf_k=60))
    assert fused["y"] == pytest.approx(2 / 63)
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] > fused["x"]


def test_rrf_ties_keep_first_seen_order_and_small_k_sharpens_the_head():
    assert [doc for doc, _ in rrf_fuse([["a"], ["b"]])] == ["a", "b"]
    assert rrf_fuse([["a", "b"]], rrf_k=0) == [("a", 1.0), ("b", 0.5)]
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], [7, 3]], rrf_k=1) == [(7, 1 / 2), (3, 1 / 3)]


# --- top-k -------------------------------------------------------------------------------


def test_top_k_orders_by_score_then_row_index():
    scores = np.array([0.2, 0.9, 0.5, 0.9, 0.5, 0.1], dtype=np.float32)
    assert top_k(scores, 3).tolist() == [1, 3, 2]  # the tie at 0.5 is cut at the lower row
    assert top_k(scores, 4).tolist() == [1, 3, 2, 4]
    assert top_k(scores, 100).tolist() == [1, 3, 2, 4, 0, 5]
    assert top_k(scores, 0).tolist() == []


def test_top_k_agrees_with_a_full_sort():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 20, size=500).astype(np.float32)  # many ties
    expected = np.lexsort((np.arange(500), -scores))[:37]
    np.testing.assert_array_equal(top_k(scores, 37), expected)


# --- metrics -----------------------------------------------------------------------------

# One query with graded judgements and a ranking that makes every metric non-trivial:
#   rank 1: d3 (judged, grade 0)   rank 2: d1 (grade 2)
#   rank 3: dX (unjudged)          rank 4: d2 (grade 1)
# A second query whose retriever returned nothing.
QRELS = {"q1": {"d1": 2, "d2": 1, "d3": 0}, "q2": {"d5": 1}}
RUN = {"q1": ["d3", "d1", "dX", "d2"], "q2": []}
IDEAL_DCG = 2 / log2(2) + 1 / log2(3)  # grades 2, 1 at ranks 1, 2


def test_ndcg_matches_hand_computation():
    # @3: only d1 (grade 2) at rank 2 contributes.
    np.testing.assert_allclose(ndcg_at_k(RUN, QRELS, k=3), [(2 / log2(3)) / IDEAL_DCG, 0.0])
    # @10: d1 at rank 2 and d2 (grade 1) at rank 4.
    np.testing.assert_allclose(
        ndcg_at_k(RUN, QRELS, k=10), [(2 / log2(3) + 1 / log2(5)) / IDEAL_DCG, 0.0]
    )
    # The same numbers as decimals, to pin the formula independently of its implementation.
    np.testing.assert_allclose(ndcg_at_k(RUN, QRELS, k=3)[0], 0.479625, atol=1e-6)
    np.testing.assert_allclose(ndcg_at_k(RUN, QRELS, k=10)[0], 0.643322, atol=1e-6)


def test_ndcg_is_one_for_the_ideal_ranking_and_uses_linear_gain():
    assert ndcg_at_k({"q": ["d1", "d2", "d3"]}, {"q": QRELS["q1"]}, k=10)[0] == pytest.approx(1.0)
    # Swapping the two relevant documents: linear gain gives (1 + 2/log2(3)) / (2 + 1/log2(3)).
    # An exponential gain (2^rel - 1) would give 0.7967 instead.
    swapped = ndcg_at_k({"q": ["d2", "d1"]}, {"q": QRELS["q1"]}, k=10)[0]
    assert swapped == pytest.approx((1 + 2 / log2(3)) / IDEAL_DCG)
    assert swapped == pytest.approx(0.859719, abs=1e-6)


def test_ideal_dcg_is_truncated_at_k():
    # Three relevant documents but k = 1: a single relevant hit at rank 1 is already ideal.
    qrels = {"q": {"a": 1, "b": 1, "c": 1}}
    assert ndcg_at_k({"q": ["a", "x"]}, qrels, k=1)[0] == pytest.approx(1.0)


def test_recall_matches_hand_computation():
    np.testing.assert_allclose(recall_at_k(RUN, QRELS, k=1), [0.0, 0.0])
    np.testing.assert_allclose(recall_at_k(RUN, QRELS, k=3), [0.5, 0.0])  # d1 of {d1, d2}
    np.testing.assert_allclose(recall_at_k(RUN, QRELS, k=4), [1.0, 0.0])


def test_mrr_matches_hand_computation():
    np.testing.assert_allclose(mrr_at_k(RUN, QRELS, k=10), [1 / 2, 0.0])  # first relevant: rank 2
    np.testing.assert_allclose(mrr_at_k(RUN, QRELS, k=1), [0.0, 0.0])


def test_average_precision_matches_hand_computation():
    # Relevant at ranks 2 and 4: AP = (1/2 + 2/4) / 2.
    np.testing.assert_allclose(average_precision(RUN, QRELS), [0.5, 0.0])
    # Cut at 3, the hit at rank 4 is lost but still counts in the denominator: (1/2) / 2.
    np.testing.assert_allclose(average_precision(RUN, QRELS, k=3), [0.25, 0.0])


def test_metrics_follow_qrels_order_and_return_one_value_per_query():
    qrels = {"b": {"x": 1}, "a": {"y": 1}}
    run = {"a": ["y"], "b": ["z", "x"], "unjudged": ["x"]}  # extra runs are ignored
    np.testing.assert_allclose(mrr_at_k(run, qrels, k=10), [0.5, 1.0])


def test_duplicate_documents_count_once():
    # Three chunks of the same relevant document must not triple its gain.
    qrels = {"q": {"d": 1, "e": 1}}
    run = {"q": ["d", "d", "d", "e"]}
    assert recall_at_k(run, qrels, k=2)[0] == pytest.approx(1.0)  # top-2 distinct: d, e
    assert ndcg_at_k(run, qrels, k=10)[0] == pytest.approx(1.0)


def test_query_without_relevant_documents_scores_zero():
    qrels = {"q": {"d": 0}}
    run = {"q": ["d"]}
    for metric in (ndcg_at_k, recall_at_k, mrr_at_k):
        assert metric(run, qrels, k=10)[0] == 0.0
    assert average_precision(run, qrels)[0] == 0.0


def test_missing_ranking_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="no ranking for 1 judged queries"):
        ndcg_at_k({}, {"q": {"d": 1}}, k=10)
