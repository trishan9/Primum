"""End-to-end checks on the real NFCorpus benchmark (need `primum data build nfcorpus`).

These guard against silent implementation errors that toy corpora cannot reveal: if our
BM25, our metrics or our index alignment were subtly wrong, the scores would fall outside
the range other implementations report for the same benchmark. The bands are deliberately
wide; they are sanity bounds, not results. Results come from `primum exp rq1`.
"""

from __future__ import annotations

import pytest

from primum.retrieval import build_retriever, ndcg_at_k, recall_at_k

pytestmark = pytest.mark.slow


def evaluate(retriever_name: str) -> tuple[float, float]:
    """(mean nDCG@10, mean Recall@100) of a retriever on the NFCorpus test queries."""
    from primum.data import load_nfcorpus

    queries, qrels = load_nfcorpus("test")
    qids = list(qrels)
    retriever = build_retriever(retriever_name, "nfcorpus")
    batches = retriever.search_batch([queries[qid] for qid in qids], k=100)
    run = {
        qid: [hit.passage.passage_id for hit in hits]
        for qid, hits in zip(qids, batches, strict=True)
    }
    return float(ndcg_at_k(run, qrels, 10).mean()), float(recall_at_k(run, qrels, 100).mean())


def test_bm25_is_in_the_range_of_published_bm25_baselines():
    # BEIR (Thakur et al., 2021) reports nDCG@10 = 0.325 and Recall@100 = 0.250 for
    # Anserini BM25 on NFCorpus. Ours uses a lighter stemmer, so a small gap is expected.
    ndcg, recall = evaluate("bm25")
    assert 0.29 <= ndcg <= 0.35
    assert 0.21 <= recall <= 0.29


def test_dense_and_hybrid_are_sane_and_fusion_does_not_hurt_recall():
    # Downloads `settings.base_encoder` on first use.
    bm25_ndcg, bm25_recall = evaluate("bm25")
    dense_ndcg, dense_recall = evaluate("dense-base")
    hybrid_ndcg, hybrid_recall = evaluate("hybrid-base")
    assert 0.25 <= dense_ndcg <= 0.40
    # Fusing two rankers that fail differently should recover documents either one misses.
    assert hybrid_recall >= max(bm25_recall, dense_recall) - 0.01
    assert hybrid_ndcg >= min(bm25_ndcg, dense_ndcg)
