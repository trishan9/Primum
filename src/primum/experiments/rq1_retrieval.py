"""RQ1: does a contrastively fine-tuned encoder, fused with BM25, retrieve biomedical evidence
better than either alone?

Two benchmarks, every system in `RETRIEVERS` plus the query-expansion ablation `bm25+concepts`:

- NFCorpus *test* queries with graded qrels: nDCG@10, Recall@100, MRR@10. The fine-tuned
  encoder saw NFCorpus *train* qrels only, and the data build drops train queries whose text
  equals a dev/test query, so no test judgment reaches training.
- PubMedQA gold-abstract retrieval: each of the 1,000 labelled questions has exactly one
  relevant passage (its abstract without the conclusion) among ~62k: Recall@1/5/20, MRR@10.
  The fine-tuning pairs exclude every PQA-L PMID and question.

Inference. Every metric is computed per query; the mean gets a 95% percentile-bootstrap interval
over queries, and each system is compared with `bm25` and with `dense-base` by a two-sided
sign-flip permutation test on the per-query differences (paired by query, since all systems
answer the same queries). The p-values are reported raw and Holm-adjusted within each family
(one dataset x metric x baseline), because each family asks "which systems differ from this
baseline?" across several systems at once.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from primum.config import get_settings
from primum.data import load_nfcorpus, load_pubmedqa_labeled
from primum.experiments._common import (
    bootstrap_estimate,
    build_meta,
    resolve_results_dir,
    rng_for,
    write_json,
)
from primum.retrieval import RETRIEVERS, build_retriever, mrr_at_k, ndcg_at_k, recall_at_k
from primum.uncertainty import paired_permutation_pvalue

SYSTEMS = [*RETRIEVERS, "bm25+concepts"]
BASELINES = ("bm25", "dense-base")
DATASETS = ("nfcorpus", "pubmedqa")

Metric = Callable[[dict[str, list[str]], dict[str, dict[str, int]]], np.ndarray]


def _metric(fn: Callable[..., np.ndarray], k: int) -> Metric:
    return lambda run, qrels: fn(run, qrels, k=k)


METRICS: dict[str, dict[str, Metric]] = {
    "nfcorpus": {
        "ndcg@10": _metric(ndcg_at_k, 10),
        "recall@100": _metric(recall_at_k, 100),
        "mrr@10": _metric(mrr_at_k, 10),
    },
    "pubmedqa": {
        "recall@1": _metric(recall_at_k, 1),
        "recall@5": _metric(recall_at_k, 5),
        "recall@20": _metric(recall_at_k, 20),
        "mrr@10": _metric(mrr_at_k, 10),
    },
}
# Ranking depth retrieved per query: the deepest cut-off any metric of that dataset reads.
DEPTH = {"nfcorpus": 100, "pubmedqa": 20}


def run(
    *,
    created: str,
    retrievers: Sequence[str] | None = None,
    datasets: Sequence[str] = DATASETS,
    limit: int | None = None,
    skip_unavailable: bool = False,
    n_boot: int = 2000,
    n_perm: int = 10_000,
    results_dir: Path | None = None,
) -> Path:
    """Evaluate the retrievers and write `results/rq1_retrieval.json`; returns its path.

    A system that cannot be built (e.g. `dense-ft` before `primum train encoder`) raises its
    `FileNotFoundError`, unless `skip_unavailable` is set, in which case it is left out and
    listed under `meta.skipped` with the reason. It is never replaced by another system.
    """
    settings = get_settings()
    systems = list(retrievers or SYSTEMS)
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets {unknown}; choose from {DATASETS}")

    payload: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    per_query: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    n_queries: dict[str, int] = {}
    for dataset in datasets:
        corpus, queries, qrels = _benchmark(dataset, limit)
        n_queries[dataset] = len(queries)
        values: dict[str, dict[str, np.ndarray]] = {}
        for system in systems:
            try:
                retriever = build_retriever(system, corpus)
                hits = retriever.search_batch(list(queries.values()), k=DEPTH[dataset])
            except FileNotFoundError as exc:
                if not skip_unavailable:
                    raise
                skipped[system] = f"{type(exc).__name__}: {exc}"
                continue
            run_ = {
                qid: [hit.passage.passage_id for hit in ranking]
                for qid, ranking in zip(queries, hits, strict=True)
            }
            values[system] = {name: fn(run_, qrels) for name, fn in METRICS[dataset].items()}
            print(
                f"[rq1] {dataset} {system}: "
                + ", ".join(f"{name}={v.mean():.4f}" for name, v in values[system].items())
            )

        payload[dataset] = {
            system: {
                metric: bootstrap_estimate(
                    v, rng_for(settings.seed, "rq1", dataset, system, metric), n_boot=n_boot
                )
                for metric, v in by_metric.items()
            }
            for system, by_metric in values.items()
        }
        paired[dataset] = _paired_tests(dataset, values, settings.seed, n_boot, n_perm)
        per_query[dataset] = {
            "qids": list(qrels),
            **{s: {m: v.tolist() for m, v in by_metric.items()} for s, by_metric in values.items()},
        }

    meta = build_meta(
        created,
        n=sum(n_queries.values()),
        seeds={"seed": settings.seed},
        models={
            "base_encoder": settings.base_encoder,
            "biomed_encoder": settings.biomed_encoder,
            "finetuned_encoder": os.path.relpath(settings.finetuned_encoder_dir, settings.root),
            "reranker": settings.reranker,
        },
        # The concept dictionary used by `+concepts` is built from the kb corpus.
        corpora=[*datasets, *(["kb"] if any("+concepts" in s for s in systems) else [])],
        n_queries=n_queries,
        systems=systems,
        skipped=skipped,
        depth=DEPTH,
        inference={
            "ci": "95% percentile bootstrap over queries",
            "n_boot": n_boot,
            "test": "two-sided paired sign-flip permutation test on per-query differences",
            "n_perm": n_perm,
            "adjustment": "Holm within dataset x metric x baseline",
        },
    )
    payload = {"meta": meta, **payload, "paired_tests": paired, "per_query": per_query}
    return write_json(resolve_results_dir(results_dir) / "rq1_retrieval.json", payload)


def _benchmark(
    dataset: str, limit: int | None
) -> tuple[str, dict[str, str], dict[str, dict[str, int]]]:
    """(corpus to search, {qid: query text}, qrels), queries in a fixed order."""
    if dataset == "nfcorpus":
        texts, qrels = load_nfcorpus("test")
        # Only judged queries can be scored; NFCorpus passage ids are the BEIR doc ids.
        qids = [qid for qid in qrels if qid in texts][:limit]
        return "nfcorpus", {q: texts[q] for q in qids}, {q: qrels[q] for q in qids}
    examples = load_pubmedqa_labeled()[:limit]
    return (
        "pubmedqa",
        {ex.qid: ex.question for ex in examples},
        {ex.qid: {ex.passage_id: 1} for ex in examples},
    )


def _paired_tests(
    dataset: str,
    values: dict[str, dict[str, np.ndarray]],
    seed: int,
    n_boot: int,
    n_perm: int,
) -> dict[str, Any]:
    """`{"vs_<baseline>": {system: {<metric>_diff: {mean, ci}, <metric>_p, <metric>_p_holm}}}`."""
    tests: dict[str, Any] = {}
    for baseline in BASELINES:
        if baseline not in values:
            continue
        others = [s for s in values if s != baseline]
        family: dict[str, Any] = {s: {} for s in others}
        for metric in METRICS[dataset]:
            b = values[baseline][metric]
            pvalues = []
            for system in others:
                a = values[system][metric]
                key = ("rq1", dataset, metric, system, baseline)
                family[system][f"{metric}_diff"] = bootstrap_estimate(
                    a - b, rng_for(seed, *key, "diff"), n_boot=n_boot
                )
                p = paired_permutation_pvalue(a, b, n_perm=n_perm, rng=rng_for(seed, *key, "p"))
                family[system][f"{metric}_p"] = p
                pvalues.append(p)
            for system, adjusted in zip(others, holm(pvalues), strict=True):
                family[system][f"{metric}_p_holm"] = adjusted
        if others:
            tests[f"vs_{baseline}"] = family
    return tests


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values; controls the family-wise error rate.

    Sorted ascending, p_(i) is multiplied by (m - i + 1); the running maximum keeps the
    adjusted values in the same order as the raw ones, and they are capped at 1.
    """
    m = len(pvalues)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()
