"""PubMedQA benchmark run: the records RQ2 and RQ3 are computed from.

For each of the 1,000 expert-labelled PubMedQA questions and each condition

    closed_book   no evidence: what the generator knows on its own
    rag           the top-k passages of the configured retriever over the `pubmedqa` corpus
                  (1,000 gold abstracts among ~61k distractors, conclusions removed)
    oracle        the gold abstract alone: an upper bound for retrieval

the generator answers yes / no / maybe with stated probabilities and 1-4 cited claims; the
claims are checked by NLI against the passages shown; the confidence features are extracted;
and correctness against the expert label is recorded. In the `rag` condition
`settings.self_consistency_samples` independent samples are drawn (sample 0 is the answer that
is scored; the others only feed the agreement features).

Operational properties:
- Resumable: every LLM response is cached by prompt and sample id, so re-running after an
  interruption replays finished items for free and only pays for new ones.
- Budget: the LLM client raises `BudgetExceeded` before a call that would go over budget.
  The run then stops cleanly: completed records are written with `meta.complete = false`
  and the exception is re-raised. RQ2/RQ3 refuse an incomplete file.
- A single failed generation (refusal, truncation, invalid output) is recorded as a failure
  for that item and the run continues; failures are counted in the summary, never dropped.
- Triage is recorded for every question but never short-circuits: a benchmark item must not
  silently leave the evaluation set.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from primum.config import get_settings
from primum.data import load_passages, load_pubmedqa_labeled
from primum.experiments._common import (
    PUBMEDQA_RUNS,
    bootstrap_estimate,
    build_meta,
    macro_f1,
    resolve_results_dir,
    rng_for,
    write_json,
)
from primum.generation import answer_benchmark
from primum.llm import BudgetExceeded, LLMClient, LLMError, LLMUnavailable
from primum.retrieval import build_retriever
from primum.safety import triage
from primum.types import DECISIONS, Hit, PubMedQAExample, Usage
from primum.uncertainty import extract_features, paired_permutation_pvalue, scale_free_stage
from primum.verification import verify_claims
from primum.verification.verify import EntailmentScorer

CONDITIONS = ("closed_book", "rag", "oracle")
CORPUS = "pubmedqa"
# Paired comparisons of accuracy (same questions, two conditions).
COMPARISONS = (("rag", "closed_book"), ("oracle", "rag"))


@dataclass(frozen=True)
class _Task:
    example: PubMedQAExample
    condition: str
    hits: list[Hit]
    triage_level: str


def run(
    *,
    created: str,
    conditions: Sequence[str] = CONDITIONS,
    limit: int | None = None,
    retriever: str | None = None,
    k: int | None = None,
    samples: int | None = None,
    workers: int | None = None,
    n_boot: int = 2000,
    n_perm: int = 10_000,
    results_dir: Path | None = None,
    llm: LLMClient | None = None,
    nli: EntailmentScorer | None = None,
) -> Path:
    """Run the benchmark and write `results/pubmedqa_runs.json`; returns its path.

    `limit` takes the first n questions (the dataset's stable order). `llm` and `nli` replace
    the process-wide client and NLI model (tests, offline replay from another cache).
    """
    settings = get_settings()
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown or not conditions:
        raise ValueError(f"unknown conditions {unknown}; choose from {CONDITIONS}")
    conditions = [c for c in CONDITIONS if c in conditions]  # canonical order
    retriever_name = retriever or settings.default_retriever
    k = k or settings.top_k
    samples = settings.self_consistency_samples if samples is None else samples
    if samples < 1:
        raise ValueError("samples must be >= 1")

    examples = load_pubmedqa_labeled()[:limit]
    tasks = _tasks(examples, conditions, retriever_name, k)

    records: dict[tuple[str, str], dict[str, Any]] = {}
    totals = {"spend": 0.0, "cached_calls": 0}
    stopped: Exception | None = None
    pool = ThreadPoolExecutor(max_workers=workers or settings.llm_concurrency)
    futures = {pool.submit(_answer, task, samples, llm, nli): task for task in tasks}
    progress = tqdm(total=len(tasks), desc="pubmedqa", unit="item")

    def collect(future: Future) -> None:
        task = futures[future]
        try:
            record, used = future.result()
        except LLMError as exc:
            record, used = _failure(task, exc), []
        records[(task.example.qid, task.condition)] = record
        totals["spend"] += sum(u.cost_usd for u in used)
        totals["cached_calls"] += sum(u.cached for u in used)
        progress.update()
        progress.set_postfix_str(
            f"spend ${totals['spend']:.3f}, cached calls {totals['cached_calls']}"
        )

    try:
        for future in as_completed(futures):
            collect(future)
    except (BudgetExceeded, LLMUnavailable) as exc:
        # No further call can succeed: stop, but still write what was completed.
        stopped = exc
    finally:
        # Calls already in flight finish (their responses are cached for the next run);
        # queued ones are dropped, so an unexpected error cannot keep spending.
        pool.shutdown(wait=True, cancel_futures=True)
    if stopped is not None:
        # `as_completed` yields already-finished futures in arbitrary order, so items that
        # completed before (or while) the run stopped may not have been collected yet.
        for future, task in futures.items():
            if (task.example.qid, task.condition) in records or future.cancelled():
                continue
            error = future.exception()
            if error is None or isinstance(error, LLMError):
                collect(future)
    progress.close()
    spent, cached_calls = totals["spend"], totals["cached_calls"]

    ordered = [
        records[(ex.qid, c)] for ex in examples for c in conditions if (ex.qid, c) in records
    ]
    payload = {
        "meta": build_meta(
            created,
            n=len(examples),
            seeds={"seed": settings.seed, "sample_ids": list(range(samples))},
            models={
                "llm": settings.llm_model,
                "llm_effort": settings.llm_effort,
                "retriever": retriever_name,
                "nli": settings.nli_model,
            },
            corpora=[CORPUS],
            conditions=conditions,
            top_k=k,
            self_consistency_samples=samples,
            complete=stopped is None,
            stopped=None if stopped is None else f"{type(stopped).__name__}: {stopped}",
            run_spend_usd=spent,
            cached_calls=cached_calls,
        ),
        "summary": {CORPUS: _summary(ordered, conditions, settings.seed, n_boot)},
        "paired_tests": _paired_tests(ordered, settings.seed, n_boot, n_perm),
        "records": ordered,
    }
    path = write_json(resolve_results_dir(results_dir) / f"{PUBMEDQA_RUNS}.json", payload)
    print(f"wrote {path} ({len(ordered)}/{len(tasks)} items, run spend ${spent:.3f})")
    if stopped is not None:
        raise stopped
    return path


# --------------------------------------------------------------------------------------------
# Work items
# --------------------------------------------------------------------------------------------


def _tasks(
    examples: list[PubMedQAExample], conditions: Sequence[str], retriever_name: str, k: int
) -> list[_Task]:
    """Evidence for every (question, condition), retrieved up front in one batch.

    Retrieval is deterministic and comparatively cheap, so it is done before any LLM call:
    a missing index fails the run immediately instead of after the budget is partly spent.
    """
    questions = [ex.question for ex in examples]
    evidence: dict[str, list[list[Hit]]] = {"closed_book": [[] for _ in examples]}
    if "rag" in conditions:
        evidence["rag"] = build_retriever(retriever_name, CORPUS).search_batch(questions, k=k)
        # The retrieval features need a scale-free score; a retriever without one (BM25 alone)
        # raises `UnscoredRetrieval` here, before any money is spent, not after every record
        # has been written with constant-zero `ret_*` features.
        for hits in evidence["rag"]:
            scale_free_stage(hits)
    if "oracle" in conditions:
        wanted = {ex.passage_id for ex in examples}
        gold = {p.passage_id: p for p in load_passages(CORPUS) if p.passage_id in wanted}
        missing = wanted - gold.keys()
        if missing:
            raise ValueError(f"{len(missing)} gold abstracts are missing from the {CORPUS} corpus")
        evidence["oracle"] = [
            [Hit(passage=gold[ex.passage_id], score=1.0, rank=1, retriever="oracle")]
            for ex in examples
        ]
    levels = [triage(question).level for question in questions]
    return [
        _Task(example, condition, evidence[condition][i], levels[i])
        for i, example in enumerate(examples)
        for condition in conditions
    ]


def _answer(
    task: _Task, samples: int, llm: LLMClient | None, nli: EntailmentScorer | None
) -> tuple[dict[str, Any], list[Usage]]:
    question, hits = task.example.question, task.hits
    answer, usage = answer_benchmark(question, hits, sample_id=0, llm=llm)
    usages = [usage]
    sample_decisions = None
    if task.condition == "rag" and samples > 1:
        sample_decisions = [answer.decision]
        for sample_id in range(1, samples):
            extra, extra_usage = answer_benchmark(question, hits, sample_id=sample_id, llm=llm)
            sample_decisions.append(extra.decision)
            usages.append(extra_usage)

    verdicts = verify_claims(answer.claims, hits, nli=nli)
    features = extract_features(
        hits,
        verdicts,
        # The stated probability of the chosen decision is the generator's verbal confidence.
        verbal_conf=answer.probabilities[answer.decision],
        sample_decisions=sample_decisions,
        probabilities=answer.probabilities,
    )
    hit_ids = [hit.passage.passage_id for hit in hits]
    record = {
        "qid": task.example.qid,
        "condition": task.condition,
        "decision": answer.decision,
        "gold": task.example.label,
        "correct": answer.decision == task.example.label,
        "probabilities": answer.probabilities,
        "sample_decisions": sample_decisions,
        "features": features,
        "n_supported": sum(v.label == "supported" for v in verdicts),
        "n_claims": len(verdicts),
        "claims": [
            {
                "text": v.claim.text,
                "cited": v.claim.cited,
                "label": v.label,
                "p_entail": v.p_entail,
                "p_contra": v.p_contra,
            }
            for v in verdicts
        ],
        "hit_ids": hit_ids,
        "gold_in_hits": task.example.passage_id in hit_ids if hits else None,
        "triage": task.triage_level,
        "cost_usd": sum(u.cost_usd for u in usages),
        "cached": all(u.cached for u in usages),
    }
    return record, usages


def _failure(task: _Task, exc: LLMError) -> dict[str, Any]:
    return {
        "qid": task.example.qid,
        "condition": task.condition,
        "gold": task.example.label,
        "triage": task.triage_level,
        "error": f"{type(exc).__name__}: {exc}",
    }


# --------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------


def _summary(
    records: list[dict[str, Any]], conditions: Sequence[str], seed: int, n_boot: int
) -> dict[str, Any]:
    """Accuracy and macro-F1 per condition with 95% bootstrap intervals over questions.

    Failed items are excluded from the estimates and reported as `n_failed`, so the reader
    can see how much the denominator shrank.
    """
    summary: dict[str, Any] = {}
    for condition in conditions:
        rows = [r for r in records if r["condition"] == condition]
        ok = [r for r in rows if "error" not in r]
        entry: dict[str, Any] = {"n": len(ok), "n_failed": len(rows) - len(ok)}
        if ok:
            correct = np.array([r["correct"] for r in ok], dtype=float)
            labels = np.array(
                [[DECISIONS.index(r["decision"]), DECISIONS.index(r["gold"])] for r in ok]
            )
            entry["accuracy"] = bootstrap_estimate(
                correct, rng_for(seed, "pubmedqa", condition, "accuracy"), n_boot=n_boot
            )
            # Resample (prediction, gold) pairs together: macro-F1 is not a mean over items.
            entry["macro_f1"] = bootstrap_estimate(
                labels,
                rng_for(seed, "pubmedqa", condition, "macro_f1"),
                stat=lambda pairs: macro_f1(pairs[:, 0], pairs[:, 1]),
                n_boot=n_boot,
            )
            entry["supported_claim_rate"] = float(
                sum(r["n_supported"] for r in ok) / max(1, sum(r["n_claims"] for r in ok))
            )
            in_hits = [r["gold_in_hits"] for r in ok if r["gold_in_hits"] is not None]
            if in_hits:
                entry["gold_in_hits_rate"] = float(np.mean(in_hits))
        entry["triage"] = {
            level: sum(r["triage"] == level for r in rows)
            for level in ("emergency", "urgent", "routine")
        }
        summary[condition] = entry
    return summary


def _paired_tests(
    records: list[dict[str, Any]], seed: int, n_boot: int, n_perm: int
) -> dict[str, Any]:
    """Accuracy difference between two conditions on the same questions (paired by qid)."""
    correct = {(r["qid"], r["condition"]): float(r["correct"]) for r in records if "error" not in r}
    tests: dict[str, Any] = {}
    for a, b in COMPARISONS:
        qids = sorted({q for q, c in correct if c == a} & {q for q, c in correct if c == b})
        if not qids:
            continue
        x = np.array([correct[(q, a)] for q in qids])
        y = np.array([correct[(q, b)] for q in qids])
        name = f"{a}_vs_{b}"
        tests[name] = {
            "accuracy_diff": bootstrap_estimate(x - y, rng_for(seed, name, "diff"), n_boot=n_boot),
            "p_value": paired_permutation_pvalue(x, y, n_perm=n_perm, rng=rng_for(seed, name, "p")),
            "n_pairs": len(qids),
        }
    return tests
