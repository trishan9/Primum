"""RQ2: which signals predict that a generated answer is correct, and is a learned combination
better calibrated than any single signal?

Data: the `rag` records of `pubmedqa_runs.json` (one answer per PubMedQA question, with its
confidence features and whether it matched the expert label).

Protocol (repeated stratified splits, `repeats` times, seeds fixed):

    train 40%        fit the confidence model (standardisation, weights, MLP early stopping)
    calibration 30%  fit the calibrator on the model's predictions for these rows
    test 30%         every reported number

Each split is stratified by correctness, so all three parts have the data's error rate. No
calibration or test row influences a model fit, and no test row influences a calibrator.

Estimators, each with raw / temperature / Platt / isotonic calibration:
- `stated:verbal_conf`: the generator's own stated probability for its decision;
- `signal:<f>`: each feature alone, oriented and scaled by a one-feature logistic fit on train;
- `logistic:{deploy,full}`, `mlp:{deploy,full}`: learned combinations of FEATURES_DEPLOY /
  FEATURES_FULL.

Metrics on the test part: AUROC (does confidence rank right above wrong answers?), AURC and
E-AURC (selective risk averaged over coverage; E-AURC removes the part due to accuracy),
ECE (10 equal-width bins) and the Brier score. Reported as the mean over repeats and the
central 95% range of the repeats.

Also: a feature-group ablation (drop retrieval / NLI / verbal / self-consistency signals from
the full logistic model); reliability bins and risk-coverage curves of the main estimators
under the main calibrator; and paired comparisons of each learned combination with the stated
probability and with the best single signal, paired by question.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from primum.config import get_settings
from primum.experiments._common import (
    SPLIT_FRACTIONS,
    bootstrap_estimate,
    build_meta,
    load_rag_data,
    repeated_splits,
    resolve_results_dir,
    rng_for,
    spread,
    write_json,
)
from primum.experiments._models import (
    FEATURE_SETS,
    MODEL_KINDS,
    STATED,
    Estimator,
    combined,
    single_signal,
    stated,
)
from primum.uncertainty import (
    CALIBRATORS,
    FEATURES_FULL,
    aurc,
    auroc,
    brier,
    e_aurc,
    ece,
    paired_permutation_pvalue,
    reliability_bins,
    risk_coverage_curve,
)

METRICS = ("auroc", "aurc", "e_aurc", "ece", "brier")
FEATURE_GROUPS: dict[str, list[str]] = {
    "retrieval": ["ret_top", "ret_mean", "ret_margin", "ret_agree"],
    "nli": ["nli_mean_entail", "nli_min_entail", "nli_max_contra", "frac_supported", "n_claims"],
    "verbal": ["verbal_conf", "verbal_margin"],
    "self_consistency": ["sc_agreement", "sc_entropy"],
}
# Risk-coverage curves are averaged over repeats on this grid of coverages.
COVERAGE_GRID = np.round(np.arange(1, 101) / 100, 2)


def run(
    *,
    created: str,
    repeats: int = 200,
    models: Sequence[str] = MODEL_KINDS,
    feature_sets: Sequence[str] = ("deploy", "full"),
    calibrators: Sequence[str] = tuple(CALIBRATORS),
    main_calibrator: str = "platt",
    ablation_model: str = "logistic",
    n_bins: int = 10,
    n_boot: int = 2000,
    n_perm: int = 10_000,
    results_dir: Path | None = None,
) -> Path:
    """Run RQ2 on `pubmedqa_runs.json` and write `results/rq2_confidence.json`.

    `main_calibrator` is fixed in advance (not chosen from these results); it is the one used
    for the curves, the ablation and the paired comparisons.
    """
    settings = get_settings()
    if main_calibrator not in calibrators:
        raise ValueError(f"main_calibrator {main_calibrator!r} must be one of {list(calibrators)}")
    directory = resolve_results_dir(results_dir)
    data = load_rag_data(directory)
    X, y = data.X, data.correct

    # A feature with one value for every question was not measured in this run (e.g. the
    # self-consistency features of a run with one sample per question). Scored as a signal it
    # would read as "AUROC 0.5", a finding about a measurement that was never taken, so it is
    # left out everywhere and listed in the metadata instead.
    unavailable = [f for i, f in enumerate(FEATURES_FULL) if np.ptp(X[:, i]) == 0]
    available = [f for f in FEATURES_FULL if f not in unavailable]
    skipped_sets = [
        fs for fs in feature_sets if fs != "deploy" and set(FEATURE_SETS[fs]) & set(unavailable)
    ]
    feature_sets = [fs for fs in feature_sets if fs not in skipped_sets]

    estimators = [stated(), *(single_signal(f) for f in available)]
    estimators += [combined(kind, fs) for kind in models for fs in feature_sets]
    ablations = [
        Estimator("all features", ablation_model, tuple(available)),
        *(
            Estimator(
                f"without {group}",
                ablation_model,
                tuple(f for f in available if f not in names),
            )
            for group, names in FEATURE_GROUPS.items()
            if set(names) - set(unavailable)
        ),
    ]

    scores: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    ablation_scores: dict[str, list[dict[str, float]]] = defaultdict(list)
    test_predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for repeat, train, cal, test in tqdm(
        repeated_splits(y, repeats, settings.seed), total=repeats, desc="rq2", unit="split"
    ):
        seed = settings.seed + repeat
        for estimator in estimators:
            calibrated = _fit_and_calibrate(estimator, calibrators, X, y, train, cal, test, seed)
            for name, p in calibrated.items():
                scores[estimator.name][name].append(_metrics(p, y[test], n_bins))
            test_predictions[estimator.name].append((test, calibrated[main_calibrator]))
        for estimator in ablations:
            p = _fit_and_calibrate(estimator, [main_calibrator], X, y, train, cal, test, seed)
            ablation_scores[estimator.name].append(_metrics(p[main_calibrator], y[test], n_bins))

    # Chosen on the test results, so the selection favours the baseline: a learned combination
    # that beats it is compared against the strongest single signal in hindsight.
    signals = [e.name for e in estimators if e.name.startswith("signal:")]
    best_signal = max(
        signals, key=lambda s: np.mean([m["auroc"] for m in scores[s][main_calibrator]])
    )
    learned = [
        e.name for e in estimators if e.kind in MODEL_KINDS and not e.name.startswith("signal:")
    ]
    main = [*learned, STATED, best_signal]

    payload = {
        "meta": build_meta(
            created,
            n=int(y.size),
            seeds={
                "seed": settings.seed,
                "split": "rng_for(seed, 'split', repeat)",
                "repeats": repeats,
            },
            models={**data.runs_meta.get("models", {}), "confidence": list(models)},
            corpora=["pubmedqa"],
            source="pubmedqa_runs.json, condition rag",
            n_correct=int(y.sum()),
            accuracy=float(y.mean()),
            n_failed_excluded=data.n_failed,
            split_fractions={
                "train": SPLIT_FRACTIONS[0],
                "calibration": SPLIT_FRACTIONS[1],
                "test": SPLIT_FRACTIONS[2],
            },
            calibrators=list(calibrators),
            main_calibrator=main_calibrator,
            best_single_signal=best_signal,
            feature_sets={fs: FEATURE_SETS[fs] for fs in feature_sets},
            unavailable_features=unavailable,
            skipped_feature_sets=skipped_sets,
            feature_groups=FEATURE_GROUPS,
            ece_bins=n_bins,
            intervals={
                "estimators, ablation, auroc_diff": "mean over repeats and central 95% range "
                "of the per-repeat values (split variability)",
                "brier_diff": "95% percentile bootstrap over questions of per-question "
                "differences in mean test Brier loss",
                "brier_p": "two-sided paired sign-flip permutation test over questions",
                "reliability": "test predictions pooled over all repeats",
                "risk_coverage": "per-repeat curves interpolated on a coverage grid, averaged",
            },
        ),
        "estimators": {
            cal_name: {est.name: _aggregate(scores[est.name][cal_name]) for est in estimators}
            for cal_name in calibrators
        },
        "ablation": {name: _aggregate(rows) for name, rows in ablation_scores.items()},
        "comparisons": _comparisons(
            learned,
            [STATED, best_signal],
            test_predictions,
            scores,
            main_calibrator,
            y,
            settings.seed,
            n_boot,
            n_perm,
        ),
        "reliability": {name: _reliability(test_predictions[name], y, n_bins) for name in main},
        "risk_coverage": {name: _risk_coverage(test_predictions[name], y) for name in main},
    }
    return write_json(directory / "rq2_confidence.json", payload)


def _fit_and_calibrate(
    estimator: Estimator,
    calibrators: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    cal: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Test-set probabilities under each calibrator: model on train, calibrator on cal."""
    model = estimator.fit(X[train], y[train], seed)
    p_cal, p_test = estimator.score(model, X[cal]), estimator.score(model, X[test])
    return {name: CALIBRATORS[name]().fit(p_cal, y[cal]).predict(p_test) for name in calibrators}


def _metrics(p: np.ndarray, correct: np.ndarray, n_bins: int) -> dict[str, float]:
    return {
        "auroc": auroc(p, correct),
        "aurc": aurc(p, correct),
        "e_aurc": e_aurc(p, correct),
        "ece": ece(p, correct, n_bins=n_bins),
        "brier": brier(p, correct),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, Any]:
    return {metric: spread([row[metric] for row in rows]) for metric in METRICS}


def _reliability(
    predictions: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray, n_bins: int
) -> list[dict[str, float]]:
    """Reliability bins of every test prediction made in any repeat.

    Pooling the (prediction, outcome) pairs, rather than averaging each question's predictions
    first, keeps the diagram about the predictions whose ECE is reported: averaging would
    shrink extreme probabilities and make the model look better calibrated than any single
    fitted model is.
    """
    p = np.concatenate([pred for _, pred in predictions])
    outcome = np.concatenate([y[idx] for idx, _ in predictions])
    return [dict(b) for b in reliability_bins(p, outcome, n_bins=n_bins)]


def _risk_coverage(
    predictions: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray
) -> dict[str, list[float]]:
    """Mean over repeats of selective risk at each coverage of `COVERAGE_GRID`.

    Each repeat's curve comes from its own model, so confidences of different repeats are not
    on one scale and are never sorted together; the curves are aligned by coverage instead.
    """
    curves = []
    for idx, pred in predictions:
        coverage, risk = risk_coverage_curve(pred, y[idx])
        curves.append(np.interp(COVERAGE_GRID, coverage, risk))
    return {"coverage": COVERAGE_GRID.tolist(), "risk": np.mean(curves, axis=0).tolist()}


def _comparisons(
    models: list[str],
    references: list[str],
    predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    scores: dict[str, dict[str, list[dict[str, float]]]],
    main_calibrator: str,
    y: np.ndarray,
    seed: int,
    n_boot: int,
    n_perm: int,
) -> dict[str, Any]:
    """Each learned model against each reference, under the main calibrator.

    Brier loss decomposes over questions, so it is compared per question: each question's
    loss is averaged over the repeats in which it was a test item, and the two estimators'
    losses are compared by a paired permutation test over questions. Caveat: the models were
    trained on overlapping data, so the per-question differences are not strictly independent
    (the usual caveat for comparing learners by resampling; Dietterich 1998); the p-value is
    indicative. AUROC does not decompose, so its per-repeat difference is summarised instead.
    """
    losses = {name: _per_question_loss(predictions[name], y) for name in {*models, *references}}
    out: dict[str, Any] = {}
    for reference in references:
        block: dict[str, Any] = {}
        for model in models:
            seen = ~np.isnan(losses[model]) & ~np.isnan(losses[reference])
            a, b = losses[model][seen], losses[reference][seen]
            key = ("rq2", model, reference)
            auroc_diff = [
                m["auroc"] - r["auroc"]
                for m, r in zip(
                    scores[model][main_calibrator], scores[reference][main_calibrator], strict=True
                )
            ]
            block[model] = {
                "brier_diff": bootstrap_estimate(
                    a - b, rng_for(seed, *key, "brier"), n_boot=n_boot
                ),
                "brier_p": paired_permutation_pvalue(
                    a, b, n_perm=n_perm, rng=rng_for(seed, *key, "p")
                ),
                "auroc_diff": spread(auroc_diff),
                "auroc_better_share": float(np.mean(np.array(auroc_diff) > 0)),
                "n_questions": int(seen.sum()),
            }
        out[f"vs_{reference}"] = block
    return out


def _per_question_loss(
    predictions: list[tuple[np.ndarray, np.ndarray]], y: np.ndarray
) -> np.ndarray:
    """Mean squared error of each question over the repeats it was tested in (NaN if never)."""
    total, count = np.zeros(y.size), np.zeros(y.size)
    for idx, pred in predictions:
        np.add.at(total, idx, (pred - y[idx]) ** 2)
        np.add.at(count, idx, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)
