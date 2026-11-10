"""RQ3: does the certified threshold keep selective risk <= alpha on unseen questions, and how
much coverage does it cost?

Same records and the same repeated stratified splits as RQ2 (repeat r uses the identical
train / calibration / test rows in both experiments), but the roles are stricter:

    train 40%        fit the confidence model AND its calibrator. The calibrator is learnt from
                     cross-fitted (out-of-fold) predictions of the training rows, so no row
                     outside train touches either.
    calibration 30%  `certify_threshold` only: the Learn-then-Test walk over its confidences.
                     Nothing is fitted here. (Fitting isotonic regression on these labels
                     would make the candidate thresholds label-dependent and void the test.)
    test 30%         selective risk and coverage of the certified rule.

`min_selected` is a fixed fraction of n_cal, set before any label is seen; the sensitivity to
that choice is reported for a few fractions.

Per method and alpha the result is the violation rate P(test risk > alpha) over repeats with a
Clopper-Pearson interval, mean coverage, mean selective risk among the repeats that answered
anything, and the share of repeats in which a threshold was certified at all. Methods:
- `certified`: the main confidence with the exact binomial test (the system's procedure);
- `certified_hb`: the same with the Hoeffding-Bentkus p-value (valid for any bounded loss,
  more conservative for 0/1 loss);
- `plug_in`: the naive rule, the smallest threshold whose *calibration* risk is <= alpha, with
  no correction for estimation error; it is what the guarantee is meant to improve on;
- `single_signal`: each feature alone as the confidence, certified the same way.

Two caveats a reader of the violation rate needs:
- The guarantee concerns the *true* selective risk. Test risk is estimated on ~300 questions,
  so when the true risk sits just below alpha, test noise alone makes "test risk > alpha"
  fairly frequent: the empirical violation rate overstates the true violation probability
  near the boundary.
- The repeats re-split one set of questions, so they are not independent draws from the
  question distribution. The Clopper-Pearson interval quantifies split-to-split Monte-Carlo
  error given this dataset, not sampling error of the dataset itself.

Also: conformal prediction sets (LAC, APS) over the generator's stated yes/no/maybe
probabilities, with their coverage and mean size; and the deployment bundle, fitted on one
fixed split with the same fractions (train 40% / calibration 30% / held out 30%) and saved for
the live system, together with the {p_correct, correct} of its calibration and held-out rows
that `/api/calibration` serves.

What the certificate covers: new questions drawn i.i.d. from the calibration distribution
(PubMedQA expert-labelled questions) answered by the same pipeline (generator, retriever,
features), under the monotone-risk condition in `primum.uncertainty.conformal`. Among the
PubMedQA questions that means the held-out part of the deployment split only; the others were
fitted or certified on. For free-form chat questions the certified threshold is an operating
point, not a guarantee.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from primum.config import get_settings, is_safe_model_name
from primum.experiments._common import (
    SPLIT_FRACTIONS,
    RagData,
    alpha_key,
    build_meta,
    ceil_fraction,
    check_stratifiable,
    clopper_pearson,
    load_rag_data,
    repeated_splits,
    resolve_results_dir,
    rng_for,
    spread,
    stratified_split,
    write_json,
)
from primum.experiments._models import (
    FEATURE_SETS,
    Estimator,
    combined,
    fit_cross_fitted_calibrator,
    single_signal,
)
from primum.uncertainty import (
    ALPHAS,
    DeploymentBundle,
    aps_sets,
    certify_threshold,
    lac_sets,
)

N_GRID = 100
GUARANTEE = (
    "For new questions drawn i.i.d. from the calibration distribution (PubMedQA expert-labelled "
    "questions) and answered by the same pipeline, P(selective risk <= alpha) >= 1 - delta "
    "over the draw of the calibration set, under the monotone-risk condition stated in "
    "primum.uncertainty.conformal. For the deployed bundle the PubMedQA questions it covers are "
    "its held-out rows only; the rest were fitted or certified on. For free-form chat questions "
    "the threshold is an operating point, not a guarantee."
)


def run(
    *,
    created: str,
    repeats: int = 200,
    alphas: Sequence[float] = ALPHAS,
    delta: float | None = None,
    model: str = "logistic",
    feature_set: str = "deploy",
    calibrator: str = "platt",
    min_selected_frac: float = 0.10,
    sensitivity_fracs: Sequence[float] = (0.05, 0.10, 0.20),
    save_bundle: bool = True,
    results_dir: Path | None = None,
    deployment_dir: Path | None = None,
) -> Path:
    """Run RQ3 on `pubmedqa_runs.json`, write `results/rq3_conformal.json`, save the bundle.

    `model`, `feature_set` and `calibrator` choose the main confidence; they are fixed in
    advance rather than picked from RQ2's test results, which share these questions.
    `feature_set="deploy"` is the one the live chat pipeline can compute.
    """
    settings = get_settings()
    delta = settings.delta if delta is None else delta
    directory = resolve_results_dir(results_dir)
    data = load_rag_data(directory)
    if save_bundle and deployment_dir is None:
        # The bundle directory is named after the records' generator; refuse a name that is
        # missing or would not stay one path component under artifacts/deployment/.
        records_llm = data.runs_meta.get("models", {}).get("llm", "")
        if not is_safe_model_name(records_llm):
            raise ValueError(
                f"cannot save the deployment bundle: the records' generator "
                f"(meta.models.llm = {records_llm!r}) is missing or not a safe directory name; "
                "pass deployment_dir or save_bundle=False"
            )
    X, y = data.X, data.correct
    main = combined(model, feature_set)
    signals = [single_signal(f) for f in FEATURE_SETS[feature_set]]
    fracs = sorted({min_selected_frac, *sensitivity_fracs})

    # outcomes[method][alpha] -> one dict per repeat
    outcomes: dict[str, dict[float, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    sets: dict[str, dict[float, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    n_cal = 0
    for repeat, train, cal, test in tqdm(
        repeated_splits(y, repeats, settings.seed), total=repeats, desc="rq3", unit="split"
    ):
        seed = settings.seed + repeat
        n_cal = cal.size
        min_selected = ceil_fraction(min_selected_frac, n_cal)

        conf = _fit_confidence(main, calibrator, X, y, train, seed)
        conf_cal, conf_test = conf(X[cal]), conf(X[test])
        for alpha in alphas:
            for method, test_kind in (("certified", "binomial"), ("certified_hb", "hb")):
                g = certify_threshold(
                    conf_cal,
                    y[cal],
                    alpha=alpha,
                    delta=delta,
                    n_grid=N_GRID,
                    min_selected=min_selected,
                    method=test_kind,
                )
                outcomes[method][alpha].append(_evaluate(g.threshold, conf_test, y[test], alpha))
            naive = plug_in_threshold(conf_cal, y[cal], alpha)
            outcomes["plug_in"][alpha].append(_evaluate(naive, conf_test, y[test], alpha))
            for frac in fracs:
                g = certify_threshold(
                    conf_cal,
                    y[cal],
                    alpha=alpha,
                    delta=delta,
                    n_grid=N_GRID,
                    min_selected=ceil_fraction(frac, n_cal),
                )
                outcomes[_sensitivity_name(frac)][alpha].append(
                    _evaluate(g.threshold, conf_test, y[test], alpha)
                )

        for signal in signals:
            # Certification uses only the order of the confidences, so no calibrator is needed.
            fitted = signal.fit(X[train], y[train], seed)
            s_cal, s_test = signal.score(fitted, X[cal]), signal.score(fitted, X[test])
            for alpha in alphas:
                g = certify_threshold(
                    s_cal,
                    y[cal],
                    alpha=alpha,
                    delta=delta,
                    n_grid=N_GRID,
                    min_selected=min_selected,
                )
                outcomes[signal.name][alpha].append(_evaluate(g.threshold, s_test, y[test], alpha))

        for alpha in alphas:
            for name, s in _prediction_sets(
                data.probs, data.gold, cal, test, alpha, settings.seed, repeat
            ).items():
                sets[name][alpha].append(s)

    main_methods = ("certified", "certified_hb", "plug_in")
    payload: dict[str, Any] = {
        "meta": build_meta(
            created,
            n=int(y.size),
            seeds={
                "seed": settings.seed,
                "split": "rng_for(seed, 'split', repeat)",
                "repeats": repeats,
            },
            models={
                **data.runs_meta.get("models", {}),
                "confidence": main.name,
                "calibrator": calibrator,
            },
            corpora=["pubmedqa"],
            source="pubmedqa_runs.json, condition rag",
            delta=delta,
            alphas=list(alphas),
            n_correct=int(y.sum()),
            accuracy=float(y.mean()),
            n_failed_excluded=data.n_failed,
            split_fractions={
                "train": SPLIT_FRACTIONS[0],
                "calibration": SPLIT_FRACTIONS[1],
                "test": SPLIT_FRACTIONS[2],
            },
            n_cal=n_cal,
            n_grid=N_GRID,
            min_selected={
                "fraction_of_n_cal": min_selected_frac,
                "value": ceil_fraction(min_selected_frac, n_cal),
            },
            calibrator_fitting="cross-fitted on the training split (5 folds); never on calibration rows",
            guarantee=GUARANTEE,
            intervals={
                "violation_ci": "95% Clopper-Pearson over repeats (split Monte-Carlo error)",
                "coverage, risk, threshold, set metrics": "mean over repeats and central 95% "
                "range of the per-repeat values",
            },
        ),
        "selective": {m: _aggregate(outcomes[m]) for m in main_methods},
        "single_signal": {s.name: _aggregate(outcomes[s.name]) for s in signals},
        "min_selected_sensitivity": {
            _sensitivity_name(f): _aggregate(outcomes[_sensitivity_name(f)]) for f in fracs
        },
        "prediction_sets": {
            name: {
                metric: {alpha_key(a): spread([s[metric] for s in by_alpha[a]]) for a in alphas}
                for metric in ("set_coverage", "mean_set_size", "empty_set_rate")
            }
            for name, by_alpha in sets.items()
        },
    }

    bundle, calibration, n_train = _deployment(
        main, calibrator, data, alphas, delta, min_selected_frac, created, settings.seed
    )
    payload["calibration"] = calibration
    payload["deployment"] = {
        "saved": save_bundle,
        "n_train": n_train,
        "n_cal": bundle.n_cal,
        "n_heldout": len(bundle.heldout_qids),
        "features": bundle.features,
        "thresholds": calibration["thresholds"],
    }
    if save_bundle:
        # Saved under the model that generated these records, not the one currently
        # configured, so analysing another model's run cannot replace the live bundle.
        bundle.save(
            deployment_dir
            if deployment_dir is not None
            else settings.deployment_dir_for(bundle.llm_model)
        )
    return write_json(directory / "rq3_conformal.json", payload)


# --------------------------------------------------------------------------------------------
# Thresholds and their evaluation
# --------------------------------------------------------------------------------------------


def plug_in_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float) -> float | None:
    """The smallest observed confidence whose empirical calibration risk is <= alpha.

    This is the uncorrected rule: it treats the calibration error rate as if it were the true
    one. Thresholds are the observed confidences, ties are answered together, and there is no
    minimum selection size. None if no threshold qualifies.
    """
    order = np.argsort(-conf, kind="stable")
    conf_desc = conf[order]
    errors = np.cumsum(~correct[order])
    # Selecting at the value of a tie group selects the whole group: its last index.
    values, starts, counts = np.unique(-conf_desc, return_index=True, return_counts=True)
    n_selected = starts + counts
    feasible = errors[n_selected - 1] / n_selected <= alpha
    return float(-values[feasible][-1]) if feasible.any() else None


def _evaluate(
    threshold: float | None, conf_test: np.ndarray, correct_test: np.ndarray, alpha: float
) -> dict[str, Any]:
    """Coverage and selective risk on the test part. No threshold = abstain on everything."""
    answered = np.zeros(conf_test.size, bool) if threshold is None else conf_test >= threshold
    n = int(answered.sum())
    risk = float(np.mean(~correct_test[answered])) if n else None
    return {
        "threshold": threshold,
        "coverage": n / conf_test.size,
        "risk": risk,
        # Abstaining on everything makes no errors, so it never violates the target.
        "violation": risk is not None and risk > alpha,
    }


def _aggregate(by_alpha: dict[float, list[dict[str, Any]]]) -> dict[str, Any]:
    """`{"0.10": {violation_rate, violation_ci, coverage, risk, certified_rate, threshold}}`."""
    out: dict[str, Any] = {}
    for alpha, rows in sorted(by_alpha.items()):
        violations = sum(row["violation"] for row in rows)
        answered = [row for row in rows if row["threshold"] is not None]
        out[alpha_key(alpha)] = {
            "violation_rate": violations / len(rows),
            "violation_ci": clopper_pearson(violations, len(rows)),
            "coverage": spread([row["coverage"] for row in rows]),
            "risk": spread([row["risk"] for row in rows if row["risk"] is not None]),
            "certified_rate": len(answered) / len(rows),
            "threshold": spread([row["threshold"] for row in answered]),
            "n_repeats": len(rows),
        }
    return out


def _sensitivity_name(frac: float) -> str:
    return f"min_selected={frac:.0%} of n_cal"


# --------------------------------------------------------------------------------------------
# Confidence fitting (train rows only)
# --------------------------------------------------------------------------------------------


def _fit_confidence(
    estimator: Estimator,
    calibrator: str,
    X: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    seed: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Model and calibrator from the training rows; returns rows -> calibrated P(correct)."""
    model = estimator.fit(X[train], y[train], seed)
    calib = fit_cross_fitted_calibrator(estimator, calibrator, X[train], y[train], seed)
    return lambda rows: calib.predict(estimator.score(model, rows))


# --------------------------------------------------------------------------------------------
# Prediction sets
# --------------------------------------------------------------------------------------------


def _prediction_sets(
    probs: np.ndarray,
    gold: np.ndarray,
    cal: np.ndarray,
    test: np.ndarray,
    alpha: float,
    seed: int,
    repeat: int,
) -> dict[str, dict[str, float]]:
    """LAC and APS sets over the stated yes/no/maybe probabilities, calibrated on `cal`.

    The stated probabilities are fixed outputs of the generator (nothing is fitted), so
    calibration and test rows are exchangeable and the marginal coverage guarantee applies.
    """
    out = {}
    for name, mask in (
        ("lac", lac_sets(probs[cal], gold[cal], probs[test], alpha)),
        (
            "aps",
            aps_sets(
                probs[cal], gold[cal], probs[test], alpha, rng=rng_for(seed, "aps", repeat, alpha)
            ),
        ),
    ):
        sizes = mask.sum(axis=1)
        out[name] = {
            "set_coverage": float(mask[np.arange(test.size), gold[test]].mean()),
            "mean_set_size": float(sizes.mean()),
            "empty_set_rate": float(np.mean(sizes == 0)),
        }
    return out


# --------------------------------------------------------------------------------------------
# Deployment bundle and the /api/calibration payload
# --------------------------------------------------------------------------------------------


def _deployment(
    main: Estimator,
    calibrator: str,
    data: RagData,
    alphas: Sequence[float],
    delta: float,
    min_selected_frac: float,
    created: str,
    seed: int,
) -> tuple[DeploymentBundle, dict[str, Any], int]:
    """Fit the live system's rule on one fixed split, with the RQ3 proportions.

    The evaluation above validated the *procedure*; the deployed rule is one run of it, on
    the same train / calibration / test fractions, so the rule deployed is the rule measured.
    The test part is kept out of everything (it is not folded into training): every other
    PubMedQA question was fitted on or certified on, and the certificate says nothing about
    those. The held-out qids go into the bundle, so the live benchmark mode claims the
    guarantee only for them.

    The `/api/calibration` payload holds both parts, scored by the deployed model:
    - `p_correct` / `correct`: the calibration rows, on which the deployed thresholds were
      certified; re-running `certify_threshold` on them reproduces those thresholds.
    - `heldout`: the test rows, which nothing was fitted or certified on; coverage and risk
      read there are out-of-sample. Read on the calibration rows they are not, because the
      threshold was chosen on those very rows.
    """
    X, y = data.X, data.correct
    check_stratifiable(y, SPLIT_FRACTIONS)
    train, cal, heldout = stratified_split(y, SPLIT_FRACTIONS, rng_for(seed, "final"))
    model = main.fit(X[train], y[train], seed)
    calib = fit_cross_fitted_calibrator(main, calibrator, X[train], y[train], seed)
    X_main = data.columns(main.features)
    min_selected = ceil_fraction(min_selected_frac, cal.size)
    bundle = DeploymentBundle.build(
        model,
        calib,
        X_main[cal],
        y[cal],
        features=list(main.features),
        delta=delta,
        llm_model=data.runs_meta.get("models", {}).get("llm", ""),
        retriever=data.runs_meta.get("models", {}).get("retriever", ""),
        created=created,
        alphas=alphas,
        n_grid=N_GRID,
        min_selected=min_selected,
        heldout_qids=[data.qids[i] for i in heldout],
    )

    def scored(rows: np.ndarray) -> dict[str, list[float] | list[int]]:
        p_correct = calib.predict(model.predict_proba(X_main[rows]))
        return {
            "p_correct": [float(p) for p in p_correct],
            "correct": [int(c) for c in y[rows]],
        }

    calibration = {
        **scored(cal),
        "thresholds": {alpha_key(g.alpha): g.threshold for g in bundle.guarantees},
        "delta": delta,
        "n_grid": N_GRID,
        "min_selected": min_selected,
        "heldout": scored(heldout),
        "source": (
            "deployed model's calibrated P(correct). p_correct/correct: the calibration rows "
            "the deployed thresholds were certified on (in-sample for the threshold). "
            "heldout: rows never used to fit, calibrate or certify (out-of-sample)."
        ),
    }
    return bundle, calibration, int(train.size)
