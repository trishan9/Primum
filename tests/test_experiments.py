"""Fast tests for primum.experiments: statistical helpers, the split protocol, and RQ2 / RQ3 /
report run end to end on a SYNTHETIC records file in a temporary directory.

The synthetic records exist only inside these tests. Nothing here reads or writes `results/`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from primum.config import get_settings
from primum.experiments import report, rq2_confidence, rq3_conformal
from primum.experiments._common import (
    MIN_PER_CLASS_PER_PART,
    SPLIT_FRACTIONS,
    DegenerateSplit,
    check_stratifiable,
    clopper_pearson,
    load_rag_data,
    macro_f1,
    repeated_splits,
    rng_for,
    stratified_split,
)
from primum.experiments.rq1_retrieval import holm
from primum.experiments.rq3_conformal import plug_in_threshold
from primum.types import DECISIONS
from primum.uncertainty import FEATURES_DEPLOY, FEATURES_FULL, DeploymentBundle, certify_threshold

# --------------------------------------------------------------------------------------------
# Synthetic PubMedQA records (tests only)
# --------------------------------------------------------------------------------------------


def write_synthetic_runs(directory: Path, n: int = 240, seed: int = 0) -> Path:
    """A `pubmedqa_runs.json` whose `rag` records follow a known latent-variable model.

    Each answer has a latent quality z; it is correct with probability sigmoid(2z), and every
    feature is a noisy function of z (some increasing, some decreasing), so a combination of
    features should predict correctness better than chance. Labelled SYNTHETIC in its meta.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    correct = rng.random(n) < 1 / (1 + np.exp(-2 * z))
    gold = rng.integers(0, 3, size=n)
    records = []
    for i in range(n):
        noise = rng.normal(scale=0.8, size=len(FEATURES_FULL))
        signal = {name: float(z[i] + noise[j]) for j, name in enumerate(FEATURES_FULL)}
        for name in ("nli_max_contra", "sc_entropy"):  # signals that point the other way
            signal[name] = -signal[name]
        signal["verbal_conf"] = float(1 / (1 + np.exp(-(z[i] + noise[0]))))
        decision = int(gold[i]) if correct[i] else int((gold[i] + 1) % 3)
        p = np.full(3, (1 - signal["verbal_conf"]) / 2)
        p[decision] = signal["verbal_conf"]
        records.append(
            {
                "qid": f"syn-{i}",
                "condition": "rag",
                "decision": DECISIONS[decision],
                "gold": DECISIONS[int(gold[i])],
                "correct": bool(correct[i]),
                "probabilities": dict(zip(DECISIONS, p.tolist(), strict=True)),
                "features": signal,
            }
        )
    records.append({"qid": "syn-failed", "condition": "rag", "gold": "yes", "error": "LLMRefusal"})
    payload = {
        "meta": {
            "created": "SYNTHETIC",
            "complete": True,
            "models": {"llm": "synthetic-llm", "retriever": "synthetic-retriever"},
        },
        "records": records,
    }
    path = directory / "pubmedqa_runs.json"
    path.write_text(json.dumps(payload))
    return path


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def test_clopper_pearson_matches_closed_forms():
    # With k = 0 the upper limit solves (1 - p)^n = 0.025; with k = n the lower one p^n = 0.025.
    assert clopper_pearson(0, 10) == [0.0, pytest.approx(1 - 0.025 ** (1 / 10))]
    assert clopper_pearson(10, 10) == [pytest.approx(0.025 ** (1 / 10)), 1.0]
    lo, hi = clopper_pearson(5, 20)
    assert lo < 0.25 < hi
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)


def test_macro_f1_matches_sklearn_including_absent_class():
    rng = np.random.default_rng(1)
    for _ in range(20):
        gold = rng.integers(0, 3, size=30)
        pred = rng.integers(0, 2, size=30)  # "maybe" never predicted
        expected = f1_score(gold, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        assert macro_f1(pred, gold) == pytest.approx(expected)
    # A class neither predicted nor present scores 0, as in scikit-learn with zero_division=0.
    assert macro_f1(np.array([0, 1]), np.array([0, 1])) == pytest.approx(2 / 3)


def test_holm_adjustment():
    adjusted = holm([0.01, 0.04, 0.03, 0.5])
    # Sorted 0.01, 0.03, 0.04, 0.5 -> 0.04, 0.09, 0.09 (running max of 0.08), 0.5.
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert holm([0.6, 0.6]) == [1.0, 1.0]


def test_rng_for_is_deterministic_and_keyed():
    assert rng_for(13, "a", 1).random() == rng_for(13, "a", 1).random()
    assert rng_for(13, "a", 1).random() != rng_for(13, "a", 2).random()


def test_stratified_split_partitions_and_preserves_error_rate():
    correct = np.array([True] * 70 + [False] * 30)
    train, cal, test = stratified_split(correct, SPLIT_FRACTIONS, rng_for(0, "x"))
    assert np.array_equal(np.sort(np.concatenate([train, cal, test])), np.arange(100))
    assert (train.size, cal.size, test.size) == (40, 30, 30)
    for part in (train, cal, test):
        assert correct[part].mean() == pytest.approx(0.7)


def test_repeated_splits_are_reproducible_and_differ_between_repeats():
    correct = np.random.default_rng(0).random(200) < 0.6
    first = list(repeated_splits(correct, 3, seed=13))
    second = list(repeated_splits(correct, 3, seed=13))
    for a, b in zip(first, second, strict=True):
        assert all(np.array_equal(x, y) for x, y in zip(a[1:], b[1:], strict=True))
    assert not np.array_equal(first[0][1], first[1][1])


def test_degenerate_data_is_refused_explicitly():
    few_errors = np.array([True] * 100 + [False] * (MIN_PER_CLASS_PER_PART * 2))
    with pytest.raises(DegenerateSplit, match="incorrect"):
        check_stratifiable(few_errors, SPLIT_FRACTIONS)
    with pytest.raises(DegenerateSplit):
        next(repeated_splits(np.ones(50, bool), 1, seed=0))


def test_plug_in_threshold_is_the_smallest_threshold_with_calibration_risk_at_most_alpha():
    conf = np.array([0.9, 0.8, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([True, True, False, True, False, False])
    # Risks at thresholds 0.9, 0.8 (tie answered together), 0.7, 0.6, 0.5: 0, 1/3, 1/4, 2/5, 3/6.
    assert plug_in_threshold(conf, correct, alpha=0.30) == 0.7
    assert plug_in_threshold(conf, correct, alpha=0.40) == 0.6
    assert plug_in_threshold(conf, correct, alpha=0.0) == 0.9
    assert plug_in_threshold(conf[::-1], correct[::-1], alpha=0.30) == 0.7
    assert plug_in_threshold(np.array([0.9]), np.array([False]), alpha=0.1) is None


def test_certified_threshold_is_never_below_the_plug_in_threshold():
    # A rejected binomial test implies an empirical risk below alpha, so the certified
    # threshold is always feasible for the plug-in rule, whose threshold is the smallest such.
    rng = np.random.default_rng(3)
    for _ in range(200):
        conf = rng.random(150)
        correct = rng.random(150) < conf * 0.5 + 0.5
        for alpha in (0.05, 0.1, 0.2):
            g = certify_threshold(conf, correct, alpha=alpha, delta=0.1)
            if g.threshold is not None:
                assert plug_in_threshold(conf, correct, alpha) <= g.threshold


def test_feature_groups_partition_the_full_feature_list():
    grouped = [f for names in rq2_confidence.FEATURE_GROUPS.values() for f in names]
    assert sorted(grouped) == sorted(FEATURES_FULL)


# --------------------------------------------------------------------------------------------
# RQ2 / RQ3 / report end to end on synthetic records
# --------------------------------------------------------------------------------------------


def _stat(value) -> None:
    assert set(value) == {"mean", "ci"}
    assert value["ci"][0] <= value["mean"] + 1e-12 and value["mean"] <= value["ci"][1] + 1e-12


def test_rq2_end_to_end_on_synthetic_records(tmp_path):
    write_synthetic_runs(tmp_path)
    path = rq2_confidence.run(
        created="test", repeats=3, models=("logistic",), n_boot=50, n_perm=99, results_dir=tmp_path
    )
    result = json.loads(path.read_text())
    meta = result["meta"]
    assert meta["n"] == 240 and meta["n_failed_excluded"] == 1
    assert "anthropic_api_key" not in meta["config"]

    table = result["estimators"]["platt"]
    expected = {"stated:verbal_conf", "logistic:deploy", "logistic:full"}
    assert expected | {f"signal:{f}" for f in FEATURES_FULL} == set(table)
    for metrics in table.values():
        assert set(metrics) == set(rq2_confidence.METRICS)
        for value in metrics.values():
            _stat(value)
    # The synthetic features carry signal, and the one-feature fit orients decreasing ones.
    assert table["logistic:full"]["auroc"]["mean"] > 0.7
    assert table["signal:nli_max_contra"]["auroc"]["mean"] > 0.6

    assert set(result["ablation"]) == {"all features"} | {
        f"without {g}" for g in rq2_confidence.FEATURE_GROUPS
    }
    main = {"logistic:deploy", "logistic:full", "stated:verbal_conf", meta["best_single_signal"]}
    assert set(result["reliability"]) == main == set(result["risk_coverage"])
    splits = repeated_splits(load_rag_data(tmp_path).correct, 3, get_settings().seed)
    n_test_predictions = sum(test.size for *_, test in splits)
    for bins in result["reliability"].values():
        assert {"confidence", "accuracy", "count"} <= set(bins[0])
        # Pooled over the 3 repeats: every test prediction lands in exactly one bin.
        assert sum(b["count"] for b in bins) == n_test_predictions
    curve = result["risk_coverage"]["logistic:full"]
    assert len(curve["coverage"]) == len(curve["risk"]) == 100
    for block in result["comparisons"].values():
        for comparison in block.values():
            assert 0 < comparison["brier_p"] <= 1


def test_rq2_leaves_out_features_a_run_did_not_measure(tmp_path):
    # A run with one sample per question has no self-consistency: every value is 0.
    path = write_synthetic_runs(tmp_path)
    payload = json.loads(path.read_text())
    for record in payload["records"]:
        if "features" not in record:  # the synthetic run includes one failed generation
            continue
        record["features"]["sc_agreement"] = 0.0
        record["features"]["sc_entropy"] = 0.0
    path.write_text(json.dumps(payload))

    out = rq2_confidence.run(
        created="test", repeats=1, models=("logistic",), n_boot=20, n_perm=19, results_dir=tmp_path
    )
    result = json.loads(out.read_text())
    meta = result["meta"]
    assert meta["unavailable_features"] == ["sc_agreement", "sc_entropy"]
    assert meta["skipped_feature_sets"] == ["full"]
    table = result["estimators"]["platt"]
    assert "signal:sc_agreement" not in table and "logistic:full" not in table
    assert "logistic:deploy" in table
    assert "without self_consistency" not in result["ablation"]


def test_rq2_with_the_mlp_runs(tmp_path):
    write_synthetic_runs(tmp_path, n=150)
    path = rq2_confidence.run(
        created="test",
        repeats=1,
        feature_sets=("deploy",),
        calibrators=("raw", "isotonic"),
        main_calibrator="isotonic",
        n_boot=20,
        n_perm=19,
        results_dir=tmp_path,
    )
    result = json.loads(path.read_text())
    assert "mlp:deploy" in result["estimators"]["raw"]


def test_rq2_refuses_an_incomplete_run(tmp_path):
    path = write_synthetic_runs(tmp_path)
    payload = json.loads(path.read_text())
    payload["meta"]["complete"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stopped early"):
        rq2_confidence.run(created="test", repeats=1, results_dir=tmp_path)


def test_rq3_end_to_end_on_synthetic_records(tmp_path):
    write_synthetic_runs(tmp_path, n=400)
    deployment = tmp_path / "deployment"
    path = rq3_conformal.run(
        created="test", repeats=4, results_dir=tmp_path, deployment_dir=deployment
    )
    result = json.loads(path.read_text())
    meta = result["meta"]
    n_cal = meta["n_cal"]
    assert meta["delta"] == 0.1 and 118 <= n_cal <= 122
    assert meta["min_selected"] == {"fraction_of_n_cal": 0.1, "value": math.ceil(0.1 * n_cal)}

    # The layout the web app reads: {method: {"0.10": {violation_rate, violation_ci, coverage}}}.
    alphas = {"0.05", "0.10", "0.15", "0.20"}
    assert set(result["selective"]) == {"certified", "certified_hb", "plug_in"}
    for block in (result["selective"], result["single_signal"], result["min_selected_sensitivity"]):
        for by_alpha in block.values():
            assert set(by_alpha) == alphas
            for row in by_alpha.values():
                lo, hi = row["violation_ci"]
                assert lo <= row["violation_rate"] <= hi
                assert row["n_repeats"] == 4
                _stat(row["coverage"])
    assert set(result["single_signal"]) == {f"signal:{f}" for f in FEATURES_DEPLOY}
    # Same confidence, so the uncorrected rule always answers at least as much.
    for a in alphas:
        plug_in = result["selective"]["plug_in"][a]["coverage"]["mean"]
        assert plug_in >= result["selective"]["certified"][a]["coverage"]["mean"]

    for method in ("lac", "aps"):
        sets = result["prediction_sets"][method]
        assert set(sets) == {"set_coverage", "mean_set_size", "empty_set_rate"}
        assert set(sets["set_coverage"]) == alphas

    # /api/calibration payload and the saved bundle agree.
    calibration = result["calibration"]
    n_final_cal = len(calibration["correct"])
    assert len(calibration["p_correct"]) == n_final_cal and 118 <= n_final_cal <= 122
    assert set(calibration["correct"]) <= {0, 1}
    assert set(calibration["thresholds"]) == alphas
    bundle = DeploymentBundle.load(deployment)
    assert bundle.features == FEATURES_DEPLOY and bundle.n_cal == n_final_cal
    assert bundle.llm_model == "synthetic-llm" and bundle.created == "test"
    for g in bundle.guarantees:
        assert calibration["thresholds"][f"{g.alpha:.2f}"] == g.threshold
        recomputed = certify_threshold(
            np.array(calibration["p_correct"]),
            np.array(calibration["correct"]),
            alpha=g.alpha,
            delta=0.1,
            min_selected=calibration["min_selected"],
        )
        assert recomputed.threshold == pytest.approx(g.threshold) or recomputed.threshold is None
        assert (recomputed.threshold is None) == (g.threshold is None)

    # The deployment split keeps a part out of both the fit and the certification (regression:
    # it used to split every question into train / calibration, so the live benchmark mode
    # could only ever claim the guarantee for in-sample questions).
    deployment_meta = result["deployment"]
    n_rows = 400  # the failed record is excluded from the arrays
    heldout_qids = bundle.heldout_qids
    assert len(set(heldout_qids)) == len(heldout_qids) == deployment_meta["n_heldout"]
    assert deployment_meta["n_train"] + n_final_cal + len(heldout_qids) == n_rows
    assert 118 <= len(heldout_qids) <= 122 and "syn-failed" not in heldout_qids
    # The served held-out rows are those questions, scored by the deployed rule.
    records = {
        r["qid"]: r for r in json.loads((tmp_path / "pubmedqa_runs.json").read_text())["records"]
    }
    heldout = calibration["heldout"]
    assert heldout["correct"] == [int(records[q]["correct"]) for q in heldout_qids]
    scored = [bundle.score(records[q]["features"]).p_correct for q in heldout_qids]
    assert heldout["p_correct"] == pytest.approx(scored)
    # The calibration rows are not among them: their scores match none of the held-out rows'
    # (continuous features, so a coincidental equality is practically impossible).
    assert not set(calibration["p_correct"]) & set(heldout["p_correct"])


@pytest.fixture
def tmp_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point `settings.artifacts_dir` at a temporary directory for one test."""
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("PRIMUM_ARTIFACTS_DIR", str(artifacts))
    get_settings.cache_clear()
    yield artifacts
    get_settings.cache_clear()


def test_rq3_saves_the_bundle_under_the_records_generator(tmp_path, tmp_artifacts):
    results = tmp_path / "results"
    results.mkdir()
    write_synthetic_runs(results, n=300)
    live = get_settings().llm_model
    assert live != "synthetic-llm"
    rq3_conformal.run(created="test", repeats=1, results_dir=results)
    saved = tmp_artifacts / "deployment" / "synthetic-llm" / "bundle.json"
    assert saved.is_file()
    assert DeploymentBundle.load(saved.parent).llm_model == "synthetic-llm"
    # The live model's bundle directory is untouched, and nothing lands at the root.
    assert not (tmp_artifacts / "deployment" / live).exists()
    assert not (tmp_artifacts / "deployment" / "bundle.json").exists()


@pytest.mark.parametrize("llm", ["", "../x", "a/b", ".."])
def test_rq3_refuses_to_save_a_bundle_under_an_unsafe_generator_name(tmp_path, tmp_artifacts, llm):
    results = tmp_path / "results"
    results.mkdir()
    path = write_synthetic_runs(results, n=300)
    payload = json.loads(path.read_text())
    payload["meta"]["models"]["llm"] = llm
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cannot save the deployment bundle"):
        rq3_conformal.run(created="test", repeats=1, results_dir=results)
    assert not (tmp_artifacts / "deployment").exists()
    assert not (results / "rq3_conformal.json").exists()


def test_report_with_no_results_says_every_experiment_is_missing(tmp_path):
    docs = tmp_path / "docs" / "RESULTS.md"
    report.run(created="test", results_dir=tmp_path / "results", docs_path=docs)
    text = docs.read_text()
    for name, command in report.FILES.items():
        assert f"`results/{name}.json` does not exist" in text
        assert command in text
    assert not any((tmp_path / "results" / "figures").iterdir())


def test_report_draws_figures_and_tables_from_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    write_synthetic_runs(results, n=300)
    rq2_confidence.run(
        created="test", repeats=2, models=("logistic",), n_boot=20, n_perm=19, results_dir=results
    )
    rq3_conformal.run(created="test", repeats=2, results_dir=results, save_bundle=False)
    # The synthetic records file has no benchmark summary; the report must not read it.
    (results / "pubmedqa_runs.json").unlink()
    docs = tmp_path / "docs" / "RESULTS.md"
    report.run(created="test", results_dir=results, docs_path=docs)
    text = docs.read_text()

    assert "## RQ2: confidence signals" in text and "## RQ3: certified abstention" in text
    assert "`results/rq1_retrieval.json` does not exist" in text
    for figure in ("rq2_estimators", "rq2_reliability", "rq2_risk_coverage", "rq3_selective"):
        for suffix in ("png", "svg"):
            assert (results / "figures" / f"{figure}.{suffix}").stat().st_size > 0
        assert f"../results/figures/{figure}.svg" in text
    # A number in the document is the one in the JSON.
    rq2 = json.loads((results / "rq2_confidence.json").read_text())
    auroc = rq2["estimators"]["platt"]["logistic:full"]["auroc"]
    assert f"{auroc['mean']:.3f} [{auroc['ci'][0]:.3f}, {auroc['ci'][1]:.3f}]" in text
    assert not math.isnan(auroc["mean"])
    # No second-generator directory: the section is a note, and its figure is not drawn.
    assert f"## {report.SECOND_GENERATOR_TITLE}" in text
    assert "`results/sonnet/pubmedqa_runs.json` does not exist" in text
    assert not (results / "figures" / "generators_rq3.svg").exists()


def _synthetic_generator_run(directory: Path, *, llm: str, n: int, samples: int, seed: int) -> None:
    """A synthetic records file with the meta and `rag` summary the benchmark writes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = write_synthetic_runs(directory, n=n, seed=seed)
    payload = json.loads(path.read_text())
    # Records are in dataset order; put the one failed question first so that a shorter run is
    # a prefix of a longer one, as with the benchmark's `--limit`.
    payload["records"].insert(0, payload["records"].pop())
    answered = [r for r in payload["records"] if "error" not in r]
    if samples == 1:  # one sample per question: self-consistency is not measured
        for record in answered:
            record["features"]["sc_agreement"] = 0.0
            record["features"]["sc_entropy"] = 0.0
    accuracy = float(np.mean([r["correct"] for r in answered]))
    payload["meta"].update(
        {
            "n": n + 1,
            "models": {"llm": llm, "retriever": "synthetic-retriever"},
            "conditions": ["rag"],
            "top_k": 5,
            "self_consistency_samples": samples,
        }
    )
    payload["summary"] = {
        "pubmedqa": {
            "rag": {
                "n": len(answered),
                "n_failed": 1,
                "accuracy": {"mean": accuracy, "ci": [accuracy - 0.05, accuracy + 0.05]},
                "macro_f1": {"mean": 0.4, "ci": [0.35, 0.45]},
                "supported_claim_rate": 0.3,
                "gold_in_hits_rate": 0.9,
            }
        }
    }
    path.write_text(json.dumps(payload))


def test_report_second_generator_section_from_synthetic_runs(tmp_path):
    results = tmp_path / "results"
    full, subset = (results / d for d in report.SECOND_GENERATOR_DIRS)
    _synthetic_generator_run(results, llm="main-llm", n=300, samples=5, seed=0)
    _synthetic_generator_run(full, llm="second-llm", n=300, samples=1, seed=1)
    # Synthetic qids are `syn-<i>`, so the subset (149 answered + 1 failed) is the first 150 of the full run.
    _synthetic_generator_run(subset, llm="second-llm", n=149, samples=5, seed=2)
    for directory in (results, full, subset):
        rq2_confidence.run(
            created="test",
            repeats=2,
            models=("logistic",),
            n_boot=20,
            n_perm=19,
            results_dir=directory,
        )
    for directory in (results, full):
        rq3_conformal.run(created="test", repeats=2, results_dir=directory, save_bundle=False)

    docs = tmp_path / "docs" / "RESULTS.md"
    report.run(created="test", results_dir=results, docs_path=docs)
    text = docs.read_text()
    section = text.split(f"## {report.SECOND_GENERATOR_TITLE}", 1)[1]

    # Header lists the second generator's files.
    assert "`results/sonnet/rq3_conformal.json`" in text
    assert "`results/sonnet_sc300/rq2_confidence.json`" in text
    # (1) Benchmark rows for both generators and the subset, numbers from the JSON.
    for directory in (results, full, subset):
        runs = json.loads((directory / "pubmedqa_runs.json").read_text())
        acc = runs["summary"]["pubmedqa"]["rag"]["accuracy"]
        assert f"{acc['mean']:.3f} [{acc['ci'][0]:.3f}, {acc['ci'][1]:.3f}]" in section
    assert "first 150 (not random)" in section
    assert "no paired test between generators" in section
    # (2) RQ2 side by side; the one-sample run's missing features are named.
    for directory, llm in ((results, "main-llm"), (full, "second-llm")):
        rq2 = json.loads((directory / "rq2_confidence.json").read_text())
        stated = rq2["estimators"]["platt"]["stated:verbal_conf"]["auroc"]
        assert f"{stated['mean']:.3f} [{stated['ci'][0]:.3f}, {stated['ci'][1]:.3f}]" in section
        assert f"`{llm}`" in section
    full_rq2 = json.loads((full / "rq2_confidence.json").read_text())
    assert full_rq2["meta"]["unavailable_features"] == ["sc_agreement", "sc_entropy"]
    assert "features not measured and left out: `sc_agreement`, `sc_entropy`" in section
    brier = full_rq2["comparisons"]["vs_stated:verbal_conf"]["logistic:deploy"]["brier_diff"]
    assert _fmt_stat(brier) in section
    # (3) Subset: self-consistency signal against stated, with the not-random caveat.
    sub_rq2 = json.loads((subset / "rq2_confidence.json").read_text())
    sc = sub_rq2["estimators"]["platt"]["signal:sc_agreement"]["auroc"]["mean"]
    assert f"`signal:sc_agreement`: mean AUROC {sc:.3f}" in section
    assert "the first 150 questions in dataset order. This is not a random sample" in section
    subset_runs = json.loads((subset / "pubmedqa_runs.json").read_text())
    full_runs = json.loads((full / "pubmedqa_runs.json").read_text())
    matched, total = report._first_sample_matches(subset_runs, full_runs)
    assert f"On the {total} questions answered in both `second-llm` runs" in section
    assert f"decision equals the one-sample run's decision for {matched}." in section
    # (4) RQ3 side by side, the verbal_conf single signal, and both deployment thresholds.
    for directory in (results, full):
        rq3 = json.loads((directory / "rq3_conformal.json").read_text())
        row = rq3["selective"]["plug_in"]["0.10"]
        assert f"{row['violation_rate']:.1%} [{row['violation_ci'][0]:.1%}" in section
        signal = rq3["single_signal"]["signal:verbal_conf"]["0.20"]["coverage"]
        assert _fmt_stat(signal) in section
        thresholds = rq3["deployment"]["thresholds"]
        expected = ", ".join(
            f"α = {a}: {'none certified' if t is None else f'{t:.3f}'}"
            for a, t in thresholds.items()
        )
        assert expected in section
    assert "artifacts/deployment/second-llm/bundle.json" in section
    # (5) The figure.
    for suffix in ("png", "svg"):
        assert (results / "figures" / f"generators_rq3.{suffix}").stat().st_size > 0
    assert "../results/figures/generators_rq3.svg" in section


def test_report_second_generator_without_subset_says_so(tmp_path):
    results = tmp_path / "results"
    full = results / report.SECOND_GENERATOR_DIRS[0]
    _synthetic_generator_run(full, llm="second-llm", n=300, samples=1, seed=1)
    docs = tmp_path / "RESULTS.md"
    report.run(created="test", results_dir=results, docs_path=docs)
    section = docs.read_text().split(f"## {report.SECOND_GENERATOR_TITLE}", 1)[1]
    assert "`results/sonnet_sc300/rq2_confidence.json` does not exist" in section
    # Without RQ3 files there is nothing to plot.
    assert not (results / "figures" / "generators_rq3.svg").exists()


def test_failed_only_in_names_questions_answered_in_the_other_run():
    subset = {
        "records": [
            {"qid": "a", "decision": "yes"},
            {"qid": "b", "error": "LLMInvalidOutput"},  # answered in the full run
            {"qid": "c", "error": "LLMInvalidOutput"},  # failed in both
        ]
    }
    full = {
        "records": [
            {"qid": "a", "decision": "yes"},
            {"qid": "b", "decision": "no"},
            {"qid": "c", "error": "LLMInvalidOutput"},
        ]
    }
    assert report._failed_only_in(subset, full) == ["b"]


def _fmt_stat(stat) -> str:
    return f"{stat['mean']:.3f} [{stat['ci'][0]:.3f}, {stat['ci'][1]:.3f}]"
