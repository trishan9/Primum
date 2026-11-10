"""Tests for primum.uncertainty.calibration."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression, isotonic_regression

from primum.uncertainty.calibration import (
    CALIBRATORS,
    IdentityCalibration,
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
    brier,
    calibrator_from_json,
    ece,
    pool_adjacent_violators,
    reliability_bins,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _overconfident(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Scores whose logits are 2.5x too large and shifted: the model is overconfident."""
    true_logit = rng.normal(0.8, 1.2, size=n)
    y = (rng.uniform(size=n) < _sigmoid(true_logit)).astype(int)
    return _sigmoid(2.5 * true_logit + 0.7), y


# --------------------------------------------------------------------------------------------
# Pool-adjacent-violators
# --------------------------------------------------------------------------------------------


def test_pav_small_examples():
    assert pool_adjacent_violators(np.array([1.0, 3.0, 2.0, 4.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert pool_adjacent_violators(np.array([3.0, 2.0, 1.0])).tolist() == [2.0, 2.0, 2.0]
    assert pool_adjacent_violators(np.array([1.0, 2.0, 3.0])).tolist() == [1.0, 2.0, 3.0]
    assert pool_adjacent_violators(np.array([])).tolist() == []
    # Weighted: the pooled value is the weighted mean (3*1 + 1*3) / 4.
    weighted = pool_adjacent_violators(np.array([3.0, 1.0]), np.array([1.0, 3.0]))
    assert weighted.tolist() == [1.5, 1.5]


@pytest.mark.parametrize("seed", range(8))
def test_pav_is_monotone_and_matches_scikit_learn(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    y = rng.normal(size=n) + 0.01 * np.arange(n) * rng.uniform()
    weights = rng.uniform(0.1, 3.0, size=n)
    for w in (None, weights):
        fitted = pool_adjacent_violators(y, w)
        assert (np.diff(fitted) >= 0).all()
        np.testing.assert_allclose(fitted, isotonic_regression(y, sample_weight=w), atol=1e-12)


def test_pav_rejects_bad_weights():
    with pytest.raises(ValueError):
        pool_adjacent_violators(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        pool_adjacent_violators(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.parametrize("seed", range(5))
def test_isotonic_calibration_matches_scikit_learn(seed: int):
    rng = np.random.default_rng(seed)
    n = 300
    # Rounded scores create repeated x values, the case where a naive PAV goes wrong.
    p = np.round(rng.beta(2, 2, size=n), 2 if seed % 2 else 6)
    y = (rng.uniform(size=n) < p**1.5).astype(int)
    ours = IsotonicCalibration().fit(p, y)
    reference = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    grid = np.concatenate([np.linspace(-0.2, 1.2, 500), p])  # includes out-of-range inputs
    predicted = ours.predict(grid)
    np.testing.assert_allclose(predicted, reference.predict(grid), atol=1e-12)
    assert (np.diff(ours.predict(np.sort(grid))) >= -1e-15).all()
    assert predicted.min() >= 0.0 and predicted.max() <= 1.0


# --------------------------------------------------------------------------------------------
# ECE, Brier, reliability bins
# --------------------------------------------------------------------------------------------


def test_ece_by_hand():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 0])
    # Low bin: confidence 0.1, accuracy 0 (gap 0.1). High bin: confidence 0.9, accuracy 0.5
    # (gap 0.4). Each holds half of the data: 0.5 * 0.1 + 0.5 * 0.4.
    assert ece(p, y, n_bins=10) == pytest.approx(0.25)
    assert ece(p, y, n_bins=2, strategy="equal_mass") == pytest.approx(0.25)
    # A single bin compares overall confidence with overall accuracy.
    assert ece(p, y, n_bins=1) == pytest.approx(abs(0.5 - 0.25))


def test_ece_is_small_for_calibrated_and_large_for_overconfident_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=50_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    for strategy in ("equal_width", "equal_mass"):
        assert ece(p, y, n_bins=15, strategy=strategy) < 0.01
    p_bad, y_bad = _overconfident(rng, 50_000)
    assert ece(p_bad, y_bad) > 0.1


def test_brier_by_hand():
    assert brier(np.array([0.0, 1.0, 0.5, 0.8]), np.array([0, 1, 1, 0])) == pytest.approx(
        (0 + 0 + 0.25 + 0.64) / 4
    )


def test_reliability_bins_partition_the_data():
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0.5, 1.0, size=200), [1.0, 0.5]])  # nothing below 0.5
    y = (rng.uniform(size=p.size) < p).astype(int)
    width = reliability_bins(p, y, 10)
    assert sum(b["count"] for b in width) == p.size
    assert all(b["count"] > 0 for b in width)  # the five empty low bins are omitted
    assert [round(b["lo"], 1) for b in width] == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert all(b["lo"] <= b["confidence"] <= b["hi"] for b in width)
    assert width[-1]["hi"] == 1.0  # p = 1.0 falls in the last bin rather than a bin of its own
    mass = reliability_bins(p, y, 10, strategy="equal_mass")
    assert [b["count"] for b in mass] == [21, 21] + [20] * 8
    assert all(mass[i]["hi"] <= mass[i + 1]["lo"] for i in range(9))
    json.dumps(width)  # plain JSON types, ready for a results file


def test_equal_mass_bins_do_not_depend_on_row_order_when_scores_tie():
    # Regression: tied scores (isotonic outputs, rounded stated probabilities) used to be split
    # across bins by input position, so permuting the rows changed the ECE.
    rng = np.random.default_rng(9)
    p = np.repeat([0.2, 0.5, 0.8], [7, 7, 6])
    y = rng.integers(0, 2, size=p.size)
    # Each tie group lies wholly in the bin of its mid-rank (3, 10, 16.5 of runs of 5), so the
    # bins are the three groups and the ECE is their weighted calibration gap.
    expected = sum(np.mean(p == v) * abs(y[p == v].mean() - v) for v in (0.2, 0.5, 0.8))
    for _ in range(50):
        perm = rng.permutation(p.size)
        assert ece(p[perm], y[perm], n_bins=4, strategy="equal_mass") == pytest.approx(expected)
        bins = reliability_bins(p[perm], y[perm], n_bins=4, strategy="equal_mass")
        assert [(b["lo"], b["hi"], b["count"]) for b in bins] == [
            (0.2, 0.2, 7),
            (0.5, 0.5, 7),
            (0.8, 0.8, 6),
        ]


def test_metric_inputs_are_validated():
    with pytest.raises(ValueError):
        ece(np.array([0.2, 1.2]), np.array([0, 1]))
    with pytest.raises(ValueError):
        brier(np.array([0.2, 0.4]), np.array([0, 2]))
    with pytest.raises(ValueError):
        brier(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        reliability_bins(np.array([0.2]), np.array([1]), strategy="quantile")


# --------------------------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------------------------


def test_temperature_scaling_recovers_the_true_temperature():
    rng = np.random.default_rng(2)
    logits = rng.normal(0.5, 2.0, size=40_000)
    y = (rng.uniform(size=logits.size) < _sigmoid(logits / 2.5)).astype(int)
    calibrator = TemperatureScaling().fit(_sigmoid(logits), y)
    assert calibrator.temperature == pytest.approx(2.5, rel=0.05)


def test_temperature_scaling_is_the_maximum_likelihood_fit():
    rng = np.random.default_rng(3)
    p, y = _overconfident(rng, 500)
    logits = np.log(p / (1 - p))

    def nll(temperature: float) -> float:
        z = logits / temperature
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    fitted = TemperatureScaling().fit(p, y).temperature
    assert nll(fitted) <= min(nll(t) for t in np.linspace(0.2, 10.0, 2000)) + 1e-9
    # One parameter and T > 0: the ranking of the scores is untouched.
    out = TemperatureScaling().fit(p, y).predict(p)
    assert (np.argsort(p, kind="stable") == np.argsort(out, kind="stable")).all()


def test_temperature_is_bounded_on_degenerate_data():
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    separable = TemperatureScaling().fit(p, np.array([1, 1, 1, 0, 0, 0]))
    assert separable.temperature == pytest.approx(0.05)  # would be 0 without the bound
    reversed_ = TemperatureScaling().fit(p, np.array([0, 0, 0, 1, 1, 1]))
    assert reversed_.temperature == pytest.approx(100.0)  # anti-correlated: flatten to 0.5
    assert np.isfinite(separable.predict(np.array([0.0, 0.5, 1.0]))).all()


def test_platt_scaling_matches_a_generic_optimiser():
    rng = np.random.default_rng(4)
    p, y = _overconfident(rng, 400)
    logits = np.log(p / (1 - p))
    n_pos, n_neg = y.sum(), (1 - y).sum()
    targets = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def objective(w: np.ndarray) -> float:
        z = w[0] * logits + w[1]
        return float(np.mean(np.logaddexp(0.0, z) - targets * z))

    reference = minimize(objective, np.zeros(2), method="BFGS", tol=1e-12).x
    calibrator = PlattScaling().fit(p, y)
    assert objective(np.array([calibrator.a, calibrator.b])) <= objective(reference) + 1e-10
    np.testing.assert_allclose([calibrator.a, calibrator.b], reference, atol=1e-4)
    # The data were generated with logit = (true - 0.7) / 2.5, so a ~ 0.4 and b ~ -0.28.
    assert calibrator.a == pytest.approx(0.4, abs=0.1)


def test_platt_scaling_stays_finite_on_separable_data_and_accepts_raw_scores():
    scores = np.array([5.0, 4.0, 3.0, -3.0, -4.0, -5.0])
    y = np.array([1, 1, 1, 0, 0, 0])
    calibrator = PlattScaling(logit_input=False).fit(scores, y)
    assert np.isfinite([calibrator.a, calibrator.b]).all() and calibrator.a > 0
    out = calibrator.predict(scores)
    assert (out[:3] > 0.5).all() and (out[3:] < 0.5).all()
    with pytest.raises(ValueError):
        PlattScaling().fit(scores, y)  # logit_input=True requires probabilities


@pytest.mark.parametrize("logit_input", [True, False])
def test_platt_scaling_never_reverses_the_ranking(logit_input: bool):
    # Regression: an unconstrained slope fitted a < 0 on a score anti-correlated with correctness,
    # turning an AUROC of ~0.16 into ~0.84 using the calibration labels. The slope now sits at
    # its positive floor: predictions stay near the base rate and in the original order.
    rng = np.random.default_rng(10)
    s = rng.uniform(size=2000)
    y = (rng.uniform(size=s.size) < 1.0 - s).astype(int)
    calibrator = PlattScaling(logit_input=logit_input).fit(s, y)
    assert calibrator.a > 0
    out = calibrator.predict(s)
    assert (np.argsort(s, kind="stable") == np.argsort(out, kind="stable")).all()
    assert abs(out.mean() - y.mean()) < 0.01 and np.ptp(out) < 0.05

    # On the boundary the intercept is the maximum-likelihood one for the fixed slope.
    targets = np.where(y == 1, (y.sum() + 1) / (y.sum() + 2), 1 / ((1 - y).sum() + 2))
    score = np.log(s / (1 - s)) if logit_input else s

    def objective(b: float) -> float:
        z = calibrator.a * score + b
        return float(np.mean(np.logaddexp(0.0, z) - targets * z))

    grid = calibrator.b + np.linspace(-0.05, 0.05, 101)
    assert objective(calibrator.b) <= min(objective(b) for b in grid) + 1e-12


def test_logit_based_calibrators_keep_distinct_confidences_distinct():
    # Regression: clipping at 1e-6 before the logit merged every confidence above 1 - 1e-6, so
    # the most confident answers tied after calibration. Now only inputs below 2^-53 are merged
    # (an exact 1.0 maps just above 1 - 2^-52, the largest float64 below 1 being 1 - 2^-53).
    p = np.array([0.5, 1 - 1e-7, 1 - 1e-9, 1 - 1e-12, 1 - 2**-52, 1.0])
    temperature = TemperatureScaling.from_json({"kind": "temperature", "temperature": 2.0})
    platt = PlattScaling.from_json({"kind": "platt", "a": 0.5, "b": 0.0, "logit_input": True})
    for calibrator in (temperature, platt):
        out = calibrator.predict(p)
        assert (np.diff(out) > 0).all()
        assert np.isfinite(calibrator.predict(np.array([0.0, 1.0]))).all()


@pytest.mark.parametrize("name", ["temperature", "platt", "isotonic"])
def test_calibrators_reduce_calibration_error_on_held_out_data(name: str):
    rng = np.random.default_rng(5)
    p_fit, y_fit = _overconfident(rng, 2000)
    p_test, y_test = _overconfident(rng, 20_000)
    calibrated = CALIBRATORS[name]().fit(p_fit, y_fit).predict(p_test)
    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0
    assert brier(calibrated, y_test) < brier(p_test, y_test)
    if name != "temperature":
        # Temperature has no intercept, so it cannot remove the shift in these scores and ECE need
        # not improve much; the two-parameter and non-parametric maps must fix it.
        assert ece(calibrated, y_test) < 0.03 < ece(p_test, y_test)


def test_identity_calibration_returns_its_input():
    p = np.array([0.1, 0.5, 0.9])
    assert IdentityCalibration().fit(p, np.array([0, 1, 1])).predict(p).tolist() == p.tolist()


@pytest.mark.parametrize("name", sorted(CALIBRATORS))
def test_calibrator_json_round_trip(name: str):
    rng = np.random.default_rng(6)
    p, y = _overconfident(rng, 300)
    calibrator = CALIBRATORS[name]().fit(p, y)
    restored = calibrator_from_json(json.loads(json.dumps(calibrator.to_json())))
    assert type(restored) is type(calibrator)
    grid = np.linspace(0, 1, 101)
    np.testing.assert_array_equal(restored.predict(grid), calibrator.predict(grid))


def test_calibrator_from_json_rejects_unknown_payloads():
    with pytest.raises(ValueError):
        calibrator_from_json({"kind": "beta"})
    with pytest.raises(ValueError):
        calibrator_from_json({})
