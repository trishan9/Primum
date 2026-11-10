"""Tests for primum.uncertainty.metrics and primum.uncertainty.stats."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from primum.uncertainty.metrics import (
    aurc,
    auroc,
    coverage_at_risk,
    e_aurc,
    risk_coverage_curve,
    selective_risk,
)
from primum.uncertainty.stats import (
    bootstrap_ci,
    paired_bootstrap_pvalue,
    paired_permutation_pvalue,
)

# --------------------------------------------------------------------------------------------
# AUROC
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_auroc_matches_scikit_learn(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 500))
    labels = rng.uniform(size=n) < 0.6
    labels[:2] = [True, False]  # both classes present
    scores = rng.normal(size=n) + labels
    if seed % 2:
        scores = np.round(scores, 1)  # heavy ties
    assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_auroc_extremes_and_ties():
    labels = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert auroc(np.full(4, 0.5), labels) == 0.5  # an uninformative constant score
    with pytest.raises(ValueError):
        auroc(np.array([0.1, 0.2]), np.array([1, 1]))


# --------------------------------------------------------------------------------------------
# Risk-coverage
# --------------------------------------------------------------------------------------------


def test_risk_coverage_curve_by_hand():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, False, True, True])
    coverage, risk = risk_coverage_curve(conf, correct)
    np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.5, 1 / 3, 0.25])
    assert aurc(conf, correct) == pytest.approx(np.mean([0.0, 0.5, 1 / 3, 0.25]))


def test_aurc_matches_the_definition_without_ties():
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=300)
    correct = rng.uniform(size=300) < conf
    errors = (~correct[np.argsort(-conf)]).astype(float)
    naive = np.mean([errors[:k].mean() for k in range(1, 301)])
    assert aurc(conf, correct) == pytest.approx(naive)


def test_ties_are_averaged_not_broken_by_input_order():
    rng = np.random.default_rng(1)
    conf = np.round(rng.uniform(size=200), 1)  # 11 distinct values
    correct = rng.uniform(size=200) < conf
    shuffle = rng.permutation(200)
    assert aurc(conf, correct) == pytest.approx(aurc(conf[shuffle], correct[shuffle]), abs=1e-12)
    # A constant score cannot rank anything: every prefix has the overall error rate.
    flat = np.full(200, 0.5)
    _, risk = risk_coverage_curve(flat, correct)
    np.testing.assert_allclose(risk, np.mean(~correct))
    # The tie-averaged curve is the expectation over random tie-breaking.
    broken = [aurc(conf + 1e-6 * rng.uniform(size=200), correct) for _ in range(2000)]
    assert aurc(conf, correct) == pytest.approx(
        np.mean(broken), abs=4 * np.std(broken) / math.sqrt(2000)
    )


def test_e_aurc_is_zero_for_a_perfect_ranking_and_positive_otherwise():
    correct = np.array([True] * 70 + [False] * 30)
    perfect = np.linspace(1.0, 0.0, 100)
    assert e_aurc(perfect, correct) == pytest.approx(0.0, abs=1e-12)
    assert e_aurc(-perfect, correct) > e_aurc(np.full(100, 0.5), correct) > 0.0
    # Oracle AURC approaches r + (1 - r) ln(1 - r) for error rate r (Geifman et al. 2019).
    big = np.array([True] * 7000 + [False] * 3000)
    oracle = aurc(np.linspace(1.0, 0.0, 10_000), big)
    assert oracle == pytest.approx(0.3 + 0.7 * math.log(0.7), abs=1e-3)


def test_coverage_at_risk_and_selective_risk_by_hand():
    conf = np.array([0.9, 0.9, 0.8, 0.7, 0.7, 0.6])
    correct = np.array([True, True, False, True, True, False])
    # Operating points (threshold: selected, errors): 0.9: 2, 0 | 0.8: 3, 1 | 0.7: 5, 1 | 0.6: 6, 2.
    assert coverage_at_risk(conf, correct, 0.0) == pytest.approx(2 / 6)
    assert coverage_at_risk(conf, correct, 0.2) == pytest.approx(
        5 / 6
    )  # not the first feasible point
    assert coverage_at_risk(conf, correct, 0.25) == pytest.approx(5 / 6)
    assert coverage_at_risk(conf, correct, 1 / 3) == 1.0
    assert coverage_at_risk(conf, ~correct, 0.1) == 0.0  # no threshold is good enough
    assert selective_risk(conf, correct, 0.8) == pytest.approx(1 / 3)
    assert selective_risk(conf, correct, 0.7) == pytest.approx(1 / 5)  # ties are answered (>=)
    assert selective_risk(conf, correct, 0.95) == 0.0  # nothing answered, nothing wrong


def test_metric_inputs_are_validated():
    with pytest.raises(ValueError):
        aurc(np.array([0.1, 0.2]), np.array([True]))
    with pytest.raises(ValueError):
        selective_risk(np.array([np.nan, 0.2]), np.array([True, False]), 0.1)
    with pytest.raises(ValueError):
        risk_coverage_curve(np.array([]), np.array([]))


# --------------------------------------------------------------------------------------------
# Bootstrap and permutation inference
# --------------------------------------------------------------------------------------------


def test_bootstrap_ci_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(2)
    values = rng.exponential(size=200)
    lo, hi = bootstrap_ci(values, rng=0)
    assert (lo, hi) == bootstrap_ci(values, rng=0)
    assert lo < values.mean() < hi
    # For a mean the interval should be close to the normal-theory one.
    half_width = 1.96 * values.std(ddof=1) / math.sqrt(200)
    assert (hi - lo) / 2 == pytest.approx(half_width, rel=0.15)
    narrow_lo, narrow_hi = bootstrap_ci(values, level=0.5, rng=0)
    assert lo < narrow_lo < narrow_hi < hi


def test_bootstrap_ci_coverage():
    # A 90% interval for the mean of n = 60 normal draws should contain the true mean about 90% of
    # the time (the percentile bootstrap is slightly liberal at this n).
    rng = np.random.default_rng(3)
    trials, hits = 400, 0
    for _ in range(trials):
        lo, hi = bootstrap_ci(rng.normal(1.0, 2.0, size=60), n_boot=400, level=0.9, rng=rng)
        hits += lo <= 1.0 <= hi
    assert 0.84 <= hits / trials <= 0.95


def test_bootstrap_ci_resamples_rows_for_multi_column_statistics():
    rng = np.random.default_rng(4)
    conf = rng.uniform(size=300)
    correct = rng.uniform(size=300) < conf

    def auroc_of_rows(rows: np.ndarray) -> float:
        return auroc(rows[:, 0], rows[:, 1] > 0.5)

    lo, hi = bootstrap_ci(np.column_stack([conf, correct]), stat=auroc_of_rows, n_boot=300, rng=0)
    assert 0.5 < lo < auroc(conf, correct) < hi < 1.0
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


def test_permutation_pvalue_matches_exact_enumeration():
    rng = np.random.default_rng(5)
    a = rng.normal(0.6, 1.0, size=10)
    b = rng.normal(0.0, 1.0, size=10)
    diff = a - b
    sums = np.array([np.dot(signs, diff) for signs in itertools.product((-1, 1), repeat=10)])
    exact = np.mean(np.abs(sums) >= abs(diff.sum()) - 1e-12)
    n_perm = 40_000
    estimate = paired_permutation_pvalue(a, b, n_perm=n_perm, rng=0)
    assert estimate == pytest.approx(exact, abs=4 * math.sqrt(exact * (1 - exact) / n_perm))


@pytest.mark.parametrize("test", [paired_permutation_pvalue, paired_bootstrap_pvalue])
def test_paired_tests_at_the_extremes(test):
    rng = np.random.default_rng(6)
    a = rng.uniform(size=100)
    assert test(a, a, 500, 0) == 1.0  # identical systems
    assert test(a + 1.0, a + rng.normal(0, 0.01, size=100), 999, 0) == pytest.approx(1 / 1000)
    assert test(a, a + 0.3, 2000, 0) == test(a + 0.3, a, 2000, 0)  # two-sided: symmetric in (a, b)
    with pytest.raises(ValueError):
        test(a, a[:-1], 100, 0)


@pytest.mark.parametrize("test", [paired_permutation_pvalue, paired_bootstrap_pvalue])
def test_paired_tests_control_type_one_error(test):
    # Under H0 (both systems draw from the same distribution) a level-0.1 test should reject
    # about 10% of the time; the bootstrap version is slightly liberal at n = 40.
    rng = np.random.default_rng(7)
    trials = 600
    rejections = sum(
        test(rng.normal(size=40), rng.normal(size=40), 400, rng) <= 0.1 for _ in range(trials)
    )
    assert rejections / trials <= 0.1 + 3 * math.sqrt(0.1 * 0.9 / trials) + 0.02


def test_bootstrap_pvalue_is_not_badly_liberal_on_small_skewed_samples():
    # Regression: the recentred (non-studentised) bootstrap of the mean rejected a true H0 about
    # 15% of the time at a nominal 5% for n = 8 centred-exponential differences, because the
    # empirical distribution understates the spread. The bootstrap-t brings this to ~6-7%; it is
    # still approximate, which the bound below allows for (Monte-Carlo SE ~ 0.007).
    rng = np.random.default_rng(12)
    trials = 1000
    rejections = sum(
        paired_bootstrap_pvalue(rng.exponential(size=8) - 1.0, np.zeros(8), 499, rng) <= 0.05
        for _ in range(trials)
    )
    assert rejections / trials < 0.10


def test_bootstrap_pvalue_degenerate_samples():
    assert paired_bootstrap_pvalue(np.array([0.7]), np.array([0.2]), 99, 0) == 1.0  # one item
    assert paired_bootstrap_pvalue(np.full(5, 0.7), np.full(5, 0.2), 99, 0) == pytest.approx(0.01)


def test_paired_tests_agree_and_have_power():
    rng = np.random.default_rng(8)
    base = rng.normal(size=200)
    better = base + rng.normal(0.25, 1.0, size=200)
    perm = paired_permutation_pvalue(better, base, n_perm=5000, rng=0)
    boot = paired_bootstrap_pvalue(better, base, n_boot=5000, rng=0)
    assert perm < 0.05 and boot < 0.05
    assert perm == pytest.approx(boot, abs=0.02)
