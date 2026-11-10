"""Tests for primum.uncertainty.conformal.

The simulation tests check the *guarantees* themselves on synthetic data where the truth is known
in closed form. Their tolerances are Monte-Carlo standard errors, never slack on the guarantee:
if one fails beyond that, the implementation is wrong.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import binom

from primum.uncertainty.conformal import (
    SelectiveGuarantee,
    aps_sets,
    binomial_pvalue,
    certify_threshold,
    conformal_quantile,
    hoeffding_bentkus_pvalue,
    lac_sets,
)

# --------------------------------------------------------------------------------------------
# conformal_quantile
# --------------------------------------------------------------------------------------------


def test_conformal_quantile_known_values():
    scores = np.arange(1.0, 10.0)  # 1..9
    assert conformal_quantile(scores, 0.1) == 9.0  # rank ceil(10 * 0.9) = 9
    assert conformal_quantile(scores, 0.5) == 5.0  # rank ceil(10 * 0.5) = 5
    assert conformal_quantile(scores[::-1], 0.5) == 5.0  # order of the input is irrelevant
    assert conformal_quantile(scores[:8], 0.1) == math.inf  # rank ceil(9 * 0.9) = 9 > 8
    assert conformal_quantile(np.array([]), 0.1) == math.inf


@pytest.mark.parametrize("alpha", ["0.05", "0.1", "0.15", "0.2", "0.3"])
def test_conformal_quantile_rank_is_exact_despite_floating_point(alpha):
    # (n + 1)(1 - alpha) is often an integer in exact arithmetic but not in binary floating point;
    # an off-by-one in the rank silently changes the coverage level.
    for n in range(1, 400):
        exact_rank = math.ceil((n + 1) * (1 - Fraction(alpha)))
        got = conformal_quantile(np.arange(1.0, n + 1.0), float(alpha))
        assert got == (math.inf if exact_rank > n else float(exact_rank))


def test_conformal_quantile_coverage_is_exactly_the_rank_fraction():
    # With n = 19 and alpha = 0.1 the rank is 18, so P(new score <= q) = 18/20 exactly.
    rng = np.random.default_rng(0)
    trials, n = 40_000, 19
    scores = rng.normal(size=(trials, n + 1))
    q = np.array([conformal_quantile(row[:n], 0.1) for row in scores])
    covered = scores[:, n] <= q
    se = math.sqrt(0.9 * 0.1 / trials)
    assert abs(covered.mean() - 0.9) < 4 * se


def test_levels_are_validated():
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(5), 0.0)
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(5), 1.0)


# --------------------------------------------------------------------------------------------
# LAC and APS prediction sets
# --------------------------------------------------------------------------------------------


def _three_class_problem(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels from the true posterior; the 'model' is a deliberately distorted version of it.

    Coverage must hold for any model, so the test uses a miscalibrated one.
    """
    true_probs = rng.dirichlet([2.0, 1.0, 0.5], size=n)
    labels = (rng.uniform(size=(n, 1)) > np.cumsum(true_probs, axis=1)).sum(axis=1)
    model_probs = true_probs**2.5 + 0.05 * rng.uniform(size=true_probs.shape)
    return model_probs / model_probs.sum(axis=1, keepdims=True), labels


def _mean_coverage(make_sets: Callable, *, seed: int = 1) -> tuple[float, float]:
    """Mean coverage of the true label, and its standard error, over many calibration draws."""
    rng = np.random.default_rng(seed)
    trials, n_cal, n_test = 600, 99, 50
    coverage = np.empty(trials)
    for t in range(trials):
        probs, labels = _three_class_problem(rng, n_cal + n_test)
        sets = make_sets(probs[:n_cal], labels[:n_cal], probs[n_cal:], rng)
        coverage[t] = sets[np.arange(n_test), labels[n_cal:]].mean()
    # Test points within a trial share a calibration set, so the trial is the independent unit.
    return float(coverage.mean()), float(coverage.std(ddof=1) / math.sqrt(trials))


@pytest.mark.parametrize(
    "make_sets",
    [
        lambda pc, yc, pt, rng: lac_sets(pc, yc, pt, 0.1),
        lambda pc, yc, pt, rng: aps_sets(pc, yc, pt, 0.1, rng=rng),
    ],
    ids=["lac", "aps-randomized"],
)
def test_prediction_sets_have_exact_marginal_coverage(make_sets: Callable):
    # n_cal = 99 and alpha = 0.1 give rank 90 of 100. LAC and randomised APS scores are
    # continuous, so the marginal coverage is exactly 0.90 and the test is two-sided:
    # under-coverage would break the guarantee, over-coverage would mean needlessly large sets.
    mean, se = _mean_coverage(make_sets)
    assert abs(mean - 0.9) < 4 * se


def test_deterministic_aps_covers_but_is_conservative():
    # Without randomisation the score of the least probable label is always exactly 1, so the
    # scores have an atom: coverage is still >= 1 - alpha but no longer tight. This is the reason
    # APS is randomised.
    mean, se = _mean_coverage(lambda pc, yc, pt, rng: aps_sets(pc, yc, pt, 0.1, randomized=False))
    assert mean >= 0.9 - 4 * se
    assert mean > 0.9 + 4 * se


def test_randomized_aps_on_stated_probabilities_with_zeros_is_valid_but_not_tight():
    # One-hot "stated" probabilities: randomisation cannot spread the score of a label given
    # probability 0, which is exactly its mass_above = 1. With the top label right only ~73% of
    # the time (< 1 - alpha) the quantile lands on that atom: coverage stays >= 1 - alpha, but
    # the sets are essentially every label (the documented limit of APS on such inputs).
    rng = np.random.default_rng(11)
    trials, n_cal, n_test, k, alpha = 300, 50, 50, 3, 0.1
    coverage, size = np.empty(trials), np.empty(trials)
    for t in range(trials):
        gold = rng.integers(0, k, size=n_cal + n_test)
        top = np.where(rng.uniform(size=gold.size) < 0.6, gold, rng.integers(0, k, size=gold.size))
        probs = np.eye(k)[top]
        sets = aps_sets(probs[:n_cal], gold[:n_cal], probs[n_cal:], alpha, rng=rng)
        coverage[t] = sets[np.arange(n_test), gold[n_cal:]].mean()
        size[t] = sets.sum(axis=1).mean()
    assert coverage.mean() >= 1 - alpha
    assert size.mean() > 2.95


def test_lac_set_by_hand():
    probs_cal = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]])
    y_cal = np.array([0, 1, 2, 1])  # scores 0.3, 0.2, 0.6, 0.7
    # n = 4, alpha = 0.4: rank ceil(5 * 0.6) = 3, so q = 0.6 and a label is in iff p >= 0.4.
    sets = lac_sets(probs_cal, y_cal, np.array([[0.5, 0.45, 0.05], [0.34, 0.33, 0.33]]), 0.4)
    assert sets.dtype == bool
    assert sets.tolist() == [[True, True, False], [False, False, False]]


def test_lac_and_aps_return_every_label_when_calibration_set_is_too_small():
    rng = np.random.default_rng(2)
    probs, labels = _three_class_problem(rng, 8)  # rank ceil(6 * 0.9) = 6 > 5
    assert lac_sets(probs[:5], labels[:5], probs[5:], 0.1).all()
    assert aps_sets(probs[:5], labels[:5], probs[5:], 0.1, rng=0).all()


def test_aps_sets_are_prefixes_of_the_ranking_and_nested_in_alpha():
    rng = np.random.default_rng(3)
    probs, labels = _three_class_problem(rng, 400)
    cal, test = slice(0, 200), slice(200, 400)
    for randomized in (True, False):
        tight = aps_sets(probs[cal], labels[cal], probs[test], 0.3, randomized=randomized, rng=7)
        loose = aps_sets(probs[cal], labels[cal], probs[test], 0.05, randomized=randomized, rng=7)
        assert (loose | ~tight).all()  # tight subset of loose (same seed, hence same u)
        # A label can only be included if every more probable label is.
        order = np.argsort(-probs[test], axis=1, kind="stable")
        ranked = np.take_along_axis(loose, order, axis=1)
        assert (ranked[:, :-1] >= ranked[:, 1:]).all()


def test_aps_deterministic_set_by_hand():
    probs_cal = np.array([[0.6, 0.3, 0.1], [0.5, 0.4, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    y_cal = np.array([0, 1, 0, 2])  # cumulative mass through the true label: 0.6, 0.9, 0.9, 0.7
    # n = 4, alpha = 0.4: rank 3, q = 0.9. Include labels whose cumulative mass is <= 0.9.
    sets = aps_sets(
        probs_cal, y_cal, np.array([[0.5, 0.3, 0.2], [0.95, 0.03, 0.02]]), 0.4, randomized=False
    )
    assert sets.tolist() == [[True, True, False], [False, False, False]]


def test_classification_inputs_are_validated():
    probs = np.full((4, 3), 1 / 3)
    with pytest.raises(ValueError):
        lac_sets(probs, np.array([0, 1, 2, 3]), probs, 0.1)  # class index out of range
    with pytest.raises(ValueError):
        lac_sets(probs, np.array([0.0, 1.0, 2.0, 1.0]), probs, 0.1)  # not integer labels
    with pytest.raises(ValueError):
        aps_sets(probs, np.array([0, 1, 2]), probs, 0.1)  # wrong length


# --------------------------------------------------------------------------------------------
# p-values
# --------------------------------------------------------------------------------------------


def test_binomial_pvalue_matches_the_definition():
    n, alpha = 12, 0.3
    for k in range(n + 1):
        direct = sum(math.comb(n, j) * alpha**j * (1 - alpha) ** (n - j) for j in range(k + 1))
        assert binomial_pvalue(k, n, alpha) == pytest.approx(direct, rel=1e-12)
    assert binomial_pvalue(0, 0, 0.1) == 1.0  # no data, no evidence
    with pytest.raises(ValueError):
        binomial_pvalue(5, 4, 0.1)


@pytest.mark.parametrize("pvalue", ["binomial", "hb"])
@pytest.mark.parametrize("true_risk", [0.1, 0.12, 0.2, 0.5])
def test_pvalues_are_super_uniform_under_the_null(pvalue: str, true_risk: float):
    # Exact check by enumeration: for K ~ Bin(n, r) with r >= alpha, P(p(K) <= u) <= u for all u.
    n, alpha = 60, 0.1
    k = np.arange(n + 1)
    if pvalue == "binomial":
        p = np.array([binomial_pvalue(int(j), n, alpha) for j in k])
    else:
        p = np.array([hoeffding_bentkus_pvalue(j / n, n, alpha) for j in k])
    pmf = binom.pmf(k, n, true_risk)
    for u in np.linspace(0.001, 0.999, 500):
        assert pmf[p <= u].sum() <= u + 1e-12


def test_hoeffding_bentkus_formula_and_ordering():
    n, alpha = 50, 0.2
    for k in range(n + 1):
        hb = hoeffding_bentkus_pvalue(k / n, n, alpha)
        # For 0/1 losses the exact binomial tail is never beaten.
        assert hb >= binomial_pvalue(k, n, alpha) - 1e-12
        assert 0.0 <= hb <= 1.0
    # Zero errors: both terms reduce to (1 - alpha)^n up to the factor e on the Bentkus term.
    assert hoeffding_bentkus_pvalue(0.0, n, alpha) == pytest.approx((1 - alpha) ** n)
    # One hand-computed case: risk_hat = 0.1, n = 50, alpha = 0.2.
    kl = 0.1 * math.log(0.1 / 0.2) + 0.9 * math.log(0.9 / 0.8)
    expected = min(math.exp(-50 * kl), math.e * binom.cdf(5, 50, 0.2))
    assert hoeffding_bentkus_pvalue(0.1, 50, 0.2) == pytest.approx(expected)
    assert hoeffding_bentkus_pvalue(0.3, 50, 0.2) == 1.0  # empirical risk above alpha
    assert hoeffding_bentkus_pvalue(0.0, 0, 0.2) == 1.0


# --------------------------------------------------------------------------------------------
# certify_threshold: deterministic behaviour
# --------------------------------------------------------------------------------------------


def test_certify_all_correct_certifies_full_coverage():
    conf = np.linspace(0.5, 0.99, 100)
    g = certify_threshold(conf, np.ones(100, dtype=bool), alpha=0.1, delta=0.1)
    assert isinstance(g, SelectiveGuarantee)
    assert g.threshold == conf.min()
    assert (g.n_cal, g.n_selected, g.cal_coverage, g.cal_risk) == (100, 100, 1.0, 0.0)
    assert g.pvalue == pytest.approx(0.9**100)


def test_certify_returns_none_when_the_first_test_fails():
    conf = np.linspace(0.5, 0.99, 100)
    g = certify_threshold(conf, np.zeros(100, dtype=bool), alpha=0.1, delta=0.1)
    assert g.threshold is None
    assert (g.n_selected, g.cal_coverage, g.cal_risk) == (0, 0.0, 0.0)
    assert g.pvalue > 0.1  # the p-value of the failed first test


def test_certify_needs_enough_points_to_reject_at_all():
    # With zero errors the p-value is 0.9^N: 0.109 at N = 21, 0.098 at N = 22. Below 22 points
    # nothing can be certified at alpha = delta = 0.1 no matter how good the answers are.
    for n, certified in ((10, False), (21, False), (22, True)):
        conf = np.linspace(0.5, 0.99, n)
        g = certify_threshold(conf, np.ones(n, dtype=bool), alpha=0.1, delta=0.1, min_selected=1)
        assert (g.threshold is not None) == certified
        if not certified:
            assert g.pvalue == 1.0  # no candidate was testable


def test_certify_is_a_fixed_sequence_not_a_search():
    # The 30 most confident answers are all wrong, the remaining 970 all right. Plenty of low
    # thresholds have empirical risk far below alpha (30/1000 overall), and a search over
    # thresholds would pick one. Fixed-sequence testing must stop at the first failure, because
    # skipping past a failed test is exactly the selection that voids the error control.
    conf = np.linspace(0.99, 0.01, 1000)
    correct = np.arange(1000) >= 30
    assert certify_threshold(conf, correct, alpha=0.1, delta=0.1).threshold is None


def test_certify_ties_are_answered_together():
    conf = np.array([0.9] * 50 + [0.5] * 50)
    correct = np.array([True] * 50 + [True] * 30 + [False] * 20)
    g = certify_threshold(conf, correct, alpha=0.1, delta=0.1)
    # 0.9 selects 50 points with no error (p = 0.9^50); 0.5 selects all 100 with 20 errors.
    assert g.threshold == 0.9
    assert g.n_selected == 50 == int(np.sum(conf >= g.threshold))
    assert g.cal_risk == 0.0
    # If the low group is clean too, the tie group enters as a whole and everything is certified.
    g_all = certify_threshold(conf, np.ones(100, dtype=bool), alpha=0.1, delta=0.1)
    assert (g_all.threshold, g_all.n_selected) == (0.5, 100)


def test_certify_reports_the_rule_it_certified():
    rng = np.random.default_rng(4)
    conf = rng.uniform(size=800)
    correct = rng.uniform(size=800) < conf
    for method in ("binomial", "hb"):
        g = certify_threshold(conf, correct, alpha=0.2, delta=0.1, method=method)
        assert g.threshold in conf  # an observed confidence, so `>=` reproduces the selection
        selected = conf >= g.threshold
        assert g.n_selected == selected.sum()
        assert g.cal_coverage == pytest.approx(selected.mean())
        assert g.cal_risk == pytest.approx(np.mean(~correct[selected]))
        assert g.pvalue <= 0.1
        assert (g.alpha, g.delta, g.n_cal) == (0.2, 0.1, 800)


def test_certify_depends_on_confidences_only_through_their_order():
    # A strictly increasing recalibration (e.g. temperature scaling) fitted on the calibration
    # split cannot change which answers are certified.
    rng = np.random.default_rng(5)
    conf = rng.uniform(size=600)
    correct = rng.uniform(size=600) < conf
    base = certify_threshold(conf, correct, alpha=0.2, delta=0.1)
    warped = certify_threshold(np.log(conf / (1 - conf)) / 3.0, correct, alpha=0.2, delta=0.1)
    assert base.n_selected == warped.n_selected > 0
    assert base.pvalue == warped.pvalue


def test_hoeffding_bentkus_never_certifies_more_than_the_exact_test():
    rng = np.random.default_rng(6)
    for _ in range(50):
        conf = rng.uniform(size=500)
        correct = rng.uniform(size=500) < conf
        exact = certify_threshold(conf, correct, alpha=0.2, delta=0.1)
        hb = certify_threshold(conf, correct, alpha=0.2, delta=0.1, method="hb")
        assert hb.n_selected <= exact.n_selected


def test_min_selected_is_respected():
    conf = np.linspace(0.5, 0.99, 300)
    g = certify_threshold(conf, np.ones(300, dtype=bool), alpha=0.1, delta=0.1, min_selected=301)
    assert g.threshold is None and g.pvalue == 1.0


def test_certify_validates_inputs():
    conf, correct = np.linspace(0, 1, 50), np.ones(50, dtype=bool)
    with pytest.raises(ValueError):
        certify_threshold(conf, correct[:-1], alpha=0.1, delta=0.1)
    with pytest.raises(ValueError):
        certify_threshold(conf, correct, alpha=0.0, delta=0.1)
    with pytest.raises(ValueError):
        certify_threshold(conf, correct, alpha=0.1, delta=1.0)
    with pytest.raises(ValueError):
        certify_threshold(conf, correct, alpha=0.1, delta=0.1, method="bonferroni")
    with pytest.raises(ValueError):
        certify_threshold(np.where(conf > 0.5, np.nan, conf), correct, alpha=0.1, delta=0.1)


# --------------------------------------------------------------------------------------------
# certify_threshold: simulation of the guarantee  P(R(lambda_hat) > alpha) <= delta
# --------------------------------------------------------------------------------------------
#
# Each world below defines the joint law of (confidence, correct) and the exact selective risk
# R(lambda) = P(wrong | confidence >= lambda), so a certified threshold can be judged against
# the truth rather than against another sample.

Sampler = Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]
Risk = Callable[[float], float]


def _calibrated_uniform() -> tuple[Sampler, Risk]:
    """c ~ U(0, 1), P(correct | c) = c.  R(lambda) = (1 - lambda) / 2, decreasing in lambda."""

    def sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        conf = rng.uniform(size=n)
        return conf, rng.uniform(size=n) < conf

    return sample, lambda lam: (1.0 - max(lam, 0.0)) / 2.0


def _discrete() -> tuple[Sampler, Risk]:
    """Four confidence levels (heavy ties), as isotonic calibration or vote shares produce."""
    values = np.array([0.9, 0.8, 0.6, 0.4])
    mass = np.array([0.3, 0.3, 0.2, 0.2])
    error = np.array([0.02, 0.10, 0.30, 0.50])  # R at the four levels: 0.02, 0.06, 0.12, 0.196

    def sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        level = rng.choice(4, size=n, p=mass)
        return values[level], rng.uniform(size=n) >= error[level]

    def risk(lam: float) -> float:
        selected = values >= lam
        return float((mass[selected] * error[selected]).sum() / mass[selected].sum())

    return sample, risk


def _non_monotone() -> tuple[Sampler, Risk]:
    """c ~ U(0, 1) with an error rate that is *not* monotone in confidence.

    Error 0.02 above 0.8, 0.30 on (0.7, 0.8], 0.00 on (0.6, 0.7], 0.50 below. At alpha = 0.1 the
    selective risk crosses the target three times: it is above alpha on (0.66, 0.72), back below
    on [0.585, 0.66], and above for good under 0.585. This is outside the monotone-risk condition
    under which the data-dependent grid is proven exact, so it probes that assumption.
    """
    pieces = ((0.8, 1.0, 0.02), (0.7, 0.8, 0.30), (0.6, 0.7, 0.0), (0.0, 0.6, 0.50))

    def sample(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        conf = rng.uniform(size=n)
        error = np.zeros(n)
        for lo, hi, rate in pieces:
            error[(conf > lo) & (conf <= hi)] = rate
        return conf, rng.uniform(size=n) >= error

    def risk(lam: float) -> float:
        lam = min(max(lam, 0.0), 1.0 - 1e-12)
        errors = sum((hi - max(lo, lam)) * rate for lo, hi, rate in pieces if hi > lam)
        return errors / (1.0 - lam)

    return sample, risk


def _plugin_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float) -> float | None:
    """The naive rule: the lowest threshold whose *empirical* selective risk is <= alpha."""
    order = np.argsort(-conf)
    risk = np.cumsum(~correct[order]) / np.arange(1, conf.size + 1)
    feasible = np.flatnonzero(risk <= alpha)
    return float(conf[order][feasible[-1]]) if feasible.size else None


def _simulate(
    world: tuple[Sampler, Risk],
    *,
    n: int,
    alpha: float,
    delta: float,
    reps: int,
    seed: int,
    **kwargs,
) -> dict[str, float]:
    sample, true_risk = world
    rng = np.random.default_rng(seed)
    violations = certified = plugin_violations = 0
    for _ in range(reps):
        conf, correct = sample(rng, n)
        g = certify_threshold(conf, correct, alpha=alpha, delta=delta, **kwargs)
        if g.threshold is not None:
            certified += 1
            violations += true_risk(g.threshold) > alpha
        plugin = _plugin_threshold(conf, correct, alpha)
        plugin_violations += plugin is not None and true_risk(plugin) > alpha
    return {
        "violation": violations / reps,
        "certified": certified / reps,
        "plugin_violation": plugin_violations / reps,
        # delta plus three Monte-Carlo standard errors of a Bernoulli(delta) frequency.
        "bound": delta + 3.0 * math.sqrt(delta * (1.0 - delta) / reps),
    }


@pytest.mark.parametrize(
    ("world", "n", "alpha", "kwargs"),
    [
        (_calibrated_uniform, 1000, 0.20, {}),
        (_calibrated_uniform, 1000, 0.10, {}),
        (_calibrated_uniform, 500, 0.25, {"n_grid": 10**6}),  # every distinct confidence
        (_calibrated_uniform, 1000, 0.20, {"method": "hb"}),
        (_discrete, 300, 0.10, {}),
        (_discrete, 300, 0.05, {}),
        (_non_monotone, 1000, 0.10, {"min_selected": 100}),
        (_non_monotone, 1000, 0.10, {"n_grid": 10**6}),
    ],
    ids=[
        "uniform-a20",
        "uniform-a10",
        "uniform-full-grid",
        "uniform-hb",
        "discrete-a10",
        "discrete-a05",
        "non-monotone",
        "non-monotone-full-grid",
    ],
)
def test_certified_threshold_violation_frequency_is_at_most_delta(world, n, alpha, kwargs):
    delta = 0.1
    result = _simulate(world(), n=n, alpha=alpha, delta=delta, reps=2500, seed=11, **kwargs)
    assert result["violation"] <= result["bound"], result
    # The test must not pass vacuously: the procedure has to certify something in a real share of
    # the draws, otherwise "no violations" says nothing.
    assert result["certified"] > 0.25, result


@pytest.mark.parametrize(
    "world", [_calibrated_uniform, _non_monotone], ids=["uniform", "non-monotone"]
)
def test_simulation_detects_an_invalid_procedure(world):
    # Negative control: the uncorrected plug-in threshold must be caught by the same harness.
    # It violates the risk target in roughly half of the draws.
    result = _simulate(world(), n=1000, alpha=0.1, delta=0.1, reps=1000, seed=12)
    assert result["plugin_violation"] > 0.3, result
    assert result["plugin_violation"] > 2 * result["bound"], result


@pytest.mark.slow
@pytest.mark.parametrize(
    "world",
    [_calibrated_uniform, _discrete, _non_monotone],
    ids=["uniform", "discrete", "non-monotone"],
)
@pytest.mark.parametrize("n", [300, 1000, 3000])
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
@pytest.mark.parametrize("delta", [0.05, 0.10])
def test_certified_threshold_violation_frequency_large_grid(world, n, alpha, delta):
    """Heavier Monte-Carlo sweep over worlds, sample sizes and levels (20,000 draws each)."""
    result = _simulate(world(), n=n, alpha=alpha, delta=delta, reps=20_000, seed=13)
    assert result["violation"] <= result["bound"], result
