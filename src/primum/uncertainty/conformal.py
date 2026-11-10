"""Distribution-free guarantees: conformal prediction sets and a certified abstention threshold.

Everything here is a statement about a *fixed* scoring function evaluated on calibration data
it was not fitted on. Refit the model on the calibration set and none of the guarantees hold.

1. Split conformal prediction sets  (`conformal_quantile`, `lac_sets`, `aps_sets`)
---------------------------------------------------------------------------------
Let s(x, y) be any nonconformity score and (X_1, Y_1), ..., (X_n, Y_n), (X, Y) be exchangeable.
With q the ceil((n + 1)(1 - alpha))-th smallest calibration score (q = +inf if that rank
exceeds n), the set C(x) = {y : s(x, y) <= q} satisfies

    P(Y in C(X)) >= 1 - alpha,

and <= 1 - alpha + 1/(n + 1) if scores are almost surely distinct (Vovk, Gammerman & Shafer
2005; Lei et al. 2018; Angelopoulos & Bates 2023, Thm 1). The proof is one line: the rank of
the test score among all n + 1 scores is uniform, so it lands in the lowest
ceil((n + 1)(1 - alpha)) positions with probability at least 1 - alpha. The "+1" is the test
point itself; the plain empirical (1 - alpha)-quantile forgets it and under-covers.
The probability is over calibration set and test point jointly (marginal coverage): it is not a
per-question guarantee.

- LAC (Sadinle, Lei & Wasserman 2019): s(x, y) = 1 - p(y | x). With the true posterior this
  minimises the expected set size among all sets with coverage 1 - alpha; may return the
  empty set.
- APS (Romano, Sesia & Candes 2020): s(x, y, u) = (mass of classes ranked strictly above y)
  + u * p(y | x), u ~ U(0, 1); `randomized=False` fixes u = 1. The randomisation only spreads
  a score over an interval of width p(y | x), so the scores are continuous, and coverage close
  to 1 - alpha rather than conservative, only when the true label's probability is almost
  surely positive. Stated probabilities (an LLM's "yes 0.7, no 0.3, maybe 0") have atoms: a
  label given probability 0 scores exactly its mass_above (1.0 when it is ranked last) whatever
  u is. Validity (coverage >= 1 - alpha) still holds, but when such labels are often the true
  one the quantile lands on the atom and the sets swell towards every label (with one-hot
  stated probabilities and accuracy below 1 - alpha, typically to the full label set). The same
  holds for LAC, whose score 1 - p(y | x) has the same atom at 1.

2. Certified selective risk  (`certify_threshold`)
-------------------------------------------------
Setting. Calibration pairs (c_i, e_i), i = 1..n, are i.i.d. from the deployment distribution:
c is the confidence in an answer, e = 1 if the answer is wrong. (Independence is needed, not
just exchangeability: the binomial law below relies on it.) The system answers iff
c >= lambda (ties answer). Its selective risk is the error rate among the answers it gives,

    R(lambda) = P(e = 1 | c >= lambda)                          (Geifman & El-Yaniv 2017).

Step 1, one threshold: condition on the number selected. R is a ratio of two expectations, so
no bound on a sample mean applies to it directly. But given N = #{i : c_i >= lambda}, the
selected points are i.i.d. draws from the law of (c, e) given c >= lambda, hence

    K | N ~ Binomial(N, R(lambda)),      K = number of errors among the selected.

Binomial(N, r) is stochastically increasing in r, so p = P(Bin(N, alpha) <= K) is super-uniform
whenever R(lambda) > alpha: a valid p-value for H: R(lambda) > alpha, conditionally on N and
therefore marginally. Using the unconditional n in place of N would test the wrong quantity
(the joint rate P(e = 1, c >= lambda), which is small whenever coverage is small).

Step 2, many thresholds: fixed-sequence testing (Learn then Test: Angelopoulos, Bates, Candes,
Jordan & Lei 2021). Order the hypotheses from the most conservative threshold downwards, test
each at the full level delta, stop at the first one that is not rejected, and certify the last
threshold rejected. For thresholds fixed in advance, a false certification requires rejecting
the first true null in the order, which happens with probability <= delta, so no Bonferroni
split of delta is needed:

    P( R(lambda_hat) <= alpha ) >= 1 - delta        over the draw of the calibration set.

If the very first test fails nothing is certified (threshold None) and the system abstains on
everything, which satisfies the guarantee trivially.

Step 3, where the candidate thresholds come from. They are quantiles of the calibration
confidences, restricted to those selecting enough points. Labels never enter: choosing where to
start, or which thresholds to try, after seeing where the errors fall is selection on the
outcome and invalidates every p-value above. Depending on the confidences is much milder, and
the price can be stated exactly. Let lambda* = sup{lambda : R(lambda) > alpha} be the true
risk boundary, N* the number of calibration confidences >= lambda*, and m the smallest
selection size the walk tests.

(a) No assumption beyond i.i.d. When the grid contains every distinct calibration confidence
    (always the case when there are at most `n_grid` of them), the walk evaluates the p-value
    at every *fixed* threshold between its start and its stopping point, because the selected
    set only changes at observed confidences. On the event N* >= m a false certification
    therefore implies that the fixed-threshold test at lambda* rejected:
    P(R(lambda_hat) > alpha, N* >= m) <= delta.
(b) Monotone risk. If answering more never lowers the error rate (R non-increasing in lambda,
    equivalently the answers at the margin are at least as error-prone as the average answer
    above them), then P(R(lambda_hat) > alpha) <= delta for the full grid, and for a coarser
    quantile grid whenever the confidence distribution has no atom at lambda*. Proof sketch:
    the first true null the walk meets selects {c >= lambda*} plus some points below lambda*;
    given N* the former contribute Binomial(N*, R(lambda*)) errors with R(lambda*) >= alpha,
    the latter independent errors each of probability >= alpha, so the error count dominates
    Binomial(N, alpha) and the p-value is still super-uniform.

Monotone risk is what selective prediction presupposes of a confidence score; where it fails
badly the score is not worth thresholding. tests/test_uncertainty_conformal.py checks the
violation frequency by simulation where R is known in closed form, including a discrete and a
deliberately non-monotone confidence.

p-values. `binomial_pvalue` is exact for 0/1 losses. `hoeffding_bentkus_pvalue` (the bound of
Bates, Angelopoulos, Lei, Malik & Jordan 2021, used as a p-value in Learn then Test) is valid for
any loss in [0, 1]:
min( exp(-n * kl(min(r_hat, alpha) || alpha)),  e * P(Bin(n, alpha) <= ceil(n * r_hat)) ).
For 0/1 losses it is never smaller than the exact binomial tail; it is kept as the baseline
the general theory would use.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel
from scipy.special import rel_entr
from scipy.stats import binom

# Round-off guard. Subtracted before ceil(), it recovers the exact integer when a product such as
# (n + 1) * (1 - alpha) lands one ulp above it (90.00000000000001); added to a quantile, it keeps
# a label whose score equals the quantile up to round-off (a cumulative mass of 1.0000000000000002).
_EPS = 1e-9


# --------------------------------------------------------------------------------------------
# Split conformal prediction sets
# --------------------------------------------------------------------------------------------


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; +inf when that rank exceeds n.

    +inf means "too few calibration points to exclude anything at this alpha" (n < 1/alpha - 1):
    the honest prediction set is then every label.
    """
    _check_level(alpha, "alpha")
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    rank = math.ceil((n + 1) * (1.0 - alpha) - _EPS)
    if rank > n:
        return math.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


def lac_sets(
    probs_cal: np.ndarray, y_cal: np.ndarray, probs_test: np.ndarray, alpha: float
) -> np.ndarray:
    """LAC prediction sets as a boolean [n_test, n_classes] mask.

    `y_cal` holds integer class indices into the columns of `probs_cal`.
    """
    probs_cal, y_cal, probs_test = _check_classification(probs_cal, y_cal, probs_test)
    cal_scores = 1.0 - probs_cal[np.arange(y_cal.size), y_cal]
    q = conformal_quantile(cal_scores, alpha)
    return (1.0 - probs_test) <= q + _EPS


def aps_sets(
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    probs_test: np.ndarray,
    alpha: float,
    *,
    randomized: bool = True,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """APS prediction sets as a boolean [n_test, n_classes] mask.

    The set is the exact inversion {y : s(x, y, u) <= q} of the score used for calibration, with
    one u per test point shared by all its labels, so the sets are nested in alpha and coverage
    follows from the quantile lemma with no extra argument. A consequence is that a set can be
    empty when the top class alone overshoots q; forcing the top class in would only add coverage.
    """
    probs_cal, y_cal, probs_test = _check_classification(probs_cal, y_cal, probs_test)
    rng = np.random.default_rng(rng)
    u_cal = rng.uniform(size=y_cal.size) if randomized else np.ones(y_cal.size)
    u_test = rng.uniform(size=probs_test.shape[0]) if randomized else np.ones(probs_test.shape[0])
    cal_scores = _aps_scores(probs_cal, u_cal)[np.arange(y_cal.size), y_cal]
    q = conformal_quantile(cal_scores, alpha)
    return _aps_scores(probs_test, u_test) <= q + _EPS


def _aps_scores(probs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """APS score of every label: mass ranked strictly above it plus u times its own mass."""
    # Stable sort: equal probabilities are ordered by class index, the same way for calibration
    # and test points, which is all exchangeability needs.
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    mass_above = np.cumsum(sorted_probs, axis=1) - sorted_probs
    sorted_scores = mass_above + u[:, None] * sorted_probs
    scores = np.empty_like(sorted_scores)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


# --------------------------------------------------------------------------------------------
# p-values for H0: risk > alpha
# --------------------------------------------------------------------------------------------


def binomial_pvalue(k: int, n: int, alpha: float) -> float:
    """P(Bin(n, alpha) <= k): exact p-value for H0: risk > alpha after k errors in n trials."""
    _check_level(alpha, "alpha")
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    return float(binom.cdf(k, n, alpha))


def hoeffding_bentkus_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: risk > alpha, for the mean of n losses in [0, 1]."""
    _check_level(alpha, "alpha")
    if n < 0 or not 0.0 <= risk_hat <= 1.0:
        raise ValueError(f"need n >= 0 and risk_hat in [0, 1], got n={n}, risk_hat={risk_hat}")
    if n == 0:
        return 1.0
    return float(_hoeffding_bentkus(np.array([risk_hat]), np.array([n]), alpha)[0])


def _hoeffding_bentkus(risk_hat: np.ndarray, n: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorised Hoeffding-Bentkus p-values for empirical risks over n >= 1 losses each."""
    # Hoeffding's term is the Chernoff bound exp(-n * KL(Bernoulli(a) || Bernoulli(alpha))) at
    # a = min(risk_hat, alpha); rel_entr(x, y) = x log(x / y) with rel_entr(0, y) = 0.
    a = np.minimum(risk_hat, alpha)
    kl = rel_entr(a, alpha) + rel_entr(1.0 - a, 1.0 - alpha)
    hoeffding = np.exp(-n * kl)
    bentkus = math.e * binom.cdf(np.ceil(n * risk_hat - _EPS), n, alpha)
    return np.minimum(1.0, np.minimum(hoeffding, bentkus))


# --------------------------------------------------------------------------------------------
# Certified abstention threshold
# --------------------------------------------------------------------------------------------


class SelectiveGuarantee(BaseModel):
    """Outcome of `certify_threshold`.

    If `threshold` is a number, answering iff confidence >= threshold has selective risk <= alpha
    with probability >= 1 - delta over calibration draws, and the `cal_*` fields describe that
    rule on the calibration set. If it is None nothing could be certified: the rule is "abstain
    on everything", so n_selected, cal_coverage and cal_risk are 0.

    `pvalue` belongs to the test at the certified threshold (<= delta), or to the first test when
    that one failed (> delta), or is 1.0 when no candidate selected enough points to be tested.
    """

    alpha: float
    delta: float
    threshold: float | None
    n_cal: int
    n_selected: int
    cal_coverage: float
    cal_risk: float
    pvalue: float


def certify_threshold(
    conf_cal: np.ndarray,
    correct_cal: np.ndarray,
    *,
    alpha: float,
    delta: float,
    n_grid: int = 100,
    min_selected: int = 20,
    method: Literal["binomial", "hb"] = "binomial",
) -> SelectiveGuarantee:
    """Lowest threshold whose selective risk is certified <= alpha at confidence 1 - delta.

    Fixed-sequence Learn-then-Test over quantiles of `conf_cal`, most conservative first; see the
    module docstring for the guarantee and its assumptions. `correct_cal[i]` is truthy when
    answer i was right. `method="hb"` swaps the exact binomial tail for Hoeffding-Bentkus.

    `min_selected` affects power, not validity. The walk has to pass its first test, and a test
    on few points tolerates almost no errors (22 points and none wrong at alpha = delta = 0.1), so
    one unlucky mistake among the most confident answers certifies nothing. A larger value starts
    the walk where the test can absorb a few errors, at the price of never certifying a smaller
    selection. Like the grid, it must be fixed without looking at the calibration labels: a
    constant, a fraction of n, or a value tuned on the training split.
    """
    _check_level(alpha, "alpha")
    _check_level(delta, "delta")
    if method not in ("binomial", "hb"):
        raise ValueError(f"unknown method {method!r}")
    if n_grid < 1 or min_selected < 1:
        raise ValueError("n_grid and min_selected must be positive")
    conf = np.asarray(conf_cal, dtype=float).ravel()
    correct = np.asarray(correct_cal).ravel().astype(bool)
    if conf.size != correct.size:
        raise ValueError(f"{conf.size} confidences but {correct.size} labels")
    if np.isnan(conf).any():
        raise ValueError("confidences contain NaN")
    n = conf.size

    order = np.argsort(-conf, kind="stable")
    conf_desc = conf[order]
    errors_cum = np.cumsum(~correct[order])

    # Everything down to the p-values depends on the confidences only, never on the labels.
    thresholds = _candidate_thresholds(conf_desc, n_grid)
    # Selection is by value, so points tied with the threshold are in: #{c_i >= lambda}.
    n_selected = np.searchsorted(-conf_desc, -thresholds, side="right")
    # With N points and zero errors the p-value is (1 - alpha)^N for both methods, so a candidate
    # with (1 - alpha)^N > delta cannot be rejected whatever the labels say. Testing it would
    # abort the walk by construction, so the walk starts below it; the cut-off is a function of
    # (alpha, delta) alone.
    min_rejectable = math.ceil(math.log(delta) / math.log1p(-alpha) - _EPS)
    testable = n_selected >= max(min_selected, min_rejectable)
    thresholds, n_selected = thresholds[testable], n_selected[testable]

    n_errors = errors_cum[n_selected - 1]
    if method == "binomial":
        pvalues = binom.cdf(n_errors, n_selected, alpha)
    else:
        pvalues = _hoeffding_bentkus(n_errors / n_selected, n_selected, alpha)

    # Fixed-sequence testing: the number certified is the length of the leading run of rejections.
    rejected = pvalues <= delta
    n_certified = int(rejected.size if rejected.all() else np.argmin(rejected))
    if n_certified == 0:
        return SelectiveGuarantee(
            alpha=alpha,
            delta=delta,
            threshold=None,
            n_cal=n,
            n_selected=0,
            cal_coverage=0.0,
            cal_risk=0.0,
            pvalue=float(pvalues[0]) if pvalues.size else 1.0,
        )
    last = n_certified - 1
    return SelectiveGuarantee(
        alpha=alpha,
        delta=delta,
        threshold=float(thresholds[last]),
        n_cal=n,
        n_selected=int(n_selected[last]),
        cal_coverage=float(n_selected[last] / n),
        cal_risk=float(n_errors[last] / n_selected[last]),
        pvalue=float(pvalues[last]),
    )


def _candidate_thresholds(conf_desc: np.ndarray, n_grid: int) -> np.ndarray:
    """Candidate thresholds, descending, from confidences sorted in descending order.

    Every distinct confidence when there are at most `n_grid` of them (guarantee (a) of the
    module docstring then applies); otherwise the confidences at ranks ceil(j * n / n_grid),
    j = 1..n_grid, i.e. `n_grid` evenly spaced empirical quantiles ending at the minimum.
    """
    distinct = np.unique(conf_desc)[::-1]
    if distinct.size <= n_grid:
        return distinct
    n = conf_desc.size
    ranks = np.ceil(np.arange(1, n_grid + 1) * n / n_grid - _EPS).astype(int)
    return np.unique(conf_desc[ranks - 1])[::-1]


# --------------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------------


def _check_level(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")


def _check_classification(
    probs_cal: np.ndarray, y_cal: np.ndarray, probs_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_cal = np.asarray(probs_cal, dtype=float)
    probs_test = np.asarray(probs_test, dtype=float)
    y_cal = np.asarray(y_cal)
    if probs_cal.ndim != 2 or probs_test.ndim != 2 or probs_cal.shape[1] != probs_test.shape[1]:
        raise ValueError("probs_cal and probs_test must be [n, n_classes] with the same n_classes")
    if y_cal.shape != (probs_cal.shape[0],) or not np.issubdtype(y_cal.dtype, np.integer):
        raise ValueError("y_cal must hold one integer class index per calibration row")
    if y_cal.size and (y_cal.min() < 0 or y_cal.max() >= probs_cal.shape[1]):
        raise ValueError("y_cal contains a class index outside the columns of probs_cal")
    return probs_cal, y_cal, probs_test
