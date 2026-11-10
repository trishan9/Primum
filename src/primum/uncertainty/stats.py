"""Resampling inference for per-item metric arrays (per-query nDCG, per-question correctness).

All three procedures treat the items (queries, questions) as the sampling unit and assume they
are i.i.d.; they say nothing about variability from retraining a model or resampling an LLM.

- `bootstrap_ci`: percentile bootstrap interval (Efron & Tibshirani 1993, ch. 13).
- `paired_bootstrap_pvalue`: two-sided bootstrap-t test of H0: E[a - b] = 0 (Efron & Tibshirani
  1993, sec. 16.4; Hall 1992). It is only approximate: the resamples come from the empirical
  distribution, whose spread underestimates the population's at small n, so the plain recentred
  bootstrap of the mean rejects a true H0 far too often (about 15% at a nominal 5% for n = 8
  skewed differences). Studentising each resample by its own standard error makes the statistic
  asymptotically pivotal and the error in the rejection rate second-order, but a small skewed
  sample still leaves it somewhat liberal (about 7% in the same setting).
- `paired_permutation_pvalue`: sign-flip randomisation test (Fisher 1935). Under H0 the two
  systems' scores on an item are exchangeable, so each difference is symmetric about zero and
  flipping signs at random generates the null distribution exactly, with no normality or
  variance assumption. This is the recommended test for paired IR metrics (Smucker, Allan &
  Carterette 2007), and the one the experiments use.

Monte-Carlo p-values use (1 + #{as extreme}) / (1 + n_draws), so they are never exactly zero.
For the permutation test the observed statistic is itself one draw from the exact null, which
makes this p-value valid at every n (Phipson & Smyth 2010); the bootstrap has no exact null to
draw from, so there the correction is only a convention.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Resampled statistics are compared with the observed one up to this relative tolerance, so
# that a resample reproducing the observed data is counted as "as extreme" despite round-off.
_RTOL = 1e-12
# Rows of the sign / index matrix generated at once; bounds memory at ~_CHUNK * n bytes.
_CHUNK = 1024


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    level: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval (lo, hi) for `stat` of `values`.

    Items are resampled along axis 0, so `values` may be [n] or [n, ...] when the statistic
    needs several aligned columns per item (e.g. confidence and correctness for an AUROC).
    """
    values = np.asarray(values)
    n = values.shape[0]
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level}")
    rng = np.random.default_rng(rng)
    draws = np.array([stat(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)], dtype=float)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(draws, [tail, 1.0 - tail])
    return float(lo), float(hi)


def paired_bootstrap_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Two-sided bootstrap-t p-value for H0: mean(a - b) = 0 on paired per-item scores.

    Approximate, and liberal for small skewed samples; prefer `paired_permutation_pvalue`.
    """
    diff = _paired_differences(a, b)
    rng = np.random.default_rng(rng)
    n = diff.size
    mean, se = diff.mean(), _standard_error(diff)
    if se == 0.0:
        # One item says nothing about variability, so it cannot reject. Several identical
        # differences give no evidence against H0 if they are zero, and show no sampling
        # variability at all otherwise (every resample reproduces the observed mean exactly).
        return 1.0 if mean == 0.0 or n < 2 else 1.0 / (1 + n_boot)
    observed = abs(mean) / se
    extreme = 0
    for start in range(0, n_boot, _CHUNK):
        rows = min(_CHUNK, n_boot - start)
        resamples = diff[rng.integers(0, n, size=(rows, n))]
        # Resampling estimates the law of mean* - mean, which mimics that of mean - E[diff], so
        # t* = (mean* - mean) / se* is a draw from the null law of the observed t = mean / se.
        # Dividing by each resample's own se (not the observed one) is what makes it pivotal.
        shift = np.abs(resamples.mean(axis=1) - mean)
        se_star = _standard_error(resamples)
        # A resample of one repeated item has se* = 0; it is as extreme as possible unless it
        # also reproduces the mean, in which case it carries no evidence.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_star = np.where(se_star > 0.0, shift / se_star, np.where(shift > 0.0, np.inf, 0.0))
        extreme += int(np.sum(t_star >= observed * (1.0 - _RTOL)))
    return (1 + extreme) / (1 + n_boot)


def paired_permutation_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Two-sided sign-flip permutation p-value for H0: a and b are exchangeable within items."""
    diff = _paired_differences(a, b)
    rng = np.random.default_rng(rng)
    observed = abs(diff.sum())
    extreme = 0
    for start in range(0, n_perm, _CHUNK):
        rows = min(_CHUNK, n_perm - start)
        signs = rng.integers(0, 2, size=(rows, diff.size), dtype=np.int8) * 2 - 1
        extreme += int(np.sum(np.abs(signs @ diff) >= observed * (1.0 - _RTOL)))
    return (1 + extreme) / (1 + n_perm)


def _standard_error(x: np.ndarray) -> np.ndarray | float:
    """Standard error of the mean along the last axis (0 for a single item)."""
    n = x.shape[-1]
    return x.std(axis=-1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(x.shape[:-1])


def _paired_differences(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size == 0:
        raise ValueError(f"paired samples need equal, non-zero length (got {a.size} and {b.size})")
    return a - b
