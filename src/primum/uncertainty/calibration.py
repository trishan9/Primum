"""Probability calibration for a binary event ("the answer is correct") and how to measure it.

A confidence p is calibrated when P(correct | p) = p. Each calibrator below is a non-decreasing
map fitted on held-out (p, y) pairs, so it changes the probabilities but (up to ties) not the
ranking; ranking metrics such as AUROC and AURC are unaffected, ECE and Brier are not. The maps
are constrained to be non-decreasing on purpose: a calibrator free to reverse the order would let
a score that is anti-correlated with correctness on the calibration labels (or weak and flipped
by chance) be re-oriented by those labels and then reported as informative.

- `TemperatureScaling` (Guo, Pleiss, Sun & Weinberger 2017): p' = sigmoid(logit(p) / T).
  One parameter, bounded T > 0, so the map is strictly increasing and cannot overfit a small set.
- `PlattScaling` (Platt 1999; fitting procedure of Lin, Lin & Weng 2007):
  p' = sigmoid(a * s + b), with s = logit(p) by default so that the identity is in the family,
  and the slope a bounded below by a small positive number, so it is strictly increasing too.
- `IsotonicCalibration` (Zadrozny & Elkan 2002): the non-decreasing step function minimising
  squared error, computed by pool-adjacent-violators. Non-parametric, hence the most flexible
  and the most prone to overfit; it also produces ties, which matters downstream because a
  threshold on a tied confidence answers every tied point.

Why this matters for the certified threshold: `conformal.certify_threshold` only uses the order
of the confidences, so a strictly increasing calibrator fitted on the certification split leaves
the certified rule unchanged. Isotonic regression is different: where it merges points depends
on the labels, so fitting it on the certification split makes the candidate thresholds
label-dependent and voids the guarantee. Fit calibrators on data disjoint from that split.

Metrics. With B bins, ECE = sum_b (n_b / n) * |acc_b - conf_b| (Naeini, Cooper & Hauskrecht
2015; Guo et al. 2017). It is a plug-in estimate that depends on the binning (equal-width bins
can be nearly empty, equal-mass bins adapt to the data) and is biased upwards on small samples,
so it is reported next to the Brier score, mean (p - y)^2, a proper scoring rule with no
binning choice.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self, TypedDict

import numpy as np

# Probabilities are clipped away from {0, 1} before taking logits, at 2^-53: 1 - 2^-53 is the
# largest float64 below 1, so no two distinct inputs in [2^-53, 1) are merged into one logit (a
# coarser clip such as 1e-6 would tie every confidence above 1 - 1e-6, exactly the answers a
# threshold cares most about). Only exact 0 and 1 and inputs below 1.1e-16 are clipped; their
# logit is +-36.7. Strict monotonicity is in exact arithmetic: after a steep map the output can
# still round to 1.0 in float64.
_P_EPS = float(np.finfo(np.float64).epsneg)
_T_MIN, _T_MAX = 0.05, 100.0


class ReliabilityBin(TypedDict):
    lo: float  # smallest confidence the bin can (equal_width) or does (equal_mass) contain
    hi: float
    count: int
    confidence: float  # mean predicted probability in the bin
    accuracy: float  # observed frequency of y = 1 in the bin


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def reliability_bins(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["equal_width", "equal_mass"] = "equal_width",
) -> list[ReliabilityBin]:
    """Per-bin confidence and accuracy for a reliability diagram; empty bins are omitted."""
    p, y = _check_pairs(p, y)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if strategy == "equal_width":
        # Bins are [k/B, (k+1)/B), with p = 1 placed in the last one.
        index = np.minimum((p * n_bins).astype(int), n_bins - 1)
        groups = [np.flatnonzero(index == k) for k in range(n_bins)]
        edges = [(k / n_bins, (k + 1) / n_bins) for k in range(n_bins)]
    elif strategy == "equal_mass":
        groups = _equal_mass_groups(p, n_bins)
        edges = [(float(p[g].min()), float(p[g].max())) if g.size else (0.0, 0.0) for g in groups]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return [
        ReliabilityBin(
            lo=lo,
            hi=hi,
            count=int(g.size),
            confidence=float(p[g].mean()),
            accuracy=float(y[g].mean()),
        )
        for g, (lo, hi) in zip(groups, edges, strict=True)
        if g.size
    ]


def _equal_mass_groups(p: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Indices of each equal-mass bin, never splitting a group of tied scores.

    Cutting the sorted order into `n_bins` runs (`np.array_split`) would send tied scores to both
    sides of a cut according to their input position, so the ECE would depend on row order; ties
    are common for isotonic outputs and rounded stated probabilities. Instead every tie group goes
    whole to the run that contains its mid-rank. With distinct scores this is exactly
    `array_split` (runs of n // B or n // B + 1 points); with ties a bin absorbs or cedes whole
    groups, so bins can be unequal or empty, and the result depends on the values alone.
    """
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    first_rank = np.cumsum(counts) - counts
    mid_rank = first_rank + (counts - 1) / 2.0
    sizes = np.full(n_bins, p.size // n_bins) + (np.arange(n_bins) < p.size % n_bins)
    # Run k holds the ranks [cumsum(sizes)[k-1], cumsum(sizes)[k]).
    bin_of_group = np.searchsorted(np.cumsum(sizes), mid_rank, side="right")
    bin_of_point = bin_of_group[inverse]
    return [np.flatnonzero(bin_of_point == k) for k in range(n_bins)]


def ece(
    p: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["equal_width", "equal_mass"] = "equal_width",
) -> float:
    """Expected calibration error: bin-mass-weighted mean of |accuracy - confidence|."""
    bins = reliability_bins(p, y, n_bins, strategy)
    n = sum(b["count"] for b in bins)
    return float(sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins) / n)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome."""
    p, y = _check_pairs(p, y)
    return float(np.mean((p - y) ** 2))


# --------------------------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------------------------


class IdentityCalibration:
    """No recalibration ("raw"); lets callers treat every condition uniformly."""

    name: ClassVar[str] = "raw"

    def fit(self, p: np.ndarray, y: np.ndarray) -> Self:
        _check_pairs(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p, dtype=float)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.name}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        return cls()


class TemperatureScaling:
    """p' = sigmoid(logit(p) / T), with T fitted by maximum likelihood."""

    name: ClassVar[str] = "temperature"

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> Self:
        p, y = _check_pairs(p, y)
        # The NLL is convex in beta = 1/T (logistic regression with one slope and no intercept),
        # so clipping the unconstrained optimum to [1/T_MAX, 1/T_MIN] is the constrained optimum.
        # The bounds matter when the data are separable (beta -> inf) or the score is
        # anti-correlated with correctness (beta <= 0, which no temperature can express).
        (beta,) = _fit_logistic(_logit(p)[:, None], y)
        self.temperature = float(1.0 / np.clip(beta, 1.0 / _T_MAX, 1.0 / _T_MIN))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(np.asarray(p, dtype=float)) / self.temperature)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.name, "temperature": self.temperature}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        obj = cls()
        obj.temperature = float(payload["temperature"])
        return obj


class PlattScaling:
    """p' = sigmoid(a * s + b), s = logit(p) (default) or the raw score (`logit_input=False`).

    `logit_input=False` is Platt's original formulation for an arbitrary real-valued score and is
    what a single unbounded signal (a reranker logit, a claim count) needs. Either way the score
    must already be oriented so that larger means "more likely correct" (orient on training data,
    never on the calibration labels); the fit keeps that order.
    """

    name: ClassVar[str] = "platt"

    def __init__(self, logit_input: bool = True) -> None:
        self.logit_input = logit_input
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> Self:
        p, y = _check_pairs(p, y, probabilities=self.logit_input)
        # Platt's targets: the posterior mean of each class's label under a uniform prior,
        # (N+ + 1)/(N+ + 2) and 1/(N- + 2), instead of 1 and 0. They keep the optimum finite
        # when the classes are separable and regularise against small calibration sets.
        n_pos = float(y.sum())
        n_neg = float(y.size - n_pos)
        targets = np.where(y > 0.5, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        s = self._score(p)
        a, b = _fit_logistic(np.column_stack([s, np.ones(p.size)]), targets)
        # Unconstrained, a < 0 would reverse the ranking whenever the score is anti-correlated
        # with correctness on these labels. The slope is bounded below by the flattest
        # temperature, 1 / T_MAX logits per standard deviation of the score (the same bound as
        # TemperatureScaling on a unit-variance logit, but free of the raw score's units). The
        # cross-entropy is convex in (a, b), so when the unconstrained optimum lies below the
        # bound the constrained one lies on it: fix a there and refit b alone. The result is
        # close to the base rate everywhere yet keeps the original order, so ranking metrics
        # report the score as it is instead of a flipped version.
        spread = float(s.std())
        a_min = 1.0 / (_T_MAX * spread) if spread > 0.0 else 0.0
        if a < a_min:
            a = a_min
            (b,) = _fit_logistic(np.ones((p.size, 1)), targets, offset=a * s)
        self.a, self.b = float(a), float(b)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return _sigmoid(self.a * self._score(np.asarray(p, dtype=float)) + self.b)

    def _score(self, p: np.ndarray) -> np.ndarray:
        return _logit(p) if self.logit_input else p

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.name, "a": self.a, "b": self.b, "logit_input": self.logit_input}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        obj = cls(logit_input=bool(payload["logit_input"]))
        obj.a, obj.b = float(payload["a"]), float(payload["b"])
        return obj


class IsotonicCalibration:
    """Non-decreasing least-squares fit of y on p, linearly interpolated between knots.

    Inputs may be any real-valued score. Predictions outside the fitted range are clamped to the
    end values, the same convention as scikit-learn's `IsotonicRegression(out_of_bounds="clip")`.
    """

    name: ClassVar[str] = "isotonic"

    def __init__(self) -> None:
        self.x: np.ndarray = np.array([0.0, 1.0])
        self.y: np.ndarray = np.array([0.0, 1.0])

    def fit(self, p: np.ndarray, y: np.ndarray) -> Self:
        p, y = _check_pairs(p, y, probabilities=False)
        # Points sharing an x must receive one fitted value: replace them by their mean with
        # weight = multiplicity, which leaves the weighted least-squares problem unchanged.
        x, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
        means = np.bincount(inverse, weights=y) / counts
        fitted = pool_adjacent_violators(means, counts.astype(float))
        # Keep only the knots where the step function changes; interpolation is unaffected.
        keep = np.ones(x.size, dtype=bool)
        keep[1:-1] = (fitted[1:-1] != fitted[:-2]) | (fitted[1:-1] != fitted[2:])
        self.x, self.y = x[keep], fitted[keep]
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(p, dtype=float), self.x, self.y)

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.name, "x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        obj = cls()
        obj.x = np.asarray(payload["x"], dtype=float)
        obj.y = np.asarray(payload["y"], dtype=float)
        return obj


Calibrator = IdentityCalibration | TemperatureScaling | PlattScaling | IsotonicCalibration

CALIBRATORS: dict[str, type[Calibrator]] = {
    cls.name: cls
    for cls in (IdentityCalibration, TemperatureScaling, PlattScaling, IsotonicCalibration)
}


def calibrator_from_json(payload: dict[str, Any]) -> Calibrator:
    """Rebuild any calibrator from its `to_json()` payload."""
    try:
        cls = CALIBRATORS[payload["kind"]]
    except KeyError:
        raise ValueError(f"not a calibrator payload: {payload!r}") from None
    return cls.from_json(payload)


# --------------------------------------------------------------------------------------------
# Numerical kernels
# --------------------------------------------------------------------------------------------


def pool_adjacent_violators(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted isotonic regression: argmin_f sum_i w_i (y_i - f_i)^2  s.t.  f_1 <= ... <= f_n.

    Scan left to right keeping a stack of blocks, each holding its weighted mean. A new point
    opens a block; while the last block's mean is below its predecessor's the two are merged,
    because the constrained optimum is constant across any adjacent pair that violates the order
    (Ayer et al. 1955; Barlow et al. 1972). Each point is pushed once and merged at most once,
    so the cost is O(n).
    """
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if y.shape != w.shape or y.ndim != 1:
        raise ValueError("y and weights must be 1-D arrays of equal length")
    if (w <= 0).any():
        raise ValueError("weights must be positive")
    means: list[float] = []
    block_weights: list[float] = []
    sizes: list[int] = []
    for value, weight in zip(y.tolist(), w.tolist(), strict=True):
        means.append(value)
        block_weights.append(weight)
        sizes.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            merged = block_weights[-2] + block_weights[-1]
            means[-2] = (block_weights[-2] * means[-2] + block_weights[-1] * means[-1]) / merged
            block_weights[-2] = merged
            sizes[-2] += sizes[-1]
            means.pop()
            block_weights.pop()
            sizes.pop()
    return np.repeat(means, sizes)


def _fit_logistic(
    design: np.ndarray,
    targets: np.ndarray,
    *,
    offset: np.ndarray | float = 0.0,
    max_iter: int = 100,
) -> np.ndarray:
    """Minimise the mean cross-entropy between sigmoid(design @ w + offset) and `targets`.

    Newton's method from w = 0: the objective is convex, so with a backtracking line search it
    converges to the global optimum. `targets` may be soft (Platt's smoothed labels); `offset` is
    a fixed term, used to refit an intercept with the slope held at a bound. Used for the one- and
    two-parameter calibrators only, where the Hessian is at most 2x2.
    """
    n, d = design.shape
    w = np.zeros(d)

    def loss(weights: np.ndarray) -> float:
        z = design @ weights + offset
        # Cross-entropy written with logaddexp so large |z| does not overflow.
        return float(np.mean(np.logaddexp(0.0, z) - targets * z))

    current = loss(w)
    for _ in range(max_iter):
        prob = _sigmoid(design @ w + offset)
        gradient = design.T @ (prob - targets) / n
        if np.max(np.abs(gradient)) < 1e-10:
            break
        # The ridge keeps the Hessian invertible when a column is constant or the fit saturates.
        hessian = (design * (prob * (1.0 - prob))[:, None]).T @ design / n + 1e-12 * np.eye(d)
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-10:
            candidate = loss(w - scale * step)
            if candidate <= current - 1e-4 * scale * float(gradient @ step):  # Armijo condition
                break
            scale /= 2.0
        else:
            break  # no descent possible at machine precision: converged
        w, current = w - scale * step, candidate
    return w


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _P_EPS, 1.0 - _P_EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # exp(-logaddexp(0, -z)) = 1 / (1 + exp(-z)) without overflow for large |z|.
    return np.exp(-np.logaddexp(0.0, -z))


def _check_pairs(
    p: np.ndarray, y: np.ndarray, *, probabilities: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if p.size != y.size or p.size == 0:
        raise ValueError(
            f"need equally many scores and labels (got {p.size} and {y.size}), at least one"
        )
    if np.isnan(p).any():
        raise ValueError("scores contain NaN")
    if probabilities and ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if ((y != 0.0) & (y != 1.0)).any():
        raise ValueError("labels must be 0/1")
    return p, y
