"""Ranking quality of a confidence score for selective prediction.

A selective predictor answers the questions it is most confident about and abstains on the rest.
Sweeping the threshold traces the risk-coverage curve (El-Yaniv & Wiener 2010): coverage is the
fraction answered, risk the error rate among the answered.

- AURC, the area under that curve, equals the selective risk averaged over all coverage levels
  (Geifman, Uziel & El-Yaniv 2019). It mixes two things: how accurate the answers are overall
  and how well confidence ranks them.
- E-AURC subtracts the AURC of an oracle that ranks every correct answer above every wrong one at
  the same accuracy, isolating the ranking part. 0 is a perfect ranking.
- AUROC is the probability that a random correct answer gets a higher confidence than a random
  wrong one, ties counting one half (the Mann-Whitney U statistic).

Ties. A threshold rule cannot separate answers with equal confidence, so "the top k" is undefined
inside a tie and any sort order would be an arbitrary choice that changes AURC. Tied answers
therefore share their group's mean error, which is the expected curve under uniformly random
tie-breaking and makes every function here invariant to the order of its inputs.

These are descriptive statistics of one sample. Reading a threshold off them (as
`coverage_at_risk` does) gives no guarantee on new data; that is what
`conformal.certify_threshold` is for.
"""

from __future__ import annotations

import numpy as np


def risk_coverage_curve(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and selective risk of the k most confident answers, for k = 1..n."""
    conf, correct = _check(conf, correct)
    n = conf.size
    order = np.argsort(-conf, kind="stable")
    errors = (~correct[order]).astype(float)
    # Tie groups appear as runs in the sorted order; spread each group's errors evenly over it.
    _, starts, counts = np.unique(-conf[order], return_index=True, return_counts=True)
    expected_errors = np.repeat(np.add.reduceat(errors, starts) / counts, counts)
    k = np.arange(1, n + 1)
    return k / n, np.cumsum(expected_errors) / k


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve: mean selective risk over coverage 1/n, ..., 1."""
    _, risk = risk_coverage_curve(conf, correct)
    return float(risk.mean())


def e_aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Excess AURC over the oracle ranking with the same number of errors (>= 0 up to ties)."""
    conf, correct = _check(conf, correct)
    n = conf.size
    n_correct = int(correct.sum())
    # The oracle answers all correct items first: zero errors up to k = n_correct, then every
    # further answer is an error.
    k = np.arange(1, n + 1)
    oracle_risk = np.maximum(k - n_correct, 0) / k
    return aurc(conf, correct) - float(oracle_risk.mean())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a positive > score of a negative) + 0.5 * P(tie), from midranks.

    With R+ the sum of the positives' ranks among all n scores, U = R+ - n+(n+ + 1)/2 counts the
    (positive, negative) pairs ordered correctly, and AUROC = U / (n+ * n-).
    """
    scores, labels = _check(scores, labels)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one positive and one negative label")
    # Midrank of a tie group = mean of the ranks it occupies = last rank - (size - 1) / 2.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    midranks = np.cumsum(counts) - (counts - 1) / 2.0
    rank_sum = float(midranks[inverse][labels].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def coverage_at_risk(conf: np.ndarray, correct: np.ndarray, max_risk: float) -> float:
    """Largest coverage of any threshold rule whose *empirical* selective risk is <= max_risk.

    Only thresholds at observed confidences are operating points (ties are answered together).
    Returns 0.0 when no threshold qualifies.
    """
    conf, correct = _check(conf, correct)
    order = np.argsort(-conf, kind="stable")
    errors_cum = np.cumsum(~correct[order])
    # Index of the last element of each tie group = number selected at that threshold, minus 1.
    _, starts, counts = np.unique(-conf[order], return_index=True, return_counts=True)
    n_selected = starts + counts
    risk = errors_cum[n_selected - 1] / n_selected
    feasible = n_selected[risk <= max_risk]
    return float(feasible.max() / conf.size) if feasible.size else 0.0


def selective_risk(conf: np.ndarray, correct: np.ndarray, threshold: float) -> float:
    """Error rate among answers with confidence >= threshold; 0.0 if none is answered.

    The zero for an empty selection is the convention under which abstaining on everything
    trivially meets any risk target: no answers, no errors.
    """
    conf, correct = _check(conf, correct)
    selected = conf >= threshold
    if not selected.any():
        return 0.0
    return float(np.mean(~correct[selected]))


def _check(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(conf, dtype=float).ravel()
    correct = np.asarray(correct).ravel().astype(bool)
    if conf.size != correct.size or conf.size == 0:
        raise ValueError(
            f"need equally many scores and labels (got {conf.size} and {correct.size}), at least one"
        )
    if np.isnan(conf).any():
        raise ValueError("scores contain NaN")
    return conf, correct
