"""Fitting the confidence estimators compared in RQ2 and RQ3, always on training rows only.

Three kinds of estimator share one interface, "fit on train, then score any rows":

- a learned combination: `LogisticConfidence` or `MLPConfidence` on a feature list;
- a single signal: a one-feature logistic regression. Some signals point the wrong way for a
  confidence (`nli_max_contra`, `sc_entropy`: higher means *less* likely correct) and some are
  not probabilities at all (`n_claims`). Fitting sigmoid(w x + b) on the training rows orients
  and scales the signal using training labels only; since it is monotone in x it ranks exactly
  as the signal itself, so ranking metrics are those of the raw signal (or its negation);
- the generator's stated probability for its decision (`verbal_conf`), used as is: the
  "trust the model's own number" baseline, with nothing fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from primum.experiments._common import rng_for, stratified_folds
from primum.uncertainty import (
    CALIBRATORS,
    FEATURES_DEPLOY,
    FEATURES_FULL,
    IdentityCalibration,
    LogisticConfidence,
    MLPConfidence,
)
from primum.uncertainty.calibration import Calibrator
from primum.uncertainty.models import ConfidenceModel

FEATURE_SETS: dict[str, list[str]] = {"deploy": FEATURES_DEPLOY, "full": FEATURES_FULL}
MODEL_KINDS = ("logistic", "mlp")
STATED = "stated:verbal_conf"
CROSS_FIT_FOLDS = 5


@dataclass(frozen=True)
class Estimator:
    """A named confidence estimator: a model kind on some features, or the stated probability."""

    name: str
    kind: str  # "logistic", "mlp" or "stated"
    features: tuple[str, ...]

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int) -> ConfidenceModel | None:
        """Fit on training rows (X has all FEATURES_FULL columns); None for the stated baseline."""
        if self.kind == "stated":
            return None
        return fit_model(self.kind, _select(X, self.features), y, seed)

    def score(self, model: ConfidenceModel | None, X: np.ndarray) -> np.ndarray:
        if model is None:
            return np.clip(_select(X, ("verbal_conf",))[:, 0], 0.0, 1.0)
        return model.predict_proba(_select(X, self.features))


def combined(kind: str, feature_set: str) -> Estimator:
    return Estimator(f"{kind}:{feature_set}", kind, tuple(FEATURE_SETS[feature_set]))


def single_signal(feature: str) -> Estimator:
    return Estimator(f"signal:{feature}", "logistic", (feature,))


def stated() -> Estimator:
    return Estimator(STATED, "stated", ("verbal_conf",))


def fit_model(kind: str, X: np.ndarray, y: np.ndarray, seed: int) -> ConfidenceModel:
    if kind == "logistic":
        return LogisticConfidence().fit(X, y)
    if kind == "mlp":
        return MLPConfidence(seed=seed).fit(X, y)
    raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")


def fit_cross_fitted_calibrator(
    estimator: Estimator, calibrator: str, X: np.ndarray, y: np.ndarray, seed: int
) -> Calibrator:
    """A calibrator fitted on out-of-fold predictions of the *training* rows.

    RQ3 keeps the calibration split for certification alone, so the calibrator has to be
    learnt from training rows. Fitting it on the model's in-sample predictions would teach it
    the model's overfitting (an MLP is more confident on rows it was trained on), so each
    training row is scored by a model fitted on the other folds, as in scikit-learn's
    `CalibratedClassifierCV(ensemble=False)`. The calibrator is then applied to the model
    refitted on all training rows.
    """
    if calibrator == IdentityCalibration.name:
        return IdentityCalibration()
    out_of_fold = np.empty(y.size)
    for fold, held_out in enumerate(
        stratified_folds(y.astype(bool), CROSS_FIT_FOLDS, rng_for(seed, "crossfit"))
    ):
        rest = np.setdiff1d(np.arange(y.size), held_out)
        model = estimator.fit(X[rest], y[rest], seed + fold)
        out_of_fold[held_out] = estimator.score(model, X[held_out])
    return CALIBRATORS[calibrator]().fit(out_of_fold, y)


def _select(X: np.ndarray, features: tuple[str, ...]) -> np.ndarray:
    return X[:, [FEATURES_FULL.index(name) for name in features]]
