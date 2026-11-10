"""The artefact the live system loads: a fitted confidence model plus its certified thresholds.

A `DeploymentBundle` freezes one complete decision rule

    features -> confidence model -> calibrator -> p_correct;   answer iff p_correct >= threshold(alpha)

together with the provenance needed to say what the thresholds were certified *on*. It is one
JSON file, so it can be read, diffed and audited without running any code.

What the certificate covers, and what it does not:
- The thresholds are valid for questions drawn from the same distribution as the calibration
  set, answered by the same pipeline (same generator model, retriever and feature list; all
  recorded here), and only for *new* draws: a benchmark question among the training or
  calibration rows is in-sample, so the bundle lists the held-out ones (`heldout_qids`).
  Free-form questions typed into the chat interface are a different
  distribution: for them the threshold is a principled operating point, not a guarantee.
- The calibration rows passed to `build` must be disjoint from every row used to fit the
  confidence model or the calibrator. Reusing them for a strictly increasing calibrator
  (temperature scaling) is harmless because certification only uses the order of the
  confidences; reusing them for isotonic regression is not (see `calibration`).
- When no threshold could be certified for an alpha, the bundle stores `threshold=None` and
  `score` abstains. It never falls back to an uncertified threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

from primum.types import ConfidenceReport
from primum.uncertainty.calibration import Calibrator, calibrator_from_json
from primum.uncertainty.conformal import SelectiveGuarantee, certify_threshold
from primum.uncertainty.models import ConfidenceModel, model_from_json

BUNDLE_FILENAME = "bundle.json"
ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)


class DeploymentBundle(BaseModel):
    features: list[str]  # order of the model's input columns
    confidence_model: dict[str, Any]  # `LogisticConfidence.to_json()` or `MLPConfidence.to_json()`
    calibrator: dict[str, Any]  # `<calibrator>.to_json()`
    delta: float
    guarantees: list[SelectiveGuarantee]  # one per certified alpha
    # Provenance: what the thresholds were certified on.
    n_cal: int
    llm_model: str
    retriever: str
    created: str  # date string supplied by the caller; this module never reads the clock
    # Benchmark questions that no part of this bundle saw: neither the model and calibrator
    # (fitted on the training rows) nor the certification (the calibration rows). Only these
    # are new draws from the calibration distribution, so only for them does the certificate
    # say anything; a question the threshold was certified on, or the model was fitted on, is
    # in-sample. Empty (the default) means no question is known to be held out.
    heldout_qids: list[str] = Field(default_factory=list)

    _model: ConfidenceModel = PrivateAttr()
    _calibrator: Calibrator = PrivateAttr()

    def model_post_init(self, _context: Any) -> None:
        # Deserialise once, so a corrupt bundle fails at load time and `score` stays cheap.
        self._model = model_from_json(self.confidence_model)
        self._calibrator = calibrator_from_json(self.calibrator)

    @classmethod
    def build(
        cls,
        model: ConfidenceModel,
        calibrator: Calibrator,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        *,
        features: Sequence[str],
        delta: float,
        llm_model: str,
        retriever: str,
        created: str,
        alphas: Sequence[float] = ALPHAS,
        n_grid: int = 100,
        min_selected: int = 20,
        heldout_qids: Sequence[str] = (),
    ) -> Self:
        """Certify one threshold per alpha on held-out (X_cal, y_cal) and package the result.

        `model` and `calibrator` must already be fitted, on rows disjoint from the calibration
        rows. Each alpha gets its own guarantee at level delta; they are not simultaneous, which
        is the right reading for a system run at one alpha at a time. `heldout_qids` are the
        benchmark questions kept out of both the fit and the calibration rows.
        """
        X_cal = np.asarray(X_cal, dtype=float)
        if X_cal.ndim != 2 or X_cal.shape[1] != len(features):
            raise ValueError(f"X_cal must be [n, {len(features)}] to match the feature list")
        confidence = calibrator.predict(model.predict_proba(X_cal))
        guarantees = [
            certify_threshold(
                confidence,
                y_cal,
                alpha=alpha,
                delta=delta,
                n_grid=n_grid,
                min_selected=min_selected,
            )
            for alpha in alphas
        ]
        return cls(
            features=list(features),
            confidence_model=model.to_json(),
            calibrator=calibrator.to_json(),
            delta=delta,
            guarantees=guarantees,
            n_cal=X_cal.shape[0],
            llm_model=llm_model,
            retriever=retriever,
            created=created,
            heldout_qids=list(heldout_qids),
        )

    def guarantee(self, alpha: float) -> SelectiveGuarantee:
        """The certificate for this alpha; raises if the bundle was not certified at it."""
        for guarantee in self.guarantees:
            if abs(guarantee.alpha - alpha) < 1e-9:
                return guarantee
        certified = [g.alpha for g in self.guarantees]
        raise ValueError(f"no threshold was certified for alpha={alpha}; available: {certified}")

    def score(self, features: Mapping[str, float], alpha: float = 0.10) -> ConfidenceReport:
        """Calibrated P(correct) for one answer and the answer/abstain decision at `alpha`."""
        missing = [name for name in self.features if name not in features]
        if missing:
            raise KeyError(f"features missing for the confidence model: {missing}")
        row = np.array([[float(features[name]) for name in self.features]])
        p_correct = float(self._calibrator.predict(self._model.predict_proba(row))[0])
        guarantee = self.guarantee(alpha)
        return ConfidenceReport(
            features={name: float(value) for name, value in features.items()},
            p_correct=p_correct,
            alpha=guarantee.alpha,
            delta=self.delta,
            threshold=guarantee.threshold,
            # Ties answer (>=), matching how the threshold was certified. No certified threshold
            # means the only rule known to meet the risk target is to abstain.
            answered=guarantee.threshold is not None and p_correct >= guarantee.threshold,
        )

    def save(self, directory: Path | str) -> Path:
        """Write `<directory>/bundle.json` and return its path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / BUNDLE_FILENAME
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path | str) -> Self:
        """Read `<directory>/bundle.json`; raises FileNotFoundError if no bundle was saved."""
        return cls.model_validate_json((Path(directory) / BUNDLE_FILENAME).read_text("utf-8"))
