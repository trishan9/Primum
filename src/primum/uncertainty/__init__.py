"""Uncertainty engine: from pipeline signals to a calibrated, certified answer/abstain decision.

    features.extract_features   answer            -> signal vector
    models.*Confidence          signal vector     -> P(correct)                 (learned)
    calibration.*               P(correct)        -> calibrated P(correct), ECE, Brier
    conformal.certify_threshold calibration set   -> threshold with P(risk <= alpha) >= 1 - delta
    conformal.lac_sets/aps_sets class probs       -> prediction sets with >= 1 - alpha coverage
    metrics, stats              evaluation: risk-coverage, AUROC, bootstrap and permutation tests
    deploy.DeploymentBundle     the frozen rule the live system loads

Pure numpy / PyTorch: no LLM calls and no I/O except `DeploymentBundle.save` / `load`.
"""

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
from primum.uncertainty.conformal import (
    SelectiveGuarantee,
    aps_sets,
    binomial_pvalue,
    certify_threshold,
    conformal_quantile,
    hoeffding_bentkus_pvalue,
    lac_sets,
)
from primum.uncertainty.deploy import ALPHAS, BUNDLE_FILENAME, DeploymentBundle
from primum.uncertainty.features import (
    FEATURES_DEPLOY,
    FEATURES_FULL,
    UnscoredRetrieval,
    extract_features,
    scale_free_stage,
)
from primum.uncertainty.metrics import (
    aurc,
    auroc,
    coverage_at_risk,
    e_aurc,
    risk_coverage_curve,
    selective_risk,
)
from primum.uncertainty.models import LogisticConfidence, MLPConfidence, model_from_json
from primum.uncertainty.stats import (
    bootstrap_ci,
    paired_bootstrap_pvalue,
    paired_permutation_pvalue,
)

__all__ = [
    "ALPHAS",
    "BUNDLE_FILENAME",
    "CALIBRATORS",
    "FEATURES_DEPLOY",
    "FEATURES_FULL",
    "DeploymentBundle",
    "IdentityCalibration",
    "IsotonicCalibration",
    "LogisticConfidence",
    "MLPConfidence",
    "PlattScaling",
    "SelectiveGuarantee",
    "TemperatureScaling",
    "UnscoredRetrieval",
    "aps_sets",
    "auroc",
    "aurc",
    "binomial_pvalue",
    "bootstrap_ci",
    "brier",
    "calibrator_from_json",
    "certify_threshold",
    "conformal_quantile",
    "coverage_at_risk",
    "e_aurc",
    "ece",
    "extract_features",
    "hoeffding_bentkus_pvalue",
    "lac_sets",
    "model_from_json",
    "paired_bootstrap_pvalue",
    "paired_permutation_pvalue",
    "pool_adjacent_violators",
    "reliability_bins",
    "risk_coverage_curve",
    "scale_free_stage",
    "selective_risk",
]
