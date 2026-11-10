"""Tests for primum.uncertainty.features, .models and .deploy."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from primum.types import Claim, ClaimVerdict, ConfidenceReport, Hit, Passage
from primum.uncertainty.calibration import IsotonicCalibration, TemperatureScaling
from primum.uncertainty.conformal import SelectiveGuarantee
from primum.uncertainty.deploy import ALPHAS, BUNDLE_FILENAME, DeploymentBundle
from primum.uncertainty.features import (
    FEATURES_DEPLOY,
    FEATURES_FULL,
    UnscoredRetrieval,
    extract_features,
    scale_free_stage,
)
from primum.uncertainty.metrics import auroc
from primum.uncertainty.models import (
    LogisticConfidence,
    MLPConfidence,
    _stratified_split,
    model_from_json,
)

# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------


def _hit(rank: int, **components: float) -> Hit:
    passage = Passage(passage_id=f"p{rank}", doc_id=f"d{rank}", text="text", source="pubmed")
    return Hit(
        passage=passage, score=1.0 / rank, rank=rank, retriever="hybrid", components=components
    )


def _verdict(label: str, p_entail: float, p_contra: float) -> ClaimVerdict:
    return ClaimVerdict(
        claim=Claim(text="claim"),
        label=label,
        p_entail=p_entail,
        p_neutral=1.0 - p_entail - p_contra,
        p_contra=p_contra,
    )


HITS = [
    _hit(1, dense=0.8, bm25=14.0, bm25_rank=1, dense_rank=2),
    _hit(2, dense=0.6, bm25=9.0, bm25_rank=7, dense_rank=1),  # BM25 put it outside its top 3
    _hit(3, dense=0.7, bm25=11.0, bm25_rank=3, dense_rank=3),
]
VERDICTS = [
    _verdict("supported", 0.9, 0.02),
    _verdict("unsupported", 0.3, 0.1),
    _verdict("contradicted", 0.1, 0.8),
]


def test_feature_lists():
    assert FEATURES_FULL[: len(FEATURES_DEPLOY)] == FEATURES_DEPLOY
    assert set(FEATURES_FULL) - set(FEATURES_DEPLOY) == {
        "sc_agreement",
        "sc_entropy",
        "verbal_margin",
    }
    assert len(set(FEATURES_FULL)) == len(FEATURES_FULL) == 13


def test_extract_features_by_hand():
    features = extract_features(
        HITS,
        VERDICTS,
        verbal_conf=0.85,
        sample_decisions=["yes", "yes", "no", "yes", "maybe"],
        probabilities={"yes": 0.7, "no": 0.2, "maybe": 0.1},
    )
    assert list(features) == FEATURES_FULL
    assert all(isinstance(v, float) for v in features.values())
    expected_entropy = -(0.6 * math.log(0.6) + 2 * 0.2 * math.log(0.2)) / math.log(3)
    expected = {
        "ret_top": 0.8,
        "ret_mean": 0.7,
        "ret_margin": 0.8 - 0.7,  # top minus runner-up by score, not by list position
        "ret_agree": 2 / 3,
        "nli_mean_entail": (0.9 + 0.3 + 0.1) / 3,
        "nli_min_entail": 0.1,
        "nli_max_contra": 0.8,
        "frac_supported": 1 / 3,
        "n_claims": 3.0,
        "verbal_conf": 0.85,
        "sc_agreement": 0.6,
        "sc_entropy": expected_entropy,
        "verbal_margin": 0.5,
    }
    assert features == pytest.approx(expected)


def test_retrieval_features_prefer_the_reranker_score_and_never_mix_stages():
    reranked = [_hit(1, dense=0.2, rerank=5.0), _hit(2, dense=0.9, rerank=-1.0)]
    assert extract_features(reranked, [], verbal_conf=None)["ret_top"] == 5.0
    mixed = [_hit(1, dense=0.2, rerank=5.0), _hit(2, dense=0.9)]  # only one hit was reranked
    assert extract_features(mixed, [], verbal_conf=None)["ret_top"] == 0.9
    assert extract_features([_hit(1, dense=0.4)], [], verbal_conf=None)["ret_margin"] == 0.0


def test_ranked_hits_without_a_scale_free_score_are_refused_not_zeroed():
    # BM25 alone: its raw score is not comparable across questions. Filling 0.0 would train
    # and serve the confidence model on a constant read as "no evidence found".
    lexical_only = [
        _hit(1, bm25=12.0, bm25_rank=1),
        _hit(2, bm25=3.0, bm25_rank=2),
    ]
    with pytest.raises(UnscoredRetrieval, match="hybrid"):
        extract_features(lexical_only, VERDICTS, verbal_conf=0.9)
    with pytest.raises(UnscoredRetrieval):
        scale_free_stage([_hit(1, dense=0.5), _hit(2, bm25=3.0)])  # one hit lacks it


def test_unranked_evidence_has_no_retrieval_signal_but_keeps_entailment():
    # The oracle condition's gold passage was handed over, not retrieved: no stage scores.
    oracle = [_hit(1)]
    assert scale_free_stage(oracle) is None
    features = extract_features(oracle, VERDICTS, verbal_conf=None)
    assert (features["ret_top"], features["ret_agree"]) == (0.0, 0.0)
    assert features["nli_max_contra"] == 0.8


def test_missing_signals_are_zero():
    features = extract_features(HITS, [], verbal_conf=None)
    for name in (
        "nli_mean_entail",
        "nli_min_entail",
        "nli_max_contra",
        "frac_supported",
        "n_claims",
        "verbal_conf",
        "sc_agreement",
        "sc_entropy",
        "verbal_margin",
    ):
        assert features[name] == 0.0
    assert features["ret_top"] == 0.8


def test_closed_book_zeroes_retrieval_and_entailment_features():
    features = extract_features([], VERDICTS, verbal_conf=0.9, sample_decisions=["no"] * 5)
    for name in (
        "ret_top",
        "ret_mean",
        "ret_margin",
        "ret_agree",
        "nli_mean_entail",
        "nli_min_entail",
        "nli_max_contra",
        "frac_supported",
    ):
        assert features[name] == 0.0
    assert features["n_claims"] == 3.0
    assert features["verbal_conf"] == 0.9
    assert (features["sc_agreement"], features["sc_entropy"]) == (1.0, 0.0)  # unanimous samples


def test_self_consistency_entropy_is_one_for_a_uniform_split():
    features = extract_features([], [], verbal_conf=None, sample_decisions=["yes", "no", "maybe"])
    assert features["sc_entropy"] == pytest.approx(1.0)
    assert features["sc_agreement"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------------------------
# Confidence models
# --------------------------------------------------------------------------------------------


def _linear_problem(
    rng: np.random.Generator, n: int, sharpness: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Logistic ground truth on features of very different scales; `sharpness` scales the logit."""
    scale, shift = np.array([1.0, 5.0, 0.1, 2.0, 1.0]), np.array([0.0, 3.0, 0.0, -1.0, 0.0])
    X = rng.normal(size=(n, 5)) * scale + shift
    logit = sharpness * (X[:, 0] + 0.2 * (X[:, 1] - 3.0)) - 0.5
    return X, (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)


def _interaction_problem(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Correct iff two signals agree in sign: no linear model can separate this."""
    X = rng.normal(size=(n, 4))
    return X, (X[:, 0] * X[:, 1] > 0).astype(int)


def test_logistic_matches_scikit_learn():
    rng = np.random.default_rng(0)
    X, y = _linear_problem(rng, 400)
    l2 = 0.05
    ours = LogisticConfidence(l2=l2).fit(X, y)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    # scikit-learn minimises sum(BCE) + ||w||^2 / (2C); ours mean(BCE) + l2 ||w||^2 / 2.
    reference = LogisticRegression(C=1.0 / (l2 * len(y)), tol=1e-10, max_iter=5000).fit(Z, y)
    np.testing.assert_allclose(ours.predict_proba(X), reference.predict_proba(Z)[:, 1], atol=1e-5)


def test_logistic_is_invariant_to_feature_scale_and_ignores_constant_features():
    rng = np.random.default_rng(1)
    X, y = _linear_problem(rng, 300)
    base = LogisticConfidence().fit(X, y).predict_proba(X)
    rescaled = LogisticConfidence().fit(X * 1000.0 + 7.0, y).predict_proba(X * 1000.0 + 7.0)
    np.testing.assert_allclose(base, rescaled, atol=1e-6)
    with_constant = np.column_stack([X, np.zeros(len(X))])  # e.g. a feature that is always 0.0
    model = LogisticConfidence().fit(with_constant, y)
    np.testing.assert_allclose(model.predict_proba(with_constant), base, atol=1e-6)
    # A non-zero value at prediction time in a feature never seen to vary must not blow up.
    shifted = with_constant.copy()
    shifted[:, -1] = 1.0
    assert np.isfinite(model.predict_proba(shifted)).all()


def test_mlp_learns_an_interaction_that_logistic_regression_cannot():
    rng = np.random.default_rng(2)
    X, y = _interaction_problem(rng, 1500)
    train, test = slice(0, 1000), slice(1000, 1500)
    linear = LogisticConfidence().fit(X[train], y[train])
    mlp = MLPConfidence(seed=0).fit(X[train], y[train])
    assert auroc(linear.predict_proba(X[test]), y[test]) < 0.6
    assert auroc(mlp.predict_proba(X[test]), y[test]) > 0.9


def test_mlp_is_deterministic_given_the_seed_and_leaves_global_rng_alone():
    rng = np.random.default_rng(3)
    X, y = _linear_problem(rng, 200)
    torch.manual_seed(123)
    expected_next = torch.rand(3)
    torch.manual_seed(123)
    first = MLPConfidence(seed=5, max_epochs=20).fit(X, y).predict_proba(X)
    assert torch.equal(torch.rand(3), expected_next)  # fitting consumed no global randomness
    second = MLPConfidence(seed=5, max_epochs=20).fit(X, y).predict_proba(X)
    other = MLPConfidence(seed=6, max_epochs=20).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, other)


def test_mlp_standardiser_is_fitted_on_training_rows_only():
    # Leakage check: the validation rows used for early stopping must not shape the inputs.
    rng = np.random.default_rng(4)
    X, y = _linear_problem(rng, 200)
    model = MLPConfidence(seed=9, max_epochs=5).fit(X, y)
    train_idx, val_idx = _stratified_split(y.astype(float), 0.2, np.random.default_rng(9))
    assert len(val_idx) == 40 and not set(train_idx) & set(val_idx)
    assert y[val_idx].mean() == pytest.approx(y.mean(), abs=0.02)  # stratified
    np.testing.assert_allclose(model._mean, X[train_idx].mean(axis=0))
    assert not np.allclose(model._mean, X.mean(axis=0))


def test_early_stopping_restores_the_best_epoch():
    # Pure-noise labels: more epochs can only overfit, so validation loss is best early and the
    # restored model must stay close to predicting the base rate.
    rng = np.random.default_rng(5)
    X = rng.normal(size=(120, 10))
    y = (rng.uniform(size=120) < 0.7).astype(int)
    model = MLPConfidence(
        hidden=64, dropout=0.0, weight_decay=0.0, lr=0.02, patience=10, seed=0
    ).fit(X, y)
    fresh = rng.normal(size=(2000, 10))
    assert np.abs(model.predict_proba(fresh) - 0.7).mean() < 0.15


@pytest.mark.parametrize("cls", [LogisticConfidence, MLPConfidence])
def test_model_json_round_trip(cls):
    rng = np.random.default_rng(6)
    X, y = _linear_problem(rng, 150)
    model = cls().fit(X, y)
    payload = json.loads(json.dumps(model.to_json()))  # must survive real JSON
    for restored in (model_from_json(payload), cls.from_json(json.dumps(payload))):
        assert type(restored) is cls
        np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
    other = MLPConfidence if cls is LogisticConfidence else LogisticConfidence
    with pytest.raises(ValueError):
        other.from_json(payload)


def test_predict_proba_shapes_and_errors():
    rng = np.random.default_rng(7)
    X, y = _linear_problem(rng, 100)
    model = LogisticConfidence().fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (100,) and ((probabilities > 0) & (probabilities < 1)).all()
    assert model.predict_proba(X[0]).shape == (1,)  # a single feature vector
    with pytest.raises(ValueError):
        model.predict_proba(X[:, :3])
    with pytest.raises(RuntimeError):
        LogisticConfidence().predict_proba(X)
    with pytest.raises(RuntimeError):
        MLPConfidence().to_json()
    for bad_y in (np.ones(100), np.full(100, 2)):
        with pytest.raises(ValueError):
            LogisticConfidence().fit(X, bad_y)
    with pytest.raises(ValueError):
        MLPConfidence().fit(np.where(X > 3, np.nan, X), y)
    with pytest.raises(ValueError):
        model_from_json({"kind": "forest"})


# --------------------------------------------------------------------------------------------
# Deployment bundle
# --------------------------------------------------------------------------------------------

FEATURE_NAMES = ["a", "b", "c", "d", "e"]


def _bundle(calibrator_cls=TemperatureScaling) -> tuple[DeploymentBundle, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8)
    X, y = _linear_problem(rng, 1500, sharpness=3.0)  # confident answers are rarely wrong
    train, fit_cal, cal = slice(0, 500), slice(500, 800), slice(800, 1500)
    model = LogisticConfidence().fit(X[train], y[train])
    calibrator = calibrator_cls().fit(model.predict_proba(X[fit_cal]), y[fit_cal])
    bundle = DeploymentBundle.build(
        model,
        calibrator,
        X[cal],
        y[cal],
        features=FEATURE_NAMES,
        delta=0.1,
        llm_model="claude-haiku-5-5",
        retriever="hybrid-ft",
        created="2026-11-10",
        heldout_qids=["q-1", "q-2"],
    )
    return bundle, X[cal], y[cal]


@pytest.mark.parametrize("calibrator_cls", [TemperatureScaling, IsotonicCalibration])
def test_bundle_build_save_load_score(tmp_path, calibrator_cls):
    bundle, X_cal, y_cal = _bundle(calibrator_cls)
    assert [g.alpha for g in bundle.guarantees] == list(ALPHAS)
    assert all(g.delta == 0.1 and g.n_cal == 700 for g in bundle.guarantees)
    assert bundle.guarantee(0.2).threshold is not None  # this problem is easy enough at 0.2

    path = bundle.save(tmp_path / "deployment")
    assert path.name == BUNDLE_FILENAME
    stored = json.loads(path.read_text())
    assert (stored["n_cal"], stored["llm_model"], stored["retriever"], stored["created"]) == (
        700,
        "claude-haiku-5-5",
        "hybrid-ft",
        "2026-11-10",
    )
    assert stored["heldout_qids"] == ["q-1", "q-2"]  # the provenance the guarantee check reads
    loaded = DeploymentBundle.load(tmp_path / "deployment")
    assert loaded.model_dump() == bundle.model_dump()

    # The rule the bundle applies to a new answer is the rule that was certified.
    guarantee = loaded.guarantee(0.2)
    answered = 0
    for row in X_cal:
        report = loaded.score(dict(zip(FEATURE_NAMES, row, strict=True)), alpha=0.2)
        assert isinstance(report, ConfidenceReport)
        assert report.answered == (report.p_correct >= guarantee.threshold)
        answered += report.answered
    assert answered == guarantee.n_selected
    assert (report.alpha, report.delta, report.threshold) == (0.2, 0.1, guarantee.threshold)
    assert report.prediction_set is None


def test_bundle_score_details():
    bundle, X_cal, _ = _bundle()
    features = dict(zip(FEATURE_NAMES, X_cal[0], strict=True))
    default = bundle.score(features)
    assert default.alpha == 0.10
    # Extra signals are passed through for display but do not affect the score.
    extended = bundle.score({**features, "sc_entropy": 0.4})
    assert extended.p_correct == default.p_correct and extended.features["sc_entropy"] == 0.4
    with pytest.raises(KeyError):
        bundle.score({name: 0.0 for name in FEATURE_NAMES[:-1]})
    with pytest.raises(ValueError):
        bundle.score(features, alpha=0.01)
    with pytest.raises(FileNotFoundError):
        DeploymentBundle.load("/nonexistent/primum/deployment")


def test_bundle_abstains_when_nothing_was_certified_and_answers_ties():
    bundle, X_cal, _ = _bundle()
    features = dict(zip(FEATURE_NAMES, X_cal[0], strict=True))
    p_correct = bundle.score(features).p_correct

    def with_threshold(threshold: float | None) -> DeploymentBundle:
        guarantee = SelectiveGuarantee(
            alpha=0.1,
            delta=0.1,
            threshold=threshold,
            n_cal=700,
            n_selected=0,
            cal_coverage=0.0,
            cal_risk=0.0,
            pvalue=1.0,
        )
        return bundle.model_copy(update={"guarantees": [guarantee]})

    none = with_threshold(None).score(features)
    assert none.threshold is None and none.answered is False  # never an uncertified fallback
    assert with_threshold(p_correct).score(features).answered is True  # p == threshold answers
    assert with_threshold(np.nextafter(p_correct, 1.0)).score(features).answered is False


def test_bundle_build_rejects_a_mismatched_feature_list():
    rng = np.random.default_rng(9)
    X, y = _linear_problem(rng, 200)
    model = LogisticConfidence().fit(X, y)
    with pytest.raises(ValueError):
        DeploymentBundle.build(
            model,
            TemperatureScaling(),
            X,
            y,
            features=["a", "b"],
            delta=0.1,
            llm_model="m",
            retriever="r",
            created="2026-11-10",
        )
