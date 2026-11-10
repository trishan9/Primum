"""Contrastive training and the cross-encoder, on a tiny randomly initialised transformer.

Nothing is downloaded: the BERT here has one layer and 32 hidden units and is built from a
config and a toy vocabulary, which is enough to exercise every line of the training loop,
the sentence-transformers export and the reranker on CPU in a few seconds.
"""

from __future__ import annotations

import json
import math
import sys
import types

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

import primum.retrieval.train_contrastive as train_module
from primum.config import Settings
from primum.retrieval import CrossEncoderReranker, DenseIndex
from primum.retrieval.train_contrastive import (
    _BiEncoder,
    _false_negative_mask,
    fit,
    info_nce_loss,
    mean_pool,
    train_encoder,
    warmup_linear_decay,
)
from primum.types import Passage

N_TOPICS = 60
CONDITIONS = [f"cond{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(N_TOPICS)]
DRUGS = [f"drug{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(N_TOPICS)]
# Each question shares exactly one content word with its passage, so the pairing is learnable.
PAIRS = [
    (f"what treats {condition}", f"{drug} treats {condition} in adults")
    for condition, drug in zip(CONDITIONS, DRUGS, strict=True)
]
VOCABULARY = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "what", "treats", "in", "adults",
    *CONDITIONS, *DRUGS,
]  # fmt: skip
TINY = {
    "vocab_size": len(VOCABULARY),
    "hidden_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
}


def save_tiny(model: torch.nn.Module, directory) -> str:
    model.save_pretrained(directory)
    BertTokenizer(vocab={token: i for i, token in enumerate(VOCABULARY)}).save_pretrained(directory)
    return str(directory)


@pytest.fixture(scope="module")
def tiny_encoder(tmp_path_factory) -> str:
    torch.manual_seed(0)
    return save_tiny(BertModel(BertConfig(**TINY)), tmp_path_factory.mktemp("tiny-encoder"))


@pytest.fixture(scope="module")
def trained(tiny_encoder, tmp_path_factory):
    """One training run shared by the tests that inspect its log and its saved model."""
    output_dir = tmp_path_factory.mktemp("trained") / "encoder-ft"
    log = fit(
        PAIRS,
        base_model=tiny_encoder,
        output_dir=output_dir,
        epochs=8,
        batch_size=8,
        lr=3e-3,
        temperature=0.05,
        seed=13,
        device="cpu",
    )
    return log, output_dir


# --- mean pooling ------------------------------------------------------------------------


def test_mean_pool_ignores_padding():
    tokens = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],  # third token is padding
            [[1.0, 1.0], [2.0, 2.0], [6.0, 6.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    np.testing.assert_allclose(mean_pool(tokens, mask).numpy(), [[2.0, 3.0], [3.0, 3.0]])


# --- InfoNCE -----------------------------------------------------------------------------


def numpy_info_nce(q, p, temperature, negatives=None, mask=None) -> float:
    """The loss from the module docstring, written independently in numpy."""
    candidates = p if negatives is None else np.vstack([p, negatives])
    logits = q @ candidates.T / temperature
    if mask is not None:
        logits = np.where(mask, -np.inf, logits)

    def cross_entropy_on_diagonal(scores: np.ndarray) -> float:
        peak = scores.max(axis=1, keepdims=True)
        log_partition = peak[:, 0] + np.log(np.exp(scores - peak).sum(axis=1))
        return float(np.mean(log_partition - np.diag(scores)))

    n = len(q)
    return 0.5 * (cross_entropy_on_diagonal(logits) + cross_entropy_on_diagonal(logits[:, :n].T))


def unit_rows(rng: np.random.Generator, n: int, dim: int = 8) -> np.ndarray:
    x = rng.normal(size=(n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_info_nce_two_orthogonal_pairs_by_hand():
    # q = p = I: logits are [[1, 0], [0, 1]] at tau = 1. Each row's loss is
    # -log(e / (e + 1)) = log(1 + 1/e), and by symmetry so is each column's.
    eye = torch.eye(2)
    assert info_nce_loss(eye, eye, 1.0).item() == pytest.approx(math.log(1 + math.exp(-1)))
    # With tau = 0.5 the logits double: log(1 + e^-2). A lower temperature sharpens the softmax.
    assert info_nce_loss(eye, eye, 0.5).item() == pytest.approx(math.log(1 + math.exp(-2)))


def test_info_nce_of_an_uninformative_encoder_is_log_batch_size():
    same = torch.nn.functional.normalize(torch.ones(6, 4), dim=-1)
    assert info_nce_loss(same, same, 0.05).item() == pytest.approx(math.log(6))


def uninformative_loss_with_hard_negatives(batch: int) -> float:
    same = torch.nn.functional.normalize(torch.ones(batch, 4), dim=-1)
    return info_nce_loss(same, same, 0.05, hard_negatives=same).item()


def test_info_nce_chance_level_with_hard_negatives_averages_the_two_directions():
    # Query-to-passage chooses among B + H candidates, passage-to-query among B queries only.
    assert uninformative_loss_with_hard_negatives(6) == pytest.approx(
        (math.log(12) + math.log(6)) / 2
    )


def test_info_nce_matches_numpy_reference():
    rng = np.random.default_rng(0)
    q, p = unit_rows(rng, 5), unit_rows(rng, 5)
    loss = info_nce_loss(torch.from_numpy(q), torch.from_numpy(p), 0.05)
    assert loss.item() == pytest.approx(numpy_info_nce(q, p, 0.05))


def test_info_nce_with_hard_negatives_and_false_negative_mask_matches_numpy_reference():
    rng = np.random.default_rng(1)
    q, p, negatives = unit_rows(rng, 4), unit_rows(rng, 4), unit_rows(rng, 4)
    mask = np.zeros((4, 8), dtype=bool)
    mask[0, 2] = True  # passage 2 is another positive of query 0
    mask[3, 5] = True  # the negative mined for pair 1 is in fact relevant to query 3
    loss = info_nce_loss(
        torch.from_numpy(q),
        torch.from_numpy(p),
        0.1,
        hard_negatives=torch.from_numpy(negatives),
        false_negatives=torch.from_numpy(mask),
    )
    assert loss.item() == pytest.approx(numpy_info_nce(q, p, 0.1, negatives, mask))
    # The hard negatives enlarge the denominator, the mask shrinks it.
    plain = numpy_info_nce(q, p, 0.1)
    assert (
        numpy_info_nce(q, p, 0.1, negatives) > plain > numpy_info_nce(q, p, 0.1, mask=mask[:, :4])
    )


def test_info_nce_is_symmetric_in_its_two_sides():
    rng = np.random.default_rng(2)
    q, p = torch.from_numpy(unit_rows(rng, 6)), torch.from_numpy(unit_rows(rng, 6))
    assert info_nce_loss(q, p, 0.05).item() == pytest.approx(info_nce_loss(p, q, 0.05).item())


def test_info_nce_gradient_pulls_pairs_together():
    rng = np.random.default_rng(3)
    q = torch.from_numpy(unit_rows(rng, 4)).requires_grad_()
    p = torch.from_numpy(unit_rows(rng, 4))
    info_nce_loss(q, p, 0.1).backward()
    # A gradient-descent step on q_i moves it towards its own positive p_i.
    step = -q.grad
    assert all((step[i] @ p[i]).item() > 0 for i in range(4))


def test_false_negative_mask_marks_other_known_positives_only():
    known = {0: {10, 11}, 1: {12}, 2: {10}}
    query_ids = np.array([0, 1, 2])
    candidate_ids = np.array([10, 12, 10, 11, 99, 12])  # 3 positives, then 3 hard negatives
    mask = _false_negative_mask(query_ids, candidate_ids, known).numpy()
    expected = np.zeros((3, 6), dtype=bool)
    expected[0, 2] = True  # passage 10 again, as pair 2's positive
    expected[0, 3] = True  # passage 11, a second positive of query 0, mined as a negative
    expected[1, 5] = True  # query 1's own positive, mined as a negative for another pair
    expected[2, 0] = True  # passage 10 is pair 0's positive and also relevant to query 2
    np.testing.assert_array_equal(mask, expected)
    assert not mask.diagonal().any()


# --- schedule ----------------------------------------------------------------------------


def test_warmup_then_linear_decay():
    schedule = [warmup_linear_decay(s, warmup_steps=4, total_steps=12) for s in range(13)]
    np.testing.assert_allclose(schedule[:5], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(schedule[5:], [7 / 8, 6 / 8, 5 / 8, 4 / 8, 3 / 8, 2 / 8, 1 / 8, 0.0])
    assert warmup_linear_decay(0, warmup_steps=0, total_steps=5) == 1.0


# --- training loop -----------------------------------------------------------------------


def test_training_log_is_complete_and_consistent(trained):
    log, _ = trained
    # 60 queries: 6 held out for dev, 54 pairs train; batches of 8 give 6 full + one of 6.
    assert log["dev"]["before"]["n_queries"] == 6
    assert log["n_train_pairs"] == 54 and log["n_train_queries"] == 54
    assert log["steps"] == 8 * 7
    assert log["device"] == "cpu"
    assert log["wall_time_s"] > 0
    assert log["loss_at_chance"] == pytest.approx(math.log(8))
    assert log["hyperparameters"]["temperature"] == 0.05
    assert log["hyperparameters"]["passage_tokens"] == 32  # capped by the tiny model

    curve = log["loss_curve"]
    assert [point["step"] for point in curve] == list(range(1, 57))
    assert all(math.isfinite(point["loss"]) for point in curve)
    assert curve[0]["lr"] == 0.0  # first step of the warmup
    assert max(point["lr"] for point in curve) == pytest.approx(3e-3, rel=0.05)
    json.dumps(log)  # the whole log must be JSON-serialisable


def test_training_reduces_the_loss(trained):
    log, _ = trained
    losses = [point["loss"] for point in log["loss_curve"]]
    first_epoch, last_epoch = np.mean(losses[:7]), np.mean(losses[-7:])
    assert last_epoch < 0.5 * first_epoch
    assert last_epoch < log["loss_at_chance"]


def test_trained_encoder_generalises_to_held_out_queries(trained):
    log, _ = trained
    before, after = log["dev"]["before"], log["dev"]["after"]
    assert before["pool_size"] == after["pool_size"] == 6
    # A randomly initialised mean-pooled encoder already matches on shared tokens, so the
    # untrained baseline is high; training must not lose that on questions it never saw.
    assert after["mrr@10"] >= before["mrr@10"]
    assert after["mrr@10"] > 0.9


def test_saved_encoder_loads_as_a_sentence_transformer_and_reproduces_embeddings(trained):
    _, output_dir = trained
    texts = [passage for _, passage in PAIRS[:5]]
    passages = [
        Passage(passage_id=str(i), doc_id=str(i), text=text, source="pubmed")
        for i, text in enumerate(texts)
    ]
    index = DenseIndex(output_dir, passages)  # loads through sentence-transformers
    own = _BiEncoder(output_dir, "cpu").embed(texts, max_tokens=32)
    np.testing.assert_allclose(index.embeddings, own, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(index.embeddings, axis=1), 1.0, atol=1e-5)
    # The fine-tuned encoder retrieves the passage that answers a training question.
    assert index.search(PAIRS[2][0], k=1)[0].passage.passage_id == "2"


def test_training_with_mined_hard_negatives(tiny_encoder, tmp_path):
    log = fit(
        PAIRS,
        base_model=tiny_encoder,
        output_dir=tmp_path / "encoder",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        hard_negatives=True,
        device="cpu",
    )
    assert log["steps"] == 7
    assert log["hyperparameters"]["hard_negatives"] is True
    # Regression: this used to log log(16), the q->p term alone, above the true chance level.
    assert log["loss_at_chance"] == pytest.approx(uninformative_loss_with_hard_negatives(8))
    assert all(math.isfinite(point["loss"]) for point in log["loss_curve"])
    assert (tmp_path / "encoder" / "modules.json").exists()


def test_training_needs_in_batch_negatives(tiny_encoder, tmp_path):
    with pytest.raises(ValueError, match="batch size of at least 2"):
        fit(PAIRS, base_model=tiny_encoder, output_dir=tmp_path, batch_size=1, device="cpu")
    with pytest.raises(ValueError, match="at least 2 training pairs"):
        fit(PAIRS[:1], base_model=tiny_encoder, output_dir=tmp_path, device="cpu")


def test_known_pairs_outside_the_training_sample_are_still_masked(
    tiny_encoder, tmp_path, monkeypatch
):
    # Regression: masking used to see only the (subsampled) training pairs, so a relevant
    # passage whose own pair was dropped stayed in the denominator as a negative.
    pairs = PAIRS[:19]  # too few queries for a dev split, and one batch of 19 holds them all
    masks = []

    def spy(query_ids, candidate_ids, known_positives):
        mask = _false_negative_mask(query_ids, candidate_ids, known_positives)
        masks.append((query_ids.tolist(), candidate_ids.tolist(), mask))
        return mask

    monkeypatch.setattr(train_module, "_false_negative_mask", spy)
    dropped = (PAIRS[0][0], PAIRS[1][1])  # query 0 is also answered by passage 1
    fit(
        pairs,
        known_pairs=[*pairs, dropped, (PAIRS[0][0], "a passage never trained on")],
        base_model=tiny_encoder,
        output_dir=tmp_path / "encoder",
        batch_size=19,
        device="cpu",
    )
    [(query_ids, candidate_ids, mask)] = masks
    # Texts are interned in order of first appearance, so ids equal indices into `pairs`.
    assert mask[query_ids.index(0), candidate_ids.index(1)]
    assert int(mask.sum()) == 1


def test_train_encoder_passes_every_pair_as_known_when_subsampling(monkeypatch, tmp_path):
    seen = {}

    def fake_fit(pairs, *, known_pairs, **_):
        seen.update(pairs=pairs, known_pairs=known_pairs)
        return {}

    settings = Settings(finetuned_encoder_dir=tmp_path / "ft", results_dir=tmp_path / "results")
    monkeypatch.setattr("primum.config.get_settings", lambda: settings)
    monkeypatch.setattr(train_module, "fit", fake_fit)
    monkeypatch.setitem(
        sys.modules, "primum.data", types.SimpleNamespace(load_contrastive_pairs=lambda: PAIRS)
    )
    train_encoder(max_pairs=10)
    assert len(seen["pairs"]) == 10 and set(seen["pairs"]) < set(PAIRS)
    assert list(seen["known_pairs"]) == PAIRS


def test_train_encoder_writes_model_and_log_without_the_api_key(
    tiny_encoder, tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
    settings = Settings(
        base_encoder=tiny_encoder,
        finetuned_encoder_dir=tmp_path / "artifacts" / "encoder-ft",
        results_dir=tmp_path / "results",
    )
    assert settings.anthropic_api_key == "sk-secret-value"
    monkeypatch.setattr("primum.config.get_settings", lambda: settings)
    monkeypatch.setattr(train_module, "select_device", lambda: "cpu")
    monkeypatch.setitem(
        sys.modules, "primum.data", types.SimpleNamespace(load_contrastive_pairs=lambda: PAIRS)
    )

    path = train_encoder(epochs=1, batch_size=8, max_pairs=40, created="2026-01-02T03:04:05Z")

    assert path == settings.finetuned_encoder_dir
    assert (path / "modules.json").exists()
    raw = (tmp_path / "results" / "encoder_training.json").read_text()
    assert "sk-secret-value" not in raw
    report = json.loads(raw)
    assert report["meta"]["created"] == "2026-01-02T03:04:05Z"
    assert report["meta"]["n"] == 40 and report["meta"]["n_pairs_available"] == 60
    assert report["meta"]["seeds"] == [settings.seed]
    assert report["meta"]["models"]["base_encoder"] == tiny_encoder
    assert report["meta"]["config"]["base_encoder"] == tiny_encoder
    assert "anthropic_api_key" not in report["meta"]["config"]
    # 40 pairs: 4 dev queries, 36 train pairs -> 4 full batches of 8 and one of 4.
    assert report["steps"] == 5 and report["loss_curve"]


# --- cross-encoder -----------------------------------------------------------------------


def test_cross_encoder_scores_are_the_model_logits(tmp_path):
    torch.manual_seed(0)
    model = BertForSequenceClassification(BertConfig(**TINY, num_labels=1)).eval()
    directory = save_tiny(model, tmp_path)
    pairs = [(query, passage) for query, passage in PAIRS[:5]] + [(PAIRS[0][0], PAIRS[9][1])]

    scores = CrossEncoderReranker(directory, batch_size=4).score(pairs)

    tokenizer = BertTokenizer(vocab={token: i for i, token in enumerate(VOCABULARY)})
    encoded = tokenizer(
        [q for q, _ in pairs], [p for _, p in pairs], padding=True, return_tensors="pt"
    )
    with torch.inference_mode():
        expected = model(**encoded).logits.squeeze(-1).numpy()
    assert scores.shape == (6,) and scores.dtype == np.float32
    # Batching (and therefore padding) must not change a pair's score.
    np.testing.assert_allclose(scores, expected, atol=1e-4)
    assert CrossEncoderReranker(directory).score([]).shape == (0,)


def test_cross_encoder_rejects_a_multi_class_model(tmp_path):
    model = BertForSequenceClassification(BertConfig(**TINY, num_labels=3))
    reranker = CrossEncoderReranker(save_tiny(model, tmp_path))
    with pytest.raises(ValueError, match="single relevance logit"):
        reranker.score([("what treats condaa", "drugaa treats condaa in adults")])
