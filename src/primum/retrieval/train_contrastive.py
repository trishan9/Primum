"""Contrastive fine-tuning of the bi-encoder on biomedical (question, abstract) pairs.

The training loop is written out in PyTorch rather than delegated to a trainer class.

Objective. A batch holds B pairs (q_i, p_i). Both sides are encoded by the same transformer,
mean-pooled over tokens and L2-normalised, so q_i . p_j is a cosine similarity. With
temperature tau, s_ij = q_i . c_j / tau over the candidates c = [p_1..p_B] (plus one mined
hard negative per pair, [n_1..n_B], when enabled), the symmetric InfoNCE loss is

    L_q2p = -(1/B) * sum_i log( exp(s_ii) / sum_{j not in F_i} exp(s_ij) )
    L_p2q = -(1/B) * sum_i log( exp(s_ii) / sum_{j <= B, j not in F'_i} exp(s_ji) )
    L     = (L_q2p + L_p2q) / 2

Each term is a softmax classification of the true partner among the batch: every other
pair's passage serves as a negative for q_i at no extra encoding cost ("in-batch
negatives"). InfoNCE lower-bounds the mutual information between the two views,
I(q; p) >= log(B) - L (van den Oord et al., 2018), which is why a larger batch gives a
tighter, harder objective. The temperature is needed because cosines are confined to
[-1, 1]: without it the softmax could put at most e^2 times more mass on the positive than
on a negative, and the loss would saturate far from zero.

F_i is the set of *false negatives*: candidates that are known positives of q_i other than
p_i itself (the same passage appearing twice in a batch, another relevant document of the
same query, or a mined negative that is in fact relevant). They are removed from the
denominator. Without this, a corpus with many relevant documents per query (NFCorpus) would
push true positives apart. "Known" means known to the whole pair set, not just the pairs
trained on: when training uses a subsample, a relation whose pair was left out is still a
positive and is still masked.

Chance level. An encoder that scores every candidate equally has L_q2p = log(B + H) and
L_p2q = log(B) (the passage-to-query term never sees the H hard negatives), so the symmetric
loss at chance is (log(B + H) + log B) / 2; masking false negatives lowers it further.

Leakage. Pairs come from `primum.data.load_contrastive_pairs`, which excludes every
PubMedQA-labelled PMID and all NFCorpus dev/test queries. The dev queries held out below are
split by query text, so the before/after dev metric is measured on unseen questions.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from primum.retrieval._common import select_device, top_k
from primum.retrieval.bm25 import BM25
from primum.retrieval.metrics import mrr_at_k

WARMUP_FRACTION = 0.1  # share of all optimiser steps spent ramping the learning rate up
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
QUERY_TOKENS = 64
PASSAGE_TOKENS = 256  # the sequence length all-MiniLM-L6-v2 was trained and is served at
HARD_NEGATIVE_DEPTH = 20  # BM25 ranks a hard negative is drawn from
DEV_QUERIES = 500


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average the token vectors of each sequence, ignoring padding.

    token_embeddings: [batch, tokens, dim]; attention_mask: [batch, tokens] with 1 for real
    tokens. Padding must be masked out, or the embedding of a text would depend on the
    length of the longest text it happened to be batched with.
    """
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def info_nce_loss(
    queries: torch.Tensor,
    positives: torch.Tensor,
    temperature: float,
    *,
    hard_negatives: torch.Tensor | None = None,
    false_negatives: torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives (formula in the module docstring).

    queries, positives: unit-norm [B, dim], row i of one paired with row i of the other.
    hard_negatives: optional unit-norm [H, dim], appended as extra candidates for every
        query. They enter only the query-to-passage term: a hard negative is a passage, and
        there is no query it should be classified to.
    false_negatives: optional bool [B, B + H]; True at (i, j) removes candidate j from the
        softmax of query i. The diagonal (the designated positive) must be False.
    """
    candidates = positives if hard_negatives is None else torch.cat([positives, hard_negatives])
    logits = queries @ candidates.T / temperature  # [B, B + H]
    if false_negatives is not None:
        logits = logits.masked_fill(false_negatives, float("-inf"))
    n = queries.shape[0]
    target = torch.arange(n, device=queries.device)
    query_to_passage = F.cross_entropy(logits, target)
    # Transposing the square block turns "which passage for query i" into "which query for
    # passage i"; the transposed mask removes the same known-positive pairs.
    passage_to_query = F.cross_entropy(logits[:, :n].T, target)
    return (query_to_passage + passage_to_query) / 2


def warmup_linear_decay(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Learning-rate multiplier: linear 0 -> 1 over the warmup, then linear 1 -> 0.

    The pretrained weights sit in a good basin, while AdamW's second-moment estimates start
    at zero and make the first updates disproportionately large; ramping the rate up
    protects the weights until those estimates are meaningful.
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


class _BiEncoder:
    """Tokenizer + transformer + mean pooling + L2 normalisation (one tower, shared weights)."""

    def __init__(self, model_name_or_path: str | Path, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path))
        self.model = AutoModel.from_pretrained(str(model_name_or_path)).to(device)
        self.device = device
        self.passage_tokens = min(PASSAGE_TOKENS, self.model.config.max_position_embeddings)
        self.query_tokens = min(QUERY_TOKENS, self.passage_tokens)

    def __call__(self, texts: list[str], max_tokens: int) -> torch.Tensor:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt"
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state
        return F.normalize(mean_pool(hidden, batch["attention_mask"]), dim=-1)

    @torch.inference_mode()
    def embed(self, texts: list[str], max_tokens: int, batch_size: int = 128) -> np.ndarray:
        """Embeddings for evaluation: dropout off, no gradients."""
        self.model.eval()
        chunks = [
            self(texts[start : start + batch_size], max_tokens).float().cpu().numpy()
            for start in range(0, len(texts), batch_size)
        ]
        return np.concatenate(chunks)

    def save(self, directory: Path) -> None:
        """Write a sentence-transformers model directory (Transformer -> Pooling -> Normalize).

        The library is used for its on-disk format only, so `DenseIndex` and any other
        sentence-transformers consumer can load the encoder; nothing here trains with it.
        """
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.sentence_transformer import modules

        with tempfile.TemporaryDirectory() as scratch:
            self.model.save_pretrained(scratch)
            self.tokenizer.save_pretrained(scratch)
            transformer = modules.Transformer(scratch, max_seq_length=self.passage_tokens)
            pooling = modules.Pooling(transformer.get_embedding_dimension(), pooling_mode="mean")
            packaged = SentenceTransformer(
                modules=[transformer, pooling, modules.Normalize()], device="cpu"
            )
            directory.mkdir(parents=True, exist_ok=True)
            packaged.save(str(directory), create_model_card=False)


def fit(
    pairs: Sequence[tuple[str, str]],
    *,
    known_pairs: Sequence[tuple[str, str]] = (),
    base_model: str | Path,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 2e-5,
    temperature: float = 0.05,
    hard_negatives: bool = False,
    seed: int = 13,
    device: str | None = None,
) -> dict:
    """Fine-tune `base_model` on (query, positive passage) pairs and save it to `output_dir`.

    `known_pairs` are further known relevances that are not trained on, typically the pairs
    a subsample left out. Together with `pairs` they define which candidates are false
    negatives, which passages hard-negative mining must skip, and the dev qrels.

    Returns the training log (loss curve, steps, wall time, device, hyperparameters and the
    held-out dev metric before and after training).
    """
    if batch_size < 2:
        raise ValueError("in-batch negatives need a batch size of at least 2")
    started = time.perf_counter()
    device = device or select_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Intern the texts: the same query or passage recurs across pairs, and integer ids make
    # "is this candidate a known positive of that query?" a set lookup.
    queries = list(dict.fromkeys(query for query, _ in pairs))
    passages = list(dict.fromkeys(passage for _, passage in pairs))
    query_id = {text: i for i, text in enumerate(queries)}
    passage_id = {text: i for i, text in enumerate(passages)}
    pair_ids = np.array([(query_id[q], passage_id[p]) for q, p in pairs], dtype=np.int64)
    # Only relations between texts that occur in training can ever meet in a batch.
    known_positives: dict[int, set[int]] = {}
    for q, p in [*pairs, *known_pairs]:
        if q in query_id and p in passage_id:
            known_positives.setdefault(query_id[q], set()).add(passage_id[p])

    # Hold out whole queries, so no dev question is ever seen in training.
    n_dev = min(DEV_QUERIES, len(queries) // 10)
    dev_queries = rng.permutation(len(queries))[:n_dev] if n_dev >= 2 else np.empty(0, int)
    train_ids = pair_ids[~np.isin(pair_ids[:, 0], dev_queries)]
    if len(train_ids) < 2:
        raise ValueError(f"need at least 2 training pairs, got {len(train_ids)}")

    encoder = _BiEncoder(base_model, device)
    dev_before = _dev_retrieval(encoder, queries, passages, dev_queries, known_positives)

    mined: dict[int, np.ndarray] = {}
    if hard_negatives:
        mined = _mine_hard_negatives(queries, passages, np.unique(train_ids[:, 0]), known_positives)

    # A lone trailing pair has no in-batch negative, so a final batch of one is dropped.
    full, remainder = divmod(len(train_ids), batch_size)
    total_steps = epochs * (full + (remainder >= 2))
    warmup_steps = round(WARMUP_FRACTION * total_steps)
    optimizer = torch.optim.AdamW(_parameter_groups(encoder.model), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_linear_decay(step, warmup_steps=warmup_steps, total_steps=total_steps),
    )

    log_every = max(1, total_steps // 200)
    loss_curve: list[dict[str, float]] = []
    window: list[float] = []
    step = 0
    progress = tqdm(total=total_steps, desc="contrastive", unit="step")
    for _ in range(epochs):
        order = rng.permutation(len(train_ids))
        for start in range(0, len(order), batch_size):
            batch = train_ids[order[start : start + batch_size]]
            if len(batch) < 2:
                continue
            candidate_ids = batch[:, 1]
            if hard_negatives:
                drawn = [
                    _draw_negative(mined[q], known_positives[q], len(passages), rng)
                    for q in batch[:, 0].tolist()
                ]
                candidate_ids = np.concatenate([candidate_ids, np.array(drawn)])

            encoder.model.train()
            query_vectors = encoder([queries[q] for q in batch[:, 0]], encoder.query_tokens)
            # Positives and hard negatives go through the encoder in one forward pass.
            candidate_vectors = encoder(
                [passages[p] for p in candidate_ids], encoder.passage_tokens
            )
            loss = info_nce_loss(
                query_vectors,
                candidate_vectors[: len(batch)],
                temperature,
                hard_negatives=candidate_vectors[len(batch) :] if hard_negatives else None,
                false_negatives=_false_negative_mask(
                    batch[:, 0], candidate_ids, known_positives
                ).to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}; lower the learning rate")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), MAX_GRAD_NORM)
            step_lr = optimizer.param_groups[0]["lr"]  # the rate this update is taken at
            optimizer.step()
            scheduler.step()

            step += 1
            window.append(loss.item())
            if step % log_every == 0 or step == total_steps:
                # Each point is the mean loss since the previous point, not a single batch.
                loss_curve.append({"step": step, "loss": float(np.mean(window)), "lr": step_lr})
                progress.set_postfix(loss=f"{loss_curve[-1]['loss']:.4f}")
                window.clear()
            progress.update()
    progress.close()

    dev_after = _dev_retrieval(encoder, queries, passages, dev_queries, known_positives)
    encoder.save(output_dir)

    n_candidates = batch_size * (2 if hard_negatives else 1)  # B positives (+ B hard negatives)
    return {
        "hyperparameters": {
            "base_model": str(base_model),
            "objective": "symmetric InfoNCE, in-batch negatives, false negatives masked",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "temperature": temperature,
            "hard_negatives": hard_negatives,
            "hard_negative_depth": HARD_NEGATIVE_DEPTH if hard_negatives else None,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION,
            "max_grad_norm": MAX_GRAD_NORM,
            "query_tokens": encoder.query_tokens,
            "passage_tokens": encoder.passage_tokens,
            "pooling": "mean",
            "seed": seed,
        },
        "device": device,
        "n_train_pairs": int(len(train_ids)),
        "n_train_queries": int(np.unique(train_ids[:, 0]).size),
        "n_passages": len(passages),
        "steps": step,
        "wall_time_s": round(time.perf_counter() - started, 2),
        # Reference line for the loss curve: the loss of an encoder that scores every
        # candidate equally, averaged over the two directions (module docstring).
        "loss_at_chance": (math.log(n_candidates) + math.log(batch_size)) / 2,
        "loss_curve": loss_curve,
        "dev": None if dev_before is None else {"before": dev_before, "after": dev_after},
    }


def train_encoder(
    *,
    epochs: int = 1,
    batch_size: int = 64,
    max_pairs: int | None = None,
    lr: float = 2e-5,
    temperature: float = 0.05,
    hard_negatives: bool = False,
    created: str | None = None,
) -> Path:
    """Fine-tune `settings.base_encoder` on the contrastive pairs built by `primum.data`.

    Saves the encoder to `settings.finetuned_encoder_dir` (returned) and writes the training
    log to `results/encoder_training.json`. `created` is the run timestamp recorded in the
    log; the CLI passes it so that one invocation stamps all of its outputs identically.
    """
    from primum.config import get_settings
    from primum.data import load_contrastive_pairs

    settings = get_settings()
    all_pairs = load_contrastive_pairs()
    pairs = all_pairs
    n_available = len(all_pairs)
    if max_pairs is not None and n_available > max_pairs:
        keep = np.random.default_rng(settings.seed).choice(n_available, max_pairs, replace=False)
        pairs = [all_pairs[i] for i in sorted(keep.tolist())]

    log = fit(
        pairs,
        known_pairs=all_pairs,  # pairs left out of the sample are still known positives
        base_model=settings.base_encoder,
        output_dir=settings.finetuned_encoder_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        hard_negatives=hard_negatives,
        seed=settings.seed,
    )
    report = {
        "meta": {
            "created": created or datetime.now(UTC).isoformat(timespec="seconds"),
            # The API key is the one setting that must never reach a results file.
            "config": settings.model_dump(mode="json", exclude={"anthropic_api_key"}),
            "n": len(pairs),
            "n_pairs_available": n_available,
            "seeds": [settings.seed],
            "models": {
                "base_encoder": settings.base_encoder,
                "finetuned_encoder": str(settings.finetuned_encoder_dir),
            },
        },
        **log,
    }
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "encoder_training.json").write_text(json.dumps(report, indent=2))
    return settings.finetuned_encoder_dir


def _parameter_groups(model: torch.nn.Module) -> list[dict]:
    """AdamW groups: weight decay on weight matrices only.

    Decaying biases and LayerNorm gains towards zero does not regularise capacity; it only
    fights the normalisation, so they are exempt (as in BERT's original fine-tuning recipe).
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 or "LayerNorm" in name else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _false_negative_mask(
    query_ids: np.ndarray, candidate_ids: np.ndarray, known_positives: dict[int, set[int]]
) -> torch.Tensor:
    """mask[i, j] = candidate j is a known positive of query i, other than its own (j == i)."""
    mask = torch.zeros(len(query_ids), len(candidate_ids), dtype=torch.bool)
    candidates = candidate_ids.tolist()
    for i, query in enumerate(query_ids.tolist()):
        positives = known_positives[query]
        for j, candidate in enumerate(candidates):
            if j != i and candidate in positives:
                mask[i, j] = True
    return mask


def _mine_hard_negatives(
    queries: list[str],
    passages: list[str],
    query_ids: np.ndarray,
    known_positives: dict[int, set[int]],
) -> dict[int, np.ndarray]:
    """One round of static hard-negative mining with BM25.

    For each training query, the top BM25 passages that are not known positives: passages
    that share the query's vocabulary yet were not written to answer it. Random in-batch
    negatives are mostly off-topic and are separated early in training; these lexical
    near-misses are the ones a dense encoder must learn to tell apart.
    """
    bm25 = BM25(passages)
    mined: dict[int, np.ndarray] = {}
    chunk = 128
    for start in tqdm(range(0, len(query_ids), chunk), desc="mining hard negatives"):
        ids = query_ids[start : start + chunk].tolist()
        for query, scores in zip(ids, bm25.score([queries[q] for q in ids]), strict=True):
            positives = known_positives[query]
            # Fetch enough extra ranks to still have DEPTH left once positives are removed.
            ranked = top_k(scores, HARD_NEGATIVE_DEPTH + len(positives))
            ranked = ranked[scores[ranked] > 0]
            keep = [row for row in ranked.tolist() if row not in positives]
            mined[query] = np.array(keep[:HARD_NEGATIVE_DEPTH], dtype=np.int64)
    return mined


def _draw_negative(
    mined: np.ndarray, positives: set[int], n_passages: int, rng: np.random.Generator
) -> int:
    """A uniformly drawn mined negative, or a random non-positive passage if none was mined.

    Drawing across the top ranks (not always rank 1) varies the negative between epochs and
    dilutes the unlabelled true positives that concentrate at the very top of BM25.
    """
    if mined.size:
        return int(rng.choice(mined))
    while True:
        candidate = int(rng.integers(n_passages))
        if candidate not in positives:
            return candidate


def _dev_retrieval(
    encoder: _BiEncoder,
    queries: list[str],
    passages: list[str],
    dev_queries: np.ndarray,
    known_positives: dict[int, set[int]],
) -> dict[str, float] | None:
    """Rank the pooled positives of all held-out queries for each held-out query.

    A sanity check that training generalises to unseen questions from the training
    distribution, on a small pool. It is not a benchmark: the retrieval experiment (RQ1)
    evaluates on NFCorpus test and PubMedQA-labelled, which training never touches.
    """
    if dev_queries.size == 0:
        return None
    pool = sorted(set().union(*(known_positives[q] for q in dev_queries.tolist())))
    query_vectors = encoder.embed([queries[q] for q in dev_queries], encoder.query_tokens)
    pool_vectors = encoder.embed([passages[p] for p in pool], encoder.passage_tokens)
    scores = query_vectors @ pool_vectors.T
    run = {
        str(q): [str(pool[row]) for row in top_k(row_scores, 10).tolist()]
        for q, row_scores in zip(dev_queries.tolist(), scores, strict=True)
    }
    qrels = {str(q): {str(p): 1 for p in known_positives[q]} for q in dev_queries.tolist()}
    return {
        "n_queries": int(dev_queries.size),
        "pool_size": len(pool),
        "mrr@10": float(mrr_at_k(run, qrels, 10).mean()),
        # Reciprocal rank cut at 1 is the indicator "the top-ranked passage is relevant".
        "top1_accuracy": float(mrr_at_k(run, qrels, 1).mean()),
    }
