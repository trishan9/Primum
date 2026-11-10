"""Dense retrieval: exact maximum-inner-product search over L2-normalised embeddings.

Passages and queries are embedded by the same bi-encoder and normalised to unit length, so
the inner product is the cosine similarity. Search is a single matrix product against the
whole embedding matrix. The corpora here are at most ~100k passages, where an exact scan
takes milliseconds; an approximate index (HNSW, IVF) would add build parameters and a recall
loss that would then be confounded with encoder quality in the retrieval experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from primum.retrieval._common import (
    corpus_fingerprint,
    passage_text,
    read_json,
    resolve_corpus,
    select_device,
    top_k,
)
from primum.types import Hit, Passage

# Maps texts to a float array [n, dim]. Lets tests (and callers with their own model) supply
# an encoder without sentence-transformers.
EmbedFn = Callable[[list[str]], np.ndarray]


class EncoderNotFound(FileNotFoundError):
    """A local encoder directory does not exist (e.g. the fine-tuned encoder is untrained)."""


class DenseIndex:
    """Bi-encoder retriever over a passage list; embeddings are cached for a named corpus.

    `encoder` is a sentence-transformers model id, a local model directory, or an `EmbedFn`.
    `corpus` is a corpus id (passages come from `primum.data`, embeddings are cached under
    `data/indexes/<corpus>/<name>/embeddings.npy`) or an explicit passage list. `name`
    defaults to `dense-<slug of the encoder id>`.
    """

    _CHUNK = 256  # queries scored per matrix product

    def __init__(
        self,
        encoder: str | Path | EmbedFn,
        corpus: str | Sequence[Passage],
        *,
        name: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.passages, cache_root = resolve_corpus(corpus)
        self._batch_size = batch_size
        self._lock = threading.Lock()  # one encode at a time: the model is not re-entrant
        if callable(encoder):
            self._embed_fn: EmbedFn | None = encoder
            self._encoder_id = getattr(encoder, "__name__", "custom")
            cache_root = None  # an arbitrary callable has no identity to key a cache on
        else:
            self._embed_fn = None  # the model is loaded on first use
            self._encoder_id = _encoder_identity(encoder)
            self._encoder_path = str(encoder)
        self.name = name or f"dense-{_slug(self._encoder_id.split('@')[0])}"

        # The directory is named after the retriever ("dense-ft"); what makes its contents
        # valid is the key: same passages, same encoder weights.
        key = {"fingerprint": corpus_fingerprint(self.passages), "encoder": self._encoder_id}
        directory = cache_root / self.name if cache_root else None
        if directory is not None and read_json(directory / "meta.json") == key:
            self.embeddings = np.load(directory / "embeddings.npy")
            return
        self.embeddings = self._embed([passage_text(p) for p in self.passages])
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / "embeddings.npy", self.embeddings)
            # Written last: an interrupted build leaves no valid meta and is redone.
            (directory / "meta.json").write_text(json.dumps(key))

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Unit-norm float32 embeddings [n, dim]."""
        with self._lock:
            if self._embed_fn is None:
                self._embed_fn = _sentence_transformer_embedder(
                    self._encoder_path, self._batch_size
                )
            vectors = np.asarray(self._embed_fn(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(f"encoder returned shape {vectors.shape} for {len(texts)} texts")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def score_all(self, queries: Sequence[str]) -> np.ndarray:
        """Cosine similarity of every passage to every query: [n_queries, n_passages]."""
        return self._embed(list(queries)) @ self.embeddings.T

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Hit]]:
        results: list[list[Hit]] = []
        for start in range(0, len(queries), self._CHUNK):
            for scores in self.score_all(queries[start : start + self._CHUNK]):
                results.append(
                    [
                        Hit(
                            passage=self.passages[row],
                            score=float(scores[row]),
                            rank=rank,
                            retriever=self.name,
                            components={"dense": float(scores[row]), "dense_rank": float(rank)},
                        )
                        for rank, row in enumerate(top_k(scores, k).tolist(), start=1)
                    ]
                )
        return results


def _sentence_transformer_embedder(model_name_or_path: str, batch_size: int) -> EmbedFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name_or_path, device=select_device())

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) >= 2000,
        )

    return embed


def _encoder_identity(encoder: str | Path) -> str:
    """A string that changes whenever the encoder's weights do.

    A hub model id names published weights. A local directory does not name anything stable:
    the fine-tuned encoder is overwritten by every training run, so its identity includes a
    hash of the weight files. Embeddings cached from an earlier run are then rebuilt, never
    silently reused with a newer model's queries.
    """
    path = Path(encoder)
    if isinstance(encoder, str) and not path.is_dir():
        return encoder  # a hub id such as "sentence-transformers/all-MiniLM-L6-v2"
    if not path.is_dir():
        raise EncoderNotFound(f"no encoder at {path}; train it with `primum train encoder`")
    digest = hashlib.sha256()
    for weights in sorted(path.rglob("*.safetensors")):
        with weights.open("rb") as handle:
            digest.update(hashlib.file_digest(handle, "sha256").digest())
    return f"{path.name}@{digest.hexdigest()[:16]}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
