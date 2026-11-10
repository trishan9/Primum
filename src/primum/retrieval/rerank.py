"""Cross-encoder reranking.

A bi-encoder embeds the query and the passage separately, so the only interaction between
them is one inner product. A cross-encoder reads "[CLS] query [SEP] passage [SEP]" as a
single sequence and lets every query token attend to every passage token, which is more
accurate but costs one transformer forward pass per (query, passage) pair. It is therefore
applied only to the short candidate list produced by the first-stage retrievers.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from primum.retrieval._common import select_device


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a pretrained sequence-classification model.

    The score is the model's raw relevance logit. It is not squashed through a sigmoid:
    the logit orders passages identically and keeps resolution at the confident end, where
    a probability saturates at 1.0.
    """

    def __init__(self, model: str | Path, *, batch_size: int = 64, max_length: int = 512) -> None:
        self._model_name = str(model)
        self._batch_size = batch_size
        self._max_length = max_length
        self._lock = threading.Lock()
        self._loaded = None  # (tokenizer, model, device), loaded on first use

    def _load(self):
        if self._loaded is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            if model.config.num_labels != 1:
                raise ValueError(
                    f"{self._model_name} has {model.config.num_labels} output labels; "
                    "a reranker must output a single relevance logit"
                )
            device = select_device()
            self._loaded = (tokenizer, model.to(device).eval(), device)
        return self._loaded

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Relevance logit for each (query, passage) pair: float32 [n_pairs]."""
        import torch

        scores = np.empty(len(pairs), dtype=np.float32)
        with self._lock, torch.inference_mode():
            tokenizer, model, device = self._load()
            for start in range(0, len(pairs), self._batch_size):
                batch = pairs[start : start + self._batch_size]
                encoded = tokenizer(
                    [query for query, _ in batch],
                    [passage for _, passage in batch],
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(device)
                logits = model(**encoded).logits.squeeze(-1)
                scores[start : start + len(batch)] = logits.float().cpu().numpy()
        return scores
