"""Natural-language-inference scorer: P(evidence sentence entails / is neutral to / contradicts a claim).

A thin wrapper around a pretrained cross-encoder (`settings.nli_model`). The model reads the
premise and the hypothesis jointly, which is what makes it sensitive to negation and direction
of effect ("reduced" vs "did not reduce") where embedding similarity is not.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from primum.config import get_settings

# Column order of every array this module returns.
NLI_LABELS = ("entailment", "neutral", "contradiction")


class NLIModel:
    """Batched NLI inference on MPS/CUDA/CPU.

    Loading happens in the constructor (a few seconds, plus a one-off download); use
    `get_nli()` to share one lazily created instance across the process.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
    ) -> None:
        # Imported here so that importing this package (e.g. for `split_sentences`, or in tests
        # that stub the model) does not pay for importing transformers.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_name or get_settings().nli_model
        self.device = torch.device(device or _default_device())
        self.batch_size = batch_size
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(self.name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.name)
        self._model.to(self.device).eval()
        # NLI checkpoints disagree on label order (MNLI's original order is contradiction,
        # neutral, entailment; many fine-tunes use the reverse). Reading it from the checkpoint
        # is the only way not to silently swap "supported" and "contradicted".
        self._columns = _canonical_columns(self._model.config.id2label)
        # The experiment runner verifies answers from a thread pool. Concurrent forward passes
        # through one model buy nothing on a single accelerator and are not safe on MPS.
        self._lock = threading.Lock()

    def predict(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """Return an `[n, 3]` float array of probabilities, columns ordered as `NLI_LABELS`.

        Row `i` scores the pair `(premises[i], hypotheses[i])`. Pairs longer than `max_length`
        tokens are truncated from the longer side, which is the premise in practice, so a claim
        is never cut.
        """
        if len(premises) != len(hypotheses):
            raise ValueError(
                f"premises and hypotheses differ in length: {len(premises)} vs {len(hypotheses)}"
            )
        probs = np.zeros((len(premises), 3), dtype=np.float32)
        # Batches of similar length waste little computation on padding; results are written
        # back by original index, so the caller's order is preserved.
        by_length = np.argsort([len(p) + len(h) for p, h in zip(premises, hypotheses, strict=True)])
        with self._lock, torch.inference_mode():
            for i in range(0, len(by_length), self.batch_size):
                rows = by_length[i : i + self.batch_size]
                encoded = self._tokenizer(
                    [premises[j] for j in rows],
                    [hypotheses[j] for j in rows],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**encoded).logits.float()
                probs[rows] = torch.softmax(logits, dim=-1)[:, self._columns].cpu().numpy()
        return probs


_default: NLIModel | None = None
_default_lock = threading.Lock()


def get_nli() -> NLIModel:
    """The process-wide NLI model for `settings.nli_model`, loaded on first use."""
    global _default
    # The lock keeps two threads that arrive before the first load from both loading the model.
    with _default_lock:
        if _default is None:
            _default = NLIModel()
        return _default


def _canonical_columns(id2label: Mapping[int, str]) -> list[int]:
    """Indices that reorder a checkpoint's logits into `NLI_LABELS` order.

    Raises instead of guessing when the checkpoint is not a three-way NLI model with named
    labels (e.g. "LABEL_0", or a binary entailment head): a wrong guess would not crash, it
    would quietly corrupt every verdict.
    """
    index_of = {str(label).strip().lower(): int(index) for index, label in id2label.items()}
    if set(index_of) != set(NLI_LABELS):
        raise ValueError(
            f"NLI model labels must be exactly {NLI_LABELS}; the checkpoint has "
            f"{sorted(id2label.values())}"
        )
    return [index_of[label] for label in NLI_LABELS]


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"
