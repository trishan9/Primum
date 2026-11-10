"""Confidence models: map the feature vector of an answer to P(answer is correct).

Two models of deliberately different capacity, both written directly in PyTorch:

- `LogisticConfidence`: sigmoid(w . z + b) on standardised features z, L2 penalty on w.
  The objective is convex, so it is minimised to convergence with full-batch L-BFGS; there is
  nothing to early-stop. Its weights are directly readable as "one standard deviation of this
  signal moves the log-odds of being correct by w_j".
- `MLPConfidence`: one hidden ReLU layer, so it can represent interactions the linear model
  cannot (e.g. "high stated confidence only counts when the claims are entailed"). With a few
  hundred training answers it overfits easily, hence dropout, decoupled weight decay (AdamW,
  Loshchilov & Hutter 2019) and early stopping on a held-out validation split.

Leakage rules both models follow:
- Feature means and standard deviations are estimated on the rows used for gradient steps only.
  For the MLP that excludes its validation rows, which would otherwise influence the very
  inputs they are meant to judge.
- Early stopping monitors validation cross-entropy, a proper scoring rule, because the output is
  used as a probability; accuracy would reward ranking near 0.5 only.
- Neither model sees calibration or test rows: the caller splits the data, and the threshold
  certified in `conformal.certify_threshold` is only valid for a model fitted without them.

Randomness (initialisation, validation split, mini-batch order, dropout) is drawn from generators
seeded by the `seed` argument, never from PyTorch's global state, so fitting is reproducible and
does not perturb other code.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self

import numpy as np
import torch
from torch import nn

_DTYPE = torch.float64  # tiny tabular problems: precision is free and makes JSON round-trips exact


class _ConfidenceModel(ABC):
    """Shared plumbing: standardisation, prediction and JSON (de)serialisation."""

    kind: ClassVar[str]

    def __init__(self) -> None:
        self._net: nn.Module | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit on features X [n, d] and labels y [n] (1 = the answer was correct)."""

    @abstractmethod
    def _build(self, n_features: int) -> nn.Module:
        """The untrained network mapping standardised features to one logit per row."""

    @abstractmethod
    def _config(self) -> dict[str, Any]:
        """Constructor arguments, so `from_json` can rebuild the same architecture."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(correct) for each row of X, shape [n] (a single feature vector gives shape [1])."""
        if self._net is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        with torch.no_grad():
            logits = self._net(self._standardise(X))
        return torch.sigmoid(logits).numpy()

    def to_json(self) -> dict[str, Any]:
        """JSON-serialisable snapshot: hyperparameters, standardiser and network weights."""
        if self._net is None or self._mean is None or self._std is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return {
            "kind": self.kind,
            "config": self._config(),
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "state": {name: value.tolist() for name, value in self._net.state_dict().items()},
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any] | str) -> Self:
        """Inverse of `to_json`; also accepts the JSON text."""
        data = json.loads(payload) if isinstance(payload, str) else payload
        if data.get("kind") != cls.kind:
            raise ValueError(f"payload is for {data.get('kind')!r}, not {cls.kind!r}")
        model = cls(**data["config"])
        model._mean = np.asarray(data["mean"], dtype=float)
        model._std = np.asarray(data["std"], dtype=float)
        model._net = model._build(model._mean.size)
        model._net.load_state_dict(
            {name: torch.tensor(value, dtype=_DTYPE) for name, value in data["state"].items()}
        )
        return model

    def _fit_standardiser(self, X: np.ndarray) -> None:
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        # A feature that is constant in training carries no information; dividing by 1 keeps it
        # at zero instead of amplifying whatever tiny deviation a later row has.
        self._std = np.where(std > 1e-12, std, 1.0)

    def _standardise(self, X: np.ndarray) -> torch.Tensor:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._mean is None or self._std is None or X.shape[1] != self._mean.size:
            expected = None if self._mean is None else self._mean.size
            raise ValueError(f"expected {expected} features per row, got {X.shape[1]}")
        return torch.as_tensor((X - self._mean) / self._std, dtype=_DTYPE)


class LogisticConfidence(_ConfidenceModel):
    """L2-regularised logistic regression: minimise mean BCE + (l2 / 2) * ||w||^2."""

    kind: ClassVar[str] = "logistic"

    def __init__(self, l2: float = 1e-2, max_iter: int = 500) -> None:
        super().__init__()
        self.l2 = l2
        self.max_iter = max_iter

    def _config(self) -> dict[str, Any]:
        return {"l2": self.l2, "max_iter": self.max_iter}

    def _build(self, n_features: int) -> nn.Module:
        net = _Linear(n_features)
        # Zero is the natural start for a convex problem and makes the fit seed-free.
        nn.init.zeros_(net.linear.weight)
        nn.init.zeros_(net.linear.bias)
        return net

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = _check_training_data(X, y)
        self._fit_standardiser(X)
        net = self._build(X.shape[1])
        inputs, targets = self._standardise(X), torch.as_tensor(y, dtype=_DTYPE)
        optimiser = torch.optim.LBFGS(
            net.parameters(),
            max_iter=self.max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimiser.zero_grad()
            # The bias is not penalised: shrinking it would pull predictions towards 0.5 instead
            # of towards the base rate.
            penalty = 0.5 * self.l2 * net.linear.weight.square().sum()
            loss = nn.functional.binary_cross_entropy_with_logits(net(inputs), targets) + penalty
            loss.backward()
            return loss

        optimiser.step(closure)
        self._net = net.eval()
        return self


class MLPConfidence(_ConfidenceModel):
    """One-hidden-layer network trained with AdamW, dropout and early stopping."""

    kind: ClassVar[str] = "mlp"

    def __init__(
        self,
        hidden: int = 16,
        dropout: float = 0.2,
        weight_decay: float = 1e-2,
        lr: float = 5e-3,
        batch_size: int = 64,
        max_epochs: int = 300,
        patience: int = 30,
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0 or not 0.0 < val_fraction < 1.0:
            raise ValueError("dropout must be in [0, 1) and val_fraction in (0, 1)")
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed

    def _config(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "val_fraction": self.val_fraction,
            "seed": self.seed,
        }

    def _build(self, n_features: int) -> nn.Module:
        return _MLP(n_features, self.hidden, self.dropout)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X, y = _check_training_data(X, y)
        train_idx, val_idx = _stratified_split(
            y, self.val_fraction, np.random.default_rng(self.seed)
        )
        self._fit_standardiser(X[train_idx])  # validation rows excluded: see module docstring
        generator = torch.Generator().manual_seed(self.seed)
        net = self._build(X.shape[1])
        net.reset_parameters(generator)

        x_train, x_val = self._standardise(X[train_idx]), self._standardise(X[val_idx])
        y_train = torch.as_tensor(y[train_idx], dtype=_DTYPE)
        y_val = torch.as_tensor(y[val_idx], dtype=_DTYPE)
        bce = nn.functional.binary_cross_entropy_with_logits
        optimiser = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss, best_state, stale = float("inf"), copy.deepcopy(net.state_dict()), 0
        for _ in range(self.max_epochs):
            permutation = torch.randperm(len(train_idx), generator=generator)
            for batch in permutation.split(self.batch_size):
                optimiser.zero_grad()
                bce(net(x_train[batch], dropout_generator=generator), y_train[batch]).backward()
                optimiser.step()
            with torch.no_grad():
                val_loss = float(bce(net(x_val), y_val))
            if val_loss < best_loss - 1e-6:
                best_loss, best_state, stale = val_loss, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        # Keep the weights from the best validation epoch, not the last one.
        net.load_state_dict(best_state)
        self._net = net.eval()
        return self


ConfidenceModel = LogisticConfidence | MLPConfidence


def model_from_json(payload: dict[str, Any] | str) -> ConfidenceModel:
    """Rebuild either confidence model from its `to_json()` payload."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    for cls in (LogisticConfidence, MLPConfidence):
        if data.get("kind") == cls.kind:
            return cls.from_json(data)
    raise ValueError(f"not a confidence-model payload (kind={data.get('kind')!r})")


# --------------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------------


def _uninitialised_linear(n_in: int, n_out: int) -> nn.Linear:
    """A Linear layer whose weights are left unset.

    `nn.Linear(...)` initialises itself from PyTorch's global generator. Every layer here is
    either set explicitly (zeros, or `reset_parameters` with a seeded generator) or loaded from
    JSON, so the default initialisation would only consume global randomness as a side effect.
    """
    return nn.utils.skip_init(nn.Linear, n_in, n_out, dtype=_DTYPE)


class _Linear(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.linear = _uninitialised_linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class _MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.hidden = _uninitialised_linear(n_features, hidden)
        self.out = _uninitialised_linear(hidden, 1)
        self.dropout = dropout

    def reset_parameters(self, generator: torch.Generator) -> None:
        """PyTorch's default Linear initialisation, U(-1/sqrt(fan_in), 1/sqrt(fan_in)), drawn from
        an explicit generator so that it does not depend on the global seed."""
        for layer in (self.hidden, self.out):
            bound = layer.in_features**-0.5
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.uniform_(-bound, bound, generator=generator)

    def forward(
        self, x: torch.Tensor, dropout_generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Logits. Passing a generator switches dropout on (training); omitting it is inference."""
        h = torch.relu(self.hidden(x))
        if dropout_generator is not None and self.dropout > 0.0:
            # Inverted dropout (Srivastava et al. 2014): zero each unit with probability p and
            # rescale the rest by 1/(1-p), so the expected activation matches inference.
            keep = 1.0 - self.dropout
            mask = torch.bernoulli(torch.full_like(h, keep), generator=dropout_generator)
            h = h * mask / keep
        return self.out(h).squeeze(-1)


# --------------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------------


def _check_training_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(
            f"X must be [n, d] with one label per row, got {X.shape} and {y.size} labels"
        )
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        # With a single class the unpenalised bias diverges; a confidence model cannot be learnt.
        raise ValueError("y must be 0/1 and contain both classes")
    return X, y


def _stratified_split(
    y: np.ndarray, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/validation indices with the class balance preserved in both parts."""
    train: list[np.ndarray] = []
    val: list[np.ndarray] = []
    for label in (0.0, 1.0):
        members = rng.permutation(np.flatnonzero(y == label))
        # At least one of each class on each side whenever the class has two members.
        n_val = min(max(1, round(val_fraction * members.size)), members.size - 1)
        val.append(members[:n_val])
        train.append(members[n_val:])
    if min(part.size for part in val) == 0:
        raise ValueError("need at least two examples of each class to hold out a validation split")
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))
