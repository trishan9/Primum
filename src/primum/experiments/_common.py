"""Plumbing shared by the experiment scripts: provenance, seeded randomness, interval summaries,
results I/O and the repeated stratified split protocol of RQ2 and RQ3.

Nothing here computes a result. It exists so that every experiment writes `meta` the same way,
draws its random numbers the same way, and splits the PubMedQA records the same way, which is
what makes RQ2 and RQ3 comparable repeat by repeat.
"""

from __future__ import annotations

import json
import math
import os
import zlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from primum.config import Settings, get_settings
from primum.types import DECISIONS
from primum.uncertainty import FEATURES_FULL, bootstrap_ci

PUBMEDQA_RUNS = "pubmedqa_runs"

# Fractions of the repeated split (SPEC §3.9): fit / calibrate / test.
SPLIT_FRACTIONS = (0.4, 0.3, 0.3)
# Every part of a split must hold at least this many correct and incorrect answers. Below it the
# protocol cannot run as specified: a confidence model needs both classes (and the MLP holds out
# a validation subset of its training rows), AUROC needs both classes on the test part, and the
# 5-fold cross-fitting of a calibrator in RQ3 needs both classes in every fold's training rows.
MIN_PER_CLASS_PER_PART = 5


class DegenerateSplit(ValueError):
    """The records have too few correct or incorrect answers for stratified splitting."""


# --------------------------------------------------------------------------------------------
# Provenance and I/O
# --------------------------------------------------------------------------------------------


def resolve_results_dir(override: Path | str | None) -> Path:
    """`override` if given (tests write to a temporary directory), else `settings.results_dir`."""
    directory = Path(override) if override is not None else get_settings().results_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_meta(
    created: str,
    *,
    n: int,
    seeds: dict[str, Any],
    models: dict[str, Any],
    corpora: Sequence[str] = (),
    **extra: Any,
) -> dict[str, Any]:
    """The `meta` block every results file starts with (SPEC §3.9)."""
    settings = get_settings()
    return {
        "created": created,
        "n": n,
        "seeds": seeds,
        "models": models,
        "config": config_snapshot(settings),
        "data_manifests": {corpus: _manifest(settings, corpus) for corpus in corpora},
        **extra,
    }


def config_snapshot(settings: Settings) -> dict[str, Any]:
    """Settings as JSON, without the API key and with paths relative to the project root."""
    snapshot = settings.model_dump(mode="json", exclude={"anthropic_api_key"})
    root = settings.root.resolve()
    for key, value in snapshot.items():
        if isinstance(value, str) and value.startswith(str(root)):
            snapshot[key] = os.path.relpath(value, root)
    snapshot["root"] = "."
    return snapshot


def _manifest(settings: Settings, corpus: str) -> dict[str, Any] | None:
    """The data build's manifest (counts, pinned source revisions), or None if absent."""
    path = settings.processed_dir / corpus / "manifest.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write atomically, so an interrupted run never leaves a truncated results file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def alpha_key(alpha: float) -> str:
    """Risk levels as JSON keys, e.g. 0.1 -> "0.10" (the form the web app parses)."""
    return f"{alpha:.2f}"


# --------------------------------------------------------------------------------------------
# Randomness and summaries
# --------------------------------------------------------------------------------------------


def rng_for(seed: int, *keys: object) -> np.random.Generator:
    """A generator determined by the global seed and a name.

    Each quantity (a bootstrap of one metric for one system, one repeat's split) gets its own
    stream, so adding a system or changing the order of a loop never changes another number.
    crc32 is used because Python's `hash` of a string is salted per process.
    """
    words = [zlib.crc32(str(key).encode("utf-8")) for key in keys]
    return np.random.default_rng([seed, *words])


def bootstrap_estimate(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """`{mean, ci}`: the statistic on the sample and its 95% percentile-bootstrap interval."""
    values = np.asarray(values)
    lo, hi = bootstrap_ci(values, stat=stat, n_boot=n_boot, level=0.95, rng=rng)
    return {"mean": float(stat(values)), "ci": [lo, hi]}


def spread(values: Sequence[float]) -> dict[str, Any] | None:
    """`{mean, ci}` over repeated splits: the mean and the central 95% range of the repeats.

    This range describes how much one split's estimate varies, not the uncertainty of the mean
    over repeats (the repeats share questions, so they are not independent samples).
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    lo, hi = np.quantile(array, [0.025, 0.975])
    return {"mean": float(array.mean()), "ci": [float(lo), float(hi)]}


def clopper_pearson(k: int, n: int, level: float = 0.95) -> list[float]:
    """Exact binomial interval for k successes in n trials (Clopper & Pearson 1934)."""
    if not 0 <= k <= n or n == 0:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    tail = (1.0 - level) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return [lo, hi]


def macro_f1(pred: np.ndarray, gold: np.ndarray, n_classes: int = len(DECISIONS)) -> float:
    """Unweighted mean of per-class F1 over all classes (yes / no / maybe).

    F1_c = 2 TP / (2 TP + FP + FN). A class that is neither predicted nor present has an
    undefined F1 and is scored 0 (scikit-learn's `zero_division=0`), so the average is always
    over the same three classes; with PubMedQA's 110 "maybe" items this case does not arise
    outside tiny subsets.
    """
    pred, gold = np.asarray(pred), np.asarray(gold)
    scores = []
    for c in range(n_classes):
        tp = int(np.sum((pred == c) & (gold == c)))
        denominator = 2 * tp + int(np.sum((pred == c) & (gold != c))) + int(np.sum(gold == c) - tp)
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


# --------------------------------------------------------------------------------------------
# PubMedQA records -> arrays (RQ2, RQ3)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RagData:
    """The `rag` condition of `pubmedqa_runs.json`, as aligned arrays."""

    qids: list[str]
    X: np.ndarray  # [n, len(FEATURES_FULL)], columns in FEATURES_FULL order
    correct: np.ndarray  # [n] bool
    probs: np.ndarray  # [n, 3] stated probabilities, columns in DECISIONS order
    gold: np.ndarray  # [n] int, index into DECISIONS
    n_failed: int  # rag items whose generation failed; excluded from the arrays
    runs_meta: dict[str, Any]

    def columns(self, names: Sequence[str]) -> np.ndarray:
        return self.X[:, [FEATURES_FULL.index(name) for name in names]]


def load_rag_data(directory: Path) -> RagData:
    """Read the `rag` records of a finished PubMedQA run.

    A run that stopped early (budget, missing key) is refused: the items it completed are the
    ones that happened to finish first, not a sample chosen in advance.
    """
    path = directory / f"{PUBMEDQA_RUNS}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `primum exp pubmedqa` first")
    runs = read_json(path)
    meta = runs["meta"]
    if not meta.get("complete", False):
        raise ValueError(
            f"{path} is from a run that stopped early ({meta.get('stopped')}); "
            "re-run `primum exp pubmedqa` to completion (cached calls are free)"
        )
    rag = [r for r in runs["records"] if r["condition"] == "rag"]
    ok = [r for r in rag if "error" not in r]
    if not ok:
        raise ValueError(f"{path} contains no successful `rag` records")
    return RagData(
        qids=[r["qid"] for r in ok],
        X=np.array([[r["features"][name] for name in FEATURES_FULL] for r in ok], dtype=float),
        correct=np.array([r["correct"] for r in ok], dtype=bool),
        probs=np.array([[r["probabilities"][d] for d in DECISIONS] for r in ok], dtype=float),
        gold=np.array([DECISIONS.index(r["gold"]) for r in ok], dtype=int),
        n_failed=len(rag) - len(ok),
        runs_meta=meta,
    )


# --------------------------------------------------------------------------------------------
# Stratified splitting
# --------------------------------------------------------------------------------------------


def stratified_split(
    correct: np.ndarray, fractions: Sequence[float], rng: np.random.Generator
) -> list[np.ndarray]:
    """Random partition into parts of the given fractions, each with the overall error rate.

    Stratifying by correctness makes every part's base rate equal to the data's, so the
    variability across repeats reflects the method rather than lucky splits, and it rules out
    a part with a single class. Part sizes per class are fixed (round, remainder to the last
    part), so they are checked once by `check_stratifiable`.
    """
    parts: list[list[np.ndarray]] = [[] for _ in fractions]
    for label in (False, True):
        members = rng.permutation(np.flatnonzero(correct == label))
        bounds = np.cumsum(_part_sizes(members.size, fractions))[:-1]
        for part, chunk in zip(parts, np.split(members, bounds), strict=True):
            part.append(chunk)
    return [np.sort(np.concatenate(part)) for part in parts]


def check_stratifiable(correct: np.ndarray, fractions: Sequence[float]) -> None:
    """Raise `DegenerateSplit` unless every part gets enough of both classes."""
    for label, name in ((True, "correct"), (False, "incorrect")):
        count = int(np.sum(correct == label))
        smallest = min(_part_sizes(count, fractions))
        if smallest < MIN_PER_CLASS_PER_PART:
            raise DegenerateSplit(
                f"only {count} {name} answers among {correct.size}: a stratified split with "
                f"fractions {tuple(fractions)} needs at least {MIN_PER_CLASS_PER_PART} of each "
                "class in every part"
            )


def repeated_splits(
    correct: np.ndarray, repeats: int, seed: int
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """`(repeat, train, cal, test)` for the RQ2/RQ3 protocol; identical in both experiments."""
    check_stratifiable(correct, SPLIT_FRACTIONS)
    for repeat in range(repeats):
        train, cal, test = stratified_split(
            correct, SPLIT_FRACTIONS, rng_for(seed, "split", repeat)
        )
        yield repeat, train, cal, test


def stratified_folds(correct: np.ndarray, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """k disjoint folds covering all rows, each class spread evenly over the folds."""
    folds: list[list[np.ndarray]] = [[] for _ in range(k)]
    for label in (False, True):
        members = rng.permutation(np.flatnonzero(correct == label))
        for fold, chunk in zip(folds, np.array_split(members, k), strict=True):
            fold.append(chunk)
    return [np.sort(np.concatenate(fold)) for fold in folds]


def _part_sizes(count: int, fractions: Sequence[float]) -> list[int]:
    sizes = [round(f * count) for f in fractions[:-1]]
    return [*sizes, count - sum(sizes)]


def ceil_fraction(fraction: float, n: int) -> int:
    return max(1, math.ceil(fraction * n))
