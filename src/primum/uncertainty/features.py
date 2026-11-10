"""Turns one answered question into the fixed-length signal vector the confidence model reads.

Three families of evidence about whether an answer is right, each cheap to compute and each
failing differently:

- retrieval (`ret_*`): was relevant evidence found at all? Uses a score that means the same thing
  across questions (cross-encoder score, else dense cosine). Raw BM25 is excluded on purpose:
  it grows with query length and term rarity, so "12.3" is strong for one question and weak
  for another.
- verification (`nli_*`, `frac_supported`, `n_claims`): do the retrieved passages entail the
  claims the answer makes?
- the generator's own uncertainty: its stated confidence (`verbal_conf`, `verbal_margin`) and,
  when several samples were drawn, how much they agree (`sc_*`; Wang et al. 2023 self-consistency;
  the entropy is the discrete analogue of semantic entropy for a three-way decision).

`FEATURES_DEPLOY` needs a single generation and is what the live system uses; `FEATURES_FULL`
adds the signals that cost extra samples or need a stated distribution over decisions.

Nothing here may look at the gold label, and nothing is fitted: every feature is a deterministic
function of one question's pipeline outputs, so there is no train/test leakage through features.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from primum.types import DECISIONS, ClaimVerdict, Decision, Hit

FEATURES_DEPLOY = [
    "ret_top",
    "ret_mean",
    "ret_margin",
    "ret_agree",
    "nli_mean_entail",
    "nli_min_entail",
    "nli_max_contra",
    "frac_supported",
    "n_claims",
    "verbal_conf",
]
FEATURES_FULL = FEATURES_DEPLOY + ["sc_agreement", "sc_entropy", "verbal_margin"]


def extract_features(
    hits: Sequence[Hit],
    verdicts: Sequence[ClaimVerdict],
    *,
    verbal_conf: float | None,
    sample_decisions: Sequence[Decision] | None = None,
    probabilities: Mapping[Decision, float] | None = None,
) -> dict[str, float]:
    """All `FEATURES_FULL` values for one answer; any signal that is unavailable is 0.0.

    The caller picks the feature list (`FEATURES_DEPLOY` or `FEATURES_FULL`) that matches the
    inputs it supplied; a model must be trained and served with the same list, otherwise a 0.0
    meaning "not measured" would be read as a measured zero. For the same reason, ranked hits
    without a scale-free score raise `UnscoredRetrieval` (see `scale_free_stage`).
    """
    features = dict.fromkeys(FEATURES_FULL, 0.0)
    features["n_claims"] = float(len(verdicts))
    if verbal_conf is not None:
        features["verbal_conf"] = float(verbal_conf)

    # Closed-book answers have no evidence: retrieval and entailment signals stay at zero even if
    # verdicts were produced, because there was nothing to verify the claims against.
    if hits:
        features.update(_retrieval_features(hits))
        if verdicts:
            entail = [v.p_entail for v in verdicts]
            features["nli_mean_entail"] = sum(entail) / len(entail)
            features["nli_min_entail"] = min(entail)
            features["nli_max_contra"] = max(v.p_contra for v in verdicts)
            features["frac_supported"] = sum(v.label == "supported" for v in verdicts) / len(
                verdicts
            )

    if sample_decisions:
        counts = Counter(sample_decisions)
        n = len(sample_decisions)
        features["sc_agreement"] = max(counts.values()) / n
        # Shannon entropy of the empirical decision distribution, divided by its maximum
        # log|DECISIONS| so that 0 = unanimous and 1 = uniform over yes / no / maybe.
        entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
        features["sc_entropy"] = entropy / math.log(len(DECISIONS))

    if probabilities:
        ranked = sorted((float(p) for p in probabilities.values()), reverse=True)
        features["verbal_margin"] = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)

    return features


class UnscoredRetrieval(ValueError):
    """Ranked hits carry no scale-free score, so the `ret_*` signals cannot be measured."""


def scale_free_stage(hits: Sequence[Hit]) -> Literal["rerank", "dense"] | None:
    """The stage score the `ret_*` features read for this hit list, or None if it has none.

    One score type for the whole list (never a mix), preferring the later, stronger stage.
    None means the hits were not ranked by any retriever (no stage scores at all, e.g. the
    oracle condition's gold passage); like closed-book, they have no retrieval signal.

    Hits that *were* ranked but lack a reranker or dense score on every hit (BM25 alone, whose
    raw score is not comparable across questions) raise `UnscoredRetrieval`. Filling 0.0 there
    would hand the confidence model a constant it reads as a measured "no evidence found",
    and an experiment run that way would silently report the retrieval signals as useless.
    """
    if not any(h.components for h in hits):
        return None
    for name in ("rerank", "dense"):
        if all(name in h.components for h in hits):
            return name
    retrievers = sorted({h.retriever for h in hits})
    raise UnscoredRetrieval(
        f"hits from {retrievers} carry no rerank or dense score on every hit, so the retrieval "
        "confidence features cannot be measured; use a dense, hybrid or reranked retriever"
    )


def _retrieval_features(hits: Sequence[Hit]) -> dict[str, float]:
    k = len(hits)
    stage = scale_free_stage(hits)
    scores = sorted((h.components[stage] for h in hits), reverse=True) if stage else [0.0] * k
    # Agreement of two retrievers that fail differently (lexical vs semantic): the share of the
    # final top-k that each of them, on its own, also ranked within its top k.
    agree = sum(
        h.components.get("bm25_rank", math.inf) <= k
        and h.components.get("dense_rank", math.inf) <= k
        for h in hits
    )
    return {
        "ret_top": scores[0],
        "ret_mean": sum(scores) / k,
        "ret_margin": scores[0] - scores[1] if k > 1 else 0.0,
        "ret_agree": agree / k,
    }
