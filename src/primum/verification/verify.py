"""Claim verification: is each claim in an answer supported by the evidence it cites?

Each claim is treated as an NLI hypothesis and scored against short premises cut from the
retrieved passages. The verdicts are shown to the user and, more importantly, become features
of the confidence model (`primum.uncertainty.features`), so the definitions below are part of
the experimental protocol and are kept deliberately simple.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from primum.sentences import split_sentences
from primum.types import Claim, ClaimVerdict, Hit, VerdictLabel
from primum.verification.nli import get_nli

ENTAIL_THRESHOLD = 0.5
CONTRA_THRESHOLD = 0.5


class EntailmentScorer(Protocol):
    """What `verify_claims` needs from an NLI model; `NLIModel` and test stubs both satisfy it."""

    def predict(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """Return `[n, 3]` probabilities ordered entailment, neutral, contradiction."""
        ...


def verify_claims(
    claims: list[Claim], hits: list[Hit], *, nli: EntailmentScorer | None = None
) -> list[ClaimVerdict]:
    """Label every claim `supported`, `contradicted` or `unsupported` by the retrieved evidence.

    A claim is scored against every sentence and every two-sentence window of the passages it
    cites (all passages if it cites none that exist). Then

    - `p_entail` is the highest entailment probability over those premises;
    - `p_contra` is the highest contradiction probability over those premises;
    - the label is `supported` if `p_entail >= 0.5`, else `contradicted` if `p_contra >= 0.5`,
      else `unsupported`.

    Taking maxima means one sentence is enough to support a claim, however much unrelated text
    surrounds it. Entailment is tested first on purpose: abstracts report mixed findings, and a
    claim that one sentence states outright is supported even if another sentence cuts against
    it. The contradiction is not lost — `p_contra` stays on the verdict and feeds the
    confidence model.

    `best_hit` and `best_sentence` identify the premise that decided the label: the most
    contradicting one for a `contradicted` claim, otherwise the most entailing one (for an
    `unsupported` claim, the closest the evidence came). `p_neutral` is that premise's neutral
    probability, so the three probabilities need not sum to one.

    With no evidence at all (closed-book answers) every claim is `unsupported` with zero
    entailment, and the NLI model is never loaded.
    """
    evidence = [_evidence_for(claim, hits) for claim in claims]
    premises = [premise for claim_evidence in evidence for _, premise in claim_evidence]
    hypotheses = [
        claim.text
        for claim, claim_evidence in zip(claims, evidence, strict=True)
        for _ in claim_evidence
    ]
    if not premises:
        return [_no_evidence(claim) for claim in claims]

    # One call for all claims, so the model batches across them.
    probs = np.asarray((nli or get_nli()).predict(premises, hypotheses), dtype=float)
    if probs.shape != (len(premises), 3):
        raise ValueError(f"NLI scorer returned shape {probs.shape}, expected {(len(premises), 3)}")

    verdicts: list[ClaimVerdict] = []
    offset = 0
    for claim, claim_evidence in zip(claims, evidence, strict=True):
        rows = probs[offset : offset + len(claim_evidence)]
        offset += len(claim_evidence)
        verdicts.append(
            _verdict(claim, claim_evidence, rows) if claim_evidence else _no_evidence(claim)
        )
    return verdicts


def _evidence_for(claim: Claim, hits: list[Hit]) -> list[tuple[int, str]]:
    """The `(1-based hit index, premise)` pairs a claim is checked against.

    Citations come from an LLM, so they can repeat or point past the end of the hit list. Such
    indices are dropped; a claim left with no valid citation is checked against everything, the
    same as a claim that cited nothing — it should not be marked unsupported merely because its
    citation was malformed when the evidence is on the page.
    """
    cited = [i for i in dict.fromkeys(claim.cited) if 1 <= i <= len(hits)]
    indices = cited or range(1, len(hits) + 1)
    return [(i, premise) for i in indices for premise in _premises(hits[i - 1].passage.text)]


def _premises(text: str) -> list[str]:
    """Single sentences, then adjacent pairs.

    A sentence is the smallest unit that can entail a claim, and short premises keep the model
    in the regime it was trained on (sentence-pair NLI) instead of asking it to find one fact
    in a 300-word abstract. The pairs cover claims whose support spans a sentence boundary —
    "Group A received metformin. It lowered HbA1c by 1.1%." — and repair sentences the splitter
    cut wrongly. Singles come first so that on ties the shorter premise is reported.
    """
    sentences = split_sentences(text)
    return sentences + [f"{a} {b}" for a, b in zip(sentences, sentences[1:], strict=False)]


def _verdict(claim: Claim, evidence: list[tuple[int, str]], probs: np.ndarray) -> ClaimVerdict:
    most_entailing = int(np.argmax(probs[:, 0]))
    most_contradicting = int(np.argmax(probs[:, 2]))
    p_entail = float(probs[most_entailing, 0])
    p_contra = float(probs[most_contradicting, 2])

    label: VerdictLabel
    if p_entail >= ENTAIL_THRESHOLD:
        label = "supported"
    elif p_contra >= CONTRA_THRESHOLD:
        label = "contradicted"
    else:
        label = "unsupported"

    deciding = most_contradicting if label == "contradicted" else most_entailing
    best_hit, best_sentence = evidence[deciding]
    return ClaimVerdict(
        claim=claim,
        label=label,
        p_entail=p_entail,
        p_neutral=float(probs[deciding, 1]),
        p_contra=p_contra,
        best_hit=best_hit,
        best_sentence=best_sentence,
    )


def _no_evidence(claim: Claim) -> ClaimVerdict:
    # Nothing to entail or contradict the claim: all probability mass on "neutral".
    return ClaimVerdict(claim=claim, label="unsupported", p_entail=0.0, p_neutral=1.0, p_contra=0.0)
