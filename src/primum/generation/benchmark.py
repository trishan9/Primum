"""Benchmark mode: one structured call that answers a yes/no/maybe research question."""

from __future__ import annotations

import math

from pydantic import BaseModel

from primum.generation.evidence import WireClaim, format_evidence, to_claims
from primum.llm import LLMClient, LLMInvalidOutput, get_llm, prompts
from primum.types import DECISIONS, BenchmarkAnswer, Decision, Hit, Usage

_MAX_CLAIMS = 4  # BenchmarkAnswer carries one to four claims


class _BenchmarkWire(BaseModel):
    """The JSON the model is constrained to produce.

    Field order is part of the method and matches the order the prompt asks for: claims,
    then probabilities, then the decision, so that the model states the findings it relies
    on before it commits to an answer instead of justifying an answer after the fact. The
    probabilities are three flat fields because the structured-output schema subset has no
    map type with a fixed key set (`BenchmarkAnswer.probabilities` is built from them).
    """

    claims: list[WireClaim]
    p_yes: float
    p_no: float
    p_maybe: float
    decision: Decision


def answer_benchmark(
    question: str,
    hits: list[Hit],
    *,
    sample_id: int = 0,
    llm: LLMClient | None = None,
) -> tuple[BenchmarkAnswer, Usage]:
    """Answer `question` from `hits`; with `hits=[]` this is the closed-book condition.

    `sample_id` selects an independent, reproducible sample of the same prompt (used for
    self-consistency). Claims cite passages by 1-based position in `hits`.
    """
    llm = llm or get_llm()
    question = question.strip()
    if hits:
        system = prompts.BENCHMARK_SYSTEM_EVIDENCE
        user = prompts.BENCHMARK_USER_EVIDENCE.format(
            evidence=format_evidence(hits), question=question
        )
        tag = "benchmark.evidence"
    else:
        system = prompts.BENCHMARK_SYSTEM_CLOSED_BOOK
        user = prompts.BENCHMARK_USER_CLOSED_BOOK.format(question=question)
        tag = "benchmark.closed_book"

    result = llm.complete(
        system=system, user=user, schema=_BenchmarkWire, sample_id=sample_id, tag=tag
    )
    wire: _BenchmarkWire = result.parsed
    answer = BenchmarkAnswer(
        # Kept as stated even in the rare case it is not the arg-max of the probabilities:
        # it is the answer the model gave (and breaks ties), and overriding it would hide
        # an inconsistency that is itself a sign of an unreliable answer.
        decision=wire.decision,
        probabilities=_normalise(wire.p_yes, wire.p_no, wire.p_maybe),
        claims=to_claims(wire.claims, len(hits))[:_MAX_CLAIMS],
    )
    return answer, result.usage


def _normalise(p_yes: float, p_no: float, p_maybe: float) -> dict[Decision, float]:
    """Rescale the stated probabilities to sum to exactly 1.

    The schema cannot bound numbers, so the model may return values with rounding error or
    in percent. Dividing by the sum fixes both without changing the ratios it stated.
    Anything that is not a distribution up to scale (a negative or non-finite value, or all
    zeros) is rejected: inventing a distribution for it would fabricate a confidence signal.
    """
    raw = (p_yes, p_no, p_maybe)
    total = sum(raw)
    if not all(math.isfinite(p) and p >= 0 for p in raw) or total <= 0:
        raise LLMInvalidOutput(f"stated probabilities are not a distribution: {raw}")
    return {decision: p / total for decision, p in zip(DECISIONS, raw, strict=True)}
