"""The question-answering pipeline as a stream of events (SPEC §3.7).

    triage -> concepts -> retrieve -> generate -> verify -> confidence -> done

Each stage is bracketed by `stage` events carrying its latency, and emits one result event the
interface can render as soon as it arrives. A failure anywhere ends the stream with one `error`
event naming the stage and a kind the user can act on (see `pipeline.errors`); nothing after it
is emitted and no trace is stored.

Two modes share the stages:

- chat: the answer is streamed token by token, then decomposed into claims (the first step of
  verification), and scored with the deployment features (`FEATURES_DEPLOY`).
- benchmark: one structured yes/no/maybe answer over the `pubmedqa` corpus; this is the
  pipeline the deployment bundle was calibrated on. Self-consistency samples are drawn only
  when the bundle's model reads them, because nothing else uses them and each costs a call.

Event payloads are JSON-native dicts (produced through pydantic's JSON encoder), so an event can
be written to the wire or the trace store without further conversion. In particular a float that
was not measured (the uncalibrated `p_correct`, NaN in Python) becomes `null`, never a number.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from typing import Any, Literal

from pydantic import BaseModel

from primum.config import get_settings
from primum.data import CorpusNotBuilt
from primum.generation import answer_benchmark, extract_claims, stream_chat_answer
from primum.pipeline.engines import Engines, default_engines
from primum.pipeline.errors import UnmeasurableFeatures, error_kind, error_message
from primum.safety import triage
from primum.types import (
    BenchmarkAnswer,
    Claim,
    Concept,
    ConfidenceReport,
    Decision,
    Hit,
    PipelineResult,
    Usage,
)
from primum.uncertainty import (
    FEATURES_DEPLOY,
    FEATURES_FULL,
    DeploymentBundle,
    extract_features,
    scale_free_stage,
)

Mode = Literal["chat", "benchmark"]

# Benchmark questions are PubMedQA items; they are answered from the PubMedQA corpus because
# that is the condition the confidence model and its certified thresholds were fitted on.
BENCHMARK_CORPUS = "pubmedqa"

# Signals each mode can measure. Chat has no stated distribution over decisions and draws one
# answer, so `verbal_margin` and the self-consistency features do not exist for it; a model
# that reads them must not be fed the 0.0 that `extract_features` uses for "not measured".
_SELF_CONSISTENCY = frozenset({"sc_agreement", "sc_entropy"})
_MEASURABLE: dict[Mode, frozenset[str]] = {
    "chat": frozenset(FEATURES_DEPLOY),
    "benchmark": frozenset(FEATURES_FULL),
}


class Event(BaseModel):
    type: str
    data: dict


async def run_pipeline(
    question: str,
    *,
    mode: Mode = "chat",
    retriever: str | None = None,
    alpha: float | None = None,
    gold_label: Decision | None = None,
    benchmark_qid: str | None = None,
    engines: Engines | None = None,
) -> AsyncIterator[Event]:
    """Answer `question`, yielding the events of SPEC §3.7 in order.

    `retriever` and `alpha` default to the configured ones. `gold_label` is recorded on the
    trace for benchmark items and is never shown to any stage. `benchmark_qid` names the
    PubMedQA item being asked; it decides only whether the guarantee covers the answer (see
    `_confidence_flags`). `engines` substitutes the components (tests pass stubs); by default
    the real ones are used.
    """
    settings = get_settings()
    engines = engines or default_engines()
    question = question.strip()
    retriever = retriever or settings.default_retriever
    alpha = settings.alpha if alpha is None else alpha
    corpus = BENCHMARK_CORPUS if mode == "benchmark" else settings.chat_corpus
    clock = _StageClock()
    usage = _UsageTotal()

    try:
        # -- triage: rules only, microseconds, so it runs inline.
        yield clock.start("triage")
        trace = PipelineResult(
            trace_id=uuid.uuid4().hex,
            question=question,
            mode=mode,
            triage=triage(question),
            hits=[],
            answer_text="",
            retriever=retriever,
            model=settings.llm_model,
            gold_label=gold_label,
        )
        yield _event("triage", trace.triage)
        yield clock.done()
        # An emergency in chat ends the run: the user is told to get help, and nothing clinical
        # is generated. Benchmark items are never dropped this way; the level is recorded on
        # the trace instead, so the evaluation set stays the evaluation set.
        if trace.triage.level == "emergency" and mode == "chat":
            yield await _finish(trace, clock, usage, engines, confidence_flags=None)
            return

        # -- concepts
        yield clock.start("concepts")
        trace.concepts, concepts_available = await asyncio.to_thread(
            _link_concepts, engines, question
        )
        yield Event(
            type="concepts",
            data={"concepts": _dump_all(trace.concepts), "available": concepts_available},
        )
        yield clock.done()

        # -- retrieve (the first call may build or load an index: seconds to minutes)
        yield clock.start("retrieve")
        trace.hits = await asyncio.to_thread(
            _search, engines, retriever, corpus, question, settings.top_k
        )
        yield Event(type="retrieval", data={"retriever": retriever, "hits": _dump_all(trace.hits)})
        # The confidence stage reads a scale-free retrieval score; a retriever without one
        # (BM25 alone) fails here, after showing its evidence but before any model call.
        scale_free_stage(trace.hits)
        yield clock.done()

        # -- generate
        yield clock.start("generate")
        bundle = await asyncio.to_thread(engines.bundle)
        # Checked before any model call, so a mismatched bundle costs nothing.
        _check_measurable(bundle, mode)
        llm = engines.llm()
        benchmark: BenchmarkAnswer | None = None
        sample_decisions: list[Decision] = []
        if mode == "chat":
            parts: list[str] = []
            stream = stream_chat_answer(question, trace.hits, llm=llm, on_usage=usage.add)
            # `aclosing` finalises the stream (and so records its cost in the ledger) at once
            # if the client disconnects mid-answer, rather than whenever it is collected.
            async with aclosing(stream):
                async for delta in stream:
                    parts.append(delta)
                    yield Event(type="token", data={"text": delta})
            trace.answer_text = "".join(parts)
            yield Event(type="answer", data={"text": trace.answer_text})
        else:
            benchmark, answer_usage = await asyncio.to_thread(
                answer_benchmark, question, trace.hits, sample_id=0, llm=llm
            )
            usage.add(answer_usage)
            if bundle is not None and _SELF_CONSISTENCY & set(bundle.features):
                extra = await _extra_samples(
                    question, trace.hits, settings.self_consistency_samples, llm, usage
                )
                sample_decisions = [benchmark.decision, *extra]
            trace.decision = benchmark.decision
            trace.answer_text = _benchmark_text(benchmark)
            yield Event(
                type="answer",
                data={
                    "text": trace.answer_text,
                    "decision": benchmark.decision,
                    "probabilities": benchmark.probabilities,
                },
            )
        yield clock.done()

        # -- verify: claim decomposition (chat) and entailment against the evidence
        yield clock.start("verify")
        claims: list[Claim]
        if benchmark is None:
            chat, claims_usage = await asyncio.to_thread(
                extract_claims, question, trace.answer_text, trace.hits, llm=llm
            )
            usage.add(claims_usage)
            claims, verbal_conf = chat.claims, chat.verbal_confidence
        else:
            # The stated probability of the chosen decision is the benchmark analogue of the
            # chat answer's verbal confidence.
            claims, verbal_conf = benchmark.claims, benchmark.probabilities[benchmark.decision]
        trace.verdicts = await asyncio.to_thread(engines.verify, claims, trace.hits)
        yield Event(type="verdicts", data={"verdicts": _dump_all(trace.verdicts)})
        yield clock.done()

        # -- confidence
        yield clock.start("confidence")
        all_features = extract_features(
            trace.hits,
            trace.verdicts,
            verbal_conf=verbal_conf,
            sample_decisions=sample_decisions or None,
            probabilities=benchmark.probabilities if benchmark else None,
        )
        features = {
            name: value
            for name, value in all_features.items()
            if name in _MEASURABLE[mode] and (name not in _SELF_CONSISTENCY or sample_decisions)
        }
        trace.confidence = _confidence(bundle, features, alpha, settings.delta)
        flags = _confidence_flags(bundle, trace, settings.llm_model, benchmark_qid)
        yield Event(type="confidence", data=_dump(trace.confidence) | flags)
        yield clock.done()

        yield await _finish(trace, clock, usage, engines, confidence_flags=flags)
    except Exception as exc:
        stage = clock.current or "pipeline"
        yield Event(
            type="error",
            data={
                "stage": stage,
                "message": error_message(exc, stage),  # also logs an internal failure
                "kind": error_kind(exc, stage),
            },
        )


# ---------------------------------------------------------------------------- stages


def _link_concepts(engines: Engines, text: str) -> tuple[list[Concept], bool]:
    """The concepts recognised in `text`, and whether the concept dictionary exists.

    Linking only annotates the question: no later stage reads it (the `+concepts` retrievers
    load the dictionary themselves and fail loudly without it). So a missing dictionary, which
    is the normal state of an install that built only `pubmedqa` or `nfcorpus`, must not stop
    a run that never needed the `kb` corpus.
    """
    try:
        return engines.link_concepts(text), True
    except CorpusNotBuilt:
        return [], False


def _search(engines: Engines, name: str, corpus: str, query: str, k: int) -> list[Hit]:
    return engines.retriever(name, corpus).search(query, k=k)


async def _extra_samples(
    question: str, hits: list[Hit], n: int, llm: Any, usage: _UsageTotal
) -> list[Decision]:
    """Decisions of samples 1..n-1, drawn in parallel.

    Together with the answer shown (sample 0) they are the `n` self-consistency samples, the
    same ids `experiments.run_pubmedqa` draws, so the features match what the bundle was fitted
    on and every sample is a cache hit for a question the experiment already answered.
    """
    results = await asyncio.gather(
        *(
            asyncio.to_thread(answer_benchmark, question, hits, sample_id=i, llm=llm)
            for i in range(1, n)
        )
    )
    for _, sample_usage in results:
        usage.add(sample_usage)
    return [answer.decision for answer, _ in results]


def _check_measurable(bundle: DeploymentBundle | None, mode: Mode) -> None:
    if bundle is None:
        return
    missing = [name for name in bundle.features if name not in _MEASURABLE[mode]]
    if missing:
        raise UnmeasurableFeatures(
            f"the deployment bundle's confidence model reads {missing}, which {mode} mode does "
            f"not measure; fit the deployed model on FEATURES_DEPLOY"
        )


def _confidence(
    bundle: DeploymentBundle | None, features: dict[str, float], alpha: float, delta: float
) -> ConfidenceReport:
    if bundle is not None:
        return bundle.score(features, alpha=alpha)
    # No bundle: there is no calibrated probability to report, and NaN (sent as null) says so.
    # Any number here would read as a confidence the system does not have. The answer is
    # shown, flagged as uncertified by the event's `uncalibrated` field.
    return ConfidenceReport(
        features=features,
        p_correct=math.nan,
        alpha=alpha,
        delta=delta,
        threshold=None,
        answered=True,
    )


def _confidence_flags(
    bundle: DeploymentBundle | None,
    trace: PipelineResult,
    llm_model: str,
    benchmark_qid: str | None,
) -> dict[str, bool]:
    """What the confidence report may be read as.

    `uncalibrated`: no deployment bundle exists, so the answer is released unscored.
    `guarantee_applies`: the certificate `P(risk <= alpha) >= 1 - delta` covers this answer.
    It does only for a *new* draw from the calibration distribution answered by the same
    pipeline the thresholds were certified on (same generator, same retriever): a PubMedQA
    item the bundle held out of both its fit and its calibration rows. An item among those
    rows is in-sample (the threshold was chosen on it, or the model fitted to it), and a
    free-form question, typed in either mode, comes from a different distribution: for both
    the threshold is an operating point, not a guarantee.
    """
    confidence = trace.confidence
    applies = (
        bundle is not None
        and confidence is not None
        and confidence.threshold is not None
        and trace.mode == "benchmark"
        and benchmark_qid is not None
        and benchmark_qid in bundle.heldout_qids
        and trace.retriever == bundle.retriever
        and llm_model == bundle.llm_model
    )
    return {"uncalibrated": bundle is None, "guarantee_applies": applies}


async def _finish(
    trace: PipelineResult,
    clock: _StageClock,
    usage: _UsageTotal,
    engines: Engines,
    *,
    confidence_flags: dict[str, bool] | None,
) -> Event:
    """Store the trace and build the `done` event; the stored payload is the event's data."""
    clock.current = "done"
    trace.timings_ms = dict(clock.timings)
    trace.usage = usage.total()
    payload = _dump(trace)
    if confidence_flags is not None and payload["confidence"] is not None:
        # Kept on the stored copy too: `PipelineResult` has no field for it, and a trace
        # reopened later must still say the answer was released without a guarantee.
        payload["confidence"] |= confidence_flags
    await asyncio.to_thread(engines.traces.save, payload)
    return Event(type="done", data=payload)


def _benchmark_text(answer: BenchmarkAnswer) -> str:
    """A readable rendering of a structured answer: the decision, then its cited claims."""
    lines = [f"**{answer.decision.capitalize()}.**"]
    for claim in answer.claims:
        markers = "".join(f"[{n}]" for n in claim.cited)
        lines.append(f"- {claim.text} {markers}".rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------- helpers


class _StageClock:
    """Tracks the running stage (for error attribution) and each stage's wall time."""

    def __init__(self) -> None:
        self.current: str | None = None
        self.timings: dict[str, float] = {}
        self._started = 0.0

    def start(self, name: str) -> Event:
        self.current = name
        self._started = time.perf_counter()
        return Event(type="stage", data={"name": name, "status": "start", "ms": None})

    def done(self) -> Event:
        name = self.current or ""
        ms = round((time.perf_counter() - self._started) * 1000, 1)
        self.timings[name] = ms
        self.current = None
        return Event(type="stage", data={"name": name, "status": "done", "ms": ms})


class _UsageTotal:
    """Sums the usage of every model call in one run (passed as `on_usage` to the stream)."""

    def __init__(self) -> None:
        self._parts: list[Usage] = []

    def add(self, usage: Usage) -> None:
        self._parts.append(usage)

    def total(self) -> Usage:
        parts = self._parts
        return Usage(
            input_tokens=sum(u.input_tokens for u in parts),
            output_tokens=sum(u.output_tokens for u in parts),
            cache_read_tokens=sum(u.cache_read_tokens for u in parts),
            cost_usd=sum(u.cost_usd for u in parts),
            # True only if no call reached the API, i.e. the run cost nothing.
            cached=bool(parts) and all(u.cached for u in parts),
        )


def _dump(model: BaseModel) -> dict[str, Any]:
    # Through the JSON encoder, so NaN/inf become null rather than invalid JSON.
    return json.loads(model.model_dump_json())


def _dump_all(models: Sequence[BaseModel]) -> list[dict[str, Any]]:
    return [_dump(model) for model in models]


def _event(type_: str, model: BaseModel) -> Event:
    return Event(type=type_, data=_dump(model))
