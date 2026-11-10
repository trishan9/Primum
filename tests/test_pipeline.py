"""Orchestrator tests with stub engines: event order, every outcome, every error kind.

The stubs replace only what needs a network, a model or a built index (retriever, LLM, NLI
model). Generation, verification, feature extraction, triage and the deployment bundle are the
real code, so these tests also check that the orchestrator calls them with the right arguments.

The stubs are imported by `test_api.py` and `test_pipeline_cli.py`.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from primum.config import get_settings
from primum.data import CorpusNotBuilt
from primum.llm import BudgetExceeded, LLMRefusal, LLMResult, LLMUnavailable
from primum.pipeline import Engines, Event, TraceStore, run_pipeline
from primum.pipeline.errors import display_paths, error_message
from primum.retrieval import EncoderNotFound
from primum.types import Concept, Hit, Passage, Usage
from primum.uncertainty import (
    ALPHAS,
    FEATURES_DEPLOY,
    FEATURES_FULL,
    DeploymentBundle,
    IdentityCalibration,
    LogisticConfidence,
    SelectiveGuarantee,
)
from primum.verification import verify_claims

PASSAGES = [
    Passage(
        passage_id="pubmed-1",
        doc_id="1",
        text="Metformin lowers HbA1c in type 2 diabetes. It is taken by mouth.",
        source="pubmed",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
    ),
    Passage(
        passage_id="mplus-diabetes-0",
        doc_id="mplus-diabetes",
        title="Diabetes",
        text="Exercise improves insulin sensitivity.",
        source="medlineplus",
    ),
    Passage(
        passage_id="pubmed-2",
        doc_id="2",
        text="Sulfonylureas can cause hypoglycaemia.",
        source="pubmed",
    ),
]
CLAIM = "Metformin lowers HbA1c in type 2 diabetes."
CHUNKS = ("Metformin lowers HbA1c ", "in type 2 diabetes [1].")
ROUTINE = "Does metformin lower HbA1c in type 2 diabetes?"
EMERGENCY = "I have crushing chest pain spreading to my left arm and I am sweating"
STAGES = ["triage", "concepts", "retrieve", "generate", "verify", "confidence"]
# PubMedQA ids: one the stub bundle held out, one it was fitted or certified on.
HELDOUT_QID, IN_SAMPLE_QID = "10000001", "10000002"


# ---------------------------------------------------------------------------- stubs


class FakeRetriever:
    def __init__(self, name: str, corpus: str) -> None:
        self.name, self.corpus = name, corpus

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return [
            Hit(
                passage=passage,
                score=1.0 / rank,
                rank=rank,
                retriever=self.name,
                components={
                    "bm25": 10.0 / rank,
                    "bm25_rank": float(rank),
                    "dense": 0.9 - 0.1 * rank,
                    "dense_rank": float(rank),
                },
            )
            for rank, passage in enumerate(PASSAGES[:k], start=1)
        ]

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Hit]]:
        return [self.search(query, k) for query in queries]


class FakeLLM:
    """Answers like the model would, through the real generation code's wire schemas."""

    def __init__(self, *, fail: Exception | None = None, decisions: Sequence[str] = ("yes",)):
        self.fail = fail
        self.decisions = decisions
        self.calls: list[tuple[str, int]] = []

    def complete(self, *, system, user, schema=None, sample_id=0, tag="", **_) -> LLMResult:
        self.calls.append((tag, sample_id))
        if self.fail is not None:
            raise self.fail
        claims = [{"text": CLAIM, "cited": [1]}]
        if tag.startswith("benchmark"):
            payload = {
                "claims": claims,
                "p_yes": 0.7,
                "p_no": 0.2,
                "p_maybe": 0.1,
                "decision": self.decisions[sample_id % len(self.decisions)],
            }
        else:
            payload = {"claims": claims, "verbal_confidence": 0.8}
        return LLMResult(
            text=json.dumps(payload),
            parsed=schema.model_validate(payload),
            usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.001),
            model="fake",
        )

    async def astream(self, *, system, user, tag="", on_usage=None, **_):
        self.calls.append((tag, 0))
        if self.fail is not None:
            raise self.fail
        for chunk in CHUNKS:
            yield chunk
        if on_usage is not None:
            on_usage(Usage(input_tokens=200, output_tokens=30, cost_usd=0.002))

    def spend(self) -> dict:
        return {"total_usd": 0.0, "by_tag": {}, "calls": 0, "cached_calls": 0, "budget_usd": 1.0}


class StubNLI:
    """Entails when the premise contains the claim verbatim; neutral otherwise."""

    def predict(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        rows = [
            [0.95, 0.04, 0.01] if h.rstrip(".").lower() in p.lower() else [0.1, 0.8, 0.1]
            for p, h in zip(premises, hypotheses, strict=True)
        ]
        return np.array(rows, dtype=np.float32).reshape(-1, 3)


def make_bundle(threshold: float | None, *, features: Sequence[str] = FEATURES_DEPLOY):
    """A real bundle around a tiny fitted model, with a chosen threshold at every alpha."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, len(features)))
    model = LogisticConfidence().fit(X, (X[:, 0] > 0).astype(float))
    guarantees = [
        SelectiveGuarantee(
            alpha=alpha,
            delta=0.1,
            threshold=threshold,
            n_cal=40,
            n_selected=20,
            cal_coverage=0.5,
            cal_risk=0.05,
            pvalue=0.01,
        )
        for alpha in ALPHAS
    ]
    return DeploymentBundle(
        features=list(features),
        confidence_model=model.to_json(),
        calibrator=IdentityCalibration().to_json(),
        delta=0.1,
        guarantees=guarantees,
        n_cal=40,
        llm_model=get_settings().llm_model,
        retriever="bm25",
        created="test",
        heldout_qids=[HELDOUT_QID],
    )


class Recorder:
    """Wraps the retriever factory to record which (name, corpus) the pipeline asked for."""

    def __init__(self, error: Exception | None = None) -> None:
        self.requests: list[tuple[str, str]] = []
        self.error = error

    def __call__(self, name: str, corpus: str) -> FakeRetriever:
        self.requests.append((name, corpus))
        if self.error is not None:
            raise self.error
        return FakeRetriever(name, corpus)


def make_engines(
    tmp_path: Path,
    *,
    llm: FakeLLM | None = None,
    bundle: DeploymentBundle | None = None,
    retriever: Recorder | None = None,
    verify=None,
) -> Engines:
    llm = llm or FakeLLM()
    concept = Concept(surface="metformin", start=5, end=14, preferred="Metformin")
    return Engines(
        retriever=retriever or Recorder(),
        link_concepts=lambda text: [concept] if "metformin" in text.lower() else [],
        expand_query=lambda query: f"{query} Glucophage",
        llm=lambda: llm,
        verify=verify or functools.partial(verify_claims, nli=StubNLI()),
        bundle=lambda: bundle,
        traces=TraceStore(tmp_path / "traces.sqlite"),
    )


async def collect(question: str, engines: Engines, **kwargs) -> list[Event]:
    return [event async for event in run_pipeline(question, engines=engines, **kwargs)]


def of_type(events: list[Event], type_: str) -> list[dict]:
    return [event.data for event in events if event.type == type_]


def skeleton(events: list[Event]) -> list[str]:
    """Event sequence with stage brackets spelled out and token runs collapsed to one."""
    names: list[str] = []
    for event in events:
        name = (
            f"{event.data['name']}:{event.data['status']}" if event.type == "stage" else event.type
        )
        if not (name == "token" and names and names[-1] == "token"):
            names.append(name)
    return names


# ---------------------------------------------------------------------------- event order


async def test_chat_emits_every_stage_in_order(tmp_path: Path) -> None:
    events = await collect(ROUTINE, make_engines(tmp_path))

    assert skeleton(events) == [
        "triage:start", "triage", "triage:done",
        "concepts:start", "concepts", "concepts:done",
        "retrieve:start", "retrieval", "retrieve:done",
        "generate:start", "token", "answer", "generate:done",
        "verify:start", "verdicts", "verify:done",
        "confidence:start", "confidence", "confidence:done",
        "done",
    ]  # fmt: skip
    for data in of_type(events, "stage"):
        assert (data["ms"] is None) == (data["status"] == "start")
    tokens = "".join(d["text"] for d in of_type(events, "token"))
    assert tokens == "".join(CHUNKS) == of_type(events, "answer")[0]["text"]


async def test_events_are_strict_json(tmp_path: Path) -> None:
    for event in await collect(ROUTINE, make_engines(tmp_path)):
        json.dumps(event.data, allow_nan=False)  # raises on NaN / inf


async def test_chat_result_carries_verdicts_usage_timings_and_is_stored(tmp_path: Path) -> None:
    engines = make_engines(tmp_path)
    events = await collect(ROUTINE, engines)

    (verdicts,) = of_type(events, "verdicts")
    assert [v["label"] for v in verdicts["verdicts"]] == ["supported"]
    assert verdicts["verdicts"][0]["best_hit"] == 1

    (done,) = of_type(events, "done")
    assert done["question"] == ROUTINE and done["mode"] == "chat"
    assert [h["passage"]["passage_id"] for h in done["hits"]] == [p.passage_id for p in PASSAGES]
    assert done["concepts"][0]["preferred"] == "Metformin"
    assert set(done["timings_ms"]) == set(STAGES)
    # Streamed answer (200 in / 30 out) plus the claim extraction call (100 in / 20 out).
    assert done["usage"]["input_tokens"] == 300 and done["usage"]["output_tokens"] == 50
    assert done["usage"]["cost_usd"] == pytest.approx(0.003)
    assert engines.traces.get(done["trace_id"]) == done


async def test_chat_uses_the_configured_retriever_on_the_chat_corpus(tmp_path: Path) -> None:
    recorder = Recorder()
    await collect(ROUTINE, make_engines(tmp_path, retriever=recorder))
    settings = get_settings()
    assert recorder.requests == [(settings.default_retriever, settings.chat_corpus)]

    await collect(ROUTINE, make_engines(tmp_path, retriever=recorder), retriever="bm25")
    assert recorder.requests[-1] == ("bm25", settings.chat_corpus)


# ---------------------------------------------------------------------------- confidence


async def test_without_a_bundle_the_answer_is_released_uncalibrated(tmp_path: Path) -> None:
    events = await collect(ROUTINE, make_engines(tmp_path, bundle=None))
    (confidence,) = of_type(events, "confidence")
    assert confidence["uncalibrated"] is True
    assert confidence["threshold"] is None and confidence["answered"] is True
    # Not measured, so not a number: null on the wire.
    assert confidence["p_correct"] is None
    assert confidence["guarantee_applies"] is False
    assert set(confidence["features"]) == set(FEATURES_DEPLOY)
    # The stored trace keeps the flag that PipelineResult itself has no field for.
    (done,) = of_type(events, "done")
    assert done["confidence"]["uncalibrated"] is True


async def test_below_the_threshold_the_system_abstains_but_keeps_the_draft(tmp_path: Path) -> None:
    engines = make_engines(tmp_path, bundle=make_bundle(threshold=1.0))
    events = await collect(ROUTINE, engines, alpha=0.05)

    (confidence,) = of_type(events, "confidence")
    assert confidence["uncalibrated"] is False
    assert confidence["alpha"] == 0.05 and confidence["threshold"] == 1.0
    assert 0.0 <= confidence["p_correct"] < 1.0
    assert confidence["answered"] is False

    (done,) = of_type(events, "done")
    assert done["answer_text"] == "".join(CHUNKS)  # withheld, not discarded
    (summary,) = engines.traces.list()
    assert summary["answered"] is False


async def test_above_the_threshold_the_answer_is_released(tmp_path: Path) -> None:
    events = await collect(ROUTINE, make_engines(tmp_path, bundle=make_bundle(threshold=0.0)))
    (confidence,) = of_type(events, "confidence")
    assert confidence["answered"] is True and confidence["uncalibrated"] is False
    # A free-form chat question is outside the calibration distribution.
    assert confidence["guarantee_applies"] is False


async def test_no_certified_threshold_means_abstain(tmp_path: Path) -> None:
    events = await collect(ROUTINE, make_engines(tmp_path, bundle=make_bundle(threshold=None)))
    (confidence,) = of_type(events, "confidence")
    assert confidence["threshold"] is None and confidence["answered"] is False


async def test_chat_refuses_a_bundle_that_reads_unmeasured_signals(tmp_path: Path) -> None:
    llm = FakeLLM()
    bundle = make_bundle(threshold=0.5, features=FEATURES_FULL)
    events = await collect(ROUTINE, make_engines(tmp_path, llm=llm, bundle=bundle))
    (error,) = of_type(events, "error")
    assert error["stage"] == "generate" and error["kind"] == "internal"
    assert "sc_agreement" in error["message"]
    assert llm.calls == []  # caught before any money was spent


# ---------------------------------------------------------------------------- benchmark mode


async def test_benchmark_mode_answers_from_pubmedqa_without_streaming(tmp_path: Path) -> None:
    recorder = Recorder()
    llm = FakeLLM()
    engines = make_engines(tmp_path, llm=llm, retriever=recorder)
    events = await collect(ROUTINE, engines, mode="benchmark", retriever="bm25", gold_label="no")

    assert "token" not in skeleton(events)
    assert recorder.requests == [("bm25", "pubmedqa")]
    (answer,) = of_type(events, "answer")
    assert answer["decision"] == "yes"
    assert answer["probabilities"] == pytest.approx({"yes": 0.7, "no": 0.2, "maybe": 0.1})
    assert "[1]" in answer["text"]
    (done,) = of_type(events, "done")
    assert done["decision"] == "yes" and done["gold_label"] == "no"
    # No bundle reads self-consistency, so no extra samples were paid for.
    assert llm.calls == [("benchmark.evidence", 0)]
    (confidence,) = of_type(events, "confidence")
    assert confidence["features"]["verbal_conf"] == pytest.approx(0.7)
    assert "sc_agreement" not in confidence["features"]


async def test_benchmark_draws_self_consistency_samples_for_a_full_bundle(tmp_path: Path) -> None:
    llm = FakeLLM(decisions=("yes", "yes", "no"))
    bundle = make_bundle(threshold=0.0, features=FEATURES_FULL)
    engines = make_engines(tmp_path, llm=llm, bundle=bundle)
    events = await collect(
        ROUTINE, engines, mode="benchmark", retriever="bm25", benchmark_qid=HELDOUT_QID
    )

    n = get_settings().self_consistency_samples
    # Sample 0 is the answer shown; 1..n-1 are the extra draws (as in run_pubmedqa).
    assert sorted(llm.calls) == [("benchmark.evidence", i) for i in range(n)]
    (confidence,) = of_type(events, "confidence")
    sample_decisions = [("yes", "yes", "no")[i % 3] for i in range(n)]
    modal = max(sample_decisions.count(d) for d in set(sample_decisions))
    assert confidence["features"]["sc_agreement"] == pytest.approx(modal / n)
    # Same generator and retriever as the bundle, on a question it held out: certified.
    assert confidence["guarantee_applies"] is True


async def test_benchmark_guarantee_needs_the_calibrated_retriever(tmp_path: Path) -> None:
    engines = make_engines(tmp_path, bundle=make_bundle(threshold=0.0))
    events = await collect(
        ROUTINE, engines, mode="benchmark", retriever="dense-base", benchmark_qid=HELDOUT_QID
    )
    (confidence,) = of_type(events, "confidence")
    assert confidence["guarantee_applies"] is False


@pytest.mark.parametrize("qid", [IN_SAMPLE_QID, None])
async def test_benchmark_guarantee_needs_a_question_the_bundle_held_out(
    tmp_path: Path, qid: str | None
) -> None:
    # A question the threshold was certified on (or the model fitted on) is in-sample, and a
    # question typed in benchmark mode is not known to be a draw from PubMedQA at all.
    engines = make_engines(tmp_path, bundle=make_bundle(threshold=0.0))
    events = await collect(ROUTINE, engines, mode="benchmark", retriever="bm25", benchmark_qid=qid)
    (confidence,) = of_type(events, "confidence")
    assert confidence["answered"] is True and confidence["guarantee_applies"] is False
    (done,) = of_type(events, "done")
    assert done["confidence"]["guarantee_applies"] is False


async def test_a_retriever_without_a_scale_free_score_fails_before_any_model_call(
    tmp_path: Path,
) -> None:
    class LexicalOnly(FakeRetriever):
        def search(self, query: str, k: int = 10) -> list[Hit]:
            return [
                hit.model_copy(update={"components": {"bm25": hit.score, "bm25_rank": hit.rank}})
                for hit in super().search(query, k)
            ]

    llm = FakeLLM()
    engines = dataclasses.replace(
        make_engines(tmp_path, llm=llm), retriever=lambda name, corpus: LexicalOnly(name, corpus)
    )
    events = await collect(ROUTINE, engines, retriever="bm25")
    (error,) = of_type(events, "error")
    assert error["stage"] == "retrieve" and "UnscoredRetrieval" in error["message"]
    assert "retrieval" in skeleton(events)  # the evidence is still shown
    assert llm.calls == [] and "confidence" not in skeleton(events)


# ---------------------------------------------------------------------------- triage


async def test_emergency_short_circuits_chat(tmp_path: Path) -> None:
    llm, recorder = FakeLLM(), Recorder()
    engines = make_engines(tmp_path, llm=llm, retriever=recorder)
    events = await collect(EMERGENCY, engines)

    assert skeleton(events) == ["triage:start", "triage", "triage:done", "done"]
    (triage,) = of_type(events, "triage")
    assert triage["level"] == "emergency" and triage["message"]
    assert llm.calls == [] and recorder.requests == []
    (done,) = of_type(events, "done")
    assert done["answer_text"] == "" and done["hits"] == [] and done["confidence"] is None
    (summary,) = engines.traces.list()
    assert summary["triage"] == "emergency" and summary["answered"] is None


async def test_emergency_does_not_drop_a_benchmark_item(tmp_path: Path) -> None:
    events = await collect(EMERGENCY, make_engines(tmp_path), mode="benchmark")
    (done,) = of_type(events, "done")
    assert done["triage"]["level"] == "emergency"
    assert done["decision"] == "yes" and done["confidence"] is not None


# ---------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("error", "stage", "kind"),
    [
        (LLMUnavailable("no key"), "generate", "no_api_key"),
        (BudgetExceeded("spent"), "generate", "budget"),
        (LLMRefusal("bio", None), "generate", "internal"),
    ],
)
async def test_llm_failures_map_to_their_kind(tmp_path, error, stage, kind) -> None:
    engines = make_engines(tmp_path, llm=FakeLLM(fail=error))
    events = await collect(ROUTINE, engines)
    assert events[-1].type == "error"
    assert events[-1].data == {"stage": stage, "message": str(error), "kind": kind}
    assert "done" not in skeleton(events) and engines.traces.list() == []


@pytest.mark.parametrize(
    "error",
    [
        CorpusNotBuilt("kb not built; run `primum data build kb`"),
        EncoderNotFound("no encoder; train it with `primum train encoder`"),
        FileNotFoundError("index missing"),
    ],
)
async def test_missing_artefacts_are_no_index(tmp_path: Path, error: Exception) -> None:
    events = await collect(ROUTINE, make_engines(tmp_path, retriever=Recorder(error)))
    assert events[-1].type == "error"
    assert events[-1].data["stage"] == "retrieve" and events[-1].data["kind"] == "no_index"
    assert events[-1].data["message"] == str(error)


async def test_a_missing_concept_dictionary_does_not_stop_the_run(tmp_path: Path) -> None:
    """Benchmark mode needs only `pubmedqa`; the `kb` concept dictionary is decoration."""

    def no_dictionary(text: str):
        raise CorpusNotBuilt("kb not built; run `primum data build kb`")

    engines = dataclasses.replace(make_engines(tmp_path), link_concepts=no_dictionary)
    events = await collect(ROUTINE, engines, mode="benchmark", retriever="bm25")
    (concepts,) = of_type(events, "concepts")
    assert concepts == {"concepts": [], "available": False}
    assert "error" not in skeleton(events)
    (done,) = of_type(events, "done")
    assert done["concepts"] == [] and done["decision"] == "yes"


async def test_concept_linking_failures_other_than_a_missing_dictionary_still_fail(
    tmp_path: Path,
) -> None:
    def broken(text: str):
        raise RuntimeError("bad dictionary entry")

    engines = dataclasses.replace(make_engines(tmp_path), link_concepts=broken)
    events = await collect(ROUTINE, engines)
    assert events[-1].type == "error"
    assert events[-1].data["stage"] == "concepts" and events[-1].data["kind"] == "internal"


async def test_a_bug_in_a_stage_is_internal_and_names_the_stage(tmp_path: Path, caplog) -> None:
    def broken_verify(claims, hits):
        raise RuntimeError("boom at /srv/primum/secret.db")

    with caplog.at_level(logging.ERROR, logger="primum.pipeline.errors"):
        events = await collect(ROUTINE, make_engines(tmp_path, verify=broken_verify))
    error = events[-1].data
    assert error["stage"] == "verify" and error["kind"] == "internal"
    # The client gets an id, never the exception text; the server log has both.
    assert "boom" not in error["message"] and "/srv" not in error["message"]
    (error_id,) = re.findall(r"id ([0-9a-f]{12})", error["message"])
    (record,) = caplog.records
    assert error_id in record.getMessage() and "boom" in caplog.text
    # Everything before the failure was still delivered.
    assert "answer" in skeleton(events)


def test_error_messages_show_paths_relative_to_the_project_root() -> None:
    root = get_settings().root
    error = CorpusNotBuilt(
        f"{root / 'data/processed/kb/concepts.jsonl'} does not exist; run `primum data build kb`"
    )
    message = error_message(error, "concepts")
    assert message == (
        "./data/processed/kb/concepts.jsonl does not exist; run `primum data build kb`"
    )
    assert display_paths(f"Nothing under {root}.") == "Nothing under .."  # root, full stop
    # A sibling directory that merely shares the root's name as a prefix is not rewritten.
    assert display_paths(f"{root}-old/x and {root}.v2/x") == f"{root}-old/x and {root}.v2/x"


async def test_benchmark_failure_in_a_self_consistency_sample_stops_the_run(tmp_path) -> None:
    llm = FakeLLM()
    engines = make_engines(tmp_path, llm=llm, bundle=make_bundle(0.0, features=FEATURES_FULL))
    original = llm.complete

    def fail_on_samples(**kwargs):
        if kwargs.get("sample_id", 0) > 0:
            raise BudgetExceeded("spent")
        return original(**kwargs)

    llm.complete = fail_on_samples
    events = await collect(ROUTINE, engines, mode="benchmark", retriever="bm25")
    assert events[-1].data["kind"] == "budget" and events[-1].data["stage"] == "generate"


# ---------------------------------------------------------------------------- trace store


def test_trace_store_lists_newest_first_and_round_trips(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "t.sqlite")
    for i in range(3):
        store.save(
            {
                "trace_id": f"t{i}",
                "question": f"q{i}",
                "mode": "chat",
                "triage": {"level": "routine"},
                "confidence": {"answered": i % 2 == 0},
            }
        )
    assert [row["trace_id"] for row in store.list(limit=2)] == ["t2", "t1"]
    assert store.get("t1")["confidence"] == {"answered": False}
    assert store.get("missing") is None
