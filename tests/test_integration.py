"""End-to-end check of the package boundaries, with stubs only where a network or a model is needed.

    run_pubmedqa -> pubmedqa_runs.json -> rq3 -> bundle.json + calibration block
                 -> pipeline (benchmark mode, BundleLoader) -> /api/calibration

Retrieval is the real code (BM25 + a dense index on a deterministic hashing embedder, fused by
`HybridRetriever`), so `Hit.components` are the ones the uncertainty features really receive.
Generation, verification, feature extraction, the confidence model, calibration, certification
and the bundle are all real. The LLM and the NLI model are stubs.

What this pins down is agreement *between* packages, which each package's own tests cannot see:
the features recorded by the experiment are the ones the bundle is fitted on and the ones the
pipeline computes for the same question; the bundle's provenance matches the pipeline's
settings; and the API serves RQ3's calibration rows in the shape the web explainer reads.
"""

from __future__ import annotations

import functools
import json
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from primum.api.app import create_app
from primum.config import get_settings
from primum.experiments import rq3_conformal, run_pubmedqa
from primum.llm import LLMResult
from primum.pipeline import Engines, TraceStore, run_pipeline
from primum.pipeline.engines import BundleLoader
from primum.retrieval import BM25Index, DenseIndex, HybridRetriever
from primum.types import DECISIONS, Passage, PubMedQAExample, Usage
from primum.uncertainty import FEATURES_DEPLOY, FEATURES_FULL, DeploymentBundle
from primum.verification import verify_claims

N_EXAMPLES = 120
RETRIEVER = "hybrid-test"

EXAMPLES = [
    PubMedQAExample(
        qid=str(5000 + i),
        question=f"Does compound{i} reduce marker{i} in adults?",
        contexts=[f"Compound{i} reduced marker{i} in adults. Effects were seen at twelve weeks."],
        long_answer="(never indexed)",
        label=DECISIONS[i % 3],
        passage_id=f"pubmed-{5000 + i}",
    )
    for i in range(N_EXAMPLES)
]
# Each passage gets a different length, so no two have exactly equal BM25 or cosine scores
# against a query. With exact ties, a batched and a single-query float32 matrix product can
# break the tie differently (different BLAS kernels), and the two call paths would disagree
# for a reason unrelated to the code under test. Real encoders practically never tie exactly.
PASSAGES = [
    Passage(
        passage_id=ex.passage_id,
        doc_id=ex.qid,
        text=" ".join([*ex.contexts, *["cohort"] * (i + 1)]),
        source="pubmed",
    )
    for i, ex in enumerate(EXAMPLES)
]


def hashing_embedder(texts: list[str]) -> np.ndarray:
    """Bag of hashed tokens: deterministic, model-free, and similar for overlapping texts."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            out[row, zlib.crc32(token.encode()) % 64] += 1.0
    return out


@functools.cache
def retriever() -> HybridRetriever:
    return HybridRetriever(
        BM25Index(PASSAGES), DenseIndex(hashing_embedder, PASSAGES), name=RETRIEVER, candidate_k=10
    )


class StubLLM:
    """Decisions depend on (prompt, sample id); about one answer in three matches the gold label."""

    def complete(self, *, system, user, schema=None, sample_id=0, tag="", **_):
        h = zlib.crc32(f"{user}|{sample_id}".encode())
        decision = DECISIONS[h % 3]
        confident = 0.4 + (h % 50) / 100
        p = {d: (confident if d == decision else (1 - confident) / 2) for d in DECISIONS}
        payload = {
            "claims": [{"text": "Compound reduced marker in adults.", "cited": [1]}],
            "p_yes": p["yes"],
            "p_no": p["no"],
            "p_maybe": p["maybe"],
            "decision": decision,
        }
        return LLMResult(
            text=json.dumps(payload),
            parsed=schema.model_validate(payload),
            usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.0),
            model="stub",
        )

    def spend(self) -> dict:
        return {"total_usd": 0.0, "by_tag": {}, "calls": 0, "cached_calls": 0, "budget_usd": 1.0}


class StubNLI:
    def predict(self, premises, hypotheses):
        return np.array(
            [[0.8, 0.15, 0.05] if "reduced" in p else [0.1, 0.8, 0.1] for p in premises],
            dtype=np.float32,
        ).reshape(-1, 3)


@pytest.fixture
def pipeline_outputs(tmp_path, monkeypatch):
    """A complete experiment chain on synthetic data, written under `tmp_path`."""
    monkeypatch.setattr(run_pubmedqa, "load_pubmedqa_labeled", lambda: list(EXAMPLES))
    monkeypatch.setattr(run_pubmedqa, "load_passages", lambda corpus: list(PASSAGES))
    monkeypatch.setattr(run_pubmedqa, "build_retriever", lambda name, corpus: retriever())
    results, deployment = tmp_path / "results", tmp_path / "deployment"
    llm, nli = StubLLM(), StubNLI()
    run_pubmedqa.run(
        created="it",
        conditions=("rag",),
        retriever=RETRIEVER,
        samples=3,
        workers=2,
        n_boot=20,
        n_perm=20,
        results_dir=results,
        llm=llm,
        nli=nli,
    )
    rq3_conformal.run(created="it", repeats=2, results_dir=results, deployment_dir=deployment)
    return results, deployment, llm, nli


def test_bundle_scores_the_features_the_experiment_recorded(pipeline_outputs):
    results, deployment, _, _ = pipeline_outputs
    runs = json.loads((results / "pubmedqa_runs.json").read_text())
    bundle = DeploymentBundle.load(deployment)

    assert bundle.features == FEATURES_DEPLOY
    assert bundle.retriever == RETRIEVER
    assert bundle.llm_model == get_settings().llm_model
    for record in runs["records"]:
        assert set(record["features"]) == set(FEATURES_FULL)
        # The real hybrid retriever must have produced the scale-free retrieval signals.
        assert record["features"]["ret_top"] > 0
    record = runs["records"][0]
    report = bundle.score(record["features"], alpha=0.10)
    assert 0.0 <= report.p_correct <= 1.0


async def test_pipeline_reproduces_the_experiment_features(pipeline_outputs, tmp_path):
    """The live benchmark pipeline must compute, for the same question, the same features the
    bundle was fitted on; otherwise the certified threshold is applied to a different signal."""
    results, deployment, llm, nli = pipeline_outputs
    runs = json.loads((results / "pubmedqa_runs.json").read_text())
    bundle = DeploymentBundle.load(deployment)
    # One question the bundle held out and one it was fitted or certified on.
    heldout = next(ex for ex in EXAMPLES if ex.qid in bundle.heldout_qids)
    in_sample = next(ex for ex in EXAMPLES if ex.qid not in bundle.heldout_qids)

    engines = Engines(
        retriever=lambda name, corpus: retriever(),
        link_concepts=lambda text: [],
        expand_query=lambda text: text,
        llm=lambda: llm,
        verify=functools.partial(verify_claims, nli=nli),
        bundle=BundleLoader(deployment),
        traces=TraceStore(tmp_path / "traces.sqlite"),
    )
    for example in (heldout, in_sample):
        recorded = next(r for r in runs["records"] if r["qid"] == example.qid)
        events = [
            e
            async for e in run_pipeline(
                example.question,
                mode="benchmark",
                retriever=RETRIEVER,
                alpha=0.10,
                gold_label=example.label,
                benchmark_qid=example.qid,
                engines=engines,
            )
        ]
        by_type = {e.type: e.data for e in events}
        assert "error" not in by_type, by_type.get("error")
        confidence = by_type["confidence"]
        assert confidence["uncalibrated"] is False
        # Certified only for a question neither the fit nor the certification saw.
        assert confidence["guarantee_applies"] == (
            confidence["threshold"] is not None and example is heldout
        )
        assert by_type["answer"]["decision"] == recorded["decision"]
        for name in FEATURES_DEPLOY:
            assert confidence["features"][name] == pytest.approx(recorded["features"][name]), name


def test_api_serves_rq3_calibration_rows(pipeline_outputs, tmp_path, monkeypatch):
    results, deployment, llm, _ = pipeline_outputs
    rq3 = json.loads((results / "rq3_conformal.json").read_text())
    settings = get_settings().model_copy(update={"results_dir": results})
    engines = Engines(
        retriever=lambda name, corpus: retriever(),
        link_concepts=lambda text: [],
        expand_query=lambda text: text,
        llm=lambda: llm,
        verify=lambda claims, hits: [],
        bundle=BundleLoader(deployment),
        traces=TraceStore(tmp_path / "traces.sqlite"),
    )
    client = TestClient(create_app(engines=engines, settings=settings))

    body = client.get("/api/calibration").json()
    assert body["p_correct"] == rq3["calibration"]["p_correct"]
    assert body["correct"] == rq3["calibration"]["correct"]
    assert body["thresholds"] == rq3["calibration"]["thresholds"]
    assert body["delta"] == rq3["calibration"]["delta"]
    # The explainer needs the grid and minimum selection size to reproduce the certificate.
    assert body["n_grid"] == rq3["calibration"]["n_grid"]
    assert body["min_selected"] == rq3["calibration"]["min_selected"]
    # Rows nothing was fitted or certified on, for out-of-sample coverage and risk.
    assert body["heldout"] == rq3["calibration"]["heldout"]

    # The benchmark picker can offer exactly the questions the guarantee can cover.
    monkeypatch.setattr("primum.data.load_pubmedqa_labeled", lambda: list(EXAMPLES))
    page = client.get("/api/benchmark/questions", params={"split": "heldout", "limit": 200}).json()
    heldout_qids = DeploymentBundle.load(deployment).heldout_qids
    assert [item["qid"] for item in page["items"]] == heldout_qids
    assert page["total"] == len(heldout_qids) == len(body["heldout"]["correct"])

    listed = client.get("/api/experiments").json()
    assert set(listed["available"]) == {"pubmedqa_runs", "rq3_conformal"}
