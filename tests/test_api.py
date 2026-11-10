"""API tests through FastAPI's TestClient, with the stub engines from `test_pipeline`.

Paths (results, artefacts, data) point at a temporary directory, so these tests see an empty
installation unless they write files into it themselves.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_pipeline import CHUNKS, EMERGENCY, ROUTINE, FakeLLM, Recorder, make_bundle, make_engines

import primum.data
from primum.api.app import create_app
from primum.config import Settings, get_settings
from primum.data import CorpusNotBuilt
from primum.llm import LLMUnavailable
from primum.types import PubMedQAExample
from primum.uncertainty import BUNDLE_FILENAME
from primum.uncertainty.conformal import certify_threshold

ORIGIN = "http://localhost:3000"


def make_settings(tmp_path: Path) -> Settings:
    settings = get_settings().model_copy(
        update={
            "data_dir": tmp_path / "data",
            "artifacts_dir": tmp_path / "artifacts",
            "results_dir": tmp_path / "results",
            "anthropic_api_key": "",
        }
    )
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    return settings


def make_client(tmp_path: Path, **engine_options) -> TestClient:
    engines = make_engines(tmp_path, **engine_options)
    return TestClient(create_app(engines=engines, settings=make_settings(tmp_path)))


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return make_client(tmp_path)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """(event, data) per SSE message; keep-alive comments and blank frames are skipped."""
    messages = []
    for frame in body.replace("\r\n", "\n").split("\n\n"):
        fields: dict[str, list[str]] = {}
        for line in frame.split("\n"):
            if line and not line.startswith(":"):
                key, _, value = line.partition(":")
                fields.setdefault(key, []).append(value.removeprefix(" "))
        if "event" in fields:
            messages.append((fields["event"][0], json.loads("\n".join(fields["data"]))))
    return messages


def assert_error(response, status: int, kind: str) -> str:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["kind"] == kind
    return error["message"]


# ---------------------------------------------------------------------------- /api/ask


def test_ask_streams_pipeline_events_as_sse(client: TestClient) -> None:
    response = client.post("/api/ask", json={"question": ROUTINE})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    messages = parse_sse(response.text)
    types = [event for event, _ in messages]
    assert types[0] == "stage" and types[-1] == "done"
    for expected in (
        "triage",
        "concepts",
        "retrieval",
        "token",
        "answer",
        "verdicts",
        "confidence",
    ):
        assert expected in types
    assert types.index("retrieval") < types.index("token") < types.index("answer")
    assert "".join(d["text"] for e, d in messages if e == "token") == "".join(CHUNKS)
    confidence = next(d for e, d in messages if e == "confidence")
    assert confidence["uncalibrated"] is True and confidence["p_correct"] is None


def test_ask_emergency_stream_ends_after_triage(client: TestClient) -> None:
    messages = parse_sse(client.post("/api/ask", json={"question": EMERGENCY}).text)
    assert [event for event, _ in messages] == ["stage", "triage", "stage", "done"]
    assert messages[1][1]["level"] == "emergency"


def test_ask_engine_failure_is_an_error_event(tmp_path: Path) -> None:
    client = make_client(tmp_path, llm=FakeLLM(fail=LLMUnavailable("set ANTHROPIC_API_KEY")))
    messages = parse_sse(client.post("/api/ask", json={"question": ROUTINE}).text)
    assert messages[-1] == (
        "error",
        {"stage": "generate", "message": "set ANTHROPIC_API_KEY", "kind": "no_api_key"},
    )


def test_ask_benchmark_qid_uses_the_dataset_question_and_gold_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = PubMedQAExample(
        qid="123",
        question="Is metformin effective?",
        contexts=["..."],
        long_answer="...",
        label="maybe",
        passage_id="pubmed-123",
    )
    monkeypatch.setattr(primum.data, "load_pubmedqa_labeled", lambda: [example])
    recorder = Recorder()
    client = make_client(tmp_path, retriever=recorder)

    body = {"question": "whatever the box said", "benchmark_qid": "123", "retriever": "bm25"}
    done = parse_sse(client.post("/api/ask", json=body).text)[-1][1]
    assert done["mode"] == "benchmark" and done["question"] == example.question
    assert done["gold_label"] == "maybe"
    assert recorder.requests == [("bm25", "pubmedqa")]

    missing = client.post("/api/ask", json={**body, "benchmark_qid": "999"})
    assert_error(missing, 404, "not_found")


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"question": "hi"}, "question"),
        ({"question": "   hi   "}, "question"),  # length is counted after stripping
        ({"question": "x" * 2001}, "question"),
        ({}, "question"),
        ({"question": ROUTINE, "mode": "essay"}, "mode"),
        ({"question": ROUTINE, "retriever": "google"}, "retriever"),
        ({"question": ROUTINE, "alpha": 0.3}, "alpha"),
    ],
)
def test_ask_rejects_invalid_requests(client: TestClient, body: dict, field: str) -> None:
    message = assert_error(client.post("/api/ask", json=body), 422, "invalid")
    assert field in message


def test_question_length_bounds_are_inclusive(client: TestClient) -> None:
    assert client.post("/api/ask", json={"question": "abc"}).status_code == 200
    assert client.post("/api/ask", json={"question": "x" * 2000}).status_code == 200


# ---------------------------------------------------------------------------- /api/retrieve


def test_retrieve_returns_each_ranking_with_concepts(tmp_path: Path) -> None:
    recorder = Recorder()
    client = make_client(tmp_path, retriever=recorder)
    body = {"query": "metformin and HbA1c", "k": 2, "retrievers": ["bm25", "dense-base"]}
    response = client.post("/api/retrieve", json=body)
    assert response.status_code == 200
    result = response.json()
    assert list(result["results"]) == ["bm25", "dense-base"]
    assert [len(hits) for hits in result["results"].values()] == [2, 2]
    assert result["results"]["bm25"][0]["components"]["bm25_rank"] == 1
    assert result["concepts"][0]["preferred"] == "Metformin"
    assert result["expanded_query"] == "metformin and HbA1c Glucophage"
    assert {corpus for _, corpus in recorder.requests} == {get_settings().chat_corpus}


@pytest.mark.parametrize(
    "body",
    [
        {"query": "metformin", "retrievers": []},
        {"query": "metformin", "retrievers": ["bm25", "bm25"]},
        {"query": "metformin", "retrievers": ["tf-idf"]},
        {"query": "metformin", "retrievers": ["bm25"], "k": 0},
        {"query": "metformin", "retrievers": ["bm25"], "corpus": "wikipedia"},
    ],
)
def test_retrieve_rejects_invalid_requests(client: TestClient, body: dict) -> None:
    assert_error(client.post("/api/retrieve", json=body), 422, "invalid")


def test_retrieve_without_the_concept_dictionary_still_returns_the_rankings(
    tmp_path: Path,
) -> None:
    """A pubmedqa- or nfcorpus-only install has no `kb` concept dictionary."""

    def no_dictionary(_: str):
        raise CorpusNotBuilt("run `primum data build kb`")

    engines = dataclasses.replace(
        make_engines(tmp_path), link_concepts=no_dictionary, expand_query=no_dictionary
    )
    client = TestClient(create_app(engines=engines, settings=make_settings(tmp_path)))
    body = {"query": "metformin", "retrievers": ["bm25"], "corpus": "nfcorpus"}
    response = client.post("/api/retrieve", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["results"]["bm25"]) == 3
    assert result["concepts"] == [] and result["concepts_available"] is False
    assert result["expanded_query"] == "metformin"  # not expanded, and not pretending to be


def test_retrieve_without_an_index_is_no_index(tmp_path: Path) -> None:
    client = make_client(tmp_path, retriever=Recorder(CorpusNotBuilt("run `primum data build kb`")))
    response = client.post("/api/retrieve", json={"query": "metformin", "retrievers": ["bm25"]})
    assert "primum data build kb" in assert_error(response, 503, "no_index")


# ---------------------------------------------------------------------------- results


def test_experiments_are_404_until_run_then_served_verbatim(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.get("/api/experiments").json() == {
        "run": "main",
        "available": [],
        "files": {},
        "runs": [],
    }
    message = assert_error(client.get("/api/experiments/rq1_retrieval"), 404, "not_found")
    assert "primum exp rq1" in message
    assert_error(client.get("/api/experiments/secret"), 404, "not_found")

    content = {"meta": {"created": "2026-11-10T00:00:00"}, "nfcorpus": {}}
    (tmp_path / "results" / "rq1_retrieval.json").write_text(json.dumps(content))
    index = client.get("/api/experiments").json()
    assert index["available"] == ["rq1_retrieval"] and "updated" in index["files"]["rq1_retrieval"]
    assert client.get("/api/experiments/rq1_retrieval").json() == content


def _write(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def _runs_file(llm: str, *, samples: int, n: int, failed: int) -> dict:
    return {
        "meta": {
            "n": n,
            "models": {"llm": llm},
            "conditions": ["rag"],
            "self_consistency_samples": samples,
        },
        "summary": {"pubmedqa": {"rag": {"n": n - failed, "n_failed": failed}}},
        "records": [],
    }


def test_experiments_serve_named_runs_from_their_subdirectory(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    results = tmp_path / "results"
    _write(results / "pubmedqa_runs.json", _runs_file("main-model", samples=5, n=10, failed=0))
    _write(results / "rq1_retrieval.json", {"meta": {}})
    second = _runs_file("second-model", samples=1, n=10, failed=2)
    _write(results / "sonnet" / "pubmedqa_runs.json", second)
    _write(results / "sonnet" / "rq2_confidence.json", {"meta": {"models": {"llm": "x"}}})
    _write(tmp_path / "secret.json", {"leak": True})

    index = client.get("/api/experiments").json()
    assert index["run"] == "main" and index["available"] == ["rq1_retrieval", "pubmedqa_runs"]
    runs = {r["run"]: r for r in index["runs"]}
    assert set(runs) == {"main", "sonnet"}  # sonnet_sc300 has no files, so it is not listed
    assert runs["sonnet"] == {
        "run": "sonnet",
        "generator": "second-model",
        "n": 10,
        "self_consistency_samples": 1,
        "conditions": ["rag"],
        "n_failed": {"rag": 2},
        "available": ["rq2_confidence", "pubmedqa_runs"],
        "subset_of": None,
    }
    assert runs["main"]["generator"] == "main-model"

    sonnet = client.get("/api/experiments?run=sonnet").json()
    assert sonnet["run"] == "sonnet" and sonnet["available"] == ["rq2_confidence", "pubmedqa_runs"]
    assert client.get("/api/experiments/pubmedqa_runs?run=sonnet").json() == second
    # Generator-independent files live in the main run only.
    assert_error(client.get("/api/experiments/rq1_retrieval?run=sonnet"), 404, "not_found")

    for run in ("nope", "..", "../..", "sonnet/..", "/etc", ""):
        assert_error(client.get("/api/experiments", params={"run": run}), 404, "not_found")
        assert_error(
            client.get("/api/experiments/pubmedqa_runs", params={"run": run}), 404, "not_found"
        )
        assert_error(client.get("/api/calibration", params={"run": run}), 404, "not_found")
    assert client.get("/api/experiments/..%2Fsecret").status_code == 404


def test_run_info_falls_back_to_rq2_meta_without_the_runs_file(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    meta = {"n": 8, "models": {"llm": "m"}, "n_failed_excluded": 1}
    _write(tmp_path / "results" / "sonnet_sc300" / "rq2_confidence.json", {"meta": meta})
    (info,) = client.get("/api/experiments").json()["runs"]
    assert info["run"] == "sonnet_sc300" and info["generator"] == "m" and info["n"] == 8
    assert info["n_failed"] == {"rag": 1} and info["self_consistency_samples"] is None


def test_run_info_says_when_a_run_is_a_subset_of_another_of_its_generator(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path)
    meta = {"models": {"llm": "g"}, "conditions": ["rag"]}

    def rec(qid: str, decision: str | None) -> dict:
        if decision is None:
            return {"qid": qid, "condition": "rag", "error": "LLMInvalidOutput"}
        return {"qid": qid, "condition": "rag", "decision": decision}

    full = [rec("q1", "yes"), rec("q2", "no"), rec("q3", None), rec("q4", "yes")]
    subset = [rec("q1", "yes"), rec("q2", None), rec("q3", None)]
    _write(tmp_path / "results" / "sonnet" / "pubmedqa_runs.json", {"meta": meta, "records": full})
    _write(
        tmp_path / "results" / "sonnet_sc300" / "pubmedqa_runs.json",
        {"meta": meta, "records": subset},
    )
    runs = {r["run"]: r for r in client.get("/api/experiments").json()["runs"]}
    assert runs["sonnet"]["subset_of"] is None
    assert runs["sonnet_sc300"]["subset_of"] == {
        "run": "sonnet",
        "prefix": True,
        "answered_in_both": 1,
        "first_sample_matches": 1,
        "failed_only_here": 1,
    }
    # Not the first questions of the larger run: still a subset, but not a prefix.
    _write(
        tmp_path / "results" / "sonnet_sc300" / "pubmedqa_runs.json",
        {"meta": meta, "records": [rec("q2", "yes"), rec("q4", "yes")]},
    )
    runs = {r["run"]: r for r in client.get("/api/experiments").json()["runs"]}
    assert runs["sonnet_sc300"]["subset_of"]["prefix"] is False
    assert runs["sonnet_sc300"]["subset_of"]["first_sample_matches"] == 1


def test_calibration_per_run_uses_that_generators_bundle(tmp_path: Path) -> None:
    live = make_bundle(threshold=0.8)
    client = make_client(tmp_path, bundle=live)
    settings = make_settings(tmp_path)
    other = live.model_copy(
        update={
            "llm_model": "other-model",
            "guarantees": [g.model_copy(update={"threshold": 0.6}) for g in live.guarantees],
        }
    )
    other.save(settings.deployment_dir_for("other-model"))
    rows = {"p_correct": [0.9, 0.4], "correct": [1, 0], "thresholds": {"0.10": 0.5}}
    main = {"meta": {"models": {"llm": live.llm_model}}, "calibration": rows}
    second = {"meta": {"models": {"llm": "other-model"}}, "calibration": rows}
    third = {"meta": {"models": {"llm": "no-bundle-model"}}, "calibration": rows}
    _write(tmp_path / "results" / "rq3_conformal.json", main)
    _write(tmp_path / "results" / "sonnet" / "rq3_conformal.json", second)
    _write(tmp_path / "results" / "sonnet_sc300" / "rq3_conformal.json", third)

    body = client.get("/api/calibration").json()
    assert body["run"] == "main" and body["llm"] == live.llm_model
    assert body["thresholds"]["0.10"] == 0.8  # the live bundle
    body = client.get("/api/calibration?run=sonnet").json()
    assert body["run"] == "sonnet" and body["llm"] == "other-model"
    assert body["thresholds"]["0.10"] == 0.6  # that generator's saved bundle
    body = client.get("/api/calibration?run=sonnet_sc300").json()
    assert body["thresholds"] == {"0.10": 0.5}  # no bundle: the file's own thresholds


@pytest.mark.parametrize("generator", ["../x", "a/b", ".."])
def test_calibration_ignores_an_unsafe_generator_name(tmp_path: Path, generator: str) -> None:
    client = make_client(tmp_path, bundle=make_bundle(threshold=0.8))
    rows = {"p_correct": [0.9, 0.4], "correct": [1, 0], "thresholds": {"0.10": 0.5}}
    rq3 = {"meta": {"models": {"llm": generator}}, "calibration": rows}
    _write(tmp_path / "results" / "sonnet" / "rq3_conformal.json", rq3)
    body = client.get("/api/calibration?run=sonnet").json()
    assert body["thresholds"] == {"0.10": 0.5}  # the file's own, no bundle read


def test_calibration_ignores_a_bundle_saved_for_another_model(tmp_path: Path) -> None:
    live = make_bundle(threshold=0.8)
    client = make_client(tmp_path, bundle=live)
    settings = make_settings(tmp_path)
    # A bundle under `other-model/` whose own llm_model says something else.
    mislabelled = live.model_copy(
        update={
            "llm_model": "third-model",
            "guarantees": [g.model_copy(update={"threshold": 0.6}) for g in live.guarantees],
        }
    )
    mislabelled.save(settings.deployment_dir_for("other-model"))
    rows = {"p_correct": [0.9, 0.4], "correct": [1, 0], "thresholds": {"0.10": 0.5}}
    rq3 = {"meta": {"models": {"llm": "other-model"}}, "calibration": rows}
    _write(tmp_path / "results" / "sonnet" / "rq3_conformal.json", rq3)
    body = client.get("/api/calibration?run=sonnet").json()
    assert body["thresholds"] == {"0.10": 0.5}


def test_calibration_without_a_named_generator_uses_the_live_bundle_only_for_main(
    tmp_path: Path,
) -> None:
    client = make_client(tmp_path, bundle=make_bundle(threshold=0.8))
    rows = {"p_correct": [0.9, 0.4], "correct": [1, 0], "thresholds": {"0.10": 0.5}}
    rq3 = {"meta": {}, "calibration": rows}
    _write(tmp_path / "results" / "rq3_conformal.json", rq3)
    _write(tmp_path / "results" / "sonnet" / "rq3_conformal.json", rq3)
    assert client.get("/api/calibration").json()["thresholds"]["0.10"] == 0.8
    body = client.get("/api/calibration?run=sonnet").json()
    assert body["thresholds"] == {"0.10": 0.5} and body["llm"] is None


def test_calibration_is_404_until_rq3_has_run(client: TestClient) -> None:
    assert "primum exp rq3" in assert_error(client.get("/api/calibration"), 404, "not_found")


def test_calibration_serves_held_out_arrays_with_the_bundle_thresholds(tmp_path: Path) -> None:
    client = make_client(tmp_path, bundle=make_bundle(threshold=0.8))
    rq3 = {"meta": {"delta": 0.2}, "oof": {"p_correct": [0.9, 0.4], "correct": [True, False]}}
    (tmp_path / "results" / "rq3_conformal.json").write_text(json.dumps(rq3))

    body = client.get("/api/calibration").json()
    assert body["p_correct"] == [0.9, 0.4] and body["correct"] == [1, 0]
    assert body["thresholds"] == {"0.05": 0.8, "0.10": 0.8, "0.15": 0.8, "0.20": 0.8}
    assert body["delta"] == 0.1  # the bundle's, which is the rule actually deployed


def test_calibration_passes_through_rq3_certification_settings(tmp_path: Path) -> None:
    # The layout `primum exp rq3` writes: a `calibration` block with the certification settings
    # and a nested `heldout` pair. The explainer needs n_grid and min_selected to reproduce the
    # thresholds; without a bundle, thresholds and delta (null levels included) come from here.
    client = make_client(tmp_path)
    calibration = {
        "p_correct": [0.9, 0.4],
        "correct": [1, 0],
        "thresholds": {"0.05": None, "0.10": 0.85},
        "delta": 0.1,
        "n_grid": 100,
        "min_selected": 30,
        "heldout": {"p_correct": [0.7], "correct": [1]},
        "source": "test",
    }
    rq3 = {"meta": {"delta": 0.2}, "selective": {"rows": []}, "calibration": calibration}
    (tmp_path / "results" / "rq3_conformal.json").write_text(json.dumps(rq3))

    body = client.get("/api/calibration").json()
    assert body == {**calibration, "n": 2, "run": "main", "llm": None}


def test_calibration_without_arrays_says_so(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    (tmp_path / "results" / "rq3_conformal.json").write_text(json.dumps({"meta": {}}))
    assert "p_correct" in assert_error(client.get("/api/calibration"), 404, "not_found")


def test_certification_fixture_shared_with_the_web_explainer_is_current() -> None:
    # The method page recomputes the certificate in the browser from /api/calibration, and
    # web/src/lib/stats.test.ts checks that port against these expected thresholds. This test
    # keeps the expectations tied to the reference implementation.
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/certify_threshold_cases.json").read_text()
    )
    for case in fixture["cases"]:
        for alpha in fixture["alphas"]:
            guarantee = certify_threshold(
                np.array(case["p_correct"]),
                np.array(case["correct"]),
                alpha=alpha,
                delta=case["delta"],
                n_grid=case["n_grid"],
                min_selected=case["min_selected"],
            )
            expected = case["expected"][f"{alpha:.2f}"]
            assert (guarantee.threshold, guarantee.n_selected) == (
                expected["threshold"],
                expected["n_selected"],
            ), (case["name"], alpha)


# ---------------------------------------------------------------------------- traces


def test_traces_list_and_fetch_completed_runs(client: TestClient) -> None:
    done = parse_sse(client.post("/api/ask", json={"question": ROUTINE}).text)[-1][1]
    (summary,) = client.get("/api/traces").json()["items"]
    assert summary["trace_id"] == done["trace_id"] and summary["answered"] is True
    assert client.get(f"/api/traces/{done['trace_id']}").json() == done
    assert_error(client.get("/api/traces/nope"), 404, "not_found")
    assert_error(client.get("/api/traces?limit=0"), 422, "invalid")


# ---------------------------------------------------------------------------- status


def test_health_reports_missing_artefacts(tmp_path: Path) -> None:
    # The encoder directory defaults to an absolute path in the real project, so it is moved
    # into tmp_path here; otherwise a trained encoder on disk would make this test depend on it.
    encoder_dir = tmp_path / "artifacts" / "encoder-ft"
    settings = make_settings(tmp_path).model_copy(update={"finetuned_encoder_dir": encoder_dir})

    def health() -> dict:
        app = create_app(engines=make_engines(tmp_path), settings=settings)
        return TestClient(app).get("/api/health").json()

    body = health()
    assert body["status"] == "ok"
    assert body["api_key"] is False and body["deployment_bundle"] is False
    assert body["finetuned_encoder"] is False and body["indexes"] == {}

    # Bundles live per generator model; a bundle for another model would not count.
    settings.deployment_dir.mkdir(parents=True)
    (settings.deployment_dir / BUNDLE_FILENAME).write_text("{}")
    assert settings.deployment_dir.name == settings.llm_model
    (tmp_path / "data" / "indexes" / "kb" / "bm25").mkdir(parents=True)
    (tmp_path / "data" / "indexes" / "kb" / "bm25" / "meta.json").write_text("{}")
    (tmp_path / "data" / "indexes" / "kb" / "dense-ft").mkdir()  # unfinished: no meta.json
    encoder_dir.mkdir()
    body = health()
    assert body["deployment_bundle"] is True and body["indexes"] == {"kb": ["bm25"]}
    assert body["finetuned_encoder"] is True


def test_meta_lists_options_and_spend(client: TestClient) -> None:
    body = client.get("/api/meta").json()
    assert "bm25" in body["retrievers"] and "bm25+concepts" in body["retrievers"]
    assert body["alphas"] == [0.05, 0.10, 0.15, 0.20]
    assert body["spend"]["total_usd"] == 0.0 and body["budget_usd"] == get_settings().budget_usd
    assert body["corpora"] == {"pubmedqa": None, "nfcorpus": None, "kb": None}


def test_warmup_loads_each_component(client: TestClient) -> None:
    loaded = client.get("/api/warmup").json()["loaded"]
    assert set(loaded) == {"concepts", "retriever", "nli", "deployment_bundle"}
    assert all(part["ok"] for part in loaded.values())


def test_benchmark_questions_hide_labels_unless_revealed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples = [
        PubMedQAExample(
            qid=str(i), question=f"q{i}?", contexts=[], long_answer="", label="yes", passage_id=""
        )
        for i in range(5)
    ]
    monkeypatch.setattr(primum.data, "load_pubmedqa_labeled", lambda: examples)
    page = client.get("/api/benchmark/questions?limit=2&offset=1").json()
    assert page["total"] == 5 and [item["qid"] for item in page["items"]] == ["1", "2"]
    assert "label" not in page["items"][0]
    revealed = client.get("/api/benchmark/questions?limit=1&reveal=true").json()
    assert revealed["items"][0]["label"] == "yes"
    assert_error(client.get("/api/benchmark/questions?split=test"), 422, "invalid")


# ---------------------------------------------------------------------------- HTTP plumbing


def test_cors_preflight_allows_the_web_app(client: TestClient) -> None:
    response = client.options(
        "/api/ask",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_unexpected_errors_are_json_the_browser_can_read(tmp_path: Path) -> None:
    # An exception no handler anticipated (here a corrupt deployment bundle) must still reach
    # the web app as the JSON error with a CORS header; without that header the browser hides
    # the response and the page can only say the API is unreachable.
    def corrupt_bundle():
        raise ValueError("corrupt deployment bundle")

    engines = dataclasses.replace(make_engines(tmp_path), bundle=corrupt_bundle)
    app = create_app(engines=engines, settings=make_settings(tmp_path))
    # The bundle is looked up for the generator RQ3 recorded, so the results file must exist.
    rq3 = {"calibration": {"p_correct": [0.9], "correct": [1]}}
    (tmp_path / "results" / "rq3_conformal.json").write_text(json.dumps(rq3))
    response = TestClient(app, raise_server_exceptions=False).get(
        "/api/calibration", headers={"Origin": ORIGIN}
    )
    message = assert_error(response, 500, "internal")
    # The exception text stays in the server log; the client gets an id to quote.
    assert "corrupt deployment bundle" not in message and "Internal error (id " in message
    assert response.headers.get("access-control-allow-origin") == ORIGIN


def test_error_bodies_and_meta_do_not_reveal_server_paths(tmp_path: Path) -> None:
    root = get_settings().root
    missing = CorpusNotBuilt(f"{root}/data/processed/kb/passages.jsonl does not exist")
    client = make_client(tmp_path, retriever=Recorder(missing))
    response = client.post("/api/retrieve", json={"query": "metformin", "retrievers": ["bm25"]})
    message = assert_error(response, 503, "no_index")
    assert message == "./data/processed/kb/passages.jsonl does not exist"
    encoder = client.get("/api/meta").json()["models"]["finetuned_encoder"]
    assert str(root) not in encoder and encoder.endswith("encoder-ft")


def test_unknown_route_is_a_json_404(client: TestClient) -> None:
    assert_error(client.get("/api/nothing"), 404, "not_found")
