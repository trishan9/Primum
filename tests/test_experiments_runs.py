"""Fast tests for the two experiments that call engines: `run_pubmedqa` with a stub LLM, stub NLI
and stub retriever, and `rq1_retrieval` with stub retrievers. No network, model or API key;
outputs go to a temporary directory, never to `results/`.
"""

from __future__ import annotations

import json
import threading
import zlib

import numpy as np
import pytest

from primum.experiments import report, rq1_retrieval, run_pubmedqa
from primum.llm import BudgetExceeded, LLMRefusal, LLMResult
from primum.retrieval import EncoderNotFound
from primum.types import DECISIONS, Hit, Passage, PubMedQAExample, Usage
from primum.uncertainty import UnscoredRetrieval

# --------------------------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------------------------

EXAMPLES = [
    PubMedQAExample(
        qid=str(1000 + i),
        question=f"Does treatment {i} reduce outcome {i}?",
        contexts=[f"Treatment {i} reduced outcome {i} in the trial."],
        long_answer="(never indexed)",
        label=DECISIONS[i % 3],
        passage_id=f"pubmed-{1000 + i}",
    )
    for i in range(6)
]
PASSAGES = [
    Passage(
        passage_id=ex.passage_id,
        doc_id=ex.qid,
        text=ex.contexts[0],
        source="pubmed",
    )
    for ex in EXAMPLES
]
DISTRACTOR = Passage(passage_id="pubmed-9", doc_id="9", text="Unrelated text.", source="pubmed")


class StubLLM:
    """Answers deterministically from (prompt, sample id); can refuse or run out of budget."""

    def __init__(self, refuse: str | None = None, budget_after: int | None = None) -> None:
        self.refuse = refuse
        self.budget_after = budget_after
        self.calls: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def complete(self, *, system, user, schema=None, sample_id=0, tag="", **_):
        with self._lock:
            self.calls.append((tag, sample_id))
            n_calls = len(self.calls)
        if self.budget_after is not None and n_calls > self.budget_after:
            raise BudgetExceeded("stub budget reached")
        if self.refuse and self.refuse in user:
            raise LLMRefusal("stub", None)
        decision = DECISIONS[zlib.crc32(f"{user}|{sample_id}".encode()) % 3]
        p = {d: (0.7 if d == decision else 0.15) for d in DECISIONS}
        wire = schema(
            claims=[{"text": "Treatment reduced the outcome in the trial.", "cited": [1]}],
            p_yes=p["yes"],
            p_no=p["no"],
            p_maybe=p["maybe"],
            decision=decision,
        )
        usage = Usage(input_tokens=10, output_tokens=5, cost_usd=0.001)
        return LLMResult(text="", parsed=wire, usage=usage, model="stub")


class StubNLI:
    """Entails when the premise mentions a reduction, otherwise neutral."""

    def predict(self, premises, hypotheses):
        return np.array(
            [[0.9, 0.08, 0.02] if "reduced" in p else [0.1, 0.8, 0.1] for p in premises]
        )


class StubRetriever:
    """Returns the gold abstract first for even-numbered questions, a distractor otherwise."""

    name = "stub"

    def search_batch(self, queries, k=10):
        out = []
        for query in queries:
            i = int(query.split()[2])
            first = PASSAGES[i] if i % 2 == 0 else DISTRACTOR
            out.append(
                [Hit(passage=first, score=1.0, rank=1, retriever="stub", components={"dense": 0.5})]
            )
        return out


@pytest.fixture
def stub_data(monkeypatch):
    monkeypatch.setattr(run_pubmedqa, "load_pubmedqa_labeled", lambda: list(EXAMPLES))
    monkeypatch.setattr(run_pubmedqa, "load_passages", lambda corpus: [*PASSAGES, DISTRACTOR])
    monkeypatch.setattr(run_pubmedqa, "build_retriever", lambda name, corpus: StubRetriever())


# --------------------------------------------------------------------------------------------
# run_pubmedqa
# --------------------------------------------------------------------------------------------


def test_run_pubmedqa_records_every_item_and_condition(tmp_path, stub_data):
    llm = StubLLM(refuse=EXAMPLES[1].question)
    path = run_pubmedqa.run(
        created="test",
        samples=3,
        workers=4,
        n_boot=50,
        n_perm=99,
        results_dir=tmp_path,
        llm=llm,
        nli=StubNLI(),
    )
    result = json.loads(path.read_text())
    meta, records = result["meta"], result["records"]
    assert meta["complete"] is True and meta["stopped"] is None
    assert meta["created"] == "test" and meta["self_consistency_samples"] == 3
    assert "anthropic_api_key" not in meta["config"]

    # Stable order: questions in dataset order, conditions in canonical order.
    assert [(r["qid"], r["condition"]) for r in records] == [
        (ex.qid, c) for ex in EXAMPLES for c in run_pubmedqa.CONDITIONS
    ]
    failed = [r for r in records if "error" in r]
    assert {r["qid"] for r in failed} == {EXAMPLES[1].qid} and len(failed) == 3
    assert all(r["error"].startswith("LLMRefusal") for r in failed)

    by_key = {(r["qid"], r["condition"]): r for r in records if "error" not in r}
    for (qid, condition), r in by_key.items():
        example = next(ex for ex in EXAMPLES if ex.qid == qid)
        assert r["correct"] == (r["decision"] == example.label)
        assert sum(r["probabilities"].values()) == pytest.approx(1.0)
        assert r["features"]["verbal_conf"] == pytest.approx(r["probabilities"][r["decision"]])
        assert r["triage"] == "routine"
        if condition == "rag":
            assert len(r["sample_decisions"]) == 3
            assert r["sample_decisions"][0] == r["decision"]
            assert r["features"]["sc_agreement"] > 0
            assert r["gold_in_hits"] == (int(qid) % 2 == 0)
        else:
            assert r["sample_decisions"] is None and r["features"]["sc_agreement"] == 0.0
        if condition == "closed_book":
            # No evidence: claims are unsupported and retrieval/NLI features are zero.
            assert r["n_supported"] == 0 and r["features"]["nli_mean_entail"] == 0.0
            assert r["gold_in_hits"] is None
        if condition == "oracle":
            assert r["hit_ids"] == [example.passage_id] and r["n_supported"] == r["n_claims"]

    # Self-consistency samples are drawn for `rag` only, with distinct sample ids.
    rag_samples = sorted({s for tag, s in llm.calls if tag == "benchmark.evidence"})
    assert rag_samples == [0, 1, 2]

    summary = result["summary"]["pubmedqa"]
    for condition in run_pubmedqa.CONDITIONS:
        entry = summary[condition]
        assert entry["n"] == 5 and entry["n_failed"] == 1
        assert set(entry["accuracy"]) == {"mean", "ci"} and set(entry["macro_f1"]) == {"mean", "ci"}
    assert summary["rag"]["gold_in_hits_rate"] == pytest.approx(3 / 5)  # questions 0, 2, 4
    assert set(result["paired_tests"]) == {"rag_vs_closed_book", "oracle_vs_rag"}
    assert result["paired_tests"]["rag_vs_closed_book"]["n_pairs"] == 5

    # The report renders the benchmark section from this file.
    docs = tmp_path / "RESULTS.md"
    report.run(created="test", results_dir=tmp_path, docs_path=docs)
    text = docs.read_text()
    assert "## PubMedQA benchmark runs" in text
    acc = summary["rag"]["accuracy"]
    assert f"{acc['mean']:.3f} [{acc['ci'][0]:.3f}, {acc['ci'][1]:.3f}]" in text


def test_run_pubmedqa_stops_cleanly_on_budget_and_keeps_completed_items(tmp_path, stub_data):
    llm = StubLLM(budget_after=4)
    with pytest.raises(BudgetExceeded):
        run_pubmedqa.run(
            created="test",
            conditions=("closed_book",),
            workers=1,
            n_boot=20,
            n_perm=19,
            results_dir=tmp_path,
            llm=llm,
            nli=StubNLI(),
        )
    result = json.loads((tmp_path / "pubmedqa_runs.json").read_text())
    assert result["meta"]["complete"] is False
    assert result["meta"]["stopped"].startswith("BudgetExceeded")
    assert len(result["records"]) == 4  # the four calls that succeeded, nothing invented


def test_run_pubmedqa_rejects_unknown_conditions(tmp_path):
    with pytest.raises(ValueError, match="unknown conditions"):
        run_pubmedqa.run(created="test", conditions=("rag", "open_book"), results_dir=tmp_path)


def test_run_pubmedqa_refuses_a_retriever_without_a_scale_free_score(
    tmp_path, stub_data, monkeypatch
):
    """BM25 alone gives no score comparable across questions. The run must stop before any
    call, not write records whose retrieval features are all a constant 0.0."""

    class LexicalOnly(StubRetriever):
        def search_batch(self, queries, k=10):
            return [
                [hit.model_copy(update={"components": {"bm25": 7.0, "bm25_rank": 1.0}})]
                for (hit,) in super().search_batch(queries, k)
            ]

    monkeypatch.setattr(run_pubmedqa, "build_retriever", lambda name, corpus: LexicalOnly())
    llm = StubLLM()
    with pytest.raises(UnscoredRetrieval):
        run_pubmedqa.run(
            created="test", conditions=("rag",), results_dir=tmp_path, llm=llm, nli=StubNLI()
        )
    assert llm.calls == []
    assert not (tmp_path / "pubmedqa_runs.json").exists()


# --------------------------------------------------------------------------------------------
# rq1_retrieval
# --------------------------------------------------------------------------------------------

NF_QUERIES = {f"q{i}": f"query {i}" for i in range(8)}
NF_QRELS = {f"q{i}": {f"d{i}": 2, f"d{i + 1}": 1} for i in range(8)}


class RankingRetriever:
    """`perfect` ranks the relevant documents first; `shuffled` puts them at rank 3 and 4."""

    def __init__(self, name: str, corpus: str) -> None:
        self.name, self.corpus = name, corpus

    def search_batch(self, queries, k=10):
        return [self._rank(q, k) for q in queries]

    def _rank(self, query: str, k: int) -> list[Hit]:
        if self.corpus == "nfcorpus":
            i = int(query.split()[1])
            relevant = [f"d{i}", f"d{i + 1}"]
        else:
            relevant = [next(ex.passage_id for ex in EXAMPLES if ex.question == query)]
        filler = [f"x{j}" for j in range(k)]
        ids = relevant + filler if self.name == "bm25" else filler[:2] + relevant + filler[2:]
        return [
            Hit(
                passage=Passage(passage_id=pid, doc_id=pid, text="", source="nfcorpus"),
                score=1.0 / rank,
                rank=rank,
                retriever=self.name,
            )
            for rank, pid in enumerate(ids[:k], start=1)
        ]


def _fake_build(name: str, corpus: str):
    if name == "dense-ft":
        raise EncoderNotFound("fine-tuned encoder not trained (stub)")
    return RankingRetriever(name, corpus)


@pytest.fixture
def rq1_stubs(monkeypatch):
    monkeypatch.setattr(rq1_retrieval, "load_nfcorpus", lambda split: (NF_QUERIES, NF_QRELS))
    monkeypatch.setattr(rq1_retrieval, "load_pubmedqa_labeled", lambda: list(EXAMPLES))
    monkeypatch.setattr(rq1_retrieval, "build_retriever", _fake_build)


def test_rq1_layout_metrics_and_paired_tests(tmp_path, rq1_stubs):
    systems = ["bm25", "dense-base", "dense-ft", "bm25+concepts"]
    path = rq1_retrieval.run(
        created="test",
        retrievers=systems,
        skip_unavailable=True,
        n_boot=50,
        n_perm=99,
        results_dir=tmp_path,
    )
    result = json.loads(path.read_text())
    assert set(result["meta"]["skipped"]) == {"dense-ft"}
    assert result["meta"]["n_queries"] == {"nfcorpus": 8, "pubmedqa": 6}

    nf = result["nfcorpus"]
    assert set(nf) == {"bm25", "dense-base", "bm25+concepts"}
    assert set(nf["bm25"]) == {"ndcg@10", "recall@100", "mrr@10"}
    assert nf["bm25"]["ndcg@10"] == {"mean": 1.0, "ci": [1.0, 1.0]}
    assert nf["dense-base"]["mrr@10"]["mean"] == pytest.approx(1 / 3)
    assert set(result["pubmedqa"]["bm25"]) == {"recall@1", "recall@5", "recall@20", "mrr@10"}
    assert result["pubmedqa"]["dense-base"]["recall@1"]["mean"] == 0.0

    tests = result["paired_tests"]["nfcorpus"]
    assert set(tests) == {"vs_bm25", "vs_dense-base"}
    row = tests["vs_bm25"]["dense-base"]
    assert row["mrr@10_diff"]["mean"] == pytest.approx(1 / 3 - 1)
    assert 0 < row["mrr@10_p"] <= row["mrr@10_p_holm"] <= 1
    assert len(result["per_query"]["nfcorpus"]["bm25"]["ndcg@10"]) == 8


def test_rq1_fails_loudly_when_a_system_is_unavailable(tmp_path, rq1_stubs):
    with pytest.raises(EncoderNotFound):
        rq1_retrieval.run(created="test", retrievers=["bm25", "dense-ft"], results_dir=tmp_path)
    assert not (tmp_path / "rq1_retrieval.json").exists()
