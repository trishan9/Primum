"""Corpus and pair builders, run end to end on small synthetic raw files.

The fixture writes raw files where the downloaders expect them and makes any network
call fail the test, so these tests exercise the real build code offline.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_data_text import TOPICS_XML, write_text_pdf

from primum.config import get_settings
from primum.data import (
    CorpusNotBuilt,
    LeakageError,
    SourceUnavailable,
    build_contrastive_pairs,
    build_corpus,
    corpora,
    load_concepts,
    load_contrastive_pairs,
    load_nfcorpus,
    load_passages,
    load_pubmedqa_labeled,
    pairs,
    sources,
)
from primum.data._io import normalise_text
from primum.data.pairs import MAX_PAIRS_PER_NFCORPUS_QUERY
from primum.types import DECISIONS

HttpxClient = httpx.Client  # the real class; the `raw` fixture replaces `httpx.Client`

N_UNLABELED = 30
N_ARTIFICIAL = 60
# Artificial rows that must never become training pairs: two reuse PQA-L PMIDs, one
# reuses a PQA-L question under a fresh PMID.
POISON_PMIDS = (5, 6)
POISON_QUESTION_PMID = 3999


def _labeled_row(i: int) -> dict:
    return {
        "pubid": i,
        "question": f"Does treatment {i} improve outcome {i}?",
        "context": {
            "contexts": [f"Background of study {i}.", f"Methods and results of study {i}."],
            "labels": ["BACKGROUND", "RESULTS"],
            "meshes": ["Humans"],
        },
        "long_answer": f"Treatment {i} conclusively changed outcome {i}.",
        "final_decision": DECISIONS[i % 3],
    }


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_pubmedqa(labeled: list[dict]) -> None:
    _write_parquet(sources.PQA_LABELED.path, labeled)
    unlabeled = [
        {
            "pubid": 2000 + i,
            "question": f"Is distractor {i} relevant?",
            "context": {
                "contexts": [f"Distractor abstract {i} body."],
                "labels": ["UNLABELLED"],
                "meshes": [],
            },
            "long_answer": f"Distractor {i} conclusion.",
        }
        for i in range(N_UNLABELED)
    ]
    _write_parquet(sources.PQA_UNLABELED.path, unlabeled)
    artificial = [
        {
            "pubid": 3000 + i,
            "question": f"Does artificial question {i} hold?",
            "context": {
                "contexts": [f"Artificial abstract {i} body.", f"More of abstract {i}."],
                "labels": ["BACKGROUND", "RESULTS"],
                "meshes": [],
            },
            "long_answer": f"Artificial conclusion {i}.",
            "final_decision": "yes",
        }
        for i in range(N_ARTIFICIAL)
    ]
    for pmid in POISON_PMIDS:
        artificial.append({**artificial[0], "pubid": pmid, "question": f"Poison by PMID {pmid}?"})
    artificial.append(
        {**artificial[0], "pubid": POISON_QUESTION_PMID, "question": labeled[10]["question"]}
    )
    _write_parquet(sources.PQA_ARTIFICIAL.path, artificial)


def _write_nfcorpus(qrels: dict[str, list[tuple[str, str, int]]]) -> None:
    docs = [
        {"_id": f"MED-{i}", "title": f"Title {i}", "text": f"Document {i} text."} for i in range(20)
    ]
    queries = [
        {"_id": "PLAIN-1", "title": "", "text": "many relevant documents"},
        {"_id": "PLAIN-2", "title": "", "text": "few relevant documents"},
        {"_id": "PLAIN-3", "title": "", "text": "Held-out Query!"},  # same text as PLAIN-21
        {"_id": "PLAIN-11", "title": "", "text": "dev query"},
        {"_id": "PLAIN-21", "title": "", "text": "held out query"},
    ]
    _write_parquet(sources.NFCORPUS_DOCS.path, docs)
    _write_parquet(sources.NFCORPUS_QUERIES.path, queries)
    for split, rows in qrels.items():
        path = sources.NFCORPUS_QRELS[split].path
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["query-id\tcorpus-id\tscore"] + [f"{q}\t{d}\t{g}" for q, d, g in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


NF_QRELS = {
    "train": [("PLAIN-1", f"MED-{i}", 1) for i in range(15)]
    + [("PLAIN-2", "MED-0", 1), ("PLAIN-2", "MED-1", 1), ("PLAIN-3", "MED-2", 1)],
    "dev": [("PLAIN-11", "MED-3", 2), ("PLAIN-11", "MED-4", 1)],
    "test": [("PLAIN-21", "MED-5", 2), ("PLAIN-21", "MED-6", 1), ("PLAIN-21", "MED-7", 1)],
}


@pytest.fixture
def raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data directory holding synthetic raw files; returns its `raw/` sub-directory."""
    monkeypatch.setenv("PRIMUM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("PRIMUM_INCLUDE_PDF", raising=False)
    get_settings.cache_clear()
    corpora._read_passages.cache_clear()

    def no_network(*args, **kwargs):
        raise AssertionError("fast tests must not touch the network")

    monkeypatch.setattr(sources, "hf_hub_download", no_network)
    monkeypatch.setattr(httpx, "Client", no_network)

    _write_pubmedqa([_labeled_row(i) for i in range(1, 1001)])
    _write_nfcorpus(NF_QRELS)
    snapshot = get_settings().raw_dir / "medlineplus" / "mplus_topics_compressed_2025-01-01.zip"
    snapshot.parent.mkdir(parents=True)
    with zipfile.ZipFile(snapshot, "w") as z:
        z.writestr("mplus_topics_2025-01-01.xml", TOPICS_XML)

    yield get_settings().raw_dir
    get_settings.cache_clear()
    corpora._read_passages.cache_clear()


# --- pubmedqa --------------------------------------------------------------------------


def test_pubmedqa_corpus_and_labeled_set(raw: Path):
    directory = build_corpus("pubmedqa")
    assert directory == get_settings().processed_dir / "pubmedqa"

    examples = load_pubmedqa_labeled()
    assert len(examples) == 1000
    assert [e.qid for e in examples] == [str(i) for i in range(1, 1001)]  # stable source order
    assert {e.label for e in examples} == set(DECISIONS)
    assert examples[0].contexts == ["Background of study 1.", "Methods and results of study 1."]
    assert examples[0].context_labels == ["BACKGROUND", "RESULTS"]

    passages = load_passages("pubmedqa")
    assert len(passages) == 1000 + N_UNLABELED
    by_id = {p.passage_id: p for p in passages}
    assert len(by_id) == len(passages)
    gold = by_id[examples[0].passage_id]
    assert gold.passage_id == "pubmed-1" and gold.doc_id == "1"
    assert gold.text == "Background of study 1. Methods and results of study 1."
    assert gold.source == "pubmed" and gold.url == "https://pubmed.ncbi.nlm.nih.gov/1/"

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["n_labeled"] == 1000 and manifest["n_distractors"] == N_UNLABELED
    assert sum(manifest["label_distribution"].values()) == 1000


def test_pubmedqa_passages_never_contain_the_conclusion_or_the_question(raw: Path):
    """The leak this corpus must not have: `long_answer` states the label, the title is the question."""
    build_corpus("pubmedqa")
    indexed = {
        p.passage_id: normalise_text(f"{p.title} {p.text}") for p in load_passages("pubmedqa")
    }
    everything = " ".join(indexed.values())
    for example in load_pubmedqa_labeled():
        assert normalise_text(example.long_answer) not in everything
        assert normalise_text(example.question) not in indexed[example.passage_id]
    assert all(p.title == "" for p in load_passages("pubmedqa"))


def test_pubmedqa_build_refuses_a_corpus_that_leaks_the_conclusion(raw: Path):
    rows = [_labeled_row(i) for i in range(1, 1001)]
    rows[7]["context"]["contexts"].append(rows[7]["long_answer"].upper())
    _write_pubmedqa(rows)
    with pytest.raises(LeakageError, match="PMID 8"):
        build_corpus("pubmedqa")
    with pytest.raises(CorpusNotBuilt):
        load_passages("pubmedqa")


def test_pubmedqa_build_requires_exactly_the_1000_labelled_items(raw: Path):
    _write_pubmedqa([_labeled_row(i) for i in range(1, 1000)])
    with pytest.raises(SourceUnavailable, match="1000"):
        build_corpus("pubmedqa")

    rows = [_labeled_row(i) for i in range(1, 1001)]
    rows[3]["final_decision"] = "unsure"
    _write_pubmedqa(rows)
    with pytest.raises(SourceUnavailable, match="unsure"):
        build_corpus("pubmedqa")


def test_max_distractors_is_a_seeded_sample(raw: Path):
    build_corpus("pubmedqa", max_distractors=10)
    first = [p.passage_id for p in load_passages("pubmedqa")]
    assert len(first) == 1010
    assert first[:1000] == [f"pubmed-{i}" for i in range(1, 1001)]
    build_corpus("pubmedqa", max_distractors=10, force=True)
    assert [p.passage_id for p in load_passages("pubmedqa")] == first


def test_build_is_idempotent_until_forced(raw: Path, monkeypatch: pytest.MonkeyPatch):
    directory = build_corpus("pubmedqa", max_distractors=5)
    calls = []
    real_build = corpora.pubmedqa.build
    monkeypatch.setattr(
        corpora.pubmedqa, "build", lambda *a, **k: calls.append(k) or real_build(*a, **k)
    )
    # Same request, and a request for a different size: both keep the existing corpus.
    assert build_corpus("pubmedqa", max_distractors=5) == directory
    assert build_corpus("pubmedqa") == directory
    assert calls == [] and len(load_passages("pubmedqa")) == 1005

    build_corpus("pubmedqa", force=True)
    assert calls == [{"max_distractors": None}]
    assert len(load_passages("pubmedqa")) == 1000 + N_UNLABELED  # cache was invalidated


def test_interrupted_build_is_not_mistaken_for_complete(raw: Path, monkeypatch: pytest.MonkeyPatch):
    directory = build_corpus("nfcorpus")

    def crash(directory: Path) -> dict:
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(corpora.nfcorpus, "build", crash)
        with pytest.raises(RuntimeError, match="interrupted"):
            build_corpus("nfcorpus", force=True)
    assert not (directory / "manifest.json").exists()

    # With no manifest the next plain call rebuilds instead of trusting what is on disk.
    build_corpus("nfcorpus")
    assert (directory / "manifest.json").exists()


def test_loaders_fail_specifically_before_a_build(raw: Path):
    for load in (
        lambda: load_passages("kb"),
        load_pubmedqa_labeled,
        lambda: load_nfcorpus("test"),
        load_concepts,
        load_contrastive_pairs,
    ):
        with pytest.raises(CorpusNotBuilt, match="primum data"):
            load()
    assert issubclass(CorpusNotBuilt, FileNotFoundError)
    with pytest.raises(ValueError):
        build_corpus("mimic")  # type: ignore[arg-type]


def test_load_passages_returns_an_independent_list(raw: Path):
    build_corpus("nfcorpus")
    first = load_passages("nfcorpus")
    first.clear()
    assert len(load_passages("nfcorpus")) == 20


# --- nfcorpus --------------------------------------------------------------------------


def test_nfcorpus_keeps_beir_ids_splits_and_grades(raw: Path):
    build_corpus("nfcorpus")
    passages = load_passages("nfcorpus")
    assert len(passages) == 20
    assert all(p.passage_id == p.doc_id and p.source == "nfcorpus" for p in passages)
    assert passages[3].passage_id == "MED-3" and passages[3].title == "Title 3"

    queries, qrels = load_nfcorpus("test")
    assert queries == {"PLAIN-21": "held out query"}
    assert qrels == {"PLAIN-21": {"MED-5": 2, "MED-6": 1, "MED-7": 1}}
    train_queries, train_qrels = load_nfcorpus("train")
    assert set(train_queries) == set(train_qrels) == {"PLAIN-1", "PLAIN-2", "PLAIN-3"}
    assert len(train_qrels["PLAIN-1"]) == 15
    with pytest.raises(ValueError):
        load_nfcorpus("validation")  # type: ignore[arg-type]


def test_nfcorpus_rejects_a_query_shared_between_splits(raw: Path):
    _write_nfcorpus({**NF_QRELS, "test": NF_QRELS["test"] + [("PLAIN-1", "MED-9", 1)]})
    with pytest.raises(LeakageError, match="PLAIN-1"):
        build_corpus("nfcorpus")


# --- kb --------------------------------------------------------------------------------


def test_kb_is_medlineplus_plus_pubmed_and_excludes_pdfs_by_default(raw: Path):
    (raw / "pdf").mkdir()
    write_text_pdf(raw / "pdf" / "book.pdf", [["A copyrighted sentence about medicine."]])
    build_corpus("kb", max_distractors=3)

    passages = load_passages("kb")
    assert {p.source for p in passages} == {"medlineplus", "pubmed"}
    assert [p.passage_id for p in passages if p.source == "pubmed"] == [
        p.passage_id for p in load_passages("pubmedqa")
    ]
    (topic,) = [p for p in passages if p.source == "medlineplus"]
    assert topic.text == "Asthma narrows the airways." and topic.title == "Asthma"

    assert load_concepts() == [
        {
            "preferred": "Asthma",
            "synonyms": ["Bronchial asthma", "Reactive Airway Disease"],
            "mesh_ids": ["D001249"],
            "url": "https://medlineplus.gov/asthma.html",
        },
        {
            "preferred": "Empty Topic",
            "synonyms": [],
            "mesh_ids": [],
            "url": "https://medlineplus.gov/empty.html",
        },
    ]


def test_kb_includes_local_pdfs_only_when_opted_in(raw: Path, monkeypatch: pytest.MonkeyPatch):
    (raw / "pdf").mkdir()
    write_text_pdf(raw / "pdf" / "book.pdf", [["A sentence from a local handbook."]])
    monkeypatch.setenv("PRIMUM_INCLUDE_PDF", "1")
    directory = build_corpus("kb", max_distractors=0)
    pdf = [p for p in load_passages("kb") if p.source == "pdf"]
    assert [p.text for p in pdf] == ["A sentence from a local handbook."]
    assert pdf[0].meta["page"] == 1
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["params"]["include_pdf"] is True
    assert manifest["sources"]["pdf_files"] == ["book.pdf"]
    assert manifest["by_source"] == {"medlineplus": 1, "pubmed": 1000, "pdf": 1}


# --- contrastive pairs -----------------------------------------------------------------


def test_contrastive_pairs_exclude_everything_that_is_evaluated(raw: Path):
    path = build_contrastive_pairs(n_pubmedqa=40)
    assert path == get_settings().processed_dir / "pairs" / "train.jsonl"
    assert set(json.loads(path.read_text().splitlines()[0])) == {"query", "positive"}
    built = load_contrastive_pairs()
    queries = {normalise_text(q) for q, _ in built}
    positives = {p for _, p in built}

    # PQA-L: no labelled abstract is a positive, no labelled question is a query — including
    # the artificial rows that reuse a labelled PMID or a labelled question.
    build_corpus("pubmedqa")
    labeled = load_pubmedqa_labeled()
    gold_texts = {p.text for p in load_passages("pubmedqa")[: len(labeled)]}
    assert not positives & gold_texts
    assert not queries & {normalise_text(e.question) for e in labeled}
    assert not any(q.startswith("poison by pmid") for q in queries)

    # NFCorpus: dev/test queries are absent, also when a train query repeats their text.
    assert "held out query" not in queries and "dev query" not in queries
    nf = [(q, p) for q, p in built if not q.endswith("?")]
    assert {q for q, _ in nf} == {"many relevant documents", "few relevant documents"}
    assert sum(q == "many relevant documents" for q, _ in nf) == MAX_PAIRS_PER_NFCORPUS_QUERY
    assert ("few relevant documents", "Title 0 Document 0 text.") in nf

    pubmed = [(q, p) for q, p in built if q.endswith("?")]
    assert len(pubmed) == 40
    assert all(p.startswith("Artificial abstract") and "conclusion" not in p for _, p in pubmed)
    assert len(set(built)) == len(built)


def test_contrastive_pairs_are_reproducible_and_idempotent(raw: Path):
    path = build_contrastive_pairs(n_pubmedqa=25)
    first = path.read_bytes()
    stamp = path.stat().st_mtime_ns
    build_contrastive_pairs(n_pubmedqa=25)
    assert path.stat().st_mtime_ns == stamp
    build_contrastive_pairs(n_pubmedqa=25, force=True)
    assert path.read_bytes() == first
    # Asking for more pairs than exist returns every eligible one, still without the poison.
    build_contrastive_pairs(n_pubmedqa=10_000, force=True)
    assert sum(q.endswith("?") for q, _ in load_contrastive_pairs()) == N_ARTIFICIAL


def test_contrastive_pairs_abort_if_a_labelled_pmid_slips_through(
    raw: Path, monkeypatch: pytest.MonkeyPatch
):
    """The final check is independent of the sampling code it guards."""
    monkeypatch.setattr(
        pairs, "_pubmedqa_pairs", lambda n, seed: ([("q?", "leaked abstract")], {5, 3001})
    )
    with pytest.raises(LeakageError, match="PQA-L"):
        build_contrastive_pairs(n_pubmedqa=1)
    with pytest.raises(CorpusNotBuilt):
        load_contrastive_pairs()


# --- sources ---------------------------------------------------------------------------


def _mock_medlineplus(monkeypatch: pytest.MonkeyPatch, page: str, requested: list[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.path.endswith(".zip"):
            return httpx.Response(200, content=b"zip-bytes")
        return httpx.Response(200, text=page)

    monkeypatch.setattr(
        sources.httpx,
        "Client",
        lambda **kwargs: HttpxClient(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_medlineplus_download_takes_the_newest_snapshot_from_the_official_host(
    raw: Path, monkeypatch: pytest.MonkeyPatch
):
    for old in (raw / "medlineplus").iterdir():
        old.unlink()
    page = """
      <a href="https://medlineplus.gov/xml/mplus_topics_compressed_2025-03-01.zip">old</a>
      <a href="https://medlineplus.gov/xml/mplus_topics_compressed_2025-03-04.zip">new</a>
      <a href="https://evil.example/xml/mplus_topics_compressed_2099-01-01.zip">not NLM</a>
      <a href="https://medlineplus.gov/xml/mplus_topics_2025-03-04.xml">uncompressed</a>
    """
    requested: list[str] = []
    _mock_medlineplus(monkeypatch, page, requested)

    path = sources.fetch_medlineplus_topics()
    assert path.name == "mplus_topics_compressed_2025-03-04.zip"
    assert path.read_bytes() == b"zip-bytes"
    assert requested == [
        sources.MEDLINEPLUS_INDEX_URL,
        "https://medlineplus.gov/xml/mplus_topics_compressed_2025-03-04.zip",
    ]
    # A snapshot on disk is reused: the corpus must not change under a re-run.
    assert sources.fetch_medlineplus_topics() == path
    assert len(requested) == 2


def test_medlineplus_download_fails_clearly_when_the_page_has_no_link(
    raw: Path, monkeypatch: pytest.MonkeyPatch
):
    for old in (raw / "medlineplus").iterdir():
        old.unlink()
    _mock_medlineplus(monkeypatch, "<html>redesigned</html>", [])
    with pytest.raises(SourceUnavailable, match="zip link"):
        sources.fetch_medlineplus_topics()
