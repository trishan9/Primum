"""Checks on the real built corpora (`primum data build all && primum data pairs`).

Marked slow: they need the processed files on disk and are skipped when those are absent.
The expected counts are properties of the pinned upstream datasets (PubMedQA PQA-L,
BEIR NFCorpus), so a mismatch means the build is wrong, not that the data moved.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from primum.config import get_settings
from primum.data import (
    CorpusNotBuilt,
    load_concepts,
    load_contrastive_pairs,
    load_nfcorpus,
    load_passages,
    load_pubmedqa_labeled,
)
from primum.data._io import normalise_text

pytestmark = pytest.mark.slow


def _built(load):
    try:
        return load()
    except CorpusNotBuilt as error:
        pytest.skip(str(error))


def test_pqa_labeled_is_the_published_benchmark():
    examples = _built(load_pubmedqa_labeled)
    assert len(examples) == 1000
    assert len({e.qid for e in examples}) == 1000
    assert Counter(e.label for e in examples) == {"yes": 552, "no": 338, "maybe": 110}


@pytest.mark.parametrize("corpus", ["pubmedqa", "kb"])
def test_no_indexed_passage_states_a_pqa_labeled_conclusion(corpus):
    by_id = {p.passage_id: p for p in _built(lambda: load_passages(corpus))}
    assert len(by_id) == len(load_passages(corpus)), "passage ids must be unique"
    for example in load_pubmedqa_labeled():
        gold = by_id[example.passage_id]
        assert gold.title == ""
        assert gold.text == " ".join(c.strip() for c in example.contexts)
        assert normalise_text(example.long_answer) not in normalise_text(gold.text)


def test_nfcorpus_matches_beir_statistics():
    docs = _built(lambda: load_passages("nfcorpus"))
    assert len(docs) == 3633
    assert all(p.passage_id == p.doc_id for p in docs)
    doc_ids = {p.passage_id for p in docs}
    sizes = {}
    query_ids: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        queries, qrels = load_nfcorpus(split)
        assert queries.keys() == qrels.keys()
        assert all(judged.keys() <= doc_ids for judged in qrels.values())
        sizes[split] = (len(queries), sum(len(judged) for judged in qrels.values()))
        query_ids[split] = set(queries)
    assert sizes == {"train": (2590, 110575), "dev": (324, 11385), "test": (323, 12334)}
    assert not query_ids["train"] & (query_ids["dev"] | query_ids["test"])
    assert not query_ids["dev"] & query_ids["test"]


def test_contrastive_pairs_contain_nothing_that_is_evaluated():
    pairs = _built(load_contrastive_pairs)
    queries = {normalise_text(q) for q, _ in pairs}
    positives = {p for _, p in pairs}

    labeled = load_pubmedqa_labeled()
    gold_ids = {e.passage_id for e in labeled}
    gold_texts = {p.text for p in load_passages("pubmedqa") if p.passage_id in gold_ids}
    assert len(gold_texts) == 1000
    assert not positives & gold_texts, "a PQA-L abstract is a training positive"
    assert not queries & {normalise_text(e.question) for e in labeled}

    for split in ("dev", "test"):
        held_out, _ = load_nfcorpus(split)
        assert not queries & {normalise_text(text) for text in held_out.values()}


def test_kb_sources_and_concept_dictionary():
    passages = _built(lambda: load_passages("kb"))
    manifest = json.loads((get_settings().processed_dir / "kb" / "manifest.json").read_text())
    sources = Counter(p.source for p in passages)
    assert dict(sources) == manifest["by_source"]
    assert sources["medlineplus"] > 1000 and sources["pubmed"] >= 1000
    assert ("pdf" in sources) == bool(manifest["sources"]["pdf_files"])
    assert all(
        p.url.startswith("https://medlineplus.gov/") for p in passages if p.source == "medlineplus"
    )
    assert max(len(p.text.split()) for p in passages if p.source == "medlineplus") <= 220

    concepts = load_concepts()
    assert len(concepts) == manifest["n_concepts"] > 900
    assert all(set(c) == {"preferred", "synonyms", "mesh_ids", "url"} for c in concepts)
    assert all(c["preferred"] and c["preferred"] not in c["synonyms"] for c in concepts)
    by_name = {c["preferred"]: c for c in concepts}
    assert "Asthma" in by_name and by_name["Asthma"]["mesh_ids"]
