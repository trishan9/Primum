"""BM25: tokenizer behaviour, the ranking function against a hand-computed example, caching."""

from __future__ import annotations

import math

import numpy as np
import pytest

import primum.retrieval.bm25 as bm25_module
from primum.retrieval.bm25 import BM25, BM25Index, stem, tokenize
from primum.types import Passage


def passage(pid: str, text: str, title: str = "") -> Passage:
    return Passage(passage_id=pid, doc_id=pid, title=title, text=text, source="pubmed")


# --- tokenizer ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("Does Aspirin reduce the risk of stroke?") == [
        "aspirin",
        "reduce",
        "risk",
        "stroke",
    ]


def test_tokenize_keeps_hyphenated_biomedical_terms_whole_and_in_parts():
    assert tokenize("IL-6 in COVID-19") == ["il-6", "il", "6", "covid-19", "covid", "19"]
    # A stopword inside a compound is dropped as a part but survives inside the compound.
    assert tokenize("vitamin-a") == ["vitamin-a", "vitamin"]


def test_tokenize_folds_diacritics_and_greek_letters():
    assert tokenize("Sjögren") == ["sjogren"]
    assert tokenize("TNF-α") == tokenize("tnf-alpha") == ["tnf-alpha", "tnf", "alpha"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Regression: letters missing from the map used to vanish entirely.
        ("σ-receptor", ["sigma-receptor", "sigma", "receptor"]),
        ("τ protein", ["tau", "protein"]),
        ("PKCθ", ["pkctheta"]),
        ("Σ", ["sigma"]),  # uppercase, via lowercasing
        ("ϑ", ["theta"]),  # symbol variant, via NFKD
    ],
)
def test_tokenize_spells_out_the_whole_greek_alphabet(text, expected):
    assert tokenize(text) == expected


def test_tokenize_drops_the_possessive_s():
    # Regression: "Crohn's" used to emit a term "s" shared with every other possessive.
    assert tokenize("Crohn's disease") == tokenize("Crohn’s disease") == ["crohn", "disease"]
    assert tokenize("Crohn's") == tokenize("Crohn")
    scores = BM25(["Crohn disease of the bowel", "the patient's knee"]).score(["Crohn's"])[0]
    assert scores[0] > 0 and scores[1] == 0


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("therapies", "therapy"),
        ("diseases", "disease"),
        ("tumors", "tumor"),
        ("virus", "virus"),  # -us is a Latin singular
        ("abscess", "abscess"),  # -ss is not a plural
        ("diagnosis", "diagnosis"),  # -is is a Greek singular
        ("gas", "gas"),  # too short to be a stripped plural
        ("beta-blockers", "beta-blocker"),
    ],
)
def test_s_stemmer_folds_only_regular_plurals(word, expected):
    assert stem(word) == expected


# --- ranking function --------------------------------------------------------------------

# Three documents chosen so every quantity can be worked out on paper.
#   d0 = aspirin aspirin fever   |d0| = 3
#   d1 = fever cough             |d1| = 2   ("and" is a stopword)
#   d2 = cough                   |d2| = 1
# N = 3, avgdl = 2, df(aspirin) = 1, df(fever) = df(cough) = 2.
DOCS = ["aspirin aspirin fever", "fever and cough", "cough"]
K1, B = 1.2, 0.75
IDF_ASPIRIN = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))  # ln(8/3)
IDF_FEVER = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))  # ln(1.6)
# Length normalisation k1 * (1 - b + b * |d| / avgdl):
NORM_D0 = 1.2 * (0.25 + 0.75 * 3 / 2)  # 1.65
NORM_D1 = 1.2 * (0.25 + 0.75 * 2 / 2)  # 1.20


def test_idf_matches_lucene_formula():
    model = BM25(DOCS)
    assert model.idf[model.vocabulary["aspirin"]] == pytest.approx(IDF_ASPIRIN)
    assert model.idf[model.vocabulary["fever"]] == pytest.approx(IDF_FEVER)
    assert model.idf[model.vocabulary["cough"]] == pytest.approx(IDF_FEVER)
    assert model.avgdl == pytest.approx(2.0)


def test_scores_match_hand_computation():
    scores = BM25(DOCS, k1=K1, b=B).score(["aspirin fever"])[0]
    expected_d0 = IDF_ASPIRIN * 2 * 2.2 / (2 + NORM_D0) + IDF_FEVER * 1 * 2.2 / (1 + NORM_D0)
    expected_d1 = IDF_FEVER * 1 * 2.2 / (1 + NORM_D1)  # = ln(1.6) exactly, since 2.2 / 2.2
    assert NORM_D0 == pytest.approx(1.65)
    assert expected_d1 == pytest.approx(math.log(1.6))
    np.testing.assert_allclose(scores, [expected_d0, expected_d1, 0.0], rtol=1e-6)


def test_repeated_query_term_counts_twice_and_unknown_terms_are_ignored():
    model = BM25(DOCS)
    once = model.score(["fever"])[0]
    twice = model.score(["fever fever zebrafish"])[0]
    np.testing.assert_allclose(twice, 2 * once, rtol=1e-6)


def test_term_frequency_saturates_and_long_documents_are_penalised():
    # Same length, so only tf differs: more occurrences score higher, but sub-linearly.
    model = BM25(["statin x x x", "statin statin x x", "statin statin statin statin", "y y y y"])
    one, two, four, _ = model.score(["statin"])[0]
    assert one < two < four
    assert four < 4 * one
    # Same tf, different length: the shorter document wins.
    short, long = BM25(["statin trial", "statin trial of many other unrelated words here"]).score(
        ["statin"]
    )[0]
    assert short > long


def test_empty_collection_is_rejected():
    with pytest.raises(ValueError, match="at least one document"):
        BM25([])


def test_batch_scoring_equals_single_scoring_and_handles_empty_query():
    model = BM25(DOCS)
    batch = model.score(["aspirin", "", "cough fever"])
    assert batch.shape == (3, 3)
    np.testing.assert_allclose(batch[0], model.score(["aspirin"])[0])
    np.testing.assert_array_equal(batch[1], 0.0)


# --- retriever ---------------------------------------------------------------------------

PASSAGES = [
    passage("p0", "aspirin aspirin fever"),
    passage("p1", "fever and cough"),
    passage("p2", "cough"),
    passage("p3", "cast immobilisation and healing", title="Aspirin fracture"),
]


def test_search_returns_ranked_hits_with_components():
    index = BM25Index(PASSAGES)
    hits = index.search("aspirin fever", k=10)
    # p2 shares no term with the query and is not returned at all. p1 and p3 each match one
    # query term of equal idf once; p1 is shorter, so it ranks above p3.
    assert [h.passage.passage_id for h in hits] == ["p0", "p1", "p3"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert all(h.retriever == "bm25" for h in hits)
    assert hits[0].components == {"bm25": hits[0].score, "bm25_rank": 1.0}
    assert hits[0].score > hits[1].score > hits[2].score > 0


def test_title_is_indexed():
    hits = BM25Index(PASSAGES).search("fracture aspirin", k=1)
    assert hits[0].passage.passage_id == "p3"


def test_search_batch_matches_search_and_respects_k():
    index = BM25Index(PASSAGES)
    queries = ["aspirin fever", "cough", "nothing matches this"]
    batch = index.search_batch(queries, k=2)
    assert [len(hits) for hits in batch] == [2, 2, 0]
    for query, hits in zip(queries, batch, strict=True):
        assert hits == index.search(query, k=2)


def test_named_corpus_index_is_cached_and_invalidated_by_corpus_change(tmp_path, monkeypatch):
    corpus = list(PASSAGES)
    monkeypatch.setattr(bm25_module, "resolve_corpus", lambda name: (corpus, tmp_path / name))

    built = BM25Index("toy")
    assert (tmp_path / "toy" / "bm25" / "meta.json").exists()

    # A second construction must load from disk, never refit.
    with monkeypatch.context() as patch:
        patch.setattr(BM25, "__init__", lambda *a, **k: pytest.fail("index was rebuilt"))
        loaded = BM25Index("toy")
    np.testing.assert_allclose(
        loaded.score_all(["aspirin fever"]), built.score_all(["aspirin fever"])
    )
    assert loaded.search("cough", k=3) == built.search("cough", k=3)

    # Changing the corpus changes the fingerprint: the stale index is replaced, not reused.
    corpus.append(passage("p4", "zebrafish model of fever"))
    rebuilt = BM25Index("toy")
    assert rebuilt.score_all(["zebrafish"]).shape == (1, 5)
    assert rebuilt.search("zebrafish", k=1)[0].passage.passage_id == "p4"

    # So does changing a ranking parameter: b = 0 switches length normalisation off.
    unnormalised = BM25Index("toy", b=0.0)
    assert not np.allclose(unnormalised.score_all(["fever"]), rebuilt.score_all(["fever"]))
