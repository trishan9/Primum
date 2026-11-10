"""Concept linker: longest match, normalisation, acronym handling and query expansion."""

from __future__ import annotations

import pytest

import primum.retrieval.concepts as concepts_module
from primum.retrieval.concepts import ConceptLinker, expand_query, link_concepts

DICTIONARY = [
    {
        "preferred": "Heart Attack",
        "synonyms": ["Myocardial Infarction", "MI"],
        "mesh_ids": ["D009203"],
        "url": "https://medlineplus.gov/heartattack.html",
    },
    {"preferred": "Cancer", "synonyms": ["Neoplasms", "Tumors"], "mesh_ids": ["D009369"]},
    {"preferred": "Breast Cancer", "synonyms": ["Mammary Carcinoma"], "mesh_ids": ["D001943"]},
    {"preferred": "Breast Cancer Screening", "synonyms": ["Mammography"], "mesh_ids": []},
    {"preferred": "Alzheimer's Disease", "synonyms": ["AD"], "mesh_ids": ["D000544"]},
    {"preferred": "HIV/AIDS", "synonyms": ["AIDS", "HIV"], "mesh_ids": ["D015658"]},
    {"preferred": "Migraine", "synonyms": [], "mesh_ids": ["D008881"]},
    {"preferred": "Hearing Aids", "synonyms": [], "mesh_ids": ["D006310"]},
    # "Tumors" is also a synonym of Cancer; as a preferred name it must win that conflict.
    {"preferred": "Tumors", "synonyms": [], "mesh_ids": []},
    # Names that must never be registered: stopwords and bare short tokens.
    {"preferred": "It", "synonyms": ["a", "to"], "mesh_ids": []},
]


@pytest.fixture
def linker() -> ConceptLinker:
    return ConceptLinker(DICTIONARY)


def preferred(concepts) -> list[str]:
    return [concept.preferred for concept in concepts]


def test_longest_match_wins(linker):
    text = "Is breast cancer screening useful, and does breast cancer differ from other cancer?"
    found = linker.link(text)
    assert preferred(found) == ["Breast Cancer Screening", "Breast Cancer", "Cancer"]
    assert found[0].surface == "breast cancer screening"


def test_spans_index_the_original_text(linker):
    text = "After a Myocardial  Infarction, migraines got worse."
    found = linker.link(text)
    assert preferred(found) == ["Heart Attack", "Migraine"]
    for concept in found:
        assert text[concept.start : concept.end] == concept.surface
    assert found[0].surface == "Myocardial  Infarction"  # internal whitespace is preserved
    assert found[0].mesh_ids == ["D009203"]
    assert found[0].synonyms == ["Myocardial Infarction", "MI"]
    assert found[0].url == "https://medlineplus.gov/heartattack.html"
    assert found[1].url is None


def test_plurals_and_possessives_are_normalised(linker):
    assert preferred(linker.link("heart attacks")) == ["Heart Attack"]
    assert preferred(linker.link("Alzheimer disease")) == ["Alzheimer's Disease"]
    assert preferred(linker.link("Alzheimer’s disease")) == ["Alzheimer's Disease"]
    assert preferred(linker.link("the patient's migraine")) == ["Migraine"]


def test_hyphen_and_slash_join_words_but_clause_boundaries_do_not(linker):
    assert preferred(linker.link("heart-attack risk")) == ["Heart Attack"]
    assert preferred(linker.link("living with HIV/AIDS")) == ["HIV/AIDS"]
    # "heart" ends one sentence and "attack" starts the next: not a heart attack.
    assert linker.link("It affects the heart. Attack rates were low.") == []
    assert preferred(linker.link("breast, cancer")) == ["Cancer"]


def test_acronyms_match_only_as_written(linker):
    assert preferred(linker.link("AIDS and MI are abbreviations")) == ["HIV/AIDS", "Heart Attack"]
    # Lower-case "aids" is a verb or a device, "mi" and "ad" are not words to link at all.
    assert linker.link("exercise aids recovery; mi ad") == []
    assert preferred(linker.link("she wears hearing aids")) == ["Hearing Aids"]


def test_preferred_name_beats_a_synonym_of_another_concept(linker):
    assert preferred(linker.link("benign tumors")) == ["Tumors"]
    assert preferred(linker.link("neoplasms")) == ["Cancer"]


def test_stopwords_and_single_letters_are_never_concepts(linker):
    assert linker.link("it is a route to recovery") == []


def test_no_concepts_in_empty_or_unrelated_text(linker):
    assert linker.link("") == []
    assert linker.link("the weather was fine") == []


def test_expand_appends_other_names_of_each_concept(linker):
    assert (
        linker.expand("aspirin after heart attack")
        == "aspirin after heart attack Myocardial Infarction MI"
    )
    # Linked through a synonym: the preferred name is added, the synonym already present is not.
    assert linker.expand("myocardial infarction outcomes") == (
        "myocardial infarction outcomes Heart Attack MI"
    )


def test_expand_leaves_a_query_without_concepts_unchanged(linker):
    assert linker.expand("does aspirin thin the blood") == "does aspirin thin the blood"


def test_expand_does_not_repeat_names_and_respects_the_cap(linker):
    # Mentioned twice, expanded once.
    assert linker.expand("migraine heart attack heart attack").count("Myocardial") == 1
    assert linker.expand("heart attack", max_synonyms=1) == "heart attack Myocardial Infarction"
    assert linker.expand("heart attack", max_synonyms=0) == "heart attack"
    # "Tumors" is already in the query (as "tumor"), so only "Neoplasms" is new.
    assert linker.expand("cancer tumor") == "cancer tumor Neoplasms"


def test_module_functions_use_the_built_dictionary(monkeypatch):
    linker = ConceptLinker(DICTIONARY)
    monkeypatch.setattr(concepts_module, "_default_linker", lambda: linker)
    assert preferred(link_concepts("a migraine")) == ["Migraine"]
    assert expand_query("heart attack") == "heart attack Myocardial Infarction MI"
