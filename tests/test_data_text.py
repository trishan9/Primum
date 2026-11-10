"""Text processing of the data layer: chunking, MedlinePlus HTML/XML, PDF pages. No I/O beyond tmp files."""

from __future__ import annotations

import random
import zipfile
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from primum.data import SourceUnavailable, chunk_text
from primum.data.medlineplus import (
    Topic,
    iter_topics,
    summary_sections,
    topic_concept,
    topic_passages,
)
from primum.data.pdf import clean_page_text, pdf_passages
from primum.sentences import split_sentences

# --- chunk_text ------------------------------------------------------------------------
# Sentence splitting itself is tested with the verifier (`test_verification.py`), which owns
# the one splitter; these tests check that chunks are built from its sentences.


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        # A single capital or a dotted abbreviation can end a sentence in clinical text.
        ("Patients with hepatitis B. Controls were healthy.", 2),
        ("Take vitamin D. It helps bones.", 2),
        ("Doses were given i.v. Patients improved.", 2),
        # ... while abbreviations and decimals inside a sentence do not end it.
        ("Risk fell (p = 0.05 vs. placebo) in adults.", 1),
        ("NSAIDs, e.g. ibuprofen, were allowed and 4.2% stopped.", 1),
    ],
)
def test_chunk_boundaries_are_the_verifiers_sentence_boundaries(text, sentences):
    """With room for one sentence per chunk, each chunk is one of the verifier's premises."""
    longest = max(len(s.split()) for s in split_sentences(text))
    chunks = chunk_text(text, max_words=longest, overlap=0)
    assert chunks == split_sentences(text)
    assert len(chunks) == sentences


def test_chunk_text_empty_and_short():
    assert chunk_text("") == []
    assert chunk_text("   \n ") == []
    assert chunk_text("One short sentence. And another.") == ["One short sentence. And another."]


def test_chunk_text_rejects_bad_arguments():
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=10, overlap=-1)


def test_chunk_text_overlaps_by_whole_sentences():
    sentences = [f"Sentence {i} has exactly six words." for i in range(10)]
    chunks = chunk_text(" ".join(sentences), max_words=24, overlap=6)
    # 24 words = 4 sentences; each later chunk repeats exactly the previous chunk's last one.
    assert chunks[0] == " ".join(sentences[:4])
    assert chunks[1] == " ".join(sentences[3:7])
    assert chunks[2] == " ".join(sentences[6:10])


def test_chunk_text_drops_overlap_rather_than_exceed_limit():
    a, b, c = "A a a a a a.", "B b b b.", "C c c c c c c c."
    # After [a, b] the tail "b" (4 words) fits the overlap budget but 4 + 8 > 10.
    assert chunk_text(f"{a} {b} {c}", max_words=10, overlap=5) == [f"{a} {b}", c]


def test_chunk_text_splits_a_sentence_longer_than_the_limit():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words) + ".", max_words=10, overlap=3)
    assert [len(c.split()) for c in chunks] == [10, 10, 5]
    assert " ".join(chunks) == " ".join(words) + "."


@pytest.mark.parametrize("seed", range(20))
def test_chunk_text_invariants_on_random_documents(seed):
    """Every chunk respects the limit; chunks cover the text in order; overlap is bounded."""
    rng = random.Random(seed)
    max_words = rng.randint(5, 60)
    overlap = rng.randint(0, max_words - 1)
    counter = iter(range(10**6))
    # Unique tokens make it unambiguous which words of a chunk repeat the previous one.
    sentences = [
        " ".join(f"T{next(counter)}" for _ in range(rng.randint(1, 80))) + "."
        for _ in range(rng.randint(1, 40))
    ]
    document = " ".join(sentences)
    chunks = chunk_text(document, max_words=max_words, overlap=overlap)

    seen: set[str] = set()
    rebuilt: list[str] = []
    for chunk in chunks:
        words = chunk.split()
        assert 0 < len(words) <= max_words
        repeated = [w for w in words if w in seen]
        fresh = [w for w in words if w not in seen]
        assert fresh, "every chunk must contribute new text"
        assert len(repeated) <= overlap
        assert words == repeated + fresh, "repeated words form a prefix of the chunk"
        seen.update(fresh)
        rebuilt.extend(fresh)
    assert rebuilt == document.split()


# --- MedlinePlus -----------------------------------------------------------------------

SUMMARY = """
<p>Asthma is a chronic (long-term) lung disease. It affects your airways &amp; breathing.</p>
<h3>What are the symptoms of asthma?</h3>
<p>The symptoms of asthma include:</p><ul>
<li>Chest tightness</li>
<li><a href="https://medlineplus.gov/cough.html">Coughing</a>, especially at night.</li>
<li>Wheezing</li>
</ul>
<p>Symptoms can be mild<br>or severe.</p>
<h3>What are the treatments for asthma?</h3>
<ul><li>Quick-relief medicines</li><li>Control medicines</li></ul>
<p class="">NIH: National Heart, Lung, and Blood Institute</p>
"""


def test_summary_sections_structure():
    sections = summary_sections(SUMMARY)
    assert [heading for heading, _ in sections] == [
        "",
        "What are the symptoms of asthma?",
        "What are the treatments for asthma?",
    ]
    assert sections[0][1] == (
        "Asthma is a chronic (long-term) lung disease. It affects your airways & breathing."
    )
    # The list is attached to its lead-in sentence; link text is kept, markup is not.
    assert sections[1][1] == (
        "The symptoms of asthma include: Chest tightness; Coughing, especially at night; Wheezing. "
        "Symptoms can be mild or severe."
    )
    # A list with no lead-in stands alone, and the institute credit line is not content.
    assert sections[2][1] == "Quick-relief medicines; Control medicines."
    assert "NIH" not in " ".join(body for _, body in sections)


def test_summary_sections_empty():
    assert summary_sections("") == []
    assert summary_sections("<p>  </p><h3>Heading with no body</h3>") == []


def _topic(**overrides) -> Topic:
    fields = {
        "topic_id": "42",
        "title": "Asthma",
        "url": "https://medlineplus.gov/asthma.html",
        "summary_html": SUMMARY,
        "also_called": ["Bronchial asthma", "asthma"],
        "see_references": ["Reactive Airway Disease", "bronchial asthma"],
        "mesh_ids": ["D001249"],
        "groups": ["Lungs and Breathing"],
        "institute": "National Heart, Lung, and Blood Institute",
    }
    return Topic(**{**fields, **overrides})


def test_topic_passages_carry_heading_provenance_and_respect_budget():
    passages = topic_passages(_topic())
    assert [p.passage_id for p in passages] == ["mplus-42-0", "mplus-42-1", "mplus-42-2"]
    assert {p.doc_id for p in passages} == {"mplus-42"}
    assert all(p.source == "medlineplus" and p.title == "Asthma" for p in passages)
    assert all(p.url == "https://medlineplus.gov/asthma.html" for p in passages)
    assert passages[1].text.startswith("What are the symptoms of asthma? The symptoms")
    assert passages[1].meta["section"] == "What are the symptoms of asthma?"
    assert passages[1].meta["mesh_ids"] == ["D001249"]

    long_body = " ".join(f"Sentence number {i} is here." for i in range(200))
    long_topic = _topic(summary_html=f"<h3>What is a long section?</h3><p>{long_body}</p>")
    chunks = topic_passages(long_topic, max_words=50, overlap=10)
    assert len(chunks) > 1
    assert all(len(p.text.split()) <= 50 for p in chunks)
    assert all(p.text.startswith("What is a long section?") for p in chunks)
    assert len({p.passage_id for p in chunks}) == len(chunks)


def test_topic_concept_deduplicates_synonyms_case_insensitively():
    assert topic_concept(_topic()) == {
        "preferred": "Asthma",
        "synonyms": ["Bronchial asthma", "Reactive Airway Disease"],
        "mesh_ids": ["D001249"],
        "url": "https://medlineplus.gov/asthma.html",
    }


TOPICS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<health-topics total="3" date-generated="01/01/2025 00:00:00">
<health-topic meta-desc="d" title="Asthma" url="https://medlineplus.gov/asthma.html" id="42" language="English" date-created="01/01/2001">
<also-called>Bronchial asthma</also-called>
<full-summary>&lt;p&gt;Asthma narrows the airways.&lt;/p&gt;</full-summary>
<group url="u" id="1">Lungs and Breathing</group>
<mesh-heading><descriptor id="D001249">Asthma</descriptor></mesh-heading>
<primary-institute url="u">National Heart, Lung, and Blood Institute</primary-institute>
<see-reference>Reactive Airway Disease</see-reference>
<site title="t" url="u"><information-category>c</information-category></site>
</health-topic>
<health-topic meta-desc="d" title="Asma" url="https://medlineplus.gov/spanish/asthma.html" id="43" language="Spanish" date-created="01/01/2001">
<full-summary>&lt;p&gt;El asma.&lt;/p&gt;</full-summary>
</health-topic>
<health-topic meta-desc="d" title="Empty Topic" url="https://medlineplus.gov/empty.html" id="44" language="English" date-created="01/01/2001">
<full-summary></full-summary>
</health-topic>
</health-topics>
"""


def test_iter_topics_reads_english_topics_from_the_zip(tmp_path: Path):
    archive = tmp_path / "mplus_topics_compressed_2025-01-01.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("mplus_topics_2025-01-01.xml", TOPICS_XML)
    topics = list(iter_topics(archive))
    assert [t.title for t in topics] == ["Asthma", "Empty Topic"]
    asthma = topics[0]
    assert asthma.topic_id == "42"
    assert asthma.also_called == ["Bronchial asthma"]
    assert asthma.see_references == ["Reactive Airway Disease"]
    assert asthma.mesh_ids == ["D001249"]
    assert asthma.groups == ["Lungs and Breathing"]
    assert asthma.summary_html == "<p>Asthma narrows the airways.</p>"
    assert topic_passages(topics[1]) == []


def test_iter_topics_rejects_unexpected_archive_layout(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("a.xml", TOPICS_XML)
        z.writestr("b.xml", TOPICS_XML)
    with pytest.raises(SourceUnavailable):
        list(iter_topics(archive))


# --- PDF -------------------------------------------------------------------------------


def write_text_pdf(path: Path, pages: list[list[str]]) -> None:
    """Write a minimal text PDF: one entry of `pages` per page, one string per line."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        shows = " T* ".join(f"({line}) Tj" for line in lines)
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 11 Tf 14 TL 72 720 Td {shows} ET".encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as f:
        writer.write(f)


def test_clean_page_text_rejoins_hyphenation_and_folds_bullets():
    raw = "The symp-\ntoms of poisoning include:\n• headache\n• dizziness\nSeek help at once."
    assert clean_page_text(raw) == (
        "The symptoms of poisoning include: headache; dizziness Seek help at once."
    )
    assert clean_page_text("   \n ") == ""


def test_pdf_passages_cite_their_page(tmp_path: Path):
    path = tmp_path / "Tiny_Medical Handbook.pdf"
    write_text_pdf(
        path,
        [
            ["Carbon monoxide binds to hemo-", "globin. It displaces oxygen."],
            [],  # a page with no text yields no passage
            [f"Sentence {i} of the third page is short." for i in range(12)],
        ],
    )
    passages = pdf_passages(path, max_words=40, overlap=8)
    assert passages[0].text == "Carbon monoxide binds to hemoglobin. It displaces oxygen."
    assert passages[0].passage_id == "pdf-tiny-medical-handbook-p1-0"
    assert passages[0].source == "pdf" and passages[0].url is None
    assert passages[0].title == "Tiny Medical Handbook"
    assert passages[0].meta == {"page": 1, "file": "Tiny_Medical Handbook.pdf"}
    third = [p for p in passages if p.meta["page"] == 3]
    assert len(third) > 1 and len(third) == len(passages) - 1
    assert all(len(p.text.split()) <= 40 for p in third)
    assert len({p.passage_id for p in passages}) == len(passages)
    assert {p.doc_id for p in passages} == {"pdf-tiny-medical-handbook"}
