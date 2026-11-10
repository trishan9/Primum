"""MedlinePlus health topics (U.S. National Library of Medicine) → passages and concepts.

MedlinePlus is NLM's consumer-health encyclopedia: ~1,000 English topic pages written in
plain language. NLM's own topic summaries are in the public domain. The XML also
carries, for each topic, its lay synonyms and its MeSH descriptor — a small, curated
bridge between how patients name a condition and how the literature indexes it, which is
what the concept recogniser (`primum.retrieval.concepts`) is built on.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from primum.data.chunking import chunk_text
from primum.data.errors import SourceUnavailable
from primum.types import Passage

# A zip member's declared size is checked before reading, so a decompression bomb is
# refused rather than inflated. The real file is ~30 MB.
_MAX_XML_BYTES = 500 * 1024 * 1024

_HEADINGS = frozenset({"h2", "h3", "h4"})
_ITEMS = frozenset({"li", "tr"})
_CELLS = frozenset({"td", "th"})
_BLOCKS = _HEADINGS | _ITEMS | frozenset({"p", "ul", "ol", "table", "div"})
# Summaries end with a credit line such as "NIH: National Cancer Institute". It names the
# authoring institute, not a medical fact; the institute is kept in `Passage.meta` instead.
_CREDIT_LINE = re.compile(r"^NIH: ")


@dataclass(frozen=True)
class Topic:
    topic_id: str
    title: str
    url: str
    summary_html: str
    also_called: list[str] = field(default_factory=list)
    see_references: list[str] = field(default_factory=list)
    mesh_ids: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    institute: str | None = None


def iter_topics(zip_path: Path) -> Iterator[Topic]:
    """Yield the English topics from a `mplus_topics_compressed_<date>.zip` snapshot.

    The XML is parsed straight out of the archive (never extracted, so member names cannot
    write anywhere) with the standard-library expat parser, which neither fetches external
    entities nor — from expat 2.4.1 — amplifies nested ones.
    """
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if m.filename.lower().endswith(".xml")]
        if len(members) != 1:
            raise SourceUnavailable(f"{zip_path.name}: expected one XML file, found {len(members)}")
        if members[0].file_size > _MAX_XML_BYTES:
            raise SourceUnavailable(f"{zip_path.name}: XML member is implausibly large")
        with archive.open(members[0]) as stream:
            for _, element in ElementTree.iterparse(stream, events=("end",)):
                if element.tag != "health-topic":
                    continue
                if element.get("language") == "English":
                    yield Topic(
                        topic_id=element.get("id", ""),
                        title=_clean(element.get("title", "")),
                        url=element.get("url", ""),
                        summary_html=element.findtext("full-summary") or "",
                        also_called=_texts(element, "also-called"),
                        see_references=_texts(element, "see-reference"),
                        mesh_ids=[
                            d.get("id", "")
                            for d in element.iterfind("mesh-heading/descriptor")
                            if d.get("id")
                        ],
                        groups=_texts(element, "group"),
                        institute=element.findtext("primary-institute"),
                    )
                element.clear()  # the file is 30 MB of siblings; free each topic once read


def _clean(text: str) -> str:
    return " ".join(text.split())


def _texts(element: ElementTree.Element, tag: str) -> list[str]:
    return [t for t in (_clean(e.text or "") for e in element.iterfind(tag)) if t]


class _SummaryParser(HTMLParser):
    """Turn a topic summary (a small HTML fragment) into `(heading, [blocks])` sections.

    Only structure that changes meaning is kept: headings open sections, and list items are
    remembered as items so that a bulleted list can be re-attached to its lead-in sentence.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[tuple[str, list[tuple[str, str]]]] = [("", [])]
        self._buffer: list[str] = []
        self._in_heading = False
        self._item_depth = 0  # > 0 inside <li>/<tr>, so a <p> nested in an item stays an item

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self._buffer.append(" ")
        elif tag in _BLOCKS:
            self._flush()
            self._in_heading = tag in _HEADINGS
            self._item_depth += tag in _ITEMS

    def handle_endtag(self, tag: str) -> None:
        if tag in _CELLS:
            self._buffer.append(" ")  # keep adjacent table cells from fusing into one word
        elif tag in _BLOCKS:
            self._flush()
            self._in_heading = False
            self._item_depth = max(0, self._item_depth - (tag in _ITEMS))

    def handle_data(self, data: str) -> None:
        self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = _clean("".join(self._buffer))
        self._buffer.clear()
        if not text:
            return
        if self._in_heading:
            self.sections.append((text, []))
        elif self._item_depth:
            self.sections[-1][1].append(("item", text))
        elif not _CREDIT_LINE.match(text):
            self.sections[-1][1].append(("para", text))


def _render(blocks: list[tuple[str, str]]) -> str:
    """Join a section's blocks into prose.

    A run of list items becomes one sentence, "a; b; c.", appended to the paragraph that
    introduces it ("Symptoms include: a; b; c."). Left as separate fragments, items such as
    "Fever" would be scored by the NLI verifier as premises with no subject.
    """
    sentences: list[str] = []
    items: list[str] = []

    def close_list() -> None:
        if not items:
            return
        listing = "; ".join(item.rstrip(".;, ") for item in items) + "."
        if sentences and sentences[-1].endswith(":"):
            sentences[-1] = f"{sentences[-1]} {listing}"
        else:
            sentences.append(listing)
        items.clear()

    for kind, text in blocks:
        if kind == "item":
            items.append(text)
        else:
            close_list()
            sentences.append(text)
    close_list()
    return " ".join(sentences)


def summary_sections(summary_html: str) -> list[tuple[str, str]]:
    """Split a topic summary into `(heading, body)` pairs; text before any heading gets ""."""
    parser = _SummaryParser()
    parser.feed(summary_html)
    parser.close()
    rendered = ((heading, _render(blocks)) for heading, blocks in parser.sections)
    return [(heading, body) for heading, body in rendered if body]


def topic_passages(topic: Topic, *, max_words: int = 220, overlap: int = 40) -> list[Passage]:
    """One passage per chunk of each section of the topic summary.

    Section headings on MedlinePlus are questions ("What causes asthma?"). Each chunk
    starts with its heading so that it stays interpretable on its own — to the retriever
    matching a user's question, and to a reader shown the passage as evidence.
    """
    passages: list[Passage] = []
    for heading, body in summary_sections(topic.summary_html):
        # The heading is part of the passage, so it is paid for out of the word budget.
        budget = max(max_words - len(heading.split()), overlap + 1)
        for chunk in chunk_text(body, max_words=budget, overlap=overlap):
            passages.append(
                Passage(
                    passage_id=f"mplus-{topic.topic_id}-{len(passages)}",
                    doc_id=f"mplus-{topic.topic_id}",
                    title=topic.title,
                    text=f"{heading} {chunk}" if heading else chunk,
                    source="medlineplus",
                    url=topic.url or None,
                    meta={
                        "section": heading,
                        "groups": topic.groups,
                        "mesh_ids": topic.mesh_ids,
                        "institute": topic.institute,
                    },
                )
            )
    return passages


def topic_concept(topic: Topic) -> dict[str, Any]:
    """The topic as a dictionary entry: preferred name, lay synonyms, MeSH ids, page URL."""
    seen = {topic.title.casefold()}
    synonyms: list[str] = []
    for name in topic.also_called + topic.see_references:
        if name.casefold() not in seen:
            seen.add(name.casefold())
            synonyms.append(name)
    return {
        "preferred": topic.title,
        "synonyms": synonyms,
        "mesh_ids": topic.mesh_ids,
        "url": topic.url or None,
    }
