"""Dictionary-based biomedical concept recognition and query expansion.

Concepts come from the MedlinePlus health-topic vocabulary (`primum.data.load_concepts`):
each has a preferred name, lay and clinical synonyms ("Heart Attack", "Myocardial
Infarction", "MI") and MeSH identifiers. Recognition is greedy longest-match over tokens:
at each position the longest dictionary name starting there wins, so "breast cancer
screening" links to the screening topic, not to "Breast Cancer" or "Cancer".

Names are compared after the same normalisation BM25 uses (case and diacritics folded,
regular plurals stemmed, possessive 's dropped), so "Migraines" and "Alzheimer disease"
match "Migraine" and "Alzheimer's Disease".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

from primum.retrieval.bm25 import STOPWORDS, fold, stem
from primum.types import Concept

# A word, with an optional possessive suffix that is part of the span but not of the key.
_WORD = re.compile(r"[^\W_]+(?:['’]s\b)?")
_POSSESSIVE = re.compile(r"['’]s$")
_JOINER = re.compile(r"[\s\-/]*")

Key = tuple[str, ...]


def _key(word: str) -> str:
    return stem(fold(_POSSESSIVE.sub("", word)))


def _is_acronym(name: str) -> bool:
    return name.isalpha() and name.isupper() and len(name) <= 5


class ConceptLinker:
    """Longest-match recogniser over a concept dictionary.

    Each dictionary entry is `{"preferred": str, "synonyms": [str], "mesh_ids": [str],
    "url": str | None}`.
    """

    def __init__(self, concepts: Iterable[dict]) -> None:
        self._concepts = list(concepts)
        # Normalised name -> (concept index, exact surface form required or None).
        self._names: dict[Key, tuple[int, str | None]] = {}
        # First token of a name -> length of the longest name starting with it. Bounds the
        # lookahead at each text position.
        self._longest: dict[str, int] = {}

        # Preferred names are registered before any synonym, so when one string is both the
        # preferred name of a concept and a synonym of another, it links to the former.
        for index, concept in enumerate(self._concepts):
            self._register(concept["preferred"], index)
        for index, concept in enumerate(self._concepts):
            for synonym in concept.get("synonyms", []):
                self._register(synonym, index)

    def _register(self, name: str, index: int) -> None:
        words = _WORD.findall(name)
        key = tuple(_key(word) for word in words)
        if not key or all(token in STOPWORDS or len(token) < 2 for token in key):
            return
        # An all-capitals name is an acronym and must match exactly as written: "AIDS" is a
        # disease but "aids" is a verb, "MS" a diagnosis but "ms" a unit. The price is that
        # a lower-cased "hiv" is not linked; a false concept shown to a user is worse.
        exact = words[0] if len(words) == 1 and _is_acronym(words[0]) else None
        if exact is None and len(key) == 1 and len(key[0]) < 3:
            return
        if key not in self._names:
            self._names[key] = (index, exact)
            self._longest[key[0]] = max(self._longest.get(key[0], 0), len(key))

    def link(self, text: str) -> list[Concept]:
        """Concepts mentioned in `text`, in order of appearance, non-overlapping."""
        matches = list(_WORD.finditer(text))
        keys = [_key(match.group()) for match in matches]
        # joined[i]: may a name continue from word i to word i + 1? Only across spaces,
        # hyphens and slashes, so no match can straddle a sentence or clause boundary.
        joined = [
            _JOINER.fullmatch(text, left.end(), right.start()) is not None
            for left, right in zip(matches, matches[1:], strict=False)
        ]
        found: list[Concept] = []
        position = 0
        while position < len(keys):
            length = self._match_length(matches, keys, joined, position)
            if length == 0:
                position += 1
                continue
            index, _ = self._names[tuple(keys[position : position + length])]
            concept = self._concepts[index]
            start, end = matches[position].start(), matches[position + length - 1].end()
            found.append(
                Concept(
                    surface=text[start:end],
                    start=start,
                    end=end,
                    preferred=concept["preferred"],
                    mesh_ids=list(concept.get("mesh_ids", [])),
                    synonyms=list(concept.get("synonyms", [])),
                    url=concept.get("url"),
                )
            )
            position += length
        return found

    def _match_length(
        self, matches: list[re.Match], keys: list[str], joined: list[bool], position: int
    ) -> int:
        """Token length of the longest dictionary name starting at `position` (0 if none)."""
        # The furthest a name could reach: bounded by the longest name starting with this
        # word, by the end of the text, and by the first gap that is not a word joiner.
        limit = min(self._longest.get(keys[position], 0), len(keys) - position)
        reach = min(limit, 1)
        while reach < limit and joined[position + reach - 1]:
            reach += 1
        for length in range(reach, 0, -1):
            entry = self._names.get(tuple(keys[position : position + length]))
            if entry is None:
                continue
            exact = entry[1]
            if exact is None or _POSSESSIVE.sub("", matches[position].group()) == exact:
                return length
        return 0

    def expand(self, query: str, *, max_synonyms: int = 4) -> str:
        """Append to `query` the other names of each concept it mentions.

        Lexical retrieval cannot match "heart attack" to a passage that only says
        "myocardial infarction"; adding the vocabulary's synonyms closes that gap without a
        learned model. Names already present in the query are skipped, and at most
        `max_synonyms` are added per concept so that one mention cannot swamp the query.
        """
        present = " ".join(_key(match.group()) for match in _WORD.finditer(query))
        additions: list[str] = []
        expanded: set[str] = set()
        for concept in self.link(query):
            if concept.preferred in expanded:
                continue
            expanded.add(concept.preferred)
            budget = max_synonyms
            for name in (concept.preferred, *concept.synonyms):
                if budget == 0:
                    break
                normalised = " ".join(_key(word) for word in _WORD.findall(name))
                if not normalised or f" {normalised} " in f" {present} ":
                    continue
                present += " " + normalised
                additions.append(name)
                budget -= 1
        return " ".join([query, *additions])


@lru_cache(maxsize=1)
def _default_linker() -> ConceptLinker:
    # Imported lazily: the data package is only needed for the built concept dictionary.
    from primum.data import load_concepts

    return ConceptLinker(load_concepts())


def link_concepts(text: str) -> list[Concept]:
    """Recognise MedlinePlus concepts in `text`."""
    return _default_linker().link(text)


def expand_query(query: str) -> str:
    """`query` with the synonyms of its recognised concepts appended."""
    return _default_linker().expand(query)
