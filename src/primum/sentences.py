"""Sentence segmentation for biomedical prose: the one definition of a sentence in Primum.

Two components depend on it and must agree. Verification scores a claim against single
evidence sentences, so the splitter decides what a "premise" is; and the data layer ends its
chunks on sentence boundaries so that no premise is cut in half. With one splitter, a chunk
boundary is always also a premise boundary. Kept free of heavy imports so the data layer can
use it without loading the NLI model's dependencies.

General-purpose rules ("split after every full stop") shred abstracts: they cut
at "e.g.", "vs.", "et al.", "Fig. 2" and "E. coli". This splitter is a small set of explicit
rules instead of a trained model so that its behaviour is predictable and it adds no download.

A terminator (. ! ?) followed by whitespace is a boundary unless one of these holds:

1. the next token does not look like the start of a sentence (plain lowercase word);
2. the word before the full stop is an abbreviation that never ends a sentence ("e.g.", "vs.",
   "Fig.");
3. it is an abbreviation that only sometimes ends one ("et al.", "etc.", "i.v.", "U.S.") and
   the next token is not capitalised;
4. it is an author initial ("J. Smith") or a list enumerator ("1. Take ...").

Decimals and p-values ("0.05", "p < .001") never reach these rules because their full stop is
not followed by whitespace.

The rules err towards splitting when in doubt. Over-splitting is the cheaper mistake here:
`verify_claims` also scores every two-sentence window, which puts a wrongly cut sentence back
together, whereas an under-split premise dilutes the entailment signal with unrelated text.
"""

from __future__ import annotations

import re

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# One or more terminators and optional closing quotes/brackets, followed by whitespace and the
# next token (looked at, not consumed, so the match ends where the sentence would).
_TERMINATOR = re.compile(r"(?P<marks>[.!?]+)[\"'”’)\]]*(?=\s+(?P<next>\S+))")
_OPENERS = "([{\"'“‘"

# Compared after lower-casing, except for all-caps words: "MS." (multiple sclerosis), "DR."
# (diabetic retinopathy) and "ST." (the ECG segment) end sentences; "Ms.", "Dr." and "St." (as in
# St. John's wort) do not.
_NEVER_FINAL = frozenset(
    {"e.g", "i.e", "vs", "cf", "viz", "approx", "fig", "figs", "eq", "eqs", "ref", "refs", "vol"}
    | {"dr", "prof", "mr", "mrs", "ms", "st"}
)
_SOMETIMES_FINAL = frozenset({"al", "etc", "sp", "spp"})  # "al" is the tail of "et al."
_DOTTED_ABBREVIATION = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")  # i.v, b.i.d, U.S
_ENUMERATOR = re.compile(r"\d{1,2}")
_TRAILING_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences, with whitespace normalised inside each one.

    Blank lines always separate sentences (headings, list items, PDF paragraphs); single
    newlines do not, because PDF extraction hard-wraps lines in the middle of sentences.
    """
    sentences: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        block = " ".join(paragraph.split())
        start = 0
        for match in _TERMINATOR.finditer(block):
            if _is_boundary(block, start, match):
                sentences.append(block[start : match.end()].strip())
                start = match.end()
        tail = block[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def _is_boundary(block: str, start: int, match: re.Match[str]) -> bool:
    """Decide whether the terminator at `match` ends the sentence that began at `start`."""
    if "?" in match["marks"] or "!" in match["marks"]:
        return True
    following = match["next"].lstrip(_OPENERS)
    if not _starts_sentence(following):
        return False

    word = block[start : match.start()].rsplit(" ", 1)[-1].lstrip(_OPENERS)
    before = block[start : match.start() - len(word)].rstrip()

    if _ENUMERATOR.fullmatch(word) and not before:
        return False  # "1. Take the tablet": the sentence so far is only a list number

    if len(word) == 1 and word.isupper():
        # A single capital is ambiguous: an initial ("J. Smith") or a real ending ("hepatitis
        # B. Patients ..."). Endings are far more common in clinical text, so it is treated as
        # an initial only where an ending is impossible: at the start of the sentence, after a
        # comma/semicolon/bracket, or after another initial. Genus abbreviations ("E. coli")
        # never get here: their next token is lowercase.
        return not (not before or before[-1] in ",;(" or _TRAILING_INITIAL.search(before))

    key = word if word.isupper() else word.lower()
    if key in _NEVER_FINAL:
        return False
    if key in _SOMETIMES_FINAL or _DOTTED_ABBREVIATION.fullmatch(word):
        return following[:1].isupper()
    return True


def _starts_sentence(token: str) -> bool:
    """Whether `token` can open a sentence.

    Capitals, digits and symbols can. A lowercase-initial token normally cannot, except for
    the biomedical names that are written that way even sentence-initially: "mRNA", "p53",
    "pH", "miR-21" — recognised by an internal capital or digit.
    """
    if not token or not token[0].islower():
        return True
    return any(ch.isupper() or ch.isdigit() for ch in token[1:])
