"""Okapi BM25, implemented from scratch on a sparse term-document matrix.

Ranking function (Robertson & Zaragoza, 2009). For a query q and a document d,

    score(q, d) = sum over terms t of   qtf(t, q) * idf(t) * tf(t, d) * (k1 + 1)
                                        --------------------------------------------
                                        tf(t, d) + k1 * (1 - b + b * |d| / avgdl)

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

where tf is the term's frequency in the document, qtf its frequency in the query, |d| the
document length in tokens, avgdl the mean document length, N the number of documents and
df(t) the number of documents containing t. `k1` sets how quickly repeated occurrences
saturate (the tf factor rises from 1 at tf=1, for a document of average length, towards
k1 + 1); `b` sets how strongly long documents are penalised. The idf is the Lucene variant:
the classic Robertson-Sparck-Jones idf turns negative for terms in more than half the
documents, and the `1 +` inside the logarithm keeps every weight positive.

Everything except qtf depends only on the (term, document) pair, so the whole right-hand
factor is precomputed once into a sparse matrix W[t, d]. Scoring a batch of queries is then
a single sparse product `Q @ W`, with Q[i, t] = qtf(t, q_i).
"""

from __future__ import annotations

import json
import re
import unicodedata
from array import array
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse

from primum.retrieval._common import (
    corpus_fingerprint,
    passage_text,
    read_json,
    resolve_corpus,
    top_k,
)
from primum.types import Hit, Passage

# Bump when `tokenize` changes: cached indexes built with another tokenizer are rebuilt.
TOKENIZER_VERSION = 2

# Lucene's English stop set plus the auxiliaries and interrogatives that open most clinical
# questions ("Does ...", "Is ...", "Can ..."). Deliberately small: BM25's idf already
# discounts frequent words; this only removes terms that are pure noise in a query.
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how if in
    into is it its no not of on or our should such than that the their then there these they
    this to was we were what when where which who why will with""".split()
)

# A word is a run of alphanumerics, optionally joined by intra-word hyphens: "il-6",
# "covid-19", "non-small-cell".
_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# The possessive "'s" is removed before words are matched: the apostrophe is not a word
# character, so "Crohn's" would otherwise yield a stray term "s" that matches every other
# possessive in the corpus ("the patient's knee").
_POSSESSIVE = re.compile(r"(?<=[a-z0-9])['’]s\b")

# Greek letters are spelled out so that "TNF-α" and "TNF-alpha" index to the same term. The
# whole lowercase alphabet is needed because `_WORD` keeps only ASCII, so an unmapped letter
# would vanish ("σ-receptor" -> "receptor"). Uppercase letters reach this table lowercased,
# and NFKD has already turned symbol variants (ϑ, ϕ, ϵ, the micro sign µ) into these.
_GREEK = str.maketrans(
    {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
        "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
        "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
        "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
        "χ": "chi", "ψ": "psi", "ω": "omega",
    }
)  # fmt: skip


def stem(word: str) -> str:
    """Harman's (1991) S-stemmer: fold regular English plurals, and nothing else.

    A full Porter stemmer conflates too aggressively for biomedical text ("organism" and
    "organ" share a stem), so only plural suffixes are removed. The exceptions keep Latin and
    Greek singulars intact: "virus", "abscess", "diagnosis".
    """
    if len(word) <= 3 or not word.endswith("s"):
        return word
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if not word.endswith(("us", "ss", "is")):
        return word[:-1]
    return word


@lru_cache(maxsize=1 << 18)
def _index_terms(word: str) -> tuple[str, ...]:
    """Index terms for one matched word (memoised: word frequencies are Zipfian).

    A hyphenated compound is indexed both whole and as its parts, so "covid-19" matches the
    exact compound (a rare, high-idf term) and "beta-blocker" still matches "beta blocker".
    """
    if "-" not in word:
        return () if word in STOPWORDS else (stem(word),)
    parts = tuple(stem(part) for part in word.split("-") if part not in STOPWORDS)
    return (stem(word), *parts)


def fold(text: str) -> str:
    """Lowercase, strip diacritics ("Sjögren" -> "sjogren") and spell out Greek letters."""
    if text.isascii():
        return text.lower()
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower().translate(_GREEK)


def tokenize(text: str) -> list[str]:
    """Biomedical-aware tokenizer shared by documents and queries."""
    words = _WORD.findall(_POSSESSIVE.sub("", fold(text)))
    return [term for word in words for term in _index_terms(word)]


class BM25:
    """The BM25 ranking function fitted to a fixed collection of texts."""

    def __init__(self, texts: Iterable[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocabulary: dict[str, int] = {}
        # One (term, document, tf) triple per distinct term of each document. Typed arrays
        # keep this at 12 bytes per triple; Python lists would need roughly ten times more.
        term_ids, doc_ids, frequencies = array("i"), array("i"), array("f")
        lengths: list[int] = []
        for doc, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                term_ids.append(self.vocabulary.setdefault(term, len(self.vocabulary)))
                doc_ids.append(doc)
                frequencies.append(tf)
        if not lengths:
            raise ValueError("BM25 needs at least one document")

        n_docs, n_terms = len(lengths), len(self.vocabulary)
        terms = np.frombuffer(term_ids, dtype=np.int32)
        docs = np.frombuffer(doc_ids, dtype=np.int32)
        tf = np.frombuffer(frequencies, dtype=np.float32).astype(np.float64)

        doc_len = np.asarray(lengths, dtype=np.float64)
        self.avgdl = float(doc_len.mean()) or 1.0  # all-empty collection: avoid 0/0
        # Each (term, document) pair occurs exactly once above, so counting pairs per term
        # gives the document frequency.
        df = np.bincount(terms, minlength=n_terms)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        length_norm = k1 * (1.0 - b + b * doc_len / self.avgdl)
        weights = self.idf[terms] * tf * (k1 + 1.0) / (tf + length_norm[docs])
        self._weights = sparse.csr_array(
            (weights.astype(np.float32), (terms, docs)), shape=(n_terms, n_docs)
        )

    @property
    def n_docs(self) -> int:
        return self._weights.shape[1]

    def score(self, queries: Sequence[str]) -> np.ndarray:
        """BM25 scores of every document for every query: float32 [n_queries, n_docs].

        Query terms outside the vocabulary occur in no document and contribute nothing.
        """
        rows, cols, qtf = [], [], []
        for i, query in enumerate(queries):
            for term, count in Counter(tokenize(query)).items():
                term_id = self.vocabulary.get(term)
                if term_id is not None:
                    rows.append(i)
                    cols.append(term_id)
                    qtf.append(count)
        query_matrix = sparse.csr_array(
            (np.asarray(qtf, dtype=np.float32), (np.asarray(rows, int), np.asarray(cols, int))),
            shape=(len(queries), len(self.vocabulary)),
        )
        return (query_matrix @ self._weights).toarray()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "weights.npz", self._weights)
        np.save(directory / "idf.npy", self.idf)
        # Terms in id order; dicts preserve insertion order, which is id order.
        (directory / "vocabulary.json").write_text(json.dumps(list(self.vocabulary)))
        (directory / "params.json").write_text(
            json.dumps({"k1": self.k1, "b": self.b, "avgdl": self.avgdl})
        )

    @classmethod
    def load(cls, directory: Path) -> BM25:
        model = cls.__new__(cls)
        params = json.loads((directory / "params.json").read_text())
        model.k1, model.b, model.avgdl = params["k1"], params["b"], params["avgdl"]
        terms = json.loads((directory / "vocabulary.json").read_text())
        model.vocabulary = {term: i for i, term in enumerate(terms)}
        model.idf = np.load(directory / "idf.npy")
        model._weights = sparse.load_npz(directory / "weights.npz")
        return model


class BM25Index:
    """BM25 retriever over a passage list; cached on disk when the corpus is named.

    `corpus` is either a corpus id (passages come from `primum.data`, the index is cached
    under `data/indexes/<corpus>/bm25/`) or an explicit passage list (in memory only).
    """

    name = "bm25"
    _CHUNK = 128  # queries scored per sparse product; bounds the dense score matrix

    def __init__(
        self, corpus: str | Sequence[Passage], *, k1: float = 1.2, b: float = 0.75
    ) -> None:
        self.passages, cache_root = resolve_corpus(corpus)
        key = {
            "fingerprint": corpus_fingerprint(self.passages),
            "k1": k1,
            "b": b,
            "tokenizer": TOKENIZER_VERSION,
        }
        directory = cache_root / "bm25" if cache_root else None
        if directory is not None and read_json(directory / "meta.json") == key:
            self._model = BM25.load(directory)
            return
        self._model = BM25((passage_text(p) for p in self.passages), k1=k1, b=b)
        if directory is not None:
            self._model.save(directory)
            # Written last: a build interrupted half-way leaves no valid meta, so it is redone.
            (directory / "meta.json").write_text(json.dumps(key))

    def score_all(self, queries: Sequence[str]) -> np.ndarray:
        """Scores of every passage for every query: [n_queries, n_passages]."""
        return self._model.score(queries)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.search_batch([query], k)[0]

    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Hit]]:
        results: list[list[Hit]] = []
        for start in range(0, len(queries), self._CHUNK):
            for scores in self.score_all(queries[start : start + self._CHUNK]):
                rows = top_k(scores, k)
                # A zero score means no query term occurs in the passage; such a passage is
                # not a lexical match at all and is not returned, so fewer than k hits is
                # possible.
                rows = rows[scores[rows] > 0]
                results.append(
                    [
                        Hit(
                            passage=self.passages[row],
                            score=float(scores[row]),
                            rank=rank,
                            retriever=self.name,
                            components={"bm25": float(scores[row]), "bm25_rank": float(rank)},
                        )
                        for rank, row in enumerate(rows.tolist(), start=1)
                    ]
                )
        return results
