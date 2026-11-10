"""Named retrieval systems. `build_retriever` is the only way the rest of the code obtains one.

A name is a first-stage system optionally followed by stages, applied left to right:

    bm25 | dense-{base,biomed,ft} | hybrid-{base,biomed,ft}     [+concepts] [+rerank]

`+concepts` expands the query with synonyms of its recognised concepts before retrieval;
`+rerank` re-orders the candidates with the cross-encoder. `RETRIEVERS` lists the systems
compared in the retrieval experiment.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from primum.config import get_settings
from primum.retrieval.bm25 import BM25Index
from primum.retrieval.concepts import expand_query
from primum.retrieval.dense import DenseIndex
from primum.retrieval.hybrid import (
    ExpandedQueryRetriever,
    HybridRetriever,
    RerankedRetriever,
    Retriever,
)
from primum.retrieval.rerank import CrossEncoderReranker

RETRIEVERS = [
    "bm25",
    "dense-base",
    "dense-biomed",
    "dense-ft",
    "hybrid-base",
    "hybrid-ft",
    "hybrid-ft+rerank",
]

_cache: dict[tuple[str, str], Retriever] = {}
# One lock per (name, corpus), so two concurrent first requests for the same system cannot
# both load its model or embed its corpus, while a build that takes minutes (embedding a
# corpus with a new encoder) never delays a lookup of a different, already-built system.
# Composite systems take their components' locks while holding their own; that cannot
# deadlock because a component never depends on a composite.
_key_locks: dict[tuple[str, str], threading.Lock] = {}
_key_locks_guard = threading.Lock()  # held only to create a key's lock, never during a build


def build_retriever(name: str, corpus: str) -> Retriever:
    """The retriever `name` over `corpus`, built on first use and cached per (name, corpus).

    Indexes are loaded from `data/indexes/<corpus>/` or built and saved there. Composite
    systems share their components, so `hybrid-ft` and `dense-ft` hold one embedding matrix.

    Raises `ValueError` for an unknown name and `EncoderNotFound` (a `FileNotFoundError`)
    when a system needs the fine-tuned encoder and it has not been trained. There is no
    fallback to another encoder: a result must come from the system it is labelled with.
    """
    key = (name, corpus)
    # A cache hit takes no lock. A dict read is atomic in CPython, and an entry is only ever
    # stored once fully built.
    if (retriever := _cache.get(key)) is not None:
        return retriever
    with _key_locks_guard:
        key_lock = _key_locks.setdefault(key, threading.Lock())
    with key_lock:
        if (retriever := _cache.get(key)) is None:  # another thread may have built it meanwhile
            retriever = _cache[key] = _build(name, corpus)
        return retriever


def _build(name: str, corpus: str) -> Retriever:
    settings = get_settings()
    base, plus, stage = name.rpartition("+")
    if plus and stage == "rerank":
        return RerankedRetriever(
            build_retriever(base, corpus),
            _reranker(),
            name=name,
            candidate_k=settings.candidate_k,
        )
    if plus and stage == "concepts":
        return ExpandedQueryRetriever(build_retriever(base, corpus), expand_query, name=name)

    encoders = {
        "base": settings.base_encoder,
        "biomed": settings.biomed_encoder,
        "ft": settings.finetuned_encoder_dir,  # a Path: must exist, or EncoderNotFound
    }
    kind, _, encoder = name.partition("-")
    if name == "bm25":
        return BM25Index(corpus)
    if kind == "dense" and encoder in encoders:
        return DenseIndex(encoders[encoder], corpus, name=name)
    if kind == "hybrid" and encoder in encoders:
        return HybridRetriever(
            build_retriever("bm25", corpus),
            build_retriever(f"dense-{encoder}", corpus),
            name=name,
            rrf_k=settings.rrf_k,
            candidate_k=settings.candidate_k,
        )
    raise ValueError(f"unknown retriever {name!r}; expected one of {RETRIEVERS}")


_reranker_lock = threading.Lock()


def _reranker() -> CrossEncoderReranker:
    """The one cross-encoder, shared by every `+rerank` system whatever its corpus.

    Locked because builds for different keys now run concurrently, and `lru_cache` alone may
    call the function twice when two first calls race.
    """
    with _reranker_lock:
        return _shared_reranker()


@lru_cache(maxsize=1)
def _shared_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(get_settings().reranker)
