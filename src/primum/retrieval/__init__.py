"""Retrieval engine: lexical, dense and hybrid search over a passage corpus.

    retriever = build_retriever("hybrid-ft", "kb")
    hits = retriever.search("does metformin reduce cardiovascular risk?", k=5)

Importing this package is cheap: torch and the transformer libraries are loaded only when a
dense index, the reranker or the training loop is first used.
"""

from primum.retrieval.bm25 import BM25, BM25Index, tokenize
from primum.retrieval.concepts import ConceptLinker, expand_query, link_concepts
from primum.retrieval.dense import DenseIndex, EncoderNotFound
from primum.retrieval.factory import RETRIEVERS, build_retriever
from primum.retrieval.fusion import rrf_fuse
from primum.retrieval.hybrid import (
    ExpandedQueryRetriever,
    HybridRetriever,
    RerankedRetriever,
    Retriever,
)
from primum.retrieval.metrics import average_precision, mrr_at_k, ndcg_at_k, recall_at_k
from primum.retrieval.rerank import CrossEncoderReranker

__all__ = [
    "BM25",
    "BM25Index",
    "ConceptLinker",
    "CrossEncoderReranker",
    "DenseIndex",
    "EncoderNotFound",
    "ExpandedQueryRetriever",
    "HybridRetriever",
    "RETRIEVERS",
    "RerankedRetriever",
    "Retriever",
    "average_precision",
    "build_retriever",
    "expand_query",
    "link_concepts",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "rrf_fuse",
    "tokenize",
    "train_encoder",
]


def __getattr__(name: str):
    # `train_encoder` lives in the one module that imports torch at import time; resolving
    # it lazily (PEP 562) keeps `import primum.retrieval` fast for callers that only search.
    if name == "train_encoder":
        from primum.retrieval.train_contrastive import train_encoder

        return train_encoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
