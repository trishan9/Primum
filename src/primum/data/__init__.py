"""Data layer: download the open sources, build the corpora, read them back.

Three corpora (`pubmedqa`, `nfcorpus`, `kb`), one set of contrastive training pairs and one
concept dictionary — all rebuilt from public sources by `build_corpus` /
`build_contrastive_pairs`, all read through the `load_*` functions below.
"""

from primum.data.chunking import chunk_text
from primum.data.corpora import (
    build_corpus,
    load_concepts,
    load_nfcorpus,
    load_passages,
    load_pubmedqa_labeled,
)
from primum.data.errors import CorpusNotBuilt, LeakageError, SourceUnavailable
from primum.data.pairs import build_contrastive_pairs, load_contrastive_pairs

__all__ = [
    "CorpusNotBuilt",
    "LeakageError",
    "SourceUnavailable",
    "build_contrastive_pairs",
    "build_corpus",
    "chunk_text",
    "load_concepts",
    "load_contrastive_pairs",
    "load_nfcorpus",
    "load_passages",
    "load_pubmedqa_labeled",
]
