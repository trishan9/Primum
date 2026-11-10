"""BEIR NFCorpus (Boteva et al., 2016; Thakur et al., 2021) → documents, queries, qrels.

A medical retrieval benchmark: lay-language nutrition questions against PubMed-derived
documents, with graded relevance. It is kept exactly as BEIR distributes it — document
ids, splits and grades untouched, no chunking — so RQ1's nDCG@10 is comparable with
published BEIR numbers.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from primum.data._io import atomic_write_lines, console, write_jsonl
from primum.data.errors import LeakageError, SourceUnavailable
from primum.data.sources import NFCORPUS_DOCS, NFCORPUS_QRELS, NFCORPUS_QUERIES, fetch
from primum.types import Passage

SPLITS = ("train", "dev", "test")
QRELS_HEADER = "query-id\tcorpus-id\tscore"

Qrels = dict[str, dict[str, int]]


def read_docs() -> list[Passage]:
    rows = pq.read_table(fetch(NFCORPUS_DOCS), columns=["_id", "title", "text"]).to_pylist()
    return [
        Passage(
            passage_id=row["_id"],
            doc_id=row["_id"],
            title=row["title"] or "",
            text=row["text"],
            source="nfcorpus",
        )
        for row in rows
    ]


def read_queries() -> dict[str, str]:
    rows = pq.read_table(fetch(NFCORPUS_QUERIES), columns=["_id", "text"]).to_pylist()
    return {row["_id"]: row["text"] for row in rows}


def parse_qrels(path: Path) -> Qrels:
    """Read a BEIR qrels TSV (`query-id, corpus-id, score` with a header row)."""
    qrels: Qrels = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def read_qrels(split: str) -> Qrels:
    return parse_qrels(fetch(NFCORPUS_QRELS[split]))


def assert_disjoint_splits(qrels_by_split: dict[str, Qrels]) -> None:
    """No query may sit in two splits: train queries feed the encoder, test queries score it."""
    for a, b in combinations(qrels_by_split, 2):
        shared = qrels_by_split[a].keys() & qrels_by_split[b].keys()
        if shared:
            raise LeakageError(
                f"NFCorpus splits {a!r} and {b!r} share {len(shared)} queries, e.g. {sorted(shared)[:3]}"
            )


def build(directory: Path) -> dict[str, Any]:
    """Write passages, per-split queries and per-split qrels into `directory`."""
    docs = read_docs()
    queries = read_queries()
    doc_ids = {d.passage_id for d in docs}
    qrels_by_split = {split: read_qrels(split) for split in SPLITS}
    assert_disjoint_splits(qrels_by_split)

    write_jsonl(directory / "passages.jsonl", docs)
    splits: dict[str, dict[str, int]] = {}
    for split, qrels in qrels_by_split.items():
        judged = {doc_id for docs_of_query in qrels.values() for doc_id in docs_of_query}
        if not judged <= doc_ids or not qrels.keys() <= queries.keys():
            raise SourceUnavailable(
                f"NFCorpus {split} qrels reference unknown documents or queries"
            )
        write_jsonl(
            directory / f"queries.{split}.jsonl",
            ({"_id": qid, "text": queries[qid]} for qid in qrels),
        )
        atomic_write_lines(
            directory / f"qrels.{split}.tsv",
            [QRELS_HEADER]
            + [
                f"{qid}\t{doc_id}\t{grade}"
                for qid, judgments in qrels.items()
                for doc_id, grade in judgments.items()
            ],
        )
        splits[split] = {
            "n_queries": len(qrels),
            "n_qrels": sum(len(judgments) for judgments in qrels.values()),
        }
    console.print(
        f"nfcorpus: {len(docs):,} documents; queries "
        + ", ".join(f"{split} {stats['n_queries']:,}" for split, stats in splits.items())
    )
    return {
        "n_passages": len(docs),
        "splits": splits,
        "sources": {
            "docs_and_queries": {"repo": NFCORPUS_DOCS.repo, "revision": NFCORPUS_DOCS.revision},
            "qrels": {
                "repo": NFCORPUS_QRELS["test"].repo,
                "revision": NFCORPUS_QRELS["test"].revision,
            },
        },
    }
