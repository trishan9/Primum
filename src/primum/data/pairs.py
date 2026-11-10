"""(query, relevant passage) pairs for contrastive fine-tuning of the dense encoder.

Two sources, chosen so that nothing the experiments are scored on is ever trained on:

* PubMedQA `pqa_artificial`: machine-generated questions paired with their abstracts.
  Every PMID of the expert-labelled set PQA-L is excluded, as is any artificial question
  whose text equals a PQA-L question.
* NFCorpus *train* qrels. Dev and test queries are never read into a pair, and a train
  query whose text is identical to a dev/test query is dropped (the splits are disjoint
  by id, but a handful of query strings recur across them).

Both exclusions are re-checked on the final pair list before anything is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from primum.config import get_settings
from primum.data import nfcorpus, pubmedqa
from primum.data._io import (
    console,
    iter_jsonl_lines,
    normalise_text,
    read_manifest,
    write_jsonl,
    write_manifest,
)
from primum.data.errors import CorpusNotBuilt, LeakageError
from primum.data.sources import NFCORPUS_QRELS, PQA_ARTIFICIAL, PQA_LABELED

# NFCorpus train has ~43 judged documents per query. With in-batch negatives, two pairs
# that share a query act as each other's negatives although both documents are relevant,
# so each query contributes at most this many pairs (a uniform sample of its documents).
MAX_PAIRS_PER_NFCORPUS_QUERY = 8


def _pairs_dir() -> Path:
    return get_settings().processed_dir / "pairs"


def _pubmedqa_pairs(n: int, seed: int) -> tuple[list[tuple[str, str]], set[int]]:
    """Sample `n` (question, abstract body) pairs from PQA-A; also return their PMIDs."""
    held_out_pmids = pubmedqa.labeled_pmids()
    held_out_questions = {
        normalise_text(row["question"]) for row in pubmedqa.iter_rows(PQA_LABELED, ["question"])
    }
    eligible = [
        i
        for i, row in enumerate(pubmedqa.iter_rows(PQA_ARTIFICIAL, ["pubid", "question"]))
        if row["pubid"] not in held_out_pmids
        and normalise_text(row["question"]) not in held_out_questions
    ]
    chosen = pubmedqa.sample_rows(eligible, n, seed)

    pairs: list[tuple[str, str]] = []
    pmids: set[int] = set()
    for row in pubmedqa.iter_rows(PQA_ARTIFICIAL, ["pubid", "question", "context"], only=chosen):
        # The positive is formatted exactly like an indexed PubMedQA passage (abstract
        # body, no conclusion), so training and retrieval see the same kind of text.
        positive = pubmedqa.abstract_text(row)
        if row["question"].strip() and positive:
            pairs.append((row["question"].strip(), positive))
            pmids.add(row["pubid"])
    return pairs, pmids


def _nfcorpus_pairs(rng: np.random.Generator) -> tuple[list[tuple[str, str]], set[str]]:
    """Pairs from NFCorpus train qrels; also return the query ids used."""
    queries = nfcorpus.read_queries()
    qrels = {split: nfcorpus.read_qrels(split) for split in nfcorpus.SPLITS}
    nfcorpus.assert_disjoint_splits(qrels)
    held_out_texts = {
        normalise_text(queries[qid]) for split in ("dev", "test") for qid in qrels[split]
    }
    # BEIR's convention for the text of a document: title and body joined by a space.
    doc_text = {d.passage_id: f"{d.title} {d.text}".strip() for d in nfcorpus.read_docs()}

    pairs: list[tuple[str, str]] = []
    used: set[str] = set()
    for qid, judgments in qrels["train"].items():
        if normalise_text(queries[qid]) in held_out_texts:
            continue
        relevant = sorted(doc_id for doc_id, grade in judgments.items() if grade > 0)
        if len(relevant) > MAX_PAIRS_PER_NFCORPUS_QUERY:
            relevant = rng.choice(
                relevant, size=MAX_PAIRS_PER_NFCORPUS_QUERY, replace=False
            ).tolist()
        pairs.extend((queries[qid], doc_text[doc_id]) for doc_id in relevant)
        used.add(qid)
    if used & (qrels["dev"].keys() | qrels["test"].keys()):
        raise LeakageError("NFCorpus dev/test queries reached the contrastive training pairs")
    return pairs, used


def build_contrastive_pairs(n_pubmedqa: int = 40000, *, force: bool = False) -> Path:
    """Write `pairs/train.jsonl` (one `{"query", "positive"}` per line) and return its path.

    Idempotent unless `force=True`. Sampling and the final shuffle are seeded from
    `settings.seed`; the shuffle mixes the two sources so that any prefix of the file
    (e.g. `--max-pairs`) is itself a representative sample.
    """
    directory = _pairs_dir()
    path = directory / "train.jsonl"
    existing = read_manifest(directory)
    if existing is not None and path.exists() and not force:
        if existing["params"]["n_pubmedqa"] != n_pubmedqa:
            console.print(
                f"[yellow]pairs were built with n_pubmedqa={existing['params']['n_pubmedqa']}, "
                f"not {n_pubmedqa}; keeping them. Pass force=True to rebuild.[/yellow]"
            )
        console.print(f"[dim]pairs: already built ({existing['n_pairs']:,} pairs)[/dim]")
        return path

    seed = get_settings().seed
    rng = np.random.default_rng(seed)
    pubmed_pairs, pubmed_pmids = _pubmedqa_pairs(n_pubmedqa, seed)
    leaked = pubmed_pmids & pubmedqa.labeled_pmids()
    if leaked:
        raise LeakageError(
            f"{len(leaked)} PQA-L PMIDs reached the contrastive training pairs, e.g. {sorted(leaked)[:3]}"
        )
    nf_pairs, nf_queries = _nfcorpus_pairs(rng)

    # Exact duplicates exist upstream (a few articles are in PubMed twice; some NFCorpus
    # train queries share their text under different ids). A repeated pair adds no signal
    # and would be its own false negative in a batch, so each pair is kept once.
    pubmed_pairs = list(dict.fromkeys(pubmed_pairs))
    nf_pairs = list(dict.fromkeys(nf_pairs))
    pairs = pubmed_pairs + nf_pairs
    order = rng.permutation(len(pairs))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)
    write_jsonl(path, ({"query": pairs[i][0], "positive": pairs[i][1]} for i in order))
    write_manifest(
        directory,
        {
            "n_pairs": len(pairs),
            "by_source": {
                "pubmedqa_artificial": len(pubmed_pairs),
                "nfcorpus_train": len(nf_pairs),
            },
            "n_nfcorpus_train_queries": len(nf_queries),
            "params": {
                "n_pubmedqa": n_pubmedqa,
                "max_pairs_per_nfcorpus_query": MAX_PAIRS_PER_NFCORPUS_QUERY,
                "seed": seed,
            },
            "excluded": "all PQA-L PMIDs and question texts; NFCorpus dev/test queries (by id and by text)",
            "sources": {
                "pubmedqa": {"repo": PQA_ARTIFICIAL.repo, "revision": PQA_ARTIFICIAL.revision},
                "nfcorpus_qrels": {
                    "repo": NFCORPUS_QRELS["train"].repo,
                    "revision": NFCORPUS_QRELS["train"].revision,
                },
            },
        },
    )
    console.print(
        f"pairs: {len(pairs):,} ({len(pubmed_pairs):,} PubMedQA-artificial + "
        f"{len(nf_pairs):,} NFCorpus-train from {len(nf_queries):,} queries)"
    )
    return path


def load_contrastive_pairs() -> list[tuple[str, str]]:
    """The training pairs as `(query, positive)` tuples, in file order."""
    path = _pairs_dir() / "train.jsonl"
    if not path.exists():
        raise CorpusNotBuilt(f"{path} does not exist; run `primum data pairs` first")
    pairs = []
    for line in iter_jsonl_lines(path):
        row = json.loads(line)
        pairs.append((row["query"], row["positive"]))
    return pairs
