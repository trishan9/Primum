"""PubMedQA (Jin et al., 2019) → retrievable abstracts and the 1,000 expert-labelled questions.

Each PubMedQA item is a PubMed abstract whose title is a yes/no question. The dataset
splits the abstract into the body (`context.contexts`) and the conclusion (`long_answer`);
annotators chose yes / no / maybe after reading the conclusion.

Two things are therefore kept out of every passage:

* the conclusion, because it states the answer — indexing it would turn "retrieve evidence
  and reason" into "retrieve the label";
* the article title, because it *is* the question — indexing it would make gold-abstract
  retrieval (RQ1) a string-matching exercise. Titles are left empty.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from primum.config import get_settings
from primum.data._io import console, normalise_text, write_jsonl
from primum.data.errors import LeakageError, SourceUnavailable
from primum.data.sources import PQA_LABELED, PQA_UNLABELED, HFFile, fetch
from primum.types import DECISIONS, Passage, PubMedQAExample

N_LABELED = 1000  # size of PQA-L; the benchmark every QA experiment reports on


def passage_id(pmid: int | str) -> str:
    return f"pubmed-{pmid}"


def abstract_text(row: dict[str, Any]) -> str:
    """The abstract body: its structured sections joined, conclusion excluded."""
    return " ".join(s.strip() for s in row["context"]["contexts"] if s and s.strip())


def iter_rows(
    file: HFFile, columns: Sequence[str], *, only: set[int] | None = None
) -> Iterator[dict[str, Any]]:
    """Stream rows of a PubMedQA parquet file, optionally just the row indices in `only`.

    The largest file is ~230 MB, so it is read in batches and rows that were not sampled
    are dropped while still in Arrow form, before any Python objects are created.
    """
    parquet = pq.ParquetFile(fetch(file))
    offset = 0
    for batch in parquet.iter_batches(batch_size=4096, columns=list(columns)):
        if only is None:
            yield from batch.to_pylist()
        else:
            wanted = [i for i in range(batch.num_rows) if offset + i in only]
            if wanted:
                yield from batch.take(wanted).to_pylist()
        offset += batch.num_rows


def sample_rows(eligible: Sequence[int], n: int | None, seed: int) -> set[int]:
    """Choose `n` of the `eligible` row indices uniformly without replacement (all if n is None)."""
    if n is None or n >= len(eligible):
        return set(eligible)
    rng = np.random.default_rng(seed)
    return set(rng.choice(np.asarray(eligible), size=n, replace=False).tolist())


def labeled_pmids() -> set[int]:
    """PMIDs of PQA-L: the set nothing may be trained on."""
    return {row["pubid"] for row in iter_rows(PQA_LABELED, ["pubid"])}


def _passage(row: dict[str, Any], subset: str) -> Passage:
    pmid = row["pubid"]
    return Passage(
        passage_id=passage_id(pmid),
        doc_id=str(pmid),
        title="",
        text=abstract_text(row),
        source="pubmed",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        meta={
            "subset": subset,
            "section_labels": row["context"]["labels"],
            "mesh": row["context"]["meshes"],
        },
    )


def _read_labeled() -> tuple[list[PubMedQAExample], list[Passage]]:
    examples: list[PubMedQAExample] = []
    passages: list[Passage] = []
    for row in iter_rows(
        PQA_LABELED, ["pubid", "question", "context", "long_answer", "final_decision"]
    ):
        if row["final_decision"] not in DECISIONS:
            raise SourceUnavailable(
                f"PQA-L item {row['pubid']} has label {row['final_decision']!r}; expected {DECISIONS}"
            )
        passage = _passage(row, "pqa_labeled")
        passages.append(passage)
        examples.append(
            PubMedQAExample(
                qid=str(row["pubid"]),
                question=row["question"],
                contexts=row["context"]["contexts"],
                context_labels=row["context"]["labels"],
                long_answer=row["long_answer"],
                label=row["final_decision"],
                passage_id=passage.passage_id,
            )
        )
    if len(examples) != N_LABELED or len({e.qid for e in examples}) != N_LABELED:
        raise SourceUnavailable(f"expected {N_LABELED} distinct PQA-L items, found {len(examples)}")
    return examples, passages


def _read_distractors(exclude: set[str], max_distractors: int | None, seed: int) -> list[Passage]:
    """PQA-U abstracts: realistic distractors that make retrieval over PQA-L non-trivial."""
    eligible = [
        i
        for i, row in enumerate(iter_rows(PQA_UNLABELED, ["pubid"]))
        if passage_id(row["pubid"]) not in exclude
    ]
    chosen = sample_rows(eligible, max_distractors, seed)
    passages = (
        _passage(row, "pqa_unlabeled")
        for row in iter_rows(PQA_UNLABELED, ["pubid", "context"], only=chosen)
    )
    return [p for p in passages if p.text]


def _assert_no_label_leak(examples: list[PubMedQAExample], passages: list[Passage]) -> None:
    """Refuse to write a corpus in which a gold passage carries its own conclusion."""
    by_id = {p.passage_id: p for p in passages}
    for example in examples:
        passage = by_id[example.passage_id]
        conclusion = normalise_text(example.long_answer)
        if conclusion and conclusion in normalise_text(f"{passage.title} {passage.text}"):
            raise LeakageError(f"conclusion of PMID {example.qid} is present in its passage")


def build(directory: Path, *, max_distractors: int | None) -> dict[str, Any]:
    """Write `passages.jsonl` and `labeled.jsonl` into `directory`; return the manifest."""
    examples, gold = _read_labeled()
    distractors = _read_distractors(
        {p.passage_id for p in gold}, max_distractors, get_settings().seed
    )
    passages = gold + distractors
    if len({p.passage_id for p in passages}) != len(passages):
        raise SourceUnavailable("duplicate PMIDs across PQA-L and PQA-U")
    _assert_no_label_leak(examples, passages)

    write_jsonl(directory / "labeled.jsonl", examples)
    write_jsonl(directory / "passages.jsonl", passages)
    labels = Counter(e.label for e in examples)
    console.print(
        f"pubmedqa: {len(passages):,} passages "
        f"({len(gold):,} labelled + {len(distractors):,} distractors); "
        f"labels {dict(labels)}"
    )
    return {
        "n_passages": len(passages),
        "n_labeled": len(examples),
        "n_distractors": len(distractors),
        "label_distribution": {d: labels[d] for d in DECISIONS},
        "params": {"max_distractors": max_distractors, "seed": get_settings().seed},
        "sources": {
            "repo": PQA_LABELED.repo,
            "revision": PQA_LABELED.revision,
            "files": [PQA_LABELED.filename, PQA_UNLABELED.filename],
        },
    }
