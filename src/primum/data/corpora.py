"""Build the three corpora and read them back.

Processed layout (everything under `settings.processed_dir`):

    pubmedqa/passages.jsonl   labeled.jsonl                       manifest.json
    nfcorpus/passages.jsonl   queries.<split>.jsonl  qrels.<split>.tsv  manifest.json
    kb/passages.jsonl         concepts.jsonl                      manifest.json

`manifest.json` is written last and records counts, parameters and source revisions; a
corpus directory without one is treated as not built.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args

from primum.config import get_settings
from primum.data import medlineplus, nfcorpus, pubmedqa
from primum.data._io import (
    console,
    iter_jsonl_lines,
    read_manifest,
    write_jsonl,
    write_manifest,
)
from primum.data.errors import CorpusNotBuilt, SourceUnavailable
from primum.data.pdf import INCLUDE_PDF_ENV, local_pdfs, pdf_enabled, pdf_passages
from primum.data.sources import fetch_medlineplus_topics
from primum.types import Passage, PubMedQAExample

CorpusName = Literal["pubmedqa", "nfcorpus", "kb"]
Split = Literal["train", "dev", "test"]


def build_corpus(
    name: CorpusName, *, force: bool = False, max_distractors: int | None = None
) -> Path:
    """Download (if needed) and process one corpus; return its directory.

    Idempotent: an existing build is left untouched unless `force=True`. `max_distractors`
    caps the number of PQA-U abstracts added to `pubmedqa` (None = all ~61k); it exists
    for quick smoke builds and is ignored by `nfcorpus`. `kb` reuses the `pubmedqa`
    passages as they are on disk, building that corpus first if it is missing.
    """
    if name not in get_args(CorpusName):
        raise ValueError(f"unknown corpus {name!r}; expected one of {get_args(CorpusName)}")
    directory = get_settings().processed_dir / name
    existing = read_manifest(directory)
    if existing is not None and not force:
        built_with = existing.get("params", {}).get("max_distractors")
        if name == "pubmedqa" and built_with != max_distractors:
            # Reusing a corpus of a different size is legitimate (indexes depend on it)
            # but must never go unnoticed: retrieval difficulty scales with distractors.
            console.print(
                f"[yellow]pubmedqa was built with max_distractors={built_with}, "
                f"not {max_distractors}; keeping it. Pass force=True to rebuild.[/yellow]"
            )
        console.print(f"[dim]{name}: already built ({existing['n_passages']:,} passages)[/dim]")
        return directory

    directory.mkdir(parents=True, exist_ok=True)
    # Remove the old manifest first so an interrupted rebuild is not mistaken for complete.
    (directory / "manifest.json").unlink(missing_ok=True)
    if name == "pubmedqa":
        manifest = pubmedqa.build(directory, max_distractors=max_distractors)
    elif name == "nfcorpus":
        manifest = nfcorpus.build(directory)
    else:
        manifest = _build_kb(directory, max_distractors=max_distractors)
    write_manifest(directory, {"corpus": name, **manifest})
    _read_passages.cache_clear()
    return directory


def _build_kb(directory: Path, *, max_distractors: int | None) -> dict[str, Any]:
    """The live-chat knowledge base: MedlinePlus topics + PubMed abstracts (+ opt-in PDFs)."""
    snapshot = fetch_medlineplus_topics()
    topics = list(medlineplus.iter_topics(snapshot))
    if not topics:
        raise SourceUnavailable(f"{snapshot.name} contains no English health topics")
    passages = [p for topic in topics for p in medlineplus.topic_passages(topic)]

    build_corpus("pubmedqa", max_distractors=max_distractors)
    passages += load_passages("pubmedqa")

    pdf_files: list[str] = []
    if pdf_enabled():
        for path in local_pdfs():
            passages += pdf_passages(path)
            pdf_files.append(path.name)
    elif local_pdfs():
        console.print(
            f"[dim]kb: {len(local_pdfs())} local PDF(s) not indexed; "
            f"set {INCLUDE_PDF_ENV}=1 to include them[/dim]"
        )

    if len({p.passage_id for p in passages}) != len(passages):
        raise SourceUnavailable("kb: passage ids are not unique across sources")
    write_jsonl(directory / "passages.jsonl", passages)
    write_jsonl(directory / "concepts.jsonl", map(medlineplus.topic_concept, topics))

    by_source = Counter(p.source for p in passages)
    console.print(f"kb: {len(passages):,} passages {dict(by_source)}; {len(topics):,} concepts")
    return {
        "n_passages": len(passages),
        "by_source": dict(by_source),
        "n_concepts": len(topics),
        "params": {"max_distractors": max_distractors, "include_pdf": pdf_enabled()},
        "sources": {
            "medlineplus_snapshot": snapshot.name,
            "pubmedqa": read_manifest(get_settings().processed_dir / "pubmedqa"),
            "pdf_files": pdf_files,
        },
    }


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise CorpusNotBuilt(f"{path} does not exist; run `primum data {command}` first")
    return path


@lru_cache(maxsize=4)
def _read_passages(path: Path) -> tuple[Passage, ...]:
    return tuple(Passage.model_validate_json(line) for line in iter_jsonl_lines(path))


def load_passages(corpus: str) -> list[Passage]:
    """All passages of a built corpus, in file order.

    Parsed once per process and cached (the API asks for corpus sizes on every `/api/meta`
    call); the returned list is a fresh one, the `Passage` objects are shared.
    """
    path = get_settings().processed_dir / corpus / "passages.jsonl"
    return list(_read_passages(_require(path, f"build {corpus}")))


def load_pubmedqa_labeled() -> list[PubMedQAExample]:
    """The 1,000 expert-labelled PubMedQA questions, in the dataset's original order."""
    path = get_settings().processed_dir / "pubmedqa" / "labeled.jsonl"
    return [
        PubMedQAExample.model_validate_json(line)
        for line in iter_jsonl_lines(_require(path, "build pubmedqa"))
    ]


def load_nfcorpus(split: Split) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Queries `{qid: text}` and graded judgments `qrels[qid][doc_id] = grade` of one split."""
    if split not in nfcorpus.SPLITS:
        raise ValueError(f"unknown NFCorpus split {split!r}; expected one of {nfcorpus.SPLITS}")
    directory = get_settings().processed_dir / "nfcorpus"
    queries_path = _require(directory / f"queries.{split}.jsonl", "build nfcorpus")
    qrels_path = _require(directory / f"qrels.{split}.tsv", "build nfcorpus")
    queries = {}
    for line in iter_jsonl_lines(queries_path):
        row = json.loads(line)
        queries[row["_id"]] = row["text"]
    return queries, nfcorpus.parse_qrels(qrels_path)


def load_concepts() -> list[dict]:
    """MedlinePlus concept dictionary: `{preferred, synonyms[], mesh_ids[], url}` per topic."""
    path = get_settings().processed_dir / "kb" / "concepts.jsonl"
    return [json.loads(line) for line in iter_jsonl_lines(_require(path, "build kb"))]
