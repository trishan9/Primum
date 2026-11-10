"""Where the raw data comes from and how it reaches `data/raw/<source>/`.

Every source is openly licensed and fetched by code, so the corpora can be rebuilt from
nothing. Hugging Face files are pinned to a commit: an experiment re-run next year reads
byte-identical inputs. MedlinePlus publishes a new dated snapshot most days and keeps no
archive, so the snapshot on disk is kept until it is deleted by hand and its date is
recorded in the corpus manifest.

Downloaded files are untrusted input. They are only ever parsed (parquet, TSV, XML read
from inside the zip without extracting it); nothing here executes or unpacks to disk.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from huggingface_hub import hf_hub_download

from primum.config import get_settings
from primum.data._io import console
from primum.data.errors import SourceUnavailable


@dataclass(frozen=True)
class HFFile:
    """One file of a Hugging Face dataset repository at a fixed revision."""

    repo: str
    revision: str
    filename: str
    source: str  # sub-directory of data/raw/

    @property
    def directory(self) -> Path:
        return get_settings().raw_dir / self.source

    @property
    def path(self) -> Path:
        return self.directory / self.filename


_PUBMEDQA = ("qiaojin/PubMedQA", "9001f2853fb87cab8d220904e0de81ac6973b318")
_NFCORPUS = ("BeIR/nfcorpus", "b5026a0e96e8a7ac4f95f482a596389289d46269")
_NFCORPUS_QRELS = ("BeIR/nfcorpus-qrels", "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65")

PQA_LABELED = HFFile(*_PUBMEDQA, "pqa_labeled/train-00000-of-00001.parquet", "pubmedqa")
PQA_UNLABELED = HFFile(*_PUBMEDQA, "pqa_unlabeled/train-00000-of-00001.parquet", "pubmedqa")
PQA_ARTIFICIAL = HFFile(*_PUBMEDQA, "pqa_artificial/train-00000-of-00001.parquet", "pubmedqa")
NFCORPUS_DOCS = HFFile(*_NFCORPUS, "corpus/corpus-00000-of-00001.parquet", "nfcorpus")
NFCORPUS_QUERIES = HFFile(*_NFCORPUS, "queries/queries-00000-of-00001.parquet", "nfcorpus")
NFCORPUS_QRELS = {
    split: HFFile(*_NFCORPUS_QRELS, f"{split}.tsv", "nfcorpus")
    for split in ("train", "dev", "test")
}

MEDLINEPLUS_INDEX_URL = "https://medlineplus.gov/xml.html"
# The snapshot link is read from a web page, so the pattern pins scheme, host and path:
# whatever the page says, the download can only ever come from medlineplus.gov/xml/.
_MEDLINEPLUS_ZIP = re.compile(
    r"https://medlineplus\.gov/xml/mplus_topics_compressed_(\d{4}-\d{2}-\d{2})\.zip"
)
_USER_AGENT = "primum-research/0.2 (academic corpus build)"


def fetch(file: HFFile) -> Path:
    """Return the local path of `file`, downloading it on first use."""
    if not file.path.exists():
        console.print(f"[dim]downloading[/dim] {file.repo}:{file.filename}")
        hf_hub_download(
            file.repo,
            file.filename,
            repo_type="dataset",
            revision=file.revision,
            local_dir=file.directory,
        )
    return file.path


def fetch_medlineplus_topics() -> Path:
    """Return the compressed MedlinePlus health-topics XML, downloading the newest if absent."""
    directory = get_settings().raw_dir / "medlineplus"
    # ISO dates in the file name sort chronologically.
    snapshots = sorted(directory.glob("mplus_topics_compressed_*.zip"))
    if snapshots:
        return snapshots[-1]

    with httpx.Client(
        follow_redirects=True, timeout=120, headers={"User-Agent": _USER_AGENT}
    ) as client:
        page = client.get(MEDLINEPLUS_INDEX_URL)
        page.raise_for_status()
        links = {m.group(1): m.group(0) for m in _MEDLINEPLUS_ZIP.finditer(page.text)}
        if not links:
            raise SourceUnavailable(
                f"no 'MedlinePlus Health Topic XML' zip link found on {MEDLINEPLUS_INDEX_URL}; "
                "the page layout may have changed"
            )
        url = links[max(links)]
        target = directory / url.rsplit("/", 1)[-1]
        directory.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]downloading[/dim] {url}")
        partial = target.with_name(target.name + ".part")
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                for block in response.iter_bytes():
                    f.write(block)
        os.replace(partial, target)
    return target
