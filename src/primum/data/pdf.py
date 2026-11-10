"""Optional local PDFs (`data/raw/pdf/*.pdf`) → passages for the `kb` corpus.

Off by default. The default knowledge base is built only from openly licensed sources, so
it can be rebuilt and redistributed by anyone; a user's own PDFs (a textbook, say) are
typically copyrighted, stay git-ignored, and are indexed only when the environment
variable `PRIMUM_INCLUDE_PDF=1` is set at build time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm

from primum.config import get_settings
from primum.data.chunking import chunk_text
from primum.types import Passage

INCLUDE_PDF_ENV = "PRIMUM_INCLUDE_PDF"

_LINE_BREAK_HYPHEN = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_FIRST_BULLET = re.compile(r":\s*[•▪◦]\s*")  # the bullet right after a lead-in colon
_BULLET = re.compile(r"\s*[•▪◦]\s*")


def pdf_enabled() -> bool:
    return os.environ.get(INCLUDE_PDF_ENV) == "1"


def local_pdfs() -> list[Path]:
    return sorted((get_settings().raw_dir / "pdf").glob("*.pdf"))


def clean_page_text(text: str) -> str:
    """Undo typesetting artefacts of extracted PDF text.

    Words hyphenated across a line break are re-joined ("symp-\\ntoms" → "symptoms"); the
    rare genuine hyphen that happened to fall at a line end is lost, which costs less than
    leaving thousands of split words no retriever can match. Bullets become a
    semicolon-separated list so each one is not mistaken for a sentence.
    """
    text = _LINE_BREAK_HYPHEN.sub("", text)
    text = _FIRST_BULLET.sub(": ", text)
    text = _BULLET.sub("; ", text)
    return " ".join(text.split()).lstrip("; ")


def pdf_passages(path: Path, *, max_words: int = 220, overlap: int = 40) -> list[Passage]:
    """Chunk each page of `path` into passages that cite their page number.

    Pages are chunked independently: a sentence that runs over a page break is cut, in
    exchange for every passage mapping to exactly one page a reader can open and check.
    """
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError(f"{path.name} is encrypted; decrypt it before indexing")
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:48]
    title = ((reader.metadata.title if reader.metadata else None) or "").strip()
    title = title or " ".join(path.stem.replace("_", " ").split())

    passages: list[Passage] = []
    for number, page in enumerate(tqdm(reader.pages, desc=path.name, unit="page"), start=1):
        text = clean_page_text(page.extract_text() or "")
        for i, chunk in enumerate(chunk_text(text, max_words=max_words, overlap=overlap)):
            passages.append(
                Passage(
                    passage_id=f"pdf-{slug}-p{number}-{i}",
                    doc_id=f"pdf-{slug}",
                    title=title,
                    text=chunk,
                    source="pdf",
                    meta={"page": number, "file": path.name},
                )
            )
    return passages
