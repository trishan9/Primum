"""Small file helpers shared by the builders: atomic writes, JSONL, console."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from rich.console import Console

console = Console(highlight=False)

MANIFEST = "manifest.json"


def atomic_write_lines(path: Path, lines: Iterable[str]) -> int:
    """Write `lines` to `path` via a temporary file and return how many were written.

    Builders decide whether to skip work by checking that outputs exist, so a build
    interrupted half-way must never leave a truncated file under the final name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
            n += 1
    os.replace(tmp, path)
    return n


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    return atomic_write_lines(
        path,
        (
            r.model_dump_json() if isinstance(r, BaseModel) else json.dumps(r, ensure_ascii=False)
            for r in rows
        ),
    )


def iter_jsonl_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    """Record what was built and from what. Written last: its presence marks a complete build."""
    atomic_write_lines(directory / MANIFEST, [json.dumps(manifest, indent=2, ensure_ascii=False)])


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def normalise_text(text: str) -> str:
    """Case-, punctuation- and whitespace-insensitive form used by the leakage checks."""
    return re.sub(r"\W+", " ", text.lower()).strip()
