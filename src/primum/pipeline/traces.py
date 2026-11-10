"""Persistent record of every completed pipeline run (`settings.trace_db`).

A trace is stored exactly as it was sent to the client in the `done` event, so what a reader
saw and what the record says can never drift apart. A few fields are copied into columns so
the list view can be served without parsing every payload.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    created    TEXT NOT NULL,
    question   TEXT NOT NULL,
    mode       TEXT NOT NULL,
    triage     TEXT NOT NULL,
    answered   INTEGER,          -- NULL when no confidence decision was made (emergency)
    payload    TEXT NOT NULL     -- the `done` event data, verbatim
)
"""


class TraceStore:
    """sqlite-backed trace store. One short-lived connection per call keeps it thread-safe."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # `closing` closes the connection; the inner context manager commits or rolls back.
        with closing(sqlite3.connect(self.path, timeout=10)) as conn, conn:
            yield conn

    def save(self, result: dict[str, Any]) -> None:
        """Store one `PipelineResult` payload (JSON-ready dict, as emitted in `done`)."""
        confidence = result.get("confidence") or {}
        answered = confidence.get("answered")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result["trace_id"],
                    datetime.now(UTC).isoformat(timespec="seconds"),
                    result["question"],
                    result["mode"],
                    result["triage"]["level"],
                    None if answered is None else int(answered),
                    json.dumps(result),
                ),
            )

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Summaries of the most recent traces, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT trace_id, created, question, mode, triage, answered FROM traces "
                "ORDER BY created DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "trace_id": trace_id,
                "created": created,
                "question": question,
                "mode": mode,
                "triage": triage,
                "answered": None if answered is None else bool(answered),
            }
            for trace_id, created, question, mode, triage, answered in rows
        ]

    def get(self, trace_id: str) -> dict[str, Any] | None:
        """The full stored payload, or None if no trace has this id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])
