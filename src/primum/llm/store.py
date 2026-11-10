"""SQLite persistence for the language-model layer: the response cache and the spend ledger.

Both live in one file (`settings.cache_db`) so that "what did we ask, what came back and
what did it cost" can be audited together. The connection is opened in WAL mode and the
ledger total is always read from disk, never from a process-local counter, so the API
server and a CLI experiment running at the same time see one shared budget.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key               TEXT PRIMARY KEY,
    model             TEXT NOT NULL,
    text              TEXT NOT NULL,
    input_tokens      INTEGER NOT NULL,
    output_tokens     INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    created           REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created            REAL NOT NULL,
    tag                TEXT NOT NULL,
    model              TEXT NOT NULL,
    input_tokens       INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    cache_read_tokens  INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    cost_usd           REAL NOT NULL,
    cached             INTEGER NOT NULL
);
"""


def cache_key(
    *,
    model: str,
    system: str,
    user: str,
    schema: dict | None,
    effort: str,
    max_tokens: int,
    sample_id: int,
) -> str:
    """SHA-256 over everything that determines the response distribution, plus `sample_id`.

    `sample_id` never reaches the model. It only names *which* draw from that distribution
    a caller wants: asking again with the same id replays the stored draw, a new id forces
    a fresh one. That is what makes self-consistency sampling reproducible.
    """
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "user": user,
            "schema": schema,
            "effort": effort,
            "max_tokens": max_tokens,
            "sample_id": sample_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedResponse:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


class Store:
    """Thread-safe access to the cache and ledger tables."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by worker threads and the event loop, serialised by a lock;
        # autocommit so a crash can never lose a recorded cost.
        self._conn = sqlite3.connect(
            path, check_same_thread=False, timeout=30, isolation_level=None
        )
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)

    def get(self, key: str) -> CachedResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT model, text, input_tokens, output_tokens, cache_read_tokens "
                "FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        return CachedResponse(*row) if row else None

    def put(self, key: str, response: CachedResponse) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    response.model,
                    response.text,
                    response.input_tokens,
                    response.output_tokens,
                    response.cache_read_tokens,
                    time.time(),
                ),
            )

    def record(
        self,
        *,
        tag: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost_usd: float = 0.0,
        cached: bool = False,
    ) -> None:
        """Append one ledger row. Cache hits are rows too (zero cost) so they can be counted."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO ledger (created, tag, model, input_tokens, output_tokens, "
                "cache_read_tokens, cache_write_tokens, cost_usd, cached) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    tag,
                    model,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    cost_usd,
                    int(cached),
                ),
            )

    def total_usd(self) -> float:
        with self._lock:
            (total,) = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) FROM ledger"
            ).fetchone()
        return float(total)

    def summary(self) -> dict:
        with self._lock:
            calls, cached_calls = self._conn.execute(
                "SELECT COALESCE(SUM(cached = 0), 0), COALESCE(SUM(cached = 1), 0) FROM ledger"
            ).fetchone()
            by_tag = self._conn.execute(
                "SELECT tag, SUM(cost_usd) FROM ledger GROUP BY tag ORDER BY tag"
            ).fetchall()
        return {
            "total_usd": float(sum(usd for _, usd in by_tag)),
            "by_tag": {tag or "untagged": float(usd) for tag, usd in by_tag},
            "calls": int(calls),
            "cached_calls": int(cached_calls),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
