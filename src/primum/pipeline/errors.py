"""How engine failures are reported: each exception maps to one of four kinds a user can act on.

    no_api_key  no usable Anthropic key            -> set ANTHROPIC_API_KEY
    budget      the spend ledger hit its ceiling   -> raise PRIMUM_BUDGET_USD
    no_index    a corpus, index or encoder is missing -> run the `primum` command in the message
    internal    anything else (a bug, a refused or malformed model reply, a corrupt artefact)

The pipeline and the API share this table so the same failure is named the same way in an SSE
`error` event and in a JSON error body.

Messages reach unauthenticated clients, so they never carry server internals: paths are shown
relative to the project root, and an exception nobody wrote a message for is reported only by
an error id, under which the full traceback is logged on the server.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Literal

from primum.config import Settings, get_settings
from primum.data import CorpusNotBuilt
from primum.llm import BudgetExceeded, LLMError, LLMUnavailable
from primum.retrieval import EncoderNotFound
from primum.uncertainty import UnscoredRetrieval

log = logging.getLogger(__name__)

ErrorKind = Literal["no_api_key", "budget", "no_index", "internal"]

# Stages whose only file inputs are corpora and indexes, so a missing file there can only mean
# an artefact that has not been built yet.
_INDEX_STAGES = frozenset({"concepts", "retrieve"})


def error_kind(exc: BaseException, stage: str | None = None) -> ErrorKind:
    """The user-facing kind of `exc`, raised while running `stage` (if known)."""
    if isinstance(exc, LLMUnavailable):
        return "no_api_key"
    if isinstance(exc, BudgetExceeded):
        return "budget"
    if isinstance(exc, CorpusNotBuilt | EncoderNotFound):
        return "no_index"
    if isinstance(exc, FileNotFoundError) and stage in _INDEX_STAGES:
        return "no_index"
    return "internal"


class UnmeasurableFeatures(ValueError):
    """The deployment bundle's model reads a signal this mode of the pipeline cannot produce."""


# Typed errors whose text was written for the person running Primum and names the fix.
_SELF_EXPLANATORY = (
    LLMUnavailable,
    BudgetExceeded,
    LLMError,
    FileNotFoundError,  # including CorpusNotBuilt and EncoderNotFound
)
# Configuration errors Primum raises on purpose: their text is safe to show, and their type
# name says which check failed.
_CONFIGURATION = (UnmeasurableFeatures, UnscoredRetrieval)


def error_message(exc: BaseException, stage: str | None = None) -> str:
    """A one-line description of `exc`, raised while running `stage`, safe to send to a client.

    Errors Primum raises on purpose already name the fix, so their text is used, with server
    paths made relative. Anything else (a bug, a database or validation error) may carry
    internals, so the client gets an error id and the traceback is logged under it. Every
    `internal` failure is logged here, so callers need not log it again.
    """
    text = display_paths(str(exc).strip())
    if text and isinstance(exc, _SELF_EXPLANATORY + _CONFIGURATION):
        if error_kind(exc, stage) == "internal":
            log.error("%s failed: %s", stage or "request", text, exc_info=exc)
        return text if isinstance(exc, _SELF_EXPLANATORY) else f"{type(exc).__name__}: {text}"
    error_id = uuid.uuid4().hex[:12]
    log.error("%s failed (error id %s)", stage or "request", error_id, exc_info=exc)
    return f"Internal error (id {error_id}); the server log has the details."


def display_paths(text: str, settings: Settings | None = None) -> str:
    """`text` with every absolute path under the project root written relative to it.

    Configured directories outside the root (PRIMUM_DATA_DIR=/mnt/...) are named by their
    setting, so the message still says where to look without revealing the server's layout.
    """
    settings = settings or get_settings()
    labels: dict[Path, str] = {settings.root: "."}
    for directory, variable in (
        (settings.data_dir, "PRIMUM_DATA_DIR"),
        (settings.artifacts_dir, "PRIMUM_ARTIFACTS_DIR"),
        (settings.results_dir, "PRIMUM_RESULTS_DIR"),
        (settings.finetuned_encoder_dir, "PRIMUM_FINETUNED_ENCODER_DIR"),
    ):
        if not directory.is_relative_to(settings.root):
            labels[directory] = f"${variable}"
    # Longest first, so a directory nested in another is replaced by its own label. The
    # lookahead stops "/srv/primum" from matching inside "/srv/primum-old" or "/srv/primum.v2"
    # while still matching before a sentence's closing full stop.
    for directory in sorted(labels, key=lambda d: len(str(d)), reverse=True):
        text = re.sub(re.escape(str(directory)) + r"(?![\w-]|\.\w)", labels[directory], text)
    return text
