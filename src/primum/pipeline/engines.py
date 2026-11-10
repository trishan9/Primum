"""The components the orchestrator drives, gathered in one place so they can be substituted.

`default_engines()` wires the real ones. Tests (and anything else that wants to run the pipeline
against stubs) build an `Engines` with their own callables instead; the orchestrator and the API
never import an engine directly.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from primum.config import get_settings
from primum.llm import LLMClient, get_llm
from primum.pipeline.traces import TraceStore
from primum.retrieval import Retriever, build_retriever, expand_query, link_concepts
from primum.types import Claim, ClaimVerdict, Concept, Hit
from primum.uncertainty import BUNDLE_FILENAME, DeploymentBundle
from primum.verification import verify_claims


@dataclass(frozen=True)
class Engines:
    """Every callable here may block (load a model, read an index, call the API); the orchestrator
    runs them in a worker thread. Each is responsible for its own caching."""

    retriever: Callable[[str, str], Retriever]  # (name, corpus) -> retriever
    link_concepts: Callable[[str], list[Concept]]
    expand_query: Callable[[str], str]
    llm: Callable[[], LLMClient]
    verify: Callable[[list[Claim], list[Hit]], list[ClaimVerdict]]
    bundle: Callable[[], DeploymentBundle | None]  # None until RQ3 has produced one
    traces: TraceStore


class BundleLoader:
    """Loads the deployment bundle on demand and reloads it when the file changes.

    RQ3 writes the bundle; re-reading on change means a running server picks up a newly
    certified rule without a restart, and never keeps serving thresholds that were replaced.
    """

    def __init__(self, directory: Path) -> None:
        self._path = directory / BUNDLE_FILENAME
        self._lock = threading.Lock()
        self._version: int | None = None
        self._bundle: DeploymentBundle | None = None

    def __call__(self) -> DeploymentBundle | None:
        try:
            version = self._path.stat().st_mtime_ns
        except FileNotFoundError:
            return None
        with self._lock:
            if version != self._version:
                # A corrupt bundle raises here, loudly: serving answers without the rule the
                # operator deployed would be worse than failing the confidence stage.
                self._bundle = DeploymentBundle.load(self._path.parent)
                self._version = version
            return self._bundle


@lru_cache(maxsize=1)
def default_engines() -> Engines:
    """The real engines. Models and indexes load lazily inside them, on first use, not here.

    Importing these modules is not free (torch arrives with `primum.uncertainty`), but it loads
    no weights, reads no index and makes no network call.
    """
    settings = get_settings()
    return Engines(
        retriever=build_retriever,
        link_concepts=link_concepts,
        expand_query=expand_query,
        llm=get_llm,
        verify=verify_claims,
        bundle=BundleLoader(settings.deployment_dir),
        traces=TraceStore(settings.trace_db),
    )
