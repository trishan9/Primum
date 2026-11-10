"""Central configuration. Read once via `get_settings()`; override with PRIMUM_* env vars."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

# A generator id used as one directory name under artifacts/deployment/: no separators, no "..".
SAFE_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def is_safe_model_name(name: str) -> bool:
    """True if `name` can be used as a single path component for a deployment bundle."""
    return bool(SAFE_MODEL_NAME.fullmatch(name))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_prefix="PRIMUM_", extra="ignore"
    )

    # --- paths -------------------------------------------------------------
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "artifacts"
    results_dir: Path = ROOT / "results"

    # --- LLM ---------------------------------------------------------------
    # Read without the PRIMUM_ prefix so the standard variable name works.
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    # Pinned explicitly so an ANTHROPIC_BASE_URL inherited from the shell is ignored.
    anthropic_base_url: str = "https://api.anthropic.com"
    llm_model: str = "claude-haiku-5-5"
    llm_effort: str = "low"
    budget_usd: float = 15.0
    llm_concurrency: int = 8

    # --- retrieval ---------------------------------------------------------
    base_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"  # the prototype's encoder
    biomed_encoder: str = "NeuML/pubmedbert-base-embeddings"  # off-the-shelf domain baseline
    finetuned_encoder_dir: Path = ROOT / "artifacts" / "encoder-ft"
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    rrf_k: int = 60
    top_k: int = 5  # passages given to the generator
    candidate_k: int = 50  # candidates per retriever before fusion / reranking
    default_retriever: str = "hybrid-ft"
    chat_corpus: str = "kb"

    # --- uncertainty -------------------------------------------------------
    alpha: float = 0.10  # target selective risk
    delta: float = 0.10  # guarantee holds with probability >= 1 - delta
    self_consistency_samples: int = 5
    seed: int = 13

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def deployment_dir(self) -> Path:
        """Fitted confidence model + certified thresholds used by the live system.

        One bundle per generator: thresholds certified on one model's answers say nothing
        about another's, so switching `llm_model` switches bundles (or finds none).
        """
        return self.deployment_dir_for(self.llm_model)

    def deployment_dir_for(self, llm_model: str) -> Path:
        return self.artifacts_dir / "deployment" / llm_model

    @property
    def cache_db(self) -> Path:
        return self.artifacts_dir / "llm_cache.sqlite"

    @property
    def trace_db(self) -> Path:
        return self.artifacts_dir / "traces.sqlite"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.raw_dir, s.processed_dir, s.index_dir, s.artifacts_dir, s.results_dir):
        d.mkdir(parents=True, exist_ok=True)
    return s
