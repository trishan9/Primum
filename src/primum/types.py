"""Shared data contracts. Every module exchanges these types and nothing else.

Keep this file dependency-free (pydantic only) so any module can import it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Source = Literal["pubmed", "medlineplus", "nfcorpus", "pdf"]
Decision = Literal["yes", "no", "maybe"]
DECISIONS: tuple[Decision, ...] = ("yes", "no", "maybe")
VerdictLabel = Literal["supported", "unsupported", "contradicted"]


class Passage(BaseModel):
    """One retrievable unit of text (an abstract, a topic section, a PDF chunk)."""

    passage_id: str
    doc_id: str
    title: str = ""
    text: str
    source: Source
    url: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class Hit(BaseModel):
    """A passage returned by a retriever, with the scores that put it there."""

    passage: Passage
    score: float
    rank: int  # 1-based
    retriever: str
    # Per-stage scores, e.g. {"bm25": 12.3, "dense": 0.71, "rrf": 0.032, "rerank": 4.1}
    components: dict[str, float] = Field(default_factory=dict)


class PubMedQAExample(BaseModel):
    qid: str  # the PMID as a string
    question: str
    contexts: list[str]
    context_labels: list[str] = Field(default_factory=list)
    long_answer: str
    label: Decision
    passage_id: str  # id of the gold abstract inside the "pubmedqa" corpus


class Claim(BaseModel):
    """An atomic factual statement made by the generator."""

    text: str
    cited: list[int] = Field(default_factory=list)  # 1-based indices into the hit list


class ClaimVerdict(BaseModel):
    claim: Claim
    label: VerdictLabel
    p_entail: float
    p_neutral: float
    p_contra: float
    best_hit: int | None = None  # 1-based index of the passage that best supports it
    best_sentence: str | None = None


class BenchmarkAnswer(BaseModel):
    """Structured answer for yes/no/maybe research questions (PubMedQA)."""

    decision: Decision
    probabilities: dict[Decision, float]  # the model's stated belief, sums to 1
    claims: list[Claim]  # 1-4 atomic claims justifying the decision, with citations


class ChatAnswer(BaseModel):
    text: str  # markdown with [n] citation markers
    claims: list[Claim] = Field(default_factory=list)
    verbal_confidence: float | None = None  # generator's stated confidence in [0,1]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False  # served from the local response cache (cost 0)


class ConfidenceReport(BaseModel):
    """Output of the uncertainty engine for one answer."""

    features: dict[str, float]
    p_correct: float  # calibrated probability that the answer is correct
    alpha: float  # target selective risk
    delta: float  # tolerated failure probability of the guarantee
    threshold: float | None  # lambda-hat; None means "no threshold certifies this alpha"
    answered: bool  # p_correct >= threshold
    prediction_set: list[Decision] | None = None  # conformal set, benchmark mode only


class TriageResult(BaseModel):
    level: Literal["emergency", "urgent", "routine"]
    matched: list[str] = Field(default_factory=list)
    message: str | None = None


class Concept(BaseModel):
    """A biomedical concept recognised in text and linked to a controlled vocabulary."""

    surface: str
    start: int
    end: int
    preferred: str
    mesh_ids: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    url: str | None = None


class PipelineResult(BaseModel):
    """Everything the system knows about one answered question (also the stored trace)."""

    trace_id: str
    question: str
    mode: Literal["chat", "benchmark"]
    triage: TriageResult
    concepts: list[Concept] = Field(default_factory=list)
    hits: list[Hit]
    answer_text: str
    decision: Decision | None = None
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    confidence: ConfidenceReport | None = None
    usage: Usage = Field(default_factory=Usage)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retriever: str = ""
    model: str = ""
    gold_label: Decision | None = None  # set when the question is a benchmark item
