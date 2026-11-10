"""Live smoke tests against the Claude API. Deselected by default; run with `-m llm`.

They use the configured key and model but a throwaway cache and a five-cent budget, so they
neither read nor pollute the project's ledger. Assertions are about the contract (shape,
citations in range, a second call served from cache), never about what the model answers.
"""

from __future__ import annotations

import pytest

from primum.config import Settings, get_settings
from primum.generation import answer_benchmark, extract_claims, stream_chat_answer
from primum.llm import LLMClient
from primum.types import DECISIONS, Hit, Passage

pytestmark = pytest.mark.llm

HITS = [
    Hit(
        passage=Passage(
            passage_id="demo-1",
            doc_id="demo-1",
            title="Common cold",
            text=(
                "The common cold is a viral infection of the nose and throat. Most people "
                "recover in 7 to 10 days. Antibiotics do not work against cold viruses."
            ),
            source="medlineplus",
        ),
        score=1.0,
        rank=1,
        retriever="demo",
    )
]


@pytest.fixture
def llm(tmp_path) -> LLMClient:
    configured = get_settings()
    if not configured.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is not configured")
    return LLMClient(
        Settings(
            ANTHROPIC_API_KEY=configured.anthropic_api_key,
            artifacts_dir=tmp_path,
            budget_usd=0.05,
            llm_model=configured.llm_model,
            llm_effort=configured.llm_effort,
        )
    )


def test_benchmark_answer_live(llm):
    question = "Do antibiotics shorten the common cold?"
    for hits in (HITS, []):
        answer, usage = answer_benchmark(question, hits, llm=llm)
        assert answer.decision in DECISIONS
        assert sum(answer.probabilities.values()) == pytest.approx(1.0)
        assert 1 <= len(answer.claims) <= 4
        assert all(1 <= n <= len(hits) for claim in answer.claims for n in claim.cited)
        assert usage.cost_usd > 0 and not usage.cached

    _, replay = answer_benchmark(question, HITS, llm=llm)
    assert replay.cached and replay.cost_usd == 0


async def test_chat_answer_live(llm):
    question = "How long does a cold last?"
    usages = []
    deltas = [d async for d in stream_chat_answer(question, HITS, llm=llm, on_usage=usages.append)]
    text = "".join(deltas)
    assert text.strip()
    assert len(usages) == 1 and usages[0].cost_usd > 0

    answer, _ = extract_claims(question, text, HITS, llm=llm)
    assert answer.text == text
    assert 0.0 <= answer.verbal_confidence <= 1.0
    assert all(n == 1 for claim in answer.claims for n in claim.cited)
    assert llm.spend()["calls"] == 2
