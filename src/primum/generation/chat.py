"""Chat mode: stream an evidence-bound answer, then decompose it into verifiable claims.

The two steps are separate calls on purpose. The reader gets prose as it is written, with
no JSON in the way; the decomposition then works on the finished text, so the claims that
are verified are the ones the reader was actually shown.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Callable

from pydantic import BaseModel

from primum.generation.evidence import WireClaim, evidence_block, to_claims
from primum.llm import LLMClient, LLMInvalidOutput, get_llm, prompts
from primum.types import ChatAnswer, Hit, Usage

# Up to ten claims of a sentence each, plus the model's reasoning, which shares this cap.
_CLAIMS_MAX_TOKENS = 3000


class _ClaimsWire(BaseModel):
    claims: list[WireClaim]
    verbal_confidence: float


async def stream_chat_answer(
    question: str,
    hits: list[Hit],
    *,
    llm: LLMClient | None = None,
    on_usage: Callable[[Usage], None] | None = None,
) -> AsyncIterator[str]:
    """Yield the text deltas of a plain-language answer with `[n]` citation markers.

    With no hits the model is told so and says the sources do not cover the question.
    `on_usage`, if given, receives the token usage and cost once the stream has ended.
    """
    llm = llm or get_llm()
    user = prompts.CHAT_USER.format(evidence=evidence_block(hits), question=question.strip())
    extra = {"on_usage": on_usage} if on_usage is not None else {}
    async for delta in llm.astream(
        system=prompts.CHAT_SYSTEM, user=user, tag="chat.answer", **extra
    ):
        yield delta


def extract_claims(
    question: str,
    answer_text: str,
    hits: list[Hit],
    *,
    llm: LLMClient | None = None,
) -> tuple[ChatAnswer, Usage]:
    """Decompose a finished answer into atomic claims with citations and a verbal confidence.

    The returned `ChatAnswer.text` is `answer_text` unchanged; an answer that makes no
    checkable claim (it only says the sources do not cover the question) yields `claims=[]`.
    """
    llm = llm or get_llm()
    user = prompts.CLAIMS_USER.format(
        evidence=evidence_block(hits), question=question.strip(), answer=answer_text.strip()
    )
    result = llm.complete(
        system=prompts.CLAIMS_SYSTEM,
        user=user,
        schema=_ClaimsWire,
        max_tokens=_CLAIMS_MAX_TOKENS,
        tag="chat.claims",
    )
    wire: _ClaimsWire = result.parsed
    confidence = wire.verbal_confidence
    # The schema cannot bound a number. A value outside [0, 1] is on an unknown scale, and
    # guessing the scale would put a made-up feature into the confidence model.
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise LLMInvalidOutput(f"verbal confidence {confidence!r} is not a probability")
    answer = ChatAnswer(
        text=answer_text,
        claims=to_claims(wire.claims, len(hits)),
        verbal_confidence=confidence,
    )
    return answer, result.usage
