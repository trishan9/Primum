"""Generation layer with a stub LLM.

`StubLLM` has exactly the signatures SPEC section 3.2 gives for `LLMClient`, so these tests also
pin down that the generation code calls nothing beyond that contract.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from primum.config import Settings
from primum.generation import answer_benchmark, extract_claims, format_evidence, stream_chat_answer
from primum.generation.benchmark import _BenchmarkWire
from primum.generation.chat import _ClaimsWire
from primum.llm import LLMClient, LLMInvalidOutput, LLMResult, prompts
from primum.types import Hit, Passage, Usage


def hit(n: int, text: str, title: str = "") -> Hit:
    passage = Passage(passage_id=f"p{n}", doc_id=f"d{n}", title=title, text=text, source="pubmed")
    return Hit(passage=passage, score=1.0 / n, rank=n, retriever="test")


HITS = [
    hit(1, "Statin therapy reduced LDL cholesterol by 30% versus placebo.", "Statins and LDL"),
    hit(2, "No difference in all-cause mortality\n  was observed at 12 months."),
]


class StubLLM:
    """Returns a canned structured reply and records how it was called."""

    def __init__(self, reply: dict | None = None, deltas: list[str] | None = None) -> None:
        self.reply = reply
        self.deltas = deltas or []
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel] | None = None,
        model: str | None = None,
        max_tokens: int = 2000,
        effort: str | None = None,
        sample_id: int = 0,
        tag: str = "",
    ) -> LLMResult:
        self.calls.append(
            {"system": system, "user": user, "schema": schema, "sample_id": sample_id, "tag": tag}
        )
        text = json.dumps(self.reply)
        usage = Usage(input_tokens=10, output_tokens=5, cost_usd=0.001)
        return LLMResult(
            text=text, parsed=schema.model_validate_json(text), usage=usage, model="stub"
        )

    async def astream(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 4000,
        tag: str = "",
    ) -> AsyncIterator[str]:
        self.calls.append({"system": system, "user": user, "tag": tag})
        for delta in self.deltas:
            yield delta


def benchmark_reply(**overrides: Any) -> dict:
    reply = {
        "claims": [{"text": "Statin therapy reduced LDL cholesterol by 30%.", "cited": [1]}],
        "p_yes": 0.7,
        "p_no": 0.1,
        "p_maybe": 0.2,
        "decision": "yes",
    }
    return {**reply, **overrides}


# ------------------------------------------------------------------ evidence formatting


def test_format_evidence_numbers_by_position_and_collapses_whitespace():
    assert format_evidence(HITS) == (
        "[1] (Statins and LDL) Statin therapy reduced LDL cholesterol by 30% versus placebo.\n\n"
        "[2] No difference in all-cause mortality was observed at 12 months."
    )
    # Position, not Hit.rank: a re-ordered list is renumbered from 1.
    assert format_evidence([HITS[1]]).startswith("[1] No difference")
    assert format_evidence([]) == ""


# ------------------------------------------------------------------ benchmark mode


def test_benchmark_with_evidence():
    llm = StubLLM(benchmark_reply())

    answer, usage = answer_benchmark("  Do statins lower LDL?  ", HITS, sample_id=3, llm=llm)

    (call,) = llm.calls
    assert call["system"] == prompts.BENCHMARK_SYSTEM_EVIDENCE
    assert format_evidence(HITS) in call["user"]
    assert "<question>\nDo statins lower LDL?\n</question>" in call["user"]
    assert call["sample_id"] == 3
    assert call["tag"] == "benchmark.evidence"
    assert answer.decision == "yes"
    assert answer.probabilities == pytest.approx({"yes": 0.7, "no": 0.1, "maybe": 0.2})
    assert [(c.text, c.cited) for c in answer.claims] == [
        ("Statin therapy reduced LDL cholesterol by 30%.", [1])
    ]
    assert usage.cost_usd == 0.001


def test_benchmark_closed_book_uses_its_own_wording_and_no_citations():
    llm = StubLLM(benchmark_reply())

    answer, _ = answer_benchmark("Do statins lower LDL?", [], llm=llm)

    (call,) = llm.calls
    assert call["system"] == prompts.BENCHMARK_SYSTEM_CLOSED_BOOK
    assert call["tag"] == "benchmark.closed_book"
    assert "No evidence passages are provided" in call["user"]
    assert "<evidence>" not in call["user"]
    # The model cited passage 1, but none was shown: the citation cannot stand.
    assert answer.claims[0].cited == []


def test_benchmark_conditions_share_the_task_and_differ_only_on_evidence():
    evidence, closed = prompts.BENCHMARK_SYSTEM_EVIDENCE, prompts.BENCHMARK_SYSTEM_CLOSED_BOOK
    assert evidence != closed
    shared = prompts._BENCHMARK_TASK
    assert evidence.startswith(shared) and closed.startswith(shared)
    for field in ("claims", "p_yes", "p_no", "p_maybe", "decision"):
        assert field in shared


def test_benchmark_probabilities_are_rescaled_to_sum_to_one():
    llm = StubLLM(benchmark_reply(p_yes=70, p_no=10, p_maybe=20))
    answer, _ = answer_benchmark("q?", HITS, llm=llm)
    assert answer.probabilities == pytest.approx({"yes": 0.7, "no": 0.1, "maybe": 0.2})
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "probabilities",
    [
        {"p_yes": 0.0, "p_no": 0.0, "p_maybe": 0.0},
        {"p_yes": 1.2, "p_no": -0.2, "p_maybe": 0.0},
    ],
)
def test_benchmark_rejects_stated_probabilities_that_are_not_a_distribution(probabilities):
    llm = StubLLM(benchmark_reply(**probabilities))
    with pytest.raises(LLMInvalidOutput):
        answer_benchmark("q?", HITS, llm=llm)


def test_benchmark_keeps_the_stated_decision_even_against_the_probabilities():
    llm = StubLLM(benchmark_reply(decision="maybe"))
    answer, _ = answer_benchmark("q?", HITS, llm=llm)
    assert answer.decision == "maybe"
    assert answer.probabilities["yes"] == pytest.approx(0.7)


def test_benchmark_claims_are_cleaned_and_capped_at_four():
    claims = [
        {"text": " Claim one. ", "cited": [2, 2, 1]},
        {"text": "   ", "cited": [1]},
        {"text": "Claim two.", "cited": [0, 3, -1]},
        {"text": "Claim three.", "cited": []},
        {"text": "Claim four.", "cited": [1]},
        {"text": "Claim five.", "cited": [1]},
    ]
    llm = StubLLM(benchmark_reply(claims=claims))

    answer, _ = answer_benchmark("q?", HITS, llm=llm)

    assert [(c.text, c.cited) for c in answer.claims] == [
        ("Claim one.", [2, 1]),  # duplicates removed, order kept
        ("Claim two.", []),  # passages 0, 3 and -1 were never shown
        ("Claim three.", []),
        ("Claim four.", [1]),
    ]


# ------------------------------------------------------------------ chat mode


async def test_stream_chat_answer_yields_deltas_bound_to_the_passages():
    llm = StubLLM(deltas=["Statins lower ", "LDL cholesterol [1]."])

    deltas = [d async for d in stream_chat_answer("Do statins work?", HITS, llm=llm)]

    assert deltas == ["Statins lower ", "LDL cholesterol [1]."]
    (call,) = llm.calls
    assert call["system"] == prompts.CHAT_SYSTEM
    assert call["tag"] == "chat.answer"
    assert format_evidence(HITS) in call["user"]
    assert "Do statins work?" in call["user"]


async def test_stream_chat_answer_says_when_nothing_was_retrieved():
    llm = StubLLM(deltas=["The sources I found do not address this."])
    _ = [d async for d in stream_chat_answer("q?", [], llm=llm)]
    assert prompts.NO_PASSAGES in llm.calls[0]["user"]


async def test_stream_chat_answer_forwards_on_usage_only_when_asked():
    class UsageReportingLLM(StubLLM):
        async def astream(self, *, on_usage, **kwargs) -> AsyncIterator[str]:
            async for delta in super().astream(**kwargs):
                yield delta
            on_usage(Usage(input_tokens=7, output_tokens=3, cost_usd=0.002))

    seen: list[Usage] = []
    llm = UsageReportingLLM(deltas=["ok"])
    _ = [d async for d in stream_chat_answer("q?", HITS, llm=llm, on_usage=seen.append)]
    assert [u.cost_usd for u in seen] == [0.002]


def test_extract_claims_returns_the_answer_unchanged_with_cleaned_claims():
    reply = {
        "claims": [
            {"text": "Statins lower LDL cholesterol.", "cited": [1]},
            {"text": "Statins did not change mortality at 12 months.", "cited": [2, 9]},
        ],
        "verbal_confidence": 0.85,
    }
    llm = StubLLM(reply)
    answer_text = "Statins lower LDL cholesterol [1]. Mortality did not change [2]."

    answer, usage = extract_claims("Do statins work?", answer_text, HITS, llm=llm)

    (call,) = llm.calls
    assert call["system"] == prompts.CLAIMS_SYSTEM
    assert call["tag"] == "chat.claims"
    assert f"<answer>\n{answer_text}\n</answer>" in call["user"]
    assert format_evidence(HITS) in call["user"]
    assert answer.text == answer_text
    assert [c.cited for c in answer.claims] == [[1], [2]]
    assert answer.verbal_confidence == 0.85
    assert usage.cost_usd == 0.001


def test_extract_claims_accepts_an_answer_with_no_checkable_claims():
    llm = StubLLM({"claims": [], "verbal_confidence": 0.3})
    answer, _ = extract_claims("q?", "The sources I found do not address this.", [], llm=llm)
    assert answer.claims == []
    assert answer.verbal_confidence == 0.3


@pytest.mark.parametrize("confidence", [85, -0.1])
def test_extract_claims_rejects_a_confidence_that_is_not_a_probability(confidence):
    llm = StubLLM({"claims": [], "verbal_confidence": confidence})
    with pytest.raises(LLMInvalidOutput):
        extract_claims("q?", "answer", HITS, llm=llm)


# ------------------------------------------------------------------ prompts and schemas


def test_templates_tolerate_braces_in_passages_and_questions():
    llm = StubLLM(benchmark_reply())
    braces = [hit(1, "Hazard ratio {HR} was 0.8 {95% CI}.")]
    answer_benchmark("Is {x} > 0?", braces, llm=llm)
    assert "{HR}" in llm.calls[0]["user"] and "Is {x} > 0?" in llm.calls[0]["user"]


@pytest.mark.parametrize("schema", [_BenchmarkWire, _ClaimsWire])
def test_wire_schemas_stay_inside_the_structured_output_subset(schema):
    """Constrained decoding rejects open objects and numeric/length/array-size bounds."""
    wire = json.dumps(anthropic.transform_schema(schema))
    unsupported = (
        "minimum",
        "maximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    )
    assert not any(f'"{keyword}"' in wire for keyword in unsupported)

    def objects(node: Any):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from objects(value)
        elif isinstance(node, list):
            for value in node:
                yield from objects(value)

    found = list(objects(json.loads(wire)))
    assert found
    for obj in found:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_benchmark_wire_schema_orders_claims_before_probabilities_before_decision():
    assert list(_BenchmarkWire.model_fields) == ["claims", "p_yes", "p_no", "p_maybe", "decision"]


# ------------------------------------------------------------------ through the real client


def test_answer_benchmark_through_llm_client_and_sdk(tmp_path):
    """End to end below the network: generation -> LLMClient -> anthropic SDK -> fake HTTP."""
    bodies: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        reply = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": json.dumps(benchmark_reply())},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 900, "output_tokens": 120},
        }
        return httpx2.Response(200, json=reply)

    settings = Settings(
        _env_file=None,
        ANTHROPIC_API_KEY="test-key",
        artifacts_dir=tmp_path,
        llm_model="claude-haiku-5-5",
    )
    sdk = anthropic.Anthropic(
        api_key="test-key",
        base_url="https://fake.invalid",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    llm = LLMClient(settings, sync_client=sdk)

    first, usage = answer_benchmark("Do statins lower LDL?", HITS, llm=llm)
    again, usage_again = answer_benchmark("Do statins lower LDL?", HITS, llm=llm)
    answer_benchmark("Do statins lower LDL?", HITS, sample_id=1, llm=llm)

    assert first.decision == "yes" and first.claims[0].cited == [1]
    assert again == first
    assert (usage.cached, usage_again.cached) == (False, True)
    assert len(bodies) == 2  # sample 0 once, sample 1 once
    assert bodies[0]["system"] == prompts.BENCHMARK_SYSTEM_EVIDENCE
    properties = bodies[0]["output_config"]["format"]["schema"]["properties"]
    assert list(properties) == ["claims", "p_yes", "p_no", "p_maybe", "decision"]
