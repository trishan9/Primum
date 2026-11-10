"""LLMClient against a fake Messages API.

The fake sits at the HTTP transport, below the real `anthropic` SDK. Requests are therefore
built and responses parsed by the SDK exactly as in production, and each test can assert on
the JSON body that would have gone over the wire. No network, no key, no cost.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from primum.config import Settings
from primum.llm import (
    BudgetExceeded,
    LLMClient,
    LLMInvalidOutput,
    LLMRefusal,
    LLMRequestError,
    LLMTransientError,
    LLMTruncated,
    LLMUnavailable,
)
from primum.llm.pricing import cost_usd
from primum.llm.store import cache_key
from primum.types import Usage

MODEL = "claude-haiku-5-5"
# 1,000 input and 200 output tokens on Haiku 5.5: (1000 * 0.10 + 200 * 0.50) / 1e6.
CALL_COST = 0.0002


class Verdict(BaseModel):
    label: str
    score: float


def message(
    text: str,
    *,
    stop_reason: str = "end_turn",
    stop_details: dict | None = None,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read: int = 0,
) -> dict:
    """A Messages API response body. It opens with an empty thinking block, as responses
    from adaptive-thinking models do, so the text is never simply `content[0]`."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read,
        },
    }


def api_error(status: int, kind: str) -> tuple[int, dict]:
    return status, {"type": "error", "error": {"type": kind, "message": f"fake {kind}"}}


def sse_stream(
    deltas: list[str],
    *,
    stop_reason: str = "end_turn",
    stop_details: dict | None = None,
    input_tokens: int = 120,
    output_tokens: int = 40,
) -> bytes:
    """The server-sent events of a streamed response: a thinking block, then text deltas."""
    start = message("", stop_reason="end_turn", input_tokens=input_tokens, output_tokens=1)
    start.update(content=[], stop_reason=None)
    events: list[dict] = [
        {"type": "message_start", "message": start},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    ]
    events += [
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": d}}
        for d in deltas
    ]
    events += [
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "stop_details": stop_details,
            },
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class FakeAPI:
    """Replays queued replies and records every request body it receives."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self._replies: list[object] = []
        self._default: object | None = None

    def queue(self, *replies: object) -> None:
        self._replies.extend(replies)

    def always(self, reply: object) -> None:
        """Answer every request the queue does not cover with `reply`."""
        self._default = reply

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/messages"
        self.requests.append(json.loads(request.content))
        if self._replies:
            reply = self._replies.pop(0)
        else:
            assert self._default is not None, "the client made a request the test did not expect"
            reply = self._default
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, bytes):
            return httpx2.Response(
                200, content=reply, headers={"content-type": "text/event-stream"}
            )
        if isinstance(reply, tuple):
            status, body = reply
            return httpx2.Response(status, json=body)
        return httpx2.Response(200, json=reply)


def make_llm(
    tmp_path: Path, api: FakeAPI, *, budget: float = 1.0, key: str = "test-key"
) -> LLMClient:
    settings = Settings(
        _env_file=None,
        ANTHROPIC_API_KEY=key,
        artifacts_dir=tmp_path,
        budget_usd=budget,
        llm_model=MODEL,
        llm_effort="low",
    )
    transport = httpx2.MockTransport(api)
    common = {"api_key": "test-key", "base_url": "https://fake.invalid", "max_retries": 0}
    return LLMClient(
        settings,
        sync_client=anthropic.Anthropic(
            **common, http_client=anthropic.DefaultHttpxClient(transport=transport)
        ),
        async_client=anthropic.AsyncAnthropic(
            **common, http_client=anthropic.DefaultAsyncHttpxClient(transport=transport)
        ),
    )


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


# ------------------------------------------------------------------ request and cache


def test_request_body_uses_only_parameters_current_models_accept(tmp_path, api):
    api.queue(message('{"label": "yes", "score": 0.8}'))
    llm = make_llm(tmp_path, api)

    result = llm.complete(system="SYS", user="USER", schema=Verdict, max_tokens=500, sample_id=7)

    (body,) = api.requests
    assert body["model"] == MODEL
    assert body["max_tokens"] == 500
    assert body["system"] == "SYS"
    assert body["messages"] == [{"role": "user", "content": "USER"}]
    assert body["output_config"]["effort"] == "low"
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["required"]) == {"label", "score"}
    # Rejected by current models, or deliberately unused; `sample_id` is local bookkeeping.
    forbidden = {"temperature", "top_p", "top_k", "thinking", "tool_choice", "tools", "fallbacks"}
    assert forbidden.isdisjoint(body)
    assert "sample_id" not in json.dumps(body)
    assert result.parsed == Verdict(label="yes", score=0.8)
    assert result.model == MODEL


def test_only_repeat_samples_ask_for_prompt_caching(tmp_path, api):
    api.queue(message('{"label": "yes", "score": 0.8}'))
    api.queue(message('{"label": "no", "score": 0.6}'))
    llm = make_llm(tmp_path, api)

    llm.complete(system="SYS", user="USER", schema=Verdict, sample_id=0)
    llm.complete(system="SYS", user="USER", schema=Verdict, sample_id=1)

    first, repeat = api.requests
    assert "cache_control" not in first
    assert repeat["cache_control"] == {"type": "ephemeral"}


def test_cache_miss_then_hit(tmp_path, api):
    api.queue(message('{"label": "yes", "score": 0.8}'))
    llm = make_llm(tmp_path, api)

    first = llm.complete(system="s", user="u", schema=Verdict)
    second = llm.complete(system="s", user="u", schema=Verdict)

    assert len(api.requests) == 1
    assert (first.usage.input_tokens, first.usage.output_tokens) == (1000, 200)
    assert first.usage.cost_usd == pytest.approx(CALL_COST)
    assert first.usage.cached is False
    assert second.usage.cached is True
    assert second.usage.cost_usd == 0.0
    assert (second.usage.input_tokens, second.usage.output_tokens) == (1000, 200)
    assert second.text == first.text
    assert second.parsed == first.parsed
    assert second.model == first.model


def test_sample_id_draws_separate_reproducible_samples(tmp_path, api):
    api.queue(message("sample zero"), message("sample one"))
    llm = make_llm(tmp_path, api)

    s0 = llm.complete(system="s", user="u", sample_id=0)
    s1 = llm.complete(system="s", user="u", sample_id=1)
    assert (s0.text, s1.text) == ("sample zero", "sample one")
    # The model sees identical requests: independence comes from its sampling. The repeat
    # differs only by the prompt-cache marker, which affects billing, not the answer.
    repeat = {k: v for k, v in api.requests[1].items() if k != "cache_control"}
    assert api.requests[0] == repeat

    assert llm.complete(system="s", user="u", sample_id=1).text == "sample one"
    assert llm.complete(system="s", user="u", sample_id=0).text == "sample zero"
    assert len(api.requests) == 2


def test_cache_key_depends_on_every_field():
    base = {
        "model": MODEL,
        "system": "s",
        "user": "u",
        "schema": {"type": "object"},
        "effort": "low",
        "max_tokens": 100,
        "sample_id": 0,
    }
    changed = {
        "model": "claude-opus-5-5",
        "system": "s2",
        "user": "u2",
        "schema": None,
        "effort": "high",
        "max_tokens": 101,
        "sample_id": 1,
    }
    keys = {cache_key(**base)} | {cache_key(**{**base, k: v}) for k, v in changed.items()}
    assert len(keys) == len(base) + 1
    assert cache_key(**base) == cache_key(**dict(reversed(base.items())))


def test_cache_survives_a_new_client_and_needs_no_key(tmp_path, api):
    api.queue(message("stored"))
    make_llm(tmp_path, api).complete(system="s", user="u")

    keyless = make_llm(tmp_path, api, key="")
    assert keyless.complete(system="s", user="u").usage.cached is True
    with pytest.raises(LLMUnavailable):
        keyless.complete(system="s", user="not cached")
    assert len(api.requests) == 1


async def test_acomplete_shares_the_cache_with_complete(tmp_path, api):
    api.queue(message("from async"))
    llm = make_llm(tmp_path, api)

    first = await llm.acomplete(system="s", user="u")
    second = llm.complete(system="s", user="u")

    assert first.usage.cached is False
    assert second.usage.cached is True
    assert second.text == "from async"
    assert len(api.requests) == 1


# ------------------------------------------------------------------ budget and cost


def test_budget_is_a_hard_stop(tmp_path, api):
    api.queue(message("a"), message("b"))
    llm = make_llm(tmp_path, api, budget=1.5 * CALL_COST)

    llm.complete(system="s", user="first")  # spend 1 x cost < budget
    llm.complete(system="s", user="second")  # spend 2 x cost >= budget from here on
    with pytest.raises(BudgetExceeded):
        llm.complete(system="s", user="third")
    assert len(api.requests) == 2

    # Cached answers cost nothing and stay available; the stop also binds a fresh client,
    # because the ledger is read from disk.
    assert llm.complete(system="s", user="first").usage.cached is True
    with pytest.raises(BudgetExceeded):
        make_llm(tmp_path, api, budget=1.5 * CALL_COST).complete(system="s", user="fourth")
    assert len(api.requests) == 2


def test_spend_reports_totals_by_tag(tmp_path, api):
    api.queue(message("a"), message("b"), message("c", input_tokens=2000, output_tokens=0))
    llm = make_llm(tmp_path, api)

    llm.complete(system="s", user="1", tag="benchmark")
    llm.complete(system="s", user="2", tag="benchmark")
    llm.complete(system="s", user="3")
    llm.complete(system="s", user="1", tag="benchmark")  # cache hit

    spend = llm.spend()
    assert spend["calls"] == 3
    assert spend["cached_calls"] == 1
    assert spend["total_usd"] == pytest.approx(3 * CALL_COST)
    assert spend["by_tag"] == {
        "benchmark": pytest.approx(2 * CALL_COST),
        "untagged": pytest.approx(CALL_COST),
    }
    assert spend["budget_usd"] == 1.0


def test_ledger_is_consistent_under_concurrent_calls(tmp_path, api):
    api.always(message("ok"))
    llm = make_llm(tmp_path, api)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda i: llm.complete(system="s", user=f"q{i}"), range(64)))

    assert all(not r.usage.cached for r in results)
    spend = llm.spend()
    assert spend["calls"] == 64
    assert spend["total_usd"] == pytest.approx(64 * CALL_COST)


def test_cost_arithmetic():
    million = 1_000_000
    assert cost_usd(MODEL, input_tokens=50_000, output_tokens=0) == pytest.approx(0.005)
    assert cost_usd(MODEL, input_tokens=0, output_tokens=million) == pytest.approx(0.50)
    assert cost_usd(
        "claude-sonnet-5-5", input_tokens=million, output_tokens=million
    ) == pytest.approx(12.0)
    assert cost_usd(
        "claude-opus-5-5", input_tokens=million, output_tokens=million
    ) == pytest.approx(24.0)
    # Cache reads are billed at a tenth of the input rate, cache writes at 1.25x.
    assert cost_usd(
        "claude-opus-5-5", input_tokens=0, output_tokens=0, cache_read_tokens=million
    ) == pytest.approx(0.40)
    assert cost_usd(
        "claude-opus-5-5", input_tokens=0, output_tokens=0, cache_write_tokens=million
    ) == pytest.approx(5.00)
    # Haiku 5.5 moves to its second rate card once the prompt exceeds 100K tokens.
    assert cost_usd(MODEL, input_tokens=100_000, output_tokens=1000) == pytest.approx(
        (100_000 * 0.10 + 1000 * 0.50) / million
    )
    assert cost_usd(MODEL, input_tokens=100_001, output_tokens=1000) == pytest.approx(
        (100_001 * 0.50 + 1000 * 2.50) / million
    )


def test_cache_read_tokens_are_reported_and_priced(tmp_path, api):
    api.queue(message("a", input_tokens=100, output_tokens=0, cache_read=900))
    result = make_llm(tmp_path, api).complete(system="s", user="u")

    assert result.usage.cache_read_tokens == 900
    assert result.usage.cost_usd == pytest.approx((100 + 0.1 * 900) * 0.10 / 1_000_000)


def test_unpriced_model_is_rejected_before_any_request(tmp_path, api):
    llm = make_llm(tmp_path, api)
    with pytest.raises(ValueError, match="no price"):
        llm.complete(system="s", user="u", model="claude-unknown-9")
    assert api.requests == []


# ------------------------------------------------------------------ failures


def test_missing_key_raises_unavailable_without_a_request(tmp_path, api):
    llm = make_llm(tmp_path, api, key="")
    with pytest.raises(LLMUnavailable):
        llm.complete(system="s", user="u")
    assert api.requests == []


def test_refusal_is_typed_billed_and_not_cached(tmp_path, api):
    details = {"type": "refusal", "category": "bio", "explanation": "declined"}
    api.queue(
        message("", stop_reason="refusal", stop_details=details),
        message('{"label": "no", "score": 0.5}'),
    )
    llm = make_llm(tmp_path, api)

    with pytest.raises(LLMRefusal) as caught:
        llm.complete(system="s", user="u", schema=Verdict)
    assert caught.value.category == "bio"
    assert llm.spend()["total_usd"] == pytest.approx(CALL_COST)

    # The failure was not stored, so asking again reaches the API.
    assert llm.complete(system="s", user="u", schema=Verdict).parsed.label == "no"
    assert len(api.requests) == 2


def test_max_tokens_is_typed_billed_and_not_cached(tmp_path, api):
    api.queue(
        message('{"label": "ye', stop_reason="max_tokens"),
        message('{"label": "yes", "score": 0.9}'),
    )
    llm = make_llm(tmp_path, api)

    with pytest.raises(LLMTruncated) as caught:
        llm.complete(system="s", user="u", schema=Verdict, max_tokens=64)
    assert caught.value.max_tokens == 64
    assert llm.spend()["calls"] == 1

    llm.complete(system="s", user="u", schema=Verdict, max_tokens=64)
    assert len(api.requests) == 2


def test_output_that_violates_the_schema_is_rejected_and_not_cached(tmp_path, api):
    api.queue(message('{"label": "yes"}'), message('{"label": "yes", "score": 1}'))
    llm = make_llm(tmp_path, api)

    with pytest.raises(LLMInvalidOutput):
        llm.complete(system="s", user="u", schema=Verdict)
    assert llm.complete(system="s", user="u", schema=Verdict).parsed.score == 1.0


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (api_error(401, "authentication_error"), LLMUnavailable),
        (api_error(400, "invalid_request_error"), LLMRequestError),
        (api_error(404, "not_found_error"), LLMRequestError),
        (api_error(429, "rate_limit_error"), LLMTransientError),
        (api_error(529, "overloaded_error"), LLMTransientError),
        (httpx2.ConnectError("connection refused"), LLMTransientError),
    ],
)
def test_api_errors_are_translated(tmp_path, api, reply, expected):
    api.queue(reply)
    llm = make_llm(tmp_path, api)
    with pytest.raises(expected):
        llm.complete(system="s", user="u")
    assert llm.spend()["calls"] == 0


# ------------------------------------------------------------------ streaming


async def test_astream_yields_text_deltas_and_records_usage(tmp_path, api):
    api.queue(sse_stream(["Rest ", "and fluids ", "help [1]."]))
    llm = make_llm(tmp_path, api)
    seen: list[Usage] = []

    deltas = [
        d async for d in llm.astream(system="SYS", user="USER", tag="chat", on_usage=seen.append)
    ]

    assert deltas == ["Rest ", "and fluids ", "help [1]."]
    (body,) = api.requests
    assert body["stream"] is True
    assert body["max_tokens"] == 4000
    assert body["output_config"] == {"effort": "low"}
    expected = cost_usd(MODEL, input_tokens=120, output_tokens=40)
    (usage,) = seen
    assert (usage.input_tokens, usage.output_tokens, usage.cached) == (120, 40, False)
    assert usage.cost_usd == pytest.approx(expected)
    assert llm.spend()["by_tag"] == {"chat": pytest.approx(expected)}


async def test_astream_is_never_cached(tmp_path, api):
    api.queue(sse_stream(["one"]), sse_stream(["two"]))
    llm = make_llm(tmp_path, api)

    first = [d async for d in llm.astream(system="s", user="u")]
    second = [d async for d in llm.astream(system="s", user="u")]

    assert (first, second) == (["one"], ["two"])


async def test_astream_refusal_surfaces_after_the_text(tmp_path, api):
    details = {"type": "refusal", "category": None, "explanation": None}
    api.queue(sse_stream(["partial"], stop_reason="refusal", stop_details=details))
    llm = make_llm(tmp_path, api)

    deltas: list[str] = []
    with pytest.raises(LLMRefusal):
        async for delta in llm.astream(system="s", user="u"):
            deltas.append(delta)

    assert deltas == ["partial"]
    assert llm.spend()["calls"] == 1


async def test_astream_truncation_is_typed(tmp_path, api):
    api.queue(sse_stream(["cut off mid"], stop_reason="max_tokens"))
    llm = make_llm(tmp_path, api)
    with pytest.raises(LLMTruncated):
        async for _ in llm.astream(system="s", user="u", max_tokens=16):
            pass


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (api_error(401, "authentication_error"), LLMUnavailable),
        (api_error(529, "overloaded_error"), LLMTransientError),
    ],
)
async def test_astream_translates_api_errors(tmp_path, api, reply, expected):
    api.queue(reply)
    llm = make_llm(tmp_path, api)
    with pytest.raises(expected):
        await anext(llm.astream(system="s", user="u"))
    assert llm.spend()["calls"] == 0


async def test_abandoned_stream_is_still_recorded(tmp_path, api):
    api.queue(sse_stream(["first ", "second"]))
    llm = make_llm(tmp_path, api)

    stream = llm.astream(system="s", user="u", tag="chat")
    assert await anext(stream) == "first "
    await stream.aclose()  # the reader went away

    spend = llm.spend()
    assert spend["calls"] == 1
    assert spend["total_usd"] > 0


@pytest.mark.parametrize("call", ["astream", "acomplete"])
async def test_async_calls_keep_the_event_loop_free_while_the_ledger_is_busy(tmp_path, api, call):
    # Another process (an experiment sharing the cache file) holds the write lock. The ledger
    # write waits for it; the event loop, which serves every other request, must not.
    api.queue(sse_stream(["text"]) if call == "astream" else message("text"))
    llm = make_llm(tmp_path, api)
    held = 0.6
    other = sqlite3.connect(llm.settings.cache_db, isolation_level=None, check_same_thread=False)
    other.execute("BEGIN IMMEDIATE")
    threading.Timer(held, other.commit).start()

    lags: list[float] = []

    async def ticker() -> None:
        while True:
            started = time.perf_counter()
            await asyncio.sleep(0.01)
            lags.append(time.perf_counter() - started - 0.01)

    ticking = asyncio.create_task(ticker())
    await asyncio.sleep(0.02)  # the ticker is running before the call starts
    if call == "astream":
        assert [d async for d in llm.astream(system="s", user="u")] == ["text"]
    else:
        assert (await llm.acomplete(system="s", user="u")).text == "text"
    await asyncio.sleep(0.02)  # and gets to measure any stall the call caused
    ticking.cancel()
    other.close()

    assert llm.spend()["calls"] == 1  # the write did happen, once the lock was released
    assert max(lags) < held / 3, f"event loop stalled for {max(lags):.2f}s"


async def test_astream_checks_key_and_budget_before_streaming(tmp_path, api):
    with pytest.raises(LLMUnavailable):
        await anext(make_llm(tmp_path, api, key="").astream(system="s", user="u"))

    api.queue(message("a"))
    llm = make_llm(tmp_path, api, budget=CALL_COST)
    llm.complete(system="s", user="u")
    with pytest.raises(BudgetExceeded):
        await anext(llm.astream(system="s", user="u"))
    assert len(api.requests) == 1
