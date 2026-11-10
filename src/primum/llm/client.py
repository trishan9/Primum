"""The single doorway to the Claude API.

Every model call in Primum goes through `LLMClient`, which adds three things the research
design depends on:

* a **response cache**, so an experiment can be re-run, resumed or audited without paying
  for (or re-sampling) answers it already has;
* `sample_id`, so several independent samples of one prompt are each reproducible;
* a **budget ledger** with a hard stop, so a bug in a loop cannot spend more than
  `settings.budget_usd`.

Order of checks on every call: cache, then API key, then budget, then the network. A cache
hit therefore needs neither a key nor remaining budget, which lets anyone holding the cache
file replay the experiments offline.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError

from primum.config import Settings, get_settings
from primum.llm.errors import (
    BudgetExceeded,
    LLMInvalidOutput,
    LLMRefusal,
    LLMRequestError,
    LLMTransientError,
    LLMTruncated,
    LLMUnavailable,
)
from primum.llm.pricing import cost_usd, price_for
from primum.llm.store import CachedResponse, Store, cache_key
from primum.types import Usage

# A benchmark run is thousands of calls from a thread pool; a short burst of 429s should
# be absorbed by the SDK's exponential backoff rather than cost us items.
_MAX_RETRIES = 4


class LLMResult(BaseModel):
    text: str
    parsed: Any | None = None  # an instance of the requested schema, when one was given
    usage: Usage
    model: str


@dataclass(frozen=True)
class _Call:
    """One resolved request: what is sent, and the key its response is cached under."""

    key: str
    model: str
    max_tokens: int
    tag: str
    schema: type[BaseModel] | None
    request: dict[str, Any]


@cache
def _wire_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The JSON schema actually sent for a pydantic model.

    `transform_schema` rewrites it into the subset that constrained decoding supports
    (e.g. `additionalProperties: false` on every object).
    """
    return anthropic.transform_schema(schema)


@contextmanager
def _api_errors() -> Iterator[None]:
    """Translate SDK exceptions into this layer's typed errors, most specific first."""
    try:
        yield
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable("the Anthropic API rejected the configured API key") from exc
    except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
        # 429 and 5xx (including 529 overloaded), after the SDK's own retries.
        raise LLMTransientError(f"API unavailable ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        # The remaining 4xx: malformed request, unknown model, permission, request too large.
        raise LLMRequestError(exc.status_code, exc.message) from exc
    except anthropic.APIConnectionError as exc:
        # No HTTP response at all: network failure or timeout.
        raise LLMTransientError(f"could not reach the API: {exc}") from exc


class LLMClient:
    """Cached, budgeted access to Claude. Safe to share across threads and the event loop."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sync_client: anthropic.Anthropic | None = None,
        async_client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        """`sync_client` / `async_client` let tests supply SDK clients with a fake transport."""
        self.settings = settings or get_settings()
        self._store = Store(self.settings.cache_db)
        self._sync = sync_client
        self._async = async_client

    # ------------------------------------------------------------------ public API

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
        """One request, one response. With `schema`, the reply is JSON validated into it.

        `model` and `effort` default to the configured ones; `tag` labels the call in the
        ledger. Raises `LLMUnavailable`, `BudgetExceeded`, or an `LLMError` subclass.
        """
        call = self._prepare(system, user, schema, model, max_tokens, effort, sample_id, tag)
        if (hit := self._cached(call)) is not None:
            return hit
        self._require_key_and_budget()
        with _api_errors():
            message = self._sync_client().messages.create(**call.request)
        return self._finish(call, message)

    async def acomplete(
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
        """Async twin of `complete`; shares its cache and ledger.

        The cache and ledger are sqlite on a file other processes write too (an experiment
        run), so a read or write can wait seconds for their lock. That waiting happens in a
        worker thread: on the event loop it would stall every other request the server holds.
        """
        call = self._prepare(system, user, schema, model, max_tokens, effort, sample_id, tag)
        if (hit := await asyncio.to_thread(self._cached, call)) is not None:
            return hit
        await asyncio.to_thread(self._require_key_and_budget)
        with _api_errors():
            message = await self._async_client().messages.create(**call.request)
        return await asyncio.to_thread(self._finish, call, message)

    async def astream(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 4000,
        tag: str = "",
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        """Yield the answer's text deltas as they are generated.

        Streams are never cached: this path serves the interactive answer, where the reader
        should watch a live generation. They are budgeted and recorded like any other call;
        `on_usage` receives the recorded usage once the stream ends.

        A refusal or a `max_tokens` cut-off is only known when the stream ends, so the
        corresponding error is raised *after* any text has been yielded. As in `acomplete`,
        the ledger is read and written in a worker thread, never on the event loop.
        """
        model = model or self.settings.llm_model
        price_for(model)
        await asyncio.to_thread(self._require_key_and_budget)
        request = self._request(model, system, user, max_tokens, self.settings.llm_effort, None)
        snapshot: anthropic.types.Message | None = None
        try:
            with _api_errors():
                async with self._async_client().messages.stream(**request) as stream:
                    async for event in stream:
                        snapshot = stream.current_message_snapshot
                        if event.type == "text":
                            yield event.text
        finally:
            # Reached on completion, on an API error mid-stream, and when the consumer stops
            # reading (a browser tab closing). Generated tokens are billed in every case, so
            # the ledger is written here. For an abandoned stream the output count is the
            # last one the API reported, which is a lower bound. The shield lets the write
            # finish even when the reader's task is being cancelled.
            if snapshot is not None:
                usage = await asyncio.shield(
                    asyncio.to_thread(self._record, model, tag, snapshot.usage)
                )
                if on_usage is not None:
                    on_usage(usage)
        if snapshot is None:
            raise LLMTransientError("the stream ended before the response started")
        _raise_for_stop_reason(snapshot, max_tokens)

    def spend(self) -> dict:
        """Ledger summary: `total_usd`, `by_tag`, `calls`, `cached_calls`, `budget_usd`."""
        return {**self._store.summary(), "budget_usd": self.settings.budget_usd}

    # ------------------------------------------------------------------ internals

    def _prepare(
        self,
        system: str,
        user: str,
        schema: type[BaseModel] | None,
        model: str | None,
        max_tokens: int,
        effort: str | None,
        sample_id: int,
        tag: str,
    ) -> _Call:
        # Defaults are resolved *before* hashing so the key describes the request that is
        # really sent: changing `settings.llm_model` must not replay another model's answers.
        model = model or self.settings.llm_model
        effort = effort or self.settings.llm_effort
        price_for(model)
        wire_schema = _wire_schema(schema) if schema is not None else None
        key = cache_key(
            model=model,
            system=system,
            user=user,
            schema=wire_schema,
            effort=effort,
            max_tokens=max_tokens,
            sample_id=sample_id,
        )
        request = self._request(model, system, user, max_tokens, effort, wire_schema)
        if sample_id >= 1:
            # Repeat samples resend a prompt already sent as sample 0, back to back: the first
            # repeat writes the provider's prompt cache and the rest read it at a tenth of the
            # input price. Sample 0 is usually the only call for its prompt, where a cache
            # write would be a pure surcharge, so it is left unmarked. This changes billing
            # only, not the request the model answers, so it stays out of the cache key.
            request["cache_control"] = {"type": "ephemeral"}
        return _Call(key, model, max_tokens, tag, schema, request)

    @staticmethod
    def _request(
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        effort: str,
        wire_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Keyword arguments for `messages.create` / `messages.stream`.

        Deliberately absent: `temperature`/`top_p`/`top_k` (current models reject non-default
        values, so independent samples come from the default sampling distribution);
        `thinking` (adaptive by default, its depth is set by `effort`); assistant prefill and
        forced tool choice (both rejected, and structured output replaces them); and
        refusal `fallbacks` (a silent switch to another model would put answers from a
        generator the confidence model was not calibrated on into the results).
        """
        output_config: dict[str, Any] = {"effort": effort}
        if wire_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": wire_schema}
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        if system:
            request["system"] = system
        return request

    def _cached(self, call: _Call) -> LLMResult | None:
        hit = self._store.get(call.key)
        if hit is None:
            return None
        self._store.record(tag=call.tag, model=call.model, cached=True)
        # Token counts are those of the original call (useful for reporting prompt sizes);
        # the cost is zero because nothing is spent now.
        usage = Usage(
            input_tokens=hit.input_tokens,
            output_tokens=hit.output_tokens,
            cache_read_tokens=hit.cache_read_tokens,
            cost_usd=0.0,
            cached=True,
        )
        parsed = _validate(call.schema, hit.text) if call.schema is not None else None
        return LLMResult(text=hit.text, parsed=parsed, usage=usage, model=hit.model)

    def _require_key_and_budget(self) -> None:
        if not self.settings.anthropic_api_key:
            raise LLMUnavailable(
                "no Anthropic API key is configured; set ANTHROPIC_API_KEY in .env "
                "(responses already in the cache remain available without one)"
            )
        # A hard stop, checked against the on-disk ledger before every network call. Calls
        # already in flight when the limit is crossed still complete, so the overshoot is
        # bounded by the cost of `llm_concurrency` calls.
        spent = self._store.total_usd()
        if spent >= self.settings.budget_usd:
            raise BudgetExceeded(
                f"recorded spend ${spent:.4f} has reached the budget of "
                f"${self.settings.budget_usd:.2f}; raise PRIMUM_BUDGET_USD to continue"
            )

    def _record(self, model: str, tag: str, api_usage: anthropic.types.Usage) -> Usage:
        cache_read = api_usage.cache_read_input_tokens or 0
        cache_write = api_usage.cache_creation_input_tokens or 0
        cost = cost_usd(
            model,
            input_tokens=api_usage.input_tokens,
            output_tokens=api_usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        self._store.record(
            tag=tag,
            model=model,
            input_tokens=api_usage.input_tokens,
            output_tokens=api_usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=cost,
        )
        return Usage(
            input_tokens=api_usage.input_tokens,
            output_tokens=api_usage.output_tokens,
            cache_read_tokens=cache_read,
            cost_usd=cost,
        )

    def _finish(self, call: _Call, message: anthropic.types.Message) -> LLMResult:
        # Record first: a refused or truncated response is billed like any other.
        usage = self._record(call.model, call.tag, message.usage)
        _raise_for_stop_reason(message, call.max_tokens)
        text = "".join(block.text for block in message.content if block.type == "text")
        parsed = _validate(call.schema, text) if call.schema is not None else None
        # Only complete, valid responses are cached; a failure may be transient, and the
        # decision to retry belongs to the caller.
        self._store.put(
            call.key,
            CachedResponse(
                model=message.model,
                text=text,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
            ),
        )
        return LLMResult(text=text, parsed=parsed, usage=usage, model=message.model)

    def _sync_client(self) -> anthropic.Anthropic:
        if self._sync is None:
            # Key and base URL are passed explicitly so that an ANTHROPIC_BASE_URL exported
            # by the shell cannot redirect requests.
            self._sync = anthropic.Anthropic(
                api_key=self.settings.anthropic_api_key,
                base_url=self.settings.anthropic_base_url,
                max_retries=_MAX_RETRIES,
            )
        return self._sync

    def _async_client(self) -> anthropic.AsyncAnthropic:
        if self._async is None:
            self._async = anthropic.AsyncAnthropic(
                api_key=self.settings.anthropic_api_key,
                base_url=self.settings.anthropic_base_url,
                max_retries=_MAX_RETRIES,
            )
        return self._async


def _raise_for_stop_reason(message: anthropic.types.Message, max_tokens: int) -> None:
    """Refuse to read content from a response that did not end normally."""
    if message.stop_reason == "refusal":
        details = message.stop_details
        raise LLMRefusal(
            category=details.category if details else None,
            explanation=details.explanation if details else None,
        )
    if message.stop_reason in ("max_tokens", "model_context_window_exceeded"):
        raise LLMTruncated(message.stop_reason, max_tokens)


def _validate(schema: type[BaseModel], text: str) -> BaseModel:
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise LLMInvalidOutput(f"response does not match {schema.__name__}: {exc}") from exc


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """The process-wide client, built from `get_settings()`."""
    return LLMClient(get_settings())
