"""Price table and cost arithmetic for the budget ledger.

Prices are USD per million tokens, first-party Claude API rates. The table is in code, not
fetched, so that a recorded cost can always be re-derived from the recorded token counts.
"""

from __future__ import annotations

from dataclasses import dataclass

# The API bills prompt-cache traffic relative to the base input rate: a cache read costs a
# tenth of it and a five-minute cache write a quarter more. `input_tokens` in a response
# already excludes both, so the three counts are disjoint and simply add.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    # (threshold, input, output): a second rate card that applies to the whole request
    # once its prompt exceeds `threshold` tokens.
    long_prompt: tuple[int, float, float] | None = None


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(input=0.10, output=0.50, long_prompt=(100_000, 0.50, 2.50)),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00),
    "claude-opus-5-5": Price(input=4.00, output=20.00),
}


def price_for(model: str) -> Price:
    """Look up a model's price, refusing models we cannot account for.

    An unpriced call would be recorded at zero cost and silently escape the budget, so an
    unknown model is an error raised *before* any request is sent.
    """
    try:
        return PRICES[model]
    except KeyError:
        known = ", ".join(sorted(PRICES))
        raise ValueError(
            f"no price is recorded for model {model!r}, so its spend could not be budgeted; "
            f"known models: {known}"
        ) from None


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Cost of one response, computed from the token counts in its usage block."""
    price = price_for(model)
    rate_in, rate_out = price.input, price.output
    if price.long_prompt is not None:
        threshold, long_in, long_out = price.long_prompt
        if input_tokens + cache_read_tokens + cache_write_tokens > threshold:
            rate_in, rate_out = long_in, long_out
    input_equivalent = (
        input_tokens
        + CACHE_READ_MULTIPLIER * cache_read_tokens
        + CACHE_WRITE_MULTIPLIER * cache_write_tokens
    )
    return (input_equivalent * rate_in + output_tokens * rate_out) / 1_000_000
