"""Typed failures of the language-model layer.

Two families, kept deliberately apart:

* `BudgetExceeded` and `LLMUnavailable` mean *no further call can succeed* until a human
  acts (raise the budget, configure a key). A batch job should stop when it sees one.
* `LLMError` and its subclasses describe *one call* that failed. A batch job may record
  the failure for that item and continue.

They do not share a base class, so `except LLMError` in an experiment loop can never
swallow a budget stop by accident.
"""

from __future__ import annotations


class BudgetExceeded(RuntimeError):
    """Cumulative recorded spend has reached `settings.budget_usd`; no call was made."""


class LLMUnavailable(RuntimeError):
    """No usable API key: none is configured, or the API rejected the one we have."""


class LLMError(RuntimeError):
    """One request failed. Subclasses say how."""


class LLMRefusal(LLMError):
    """The model stopped with `stop_reason == "refusal"`; any content is not an answer."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        self.category = category
        self.explanation = explanation
        detail = f" ({category})" if category else ""
        super().__init__(
            f"the model declined to answer{detail}" + (f": {explanation}" if explanation else "")
        )


class LLMTruncated(LLMError):
    """The response was cut off: text is incomplete and structured output is invalid JSON."""

    def __init__(self, stop_reason: str, max_tokens: int) -> None:
        self.stop_reason = stop_reason
        self.max_tokens = max_tokens
        super().__init__(f"response was cut off ({stop_reason}, max_tokens={max_tokens})")


class LLMInvalidOutput(LLMError):
    """The model finished normally but its output does not satisfy the requested schema."""


class LLMRequestError(LLMError):
    """The API rejected the request (4xx). Retrying the same request will not help."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"API rejected the request ({status_code}): {message}")


class LLMTransientError(LLMError):
    """Rate limit, overload, server error or network failure that outlasted the SDK's retries."""
