"""Language-model layer: a cached, budgeted client for the Claude API, and the prompts."""

from primum.llm.client import LLMClient, LLMResult, get_llm
from primum.llm.errors import (
    BudgetExceeded,
    LLMError,
    LLMInvalidOutput,
    LLMRefusal,
    LLMRequestError,
    LLMTransientError,
    LLMTruncated,
    LLMUnavailable,
)

__all__ = [
    "BudgetExceeded",
    "LLMClient",
    "LLMError",
    "LLMInvalidOutput",
    "LLMRefusal",
    "LLMRequestError",
    "LLMResult",
    "LLMTransientError",
    "LLMTruncated",
    "LLMUnavailable",
    "get_llm",
]
