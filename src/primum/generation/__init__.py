"""Answer generation: benchmark (structured yes/no/maybe) and chat (streamed, then decomposed)."""

from primum.generation.benchmark import answer_benchmark
from primum.generation.chat import extract_claims, stream_chat_answer
from primum.generation.evidence import format_evidence

__all__ = ["answer_benchmark", "extract_claims", "format_evidence", "stream_chat_answer"]
