"""Verification engine: score each claim of an answer against the retrieved evidence (SPEC §3.4)."""

from primum.sentences import split_sentences
from primum.verification.nli import NLIModel, get_nli
from primum.verification.verify import verify_claims

__all__ = ["NLIModel", "get_nli", "split_sentences", "verify_claims"]
