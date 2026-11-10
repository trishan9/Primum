"""How retrieved passages are shown to the model, and how its citations are read back.

The two directions must agree: a passage shown as `[n]` is `hits[n - 1]`, and a citation
`n` returned by the model is resolved against the same list by the verifier.
"""

from __future__ import annotations

from pydantic import BaseModel

from primum.llm import prompts
from primum.types import Claim, Hit


class WireClaim(BaseModel):
    """A claim as the model returns it, before its citations are checked against the hits."""

    text: str
    cited: list[int]


def format_evidence(hits: list[Hit]) -> str:
    """Render hits as numbered blocks, `[1] (title) text`, separated by blank lines.

    Numbers are positions in `hits`, not `Hit.rank`: callers may pass a re-ordered or
    filtered list (the oracle condition passes a single gold abstract), and the citation
    indices the model returns have to index whatever list it was actually shown.
    """
    blocks = []
    for number, hit in enumerate(hits, start=1):
        title = " ".join(hit.passage.title.split())
        text = " ".join(hit.passage.text.split())
        blocks.append(f"[{number}] ({title}) {text}" if title else f"[{number}] {text}")
    return "\n\n".join(blocks)


def evidence_block(hits: list[Hit]) -> str:
    """The passage section of a prompt; says so explicitly when retrieval found nothing."""
    return format_evidence(hits) if hits else prompts.NO_PASSAGES


def to_claims(wire_claims: list[WireClaim], n_hits: int) -> list[Claim]:
    """Convert model output to `Claim`s whose citations are guaranteed to index the hits.

    A citation outside `1..n_hits` points at a passage the model was never shown. It is
    dropped rather than kept, so the claim counts as uncited and the verifier checks it
    against every passage; keeping it would crash the lookup or, worse, credit the claim
    with support from the wrong text. Blank claims are dropped.
    """
    claims = []
    for wire in wire_claims:
        text = wire.text.strip()
        if not text:
            continue
        cited = list(dict.fromkeys(n for n in wire.cited if 1 <= n <= n_hits))
        claims.append(Claim(text=text, cited=cited))
    return claims
