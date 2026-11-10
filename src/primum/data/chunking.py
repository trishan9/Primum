"""Sentence-aware chunking for long documents (MedlinePlus sections, PDF pages).

Chunks end on sentence boundaries because the NLI verifier scores a claim against individual
sentences, and a passage cut mid-sentence gives it a premise that says something the source
never said. The boundaries come from the verifier's own splitter (`primum.sentences`), so a
chunk boundary is always one the verifier would also draw.
"""

from __future__ import annotations

from primum.sentences import split_sentences


def chunk_text(text: str, *, max_words: int = 220, overlap: int = 40) -> list[str]:
    """Split `text` into chunks of at most `max_words` words that end on sentence boundaries.

    Consecutive chunks share trailing sentences worth at most `overlap` words, so a fact
    stated across a chunk boundary is intact in at least one chunk. The overlap is made of
    whole sentences (never a cut one) and is dropped when keeping it would push the next
    sentence over the limit. A single sentence longer than `max_words` (tables, reference
    lists) is the one case that is cut mid-sentence, into consecutive word windows.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must satisfy 0 <= overlap < max_words")

    units: list[list[str]] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        units.extend(words[i : i + max_words] for i in range(0, len(words), max_words))

    chunks: list[str] = []
    current: list[list[str]] = []
    size = 0
    for unit in units:
        if current and size + len(unit) > max_words:
            chunks.append(" ".join(w for u in current for w in u))
            # Seed the next chunk with the longest sentence-aligned tail that fits both
            # the overlap budget and, together with the incoming sentence, the size limit.
            tail: list[list[str]] = []
            tail_size = 0
            for previous in reversed(current):
                grown = tail_size + len(previous)
                if grown > overlap or grown + len(unit) > max_words:
                    break
                tail.insert(0, previous)
                tail_size = grown
            current, size = tail, tail_size
        current.append(unit)
        size += len(unit)
    if current:
        chunks.append(" ".join(w for u in current for w in u))
    return chunks
