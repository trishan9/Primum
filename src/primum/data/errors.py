"""Typed failures of the data layer, so callers can react to each one specifically."""

from __future__ import annotations


class CorpusNotBuilt(FileNotFoundError):
    """A processed file was requested before its builder ran.

    Subclasses `FileNotFoundError` so generic "artefact missing" handlers still catch it.
    """


class SourceUnavailable(RuntimeError):
    """A raw source could not be located or did not have the expected shape."""


class LeakageError(RuntimeError):
    """A build would have let evaluation information reach an index or a training set.

    Raised instead of writing the output: a leaked benchmark is worse than a failed build,
    because every number computed from it afterwards looks fine and is wrong.
    """
