"""Triage against the real benchmark questions (needs `primum data build pubmedqa`).

The benchmark experiments send PubMedQA research questions through the pipeline. Any question
that triage called an `emergency` would be answered with the emergency message instead of
being scored, silently shrinking the evaluation set, so the false-alarm count on the 1,000
labelled questions must be exactly zero.

Run with: uv run pytest -m slow tests/test_safety_slow.py
"""

from __future__ import annotations

import json

import pytest

from primum.config import get_settings
from primum.safety import triage

pytestmark = pytest.mark.slow


def test_no_pubmedqa_research_question_is_triaged_as_an_emergency() -> None:
    path = get_settings().processed_dir / "pubmedqa" / "labeled.jsonl"
    if not path.exists():
        pytest.skip(f"{path} has not been built")
    with path.open(encoding="utf-8") as lines:  # not str.splitlines(): the text contains U+2029
        questions = [json.loads(line)["question"] for line in lines]
    emergencies = [q for q in questions if triage(q).level == "emergency"]
    assert emergencies == []
