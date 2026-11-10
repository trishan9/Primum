"""Checks against the real pretrained NLI cross-encoder (`settings.nli_model`).

Downloads the checkpoint on first run. The point is not to benchmark the model but to prove
the wiring: the label order read from its config is right (a swap would turn every supported
claim into a contradicted one without raising), and `verify_claims` produces sensible verdicts
end to end on medical sentences.

Run with: uv run pytest -m slow tests/test_verification_slow.py
"""

from __future__ import annotations

import numpy as np
import pytest

from primum.types import Claim, Hit, Passage
from primum.verification import NLIModel, get_nli, verify_claims

pytestmark = pytest.mark.slow

PREMISE = (
    "In this randomized trial, metformin significantly reduced HbA1c compared with placebo "
    "in adults with type 2 diabetes."
)
ENTAILED = "Metformin lowered HbA1c in adults with type 2 diabetes."
CONTRADICTED = "Metformin did not reduce HbA1c in adults with type 2 diabetes."
UNRELATED = "Amoxicillin is an effective treatment for streptococcal pharyngitis."

ENTAIL, NEUTRAL, CONTRA = 0, 1, 2


@pytest.fixture(scope="module")
def nli() -> NLIModel:
    return get_nli()


def test_get_nli_is_a_singleton(nli: NLIModel) -> None:
    assert get_nli() is nli


def test_label_mapping_on_clear_medical_pairs(nli: NLIModel) -> None:
    probs = nli.predict([PREMISE] * 3, [ENTAILED, CONTRADICTED, UNRELATED])
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert probs.argmax(axis=1).tolist() == [ENTAIL, CONTRA, NEUTRAL]
    assert probs[0, ENTAIL] > 0.9
    assert probs[1, CONTRA] > 0.9


def test_verify_claims_end_to_end(nli: NLIModel) -> None:
    passage = Passage(
        passage_id="p1",
        doc_id="d1",
        source="pubmed",
        text=(
            "BACKGROUND: Glycaemic control is central to diabetes care. " + PREMISE + " "
            "Gastrointestinal adverse events were more frequent with metformin (p < 0.05)."
        ),
    )
    hits = [Hit(passage=passage, score=1.0, rank=1, retriever="test")]
    supported, contradicted, unsupported = verify_claims(
        [
            Claim(text=ENTAILED, cited=[1]),
            Claim(text=CONTRADICTED, cited=[1]),
            Claim(text=UNRELATED, cited=[1]),
        ],
        hits,
        nli=nli,
    )
    assert supported.label == "supported"
    assert supported.best_hit == 1 and "reduced HbA1c" in (supported.best_sentence or "")
    assert contradicted.label == "contradicted"
    assert "reduced HbA1c" in (contradicted.best_sentence or "")
    assert unsupported.label == "unsupported"
