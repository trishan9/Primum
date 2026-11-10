"""Fast tests for the verification engine: no network, no pretrained weights."""

from __future__ import annotations

import numpy as np
import pytest

from primum.types import Claim, Hit, Passage
from primum.verification import split_sentences, verify_claims
from primum.verification.nli import NLI_LABELS, NLIModel, _canonical_columns

# --- split_sentences ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "Statins lower LDL cholesterol (e.g. atorvastatin). They are well tolerated.",
            ["Statins lower LDL cholesterol (e.g. atorvastatin).", "They are well tolerated."],
            id="e.g.",
        ),
        pytest.param(
            "We enrolled adults, i.e. patients over 18. Children were excluded.",
            ["We enrolled adults, i.e. patients over 18.", "Children were excluded."],
            id="i.e.",
        ),
        pytest.param(
            "Drug A vs. placebo showed no difference. The trial was stopped.",
            ["Drug A vs. placebo showed no difference.", "The trial was stopped."],
            id="vs.",
        ),
        pytest.param(
            "Smith et al. reported a benefit. Jones et al. (2019) did not.",
            ["Smith et al. reported a benefit.", "Jones et al. (2019) did not."],
            id="et al.",
        ),
        pytest.param(
            "Mortality was lower (p < 0.05). The hazard ratio was 0.82 (95% CI 0.71-0.94). "
            "P = .03 for trend.",
            [
                "Mortality was lower (p < 0.05).",
                "The hazard ratio was 0.82 (95% CI 0.71-0.94).",
                "P = .03 for trend.",
            ],
            id="decimals-and-p-values",
        ),
        pytest.param(
            "Mortality was 4.2%. Survival improved (n = 30). Then it fell.",
            ["Mortality was 4.2%.", "Survival improved (n = 30).", "Then it fell."],
            id="after-percent-and-closing-bracket",
        ),
        pytest.param(
            "E. coli and S. aureus were isolated. H. pylori was absent.",
            ["E. coli and S. aureus were isolated.", "H. pylori was absent."],
            id="genus-abbreviations",
        ),
        pytest.param(
            "Patients were infected with hepatitis A and B. Vaccination was offered.",
            ["Patients were infected with hepatitis A and B.", "Vaccination was offered."],
            id="single-capital-ends-sentence",
        ),
        pytest.param(
            "J. R. Smith wrote the protocol. It was registered.",
            ["J. R. Smith wrote the protocol.", "It was registered."],
            id="author-initials",
        ),
        pytest.param(
            "Results are shown in Fig. 2. The effect was dose dependent.",
            ["Results are shown in Fig. 2.", "The effect was dose dependent."],
            id="Fig.",
        ),
        pytest.param(
            "The dose was given i.v. at 5 mg/kg. Outcomes improved.",
            ["The dose was given i.v. at 5 mg/kg.", "Outcomes improved."],
            id="dotted-abbreviation-mid-sentence",
        ),
        pytest.param(
            "The drug was approved in the U.S. The EU followed.",
            ["The drug was approved in the U.S.", "The EU followed."],
            id="dotted-abbreviation-sentence-final",
        ),
        pytest.param(
            "She has MS. Her symptoms began in 2010. She saw Dr. Lee.",
            ["She has MS.", "Her symptoms began in 2010.", "She saw Dr. Lee."],
            id="acronym-vs-title",
        ),
        pytest.param(
            "St. John's wort induces CYP3A4. The ECG showed no change in ST. Levels fell.",
            ["St. John's wort induces CYP3A4.", "The ECG showed no change in ST.", "Levels fell."],
            id="St-vs-ST",
        ),
        pytest.param(
            "p53 was mutated in 40% of tumours. mRNA levels were unchanged.",
            ["p53 was mutated in 40% of tumours.", "mRNA levels were unchanged."],
            id="lowercase-biomedical-names",
        ),
        pytest.param(
            "Is aspirin safe in pregnancy? Evidence is limited! More trials are needed.",
            ["Is aspirin safe in pregnancy?", "Evidence is limited!", "More trials are needed."],
            id="question-and-exclamation",
        ),
        pytest.param(
            "1. Take the tablet with water. 2. Do not lie down for 30 minutes.",
            ["1. Take the tablet with water.", "2. Do not lie down for 30 minutes."],
            id="list-enumerators",
        ),
        pytest.param(
            "BACKGROUND: Sepsis is common.\n\nMETHODS\n\nWe enrolled 200\npatients.",
            ["BACKGROUND: Sepsis is common.", "METHODS", "We enrolled 200 patients."],
            id="blank-lines-split-but-line-wraps-do-not",
        ),
        pytest.param("No terminal punctuation", ["No terminal punctuation"], id="no-terminator"),
        pytest.param("  \n\n  ", [], id="whitespace-only"),
        pytest.param("", [], id="empty"),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_split_sentences_loses_no_text() -> None:
    text = (
        "Metformin (approx. 2 g/day) was compared vs. placebo in 1,204 adults; cf. Smith et al. "
        "HbA1c fell by 1.1% (p < .001). Adverse events, e.g. nausea, were mild. Is it safe? Yes."
    )
    assert " ".join(split_sentences(text)) == text


# --- label order -------------------------------------------------------------------------------


def test_canonical_columns_follow_the_checkpoint_not_a_fixed_order() -> None:
    mnli_order = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    reverse_order = {0: "entailment", 1: "neutral", 2: "contradiction"}
    assert _canonical_columns(mnli_order) == [2, 1, 0]
    assert _canonical_columns(reverse_order) == [0, 1, 2]


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        {0: "entailment", 1: "not_entailment"},
        {0: "entailment", 1: "neutral", 2: "contradiction", 3: "other"},
    ],
)
def test_canonical_columns_refuse_to_guess(id2label: dict[int, str]) -> None:
    with pytest.raises(ValueError, match="NLI model labels"):
        _canonical_columns(id2label)


# --- NLIModel on a tiny random checkpoint ------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_nli(tmp_path_factory: pytest.TempPathFactory) -> NLIModel:
    """A randomly initialised 2-layer BERT saved to disk: exercises the real loading, batching
    and label-reordering code without downloading anything. Its label order is deliberately not
    the canonical one."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    directory = tmp_path_factory.mktemp("tiny-nli")
    words = "aspirin reduces does not reduce the risk of stroke in adults fever and it".split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *sorted(set(words)), "."]
    (directory / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(directory / "vocab.txt")).save_pretrained(directory)

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=3,
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
        label2id={"contradiction": 0, "entailment": 1, "neutral": 2},
    )
    BertForSequenceClassification(config).save_pretrained(directory)
    return NLIModel(str(directory), device="cpu", batch_size=2)


PAIRS = [
    ("aspirin reduces the risk of stroke in adults .", "aspirin reduces stroke ."),
    ("fever .", "aspirin does not reduce the risk of stroke ."),
    ("aspirin does not reduce the risk of stroke in adults and it does not reduce fever .", "it ."),
    ("stroke", "fever and stroke"),
    ("the risk of stroke", "the risk of fever in adults"),
]


def test_nli_model_returns_one_distribution_per_pair(tiny_nli: NLIModel) -> None:
    premises, hypotheses = map(list, zip(*PAIRS, strict=True))
    probs = tiny_nli.predict(premises, hypotheses)
    assert probs.shape == (len(PAIRS), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_nli_model_batching_preserves_order_and_values(tiny_nli: NLIModel) -> None:
    """Length-sorted batches of 2 must give the same rows, in the caller's order, as scoring
    each pair alone (padding is masked, so batch composition cannot change a result)."""
    premises, hypotheses = map(list, zip(*PAIRS, strict=True))
    batched = tiny_nli.predict(premises, hypotheses)
    one_by_one = np.vstack([tiny_nli.predict([p], [h]) for p, h in PAIRS])
    np.testing.assert_allclose(batched, one_by_one, atol=1e-5)


def test_nli_model_reorders_columns_to_entail_neutral_contra(tiny_nli: NLIModel) -> None:
    import torch

    premise, hypothesis = PAIRS[0]
    encoded = tiny_nli._tokenizer([premise], [hypothesis], return_tensors="pt")
    with torch.inference_mode():
        raw = torch.softmax(tiny_nli._model(**encoded).logits, dim=-1)[0].numpy()
    id2label = tiny_nli._model.config.id2label
    expected = [
        raw[next(i for i, name in id2label.items() if name == label)] for label in NLI_LABELS
    ]
    np.testing.assert_allclose(tiny_nli.predict([premise], [hypothesis])[0], expected, atol=1e-6)


def test_nli_model_handles_empty_input_and_rejects_ragged_input(tiny_nli: NLIModel) -> None:
    assert tiny_nli.predict([], []).shape == (0, 3)
    with pytest.raises(ValueError, match="differ in length"):
        tiny_nli.predict(["a"], [])


# --- verify_claims -----------------------------------------------------------------------------


class StubNLI:
    """Scores a premise by lookup, so each test states exactly what the 'model' believes.

    Premises not listed are neutral. Calls are recorded to check what was (not) scored.
    """

    def __init__(self, scores: dict[str, tuple[float, float, float]] | None = None) -> None:
        self.scores = scores or {}
        self.calls: list[tuple[list[str], list[str]]] = []

    def predict(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        self.calls.append((list(premises), list(hypotheses)))
        return np.array([self.scores.get(p, (0.05, 0.9, 0.05)) for p in premises]).reshape(-1, 3)

    @property
    def premises(self) -> list[str]:
        return [p for premises, _ in self.calls for p in premises]


def make_hits(*texts: str) -> list[Hit]:
    return [
        Hit(
            passage=Passage(passage_id=f"p{i}", doc_id=f"d{i}", text=text, source="pubmed"),
            score=1.0 / i,
            rank=i,
            retriever="stub",
        )
        for i, text in enumerate(texts, start=1)
    ]


HITS = make_hits(
    "Aspirin reduced stroke. Bleeding increased. Mortality was unchanged.",
    "Statins lowered LDL. Myalgia was rare.",
)


def test_claim_is_scored_against_sentences_and_adjacent_pairs_of_cited_passages_only() -> None:
    nli = StubNLI()
    verify_claims([Claim(text="Aspirin prevents stroke.", cited=[1])], HITS, nli=nli)
    assert nli.premises == [
        "Aspirin reduced stroke.",
        "Bleeding increased.",
        "Mortality was unchanged.",
        "Aspirin reduced stroke. Bleeding increased.",
        "Bleeding increased. Mortality was unchanged.",
    ]
    assert set(nli.calls[0][1]) == {"Aspirin prevents stroke."}


def test_uncited_claim_is_scored_against_every_passage() -> None:
    nli = StubNLI()
    verify_claims([Claim(text="Statins are safe.")], HITS, nli=nli)
    assert len(nli.premises) == (3 + 2) + (2 + 1)
    assert "Myalgia was rare." in nli.premises and "Bleeding increased." in nli.premises


def test_invalid_and_duplicate_citations_are_ignored() -> None:
    nli = StubNLI()
    verify_claims([Claim(text="c", cited=[2, 2, 0, 9, -1])], HITS, nli=nli)
    assert nli.premises == [
        "Statins lowered LDL.",
        "Myalgia was rare.",
        "Statins lowered LDL. Myalgia was rare.",
    ]

    only_invalid = StubNLI()
    verify_claims([Claim(text="c", cited=[7])], HITS, nli=only_invalid)
    assert len(only_invalid.premises) == 8  # falls back to all evidence, like an uncited claim


def test_all_claims_are_scored_in_one_batched_call_and_keep_their_order() -> None:
    nli = StubNLI({"Myalgia was rare.": (0.9, 0.05, 0.05)})
    claims = [Claim(text="first", cited=[1]), Claim(text="second", cited=[2])]
    verdicts = verify_claims(claims, HITS, nli=nli)
    assert len(nli.calls) == 1
    assert nli.calls[0][1] == ["first"] * 5 + ["second"] * 3
    assert [v.claim.text for v in verdicts] == ["first", "second"]
    assert [v.label for v in verdicts] == ["unsupported", "supported"]


def test_supported_verdict_reports_the_most_entailing_premise() -> None:
    nli = StubNLI(
        {
            "Aspirin reduced stroke.": (0.70, 0.25, 0.05),
            "Statins lowered LDL.": (0.95, 0.04, 0.01),
        }
    )
    [verdict] = verify_claims([Claim(text="c")], HITS, nli=nli)
    assert verdict.label == "supported"
    assert verdict.p_entail == pytest.approx(0.95)
    assert verdict.p_neutral == pytest.approx(0.04)
    assert (verdict.best_hit, verdict.best_sentence) == (2, "Statins lowered LDL.")


def test_entail_and_contra_maxima_come_from_different_premises() -> None:
    nli = StubNLI(
        {
            "Aspirin reduced stroke.": (0.40, 0.55, 0.05),
            "Bleeding increased.": (0.02, 0.08, 0.90),
        }
    )
    [verdict] = verify_claims([Claim(text="c", cited=[1])], HITS, nli=nli)
    assert verdict.label == "contradicted"
    assert verdict.p_entail == pytest.approx(0.40)
    assert verdict.p_contra == pytest.approx(0.90)
    # A contradicted claim points at the sentence that contradicts it.
    assert (verdict.best_hit, verdict.best_sentence) == (1, "Bleeding increased.")
    assert verdict.p_neutral == pytest.approx(0.08)


def test_entailment_takes_precedence_over_contradiction() -> None:
    nli = StubNLI(
        {
            "Aspirin reduced stroke.": (0.80, 0.15, 0.05),
            "Mortality was unchanged.": (0.01, 0.04, 0.95),
        }
    )
    [verdict] = verify_claims([Claim(text="c", cited=[1])], HITS, nli=nli)
    assert verdict.label == "supported"
    assert verdict.p_contra == pytest.approx(0.95)  # still reported for the confidence model
    assert verdict.best_sentence == "Aspirin reduced stroke."


@pytest.mark.parametrize(
    ("scores", "label"),
    [
        ((0.50, 0.30, 0.20), "supported"),  # thresholds are inclusive
        ((0.49, 0.01, 0.50), "contradicted"),
        ((0.49, 0.02, 0.49), "unsupported"),
    ],
)
def test_label_thresholds(scores: tuple[float, float, float], label: str) -> None:
    nli = StubNLI({"Only sentence.": scores})
    [verdict] = verify_claims([Claim(text="c")], make_hits("Only sentence."), nli=nli)
    assert verdict.label == label


def test_window_premise_can_be_the_best_evidence() -> None:
    window = "Aspirin reduced stroke. Bleeding increased."
    nli = StubNLI({window: (0.9, 0.05, 0.05)})
    [verdict] = verify_claims([Claim(text="c", cited=[1])], HITS, nli=nli)
    assert verdict.label == "supported"
    assert verdict.best_sentence == window


class ExplodingNLI:
    def predict(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        raise AssertionError("the NLI model must not be used when there is no evidence")


def test_closed_book_claims_are_unsupported_without_touching_the_model() -> None:
    [verdict] = verify_claims([Claim(text="c", cited=[1])], [], nli=ExplodingNLI())
    assert verdict.label == "unsupported"
    assert (verdict.p_entail, verdict.p_neutral, verdict.p_contra) == (0.0, 1.0, 0.0)
    assert verdict.best_hit is None and verdict.best_sentence is None


def test_no_claims_gives_no_verdicts() -> None:
    assert verify_claims([], HITS, nli=ExplodingNLI()) == []


def test_claim_citing_an_empty_passage_is_unsupported() -> None:
    hits = make_hits("   ", "Statins lowered LDL.")
    nli = StubNLI({"Statins lowered LDL.": (0.9, 0.05, 0.05)})
    empty, supported = verify_claims(
        [Claim(text="a", cited=[1]), Claim(text="b", cited=[2])], hits, nli=nli
    )
    assert (empty.label, empty.best_hit) == ("unsupported", None)
    assert (supported.label, supported.best_hit) == ("supported", 2)


def test_scorer_returning_the_wrong_shape_is_an_error() -> None:
    class TwoColumnNLI:
        def predict(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
            return np.zeros((len(premises), 2))

    with pytest.raises(ValueError, match="shape"):
        verify_claims([Claim(text="c")], HITS, nli=TwoColumnNLI())
