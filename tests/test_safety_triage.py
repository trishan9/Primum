"""Tests for the emergency triage rail.

Three groups mirror the design note in `primum.safety.triage`:

1. true red flags described as happening to someone      -> must be `emergency` (recall);
2. benign look-alikes and general-knowledge questions    -> must not be `emergency` (precision);
3. false alarms the design accepts, pinned here so that the cost of favouring recall is
   written down rather than discovered.
"""

from __future__ import annotations

import pytest

from primum.safety import triage
from primum.safety.triage import RULES

# --- 1. red flags ------------------------------------------------------------------------------

EMERGENCIES = [
    # cardiac
    ("I have crushing chest pain spreading to my left arm and I'm sweating", "cardiac"),
    ("My chest feels really tight and I'm short of breath", "cardiac"),
    ("Why does my chest hurt and my left arm feel numb?", "cardiac"),
    ("What should I do, my chest hurts and I'm sweating", "cardiac"),
    ("I have chest pain. I also feel sick and dizzy.", "cardiac"),  # features in two sentences
    ("I think I'm having a heart attack", "stated_emergency"),
    # stroke
    ("My husband's face is drooping on one side and his speech is slurred", "stroke"),
    ("my left arm suddenly went numb and weak", "stroke"),
    ("Worst headache of my life, it came on suddenly", "stroke"),
    ("She suddenly can't speak properly", "stroke"),
    # airway and anaphylaxis
    ("I can't breathe", "airway"),
    ("I got stung by a bee and my throat is closing up", "airway"),
    ("My tongue is swelling", "airway"),
    ("he's turning blue", "airway"),
    ("My lips are swelling after eating peanuts", "anaphylaxis"),
    ("I'm covered in hives and wheezing after taking a new antibiotic", "anaphylaxis"),
    # self-harm
    ("I want to kill myself", "suicidal_intent"),
    ("I don't want to be here anymore, I'm thinking about ending it all", "suicidal_intent"),
    ("How can I end my life painlessly?", "suicidal_intent"),  # question form does not soften it
    ("My son says he wants to kill himself", "self_harm"),
    # overdose and poisoning
    ("I took too many paracetamol tablets", "overdose_or_poisoning"),
    ("My toddler swallowed a button battery", "overdose_or_poisoning"),
    ("I think my son took a whole bottle of ibuprofen", "overdose_or_poisoning"),
    # ... and the commoner phrasing as a quantity, without the word "overdose"
    ("I swallowed a bottle of pills", "overdose_or_poisoning"),
    ("I took a bottle of sleeping pills", "overdose_or_poisoning"),
    ("My son swallowed a handful of my blood pressure pills", "overdose_or_poisoning"),
    ("I took 20 paracetamol tablets", "overdose_or_poisoning"),
    ("I took 30 tylenol an hour ago", "overdose_or_poisoning"),
    ("My daughter ate a pack of my iron tablets", "overdose_or_poisoning"),
    # bleeding
    ("The bleeding won't stop", "severe_bleeding"),
    ("He is coughing up blood", "severe_bleeding"),
    # infants
    ("My 6 week old has a fever of 38.5", "infant_fever"),
    ("Our newborn has a temperature of 39", "infant_fever"),
    ("my baby is 5 weeks and has a fever", "infant_fever"),
    # unresponsive, seizure
    ("My dad collapsed and is not breathing", "unresponsive"),
    ("Help, my wife is unresponsive", "unresponsive"),
    ("She is having a seizure", "seizure"),
    # meningitis, head injury
    ("I have a stiff neck, fever and the light hurts my eyes", "meningitis"),
    ("He hit his head and now he is vomiting and confused", "head_injury"),
]


@pytest.mark.parametrize(("question", "rule"), EMERGENCIES)
def test_red_flag_presentations_are_emergencies(question: str, rule: str) -> None:
    result = triage(question)
    assert result.level == "emergency"
    assert rule in result.matched
    assert result.message and "has not answered" in result.message


@pytest.mark.parametrize(
    "question",
    [
        "crushing chest pain left arm sweating",  # no pronoun, no question word
        "Baby not breathing",
        "Can't breathe, throat closing",  # "Can" must not be read as a question opener
        "Dad having chest pain and sweating, what to do?",
        "husband has chest pain and is sweating",
        "Had a seizure, won't wake up",  # subject dropped; "Had" is not a question opener here
        "Please help me, crushing chest pain and sweating",  # a plea, not a request to be taught
        "im having chest pain and sweating",
        "baby is not responding to me",
        "Has a fever and is 3 weeks old, is this serious?",
    ],
)
def test_unclear_framing_is_treated_as_an_emergency(question: str) -> None:
    assert triage(question).level == "emergency"


def test_history_does_not_downgrade_a_current_presentation() -> None:
    result = triage("Crushing chest pain going down my arm. I had a heart attack years ago.")
    assert result.level == "emergency"


def test_a_person_outranks_a_knowledge_seeking_preamble() -> None:
    assert triage("I'm a medical student and my chest hurts and I'm sweating").level == "emergency"
    assert triage("I was wondering why my face is drooping").level == "emergency"


def test_loose_negation_does_not_hide_a_symptom() -> None:
    assert triage("My chest pain is not going away and I'm sweating").level == "emergency"
    assert triage("I have chest pain and no fever, but I'm sweating a lot").level == "emergency"
    assert triage("He has no pulse").level == "emergency"


# --- 2. look-alikes ----------------------------------------------------------------------------

ROUTINE = [
    "What causes chest pain after exercise in general?",
    "What are the warning signs of a stroke?",
    "What is anaphylaxis?",
    "Is aspirin useful for stroke prevention?",
    "How is a collapsed lung treated?",
    "I can't breathe through my nose because of a cold",
    "I have a sore, swollen throat and a runny nose",
    "I have a cough but no chest pain or shortness of breath",  # negated symptoms
    "I hurt myself lifting weights yesterday",
    "I cut myself shaving",
    "I ate too much pizza and feel sick",
    "How much paracetamol is too much?",
    "What happens if you take too much vitamin D?",
    # an ordinary dose is not an overdose: a count must be ten or more, and "mg" is not a count
    "I took 2 tablets of ibuprofen, is that ok?",
    "I took 25 mg of my medication this morning",
    "I took 10 mg of amlodipine tablets",
    "I took a bottle of water with my pills",
    "I have had a fever for 2 weeks",  # "2 weeks" is a duration, not an infant's age
    "What room temperature should a newborn sleep in?",
    "Is heavy menstrual bleeding normal?",
    "Can ibuprofen cause stomach bleeding?",
    "My child fell and hit his head but seems fine",
    "My mother had a stroke last year, what rehab helps?",
    "My infection is not responding to antibiotics",
    "What are the side effects of ibuprofen, such as nausea and vomiting?",
    "Does metformin cause weight loss?",
    "What causes blood in urine?",
    "",
]


@pytest.mark.parametrize("question", ROUTINE)
def test_benign_look_alikes_are_routine(question: str) -> None:
    result = triage(question)
    assert result.level == "routine"
    assert result.matched == [] and result.message is None


GENERAL_QUESTIONS_ABOUT_RED_FLAGS = [
    ("What causes chest pain that radiates to the arm?", "cardiac"),
    ("What should I know about heart attack symptoms like chest pain and sweating?", "cardiac"),
    ("I'm a nursing student: how is chest pain with sweating managed?", "cardiac"),
    ("I read that chest pain with sweating means a heart attack, is that true?", "cardiac"),
    ("Is facial droop always a sign of stroke?", "stroke"),
    ("What is the first aid for someone who is choking?", "airway"),
    ("Tell me about overdose treatment", "overdose_or_poisoning"),
    ("Help me understand stroke signs like slurred speech", "stroke"),
    # Once lower-cased, "IM" and "ME" must not be mistaken for the asker.
    ("What is the dose of IM epinephrine for hives after a bee sting?", "anaphylaxis"),
    ("What are symptoms of ME/CFS, e.g. chest pain and dizziness?", "cardiac"),
    ("Have prescription claims changed following the FDA suicidality warning?", "self_harm"),
    ("What are the risk factors for suicide in older adults?", "self_harm"),
]


@pytest.mark.parametrize(("question", "rule"), GENERAL_QUESTIONS_ABOUT_RED_FLAGS)
def test_general_questions_about_red_flags_are_answered_with_a_warning(
    question: str, rule: str
) -> None:
    """Not a hard stop — but the emergency advice stays on screen in case the framing was misread."""
    result = triage(question)
    assert result.level == "urgent"
    assert result.matched == [rule]
    assert result.message and "has not answered" not in result.message
    assert "emergency number" in result.message


# --- urgent ------------------------------------------------------------------------------------

URGENT = [
    ("I have chest pain when I climb stairs", "chest_pain"),
    ("I get short of breath walking to the shops", "breathlessness"),
    ("I passed out yesterday after standing up", "fainting"),
    ("My baby has a fever", "infant_fever_age_unstated"),
    ("my 8 month old has a fever", "infant_fever_age_unstated"),
    ("I've had a fever for 5 days", "high_or_lasting_fever"),
    ("My temperature is 40.2 C", "high_or_lasting_fever"),
    ("There is blood in my urine", "blood_in_stool_or_urine"),
    ("I'm pregnant and bleeding", "pregnancy_warning"),
    ("severe abdominal pain since last night", "severe_pain"),
]


@pytest.mark.parametrize(("question", "rule"), URGENT)
def test_urgent_presentations(question: str, rule: str) -> None:
    result = triage(question)
    assert result.level == "urgent"
    assert rule in result.matched
    assert result.message and "prompt medical attention" in result.message


def test_lethal_dose_question_is_urgent_however_it_is_framed() -> None:
    result = triage("What is the lethal dose of acetaminophen?")
    assert result.level == "urgent"
    assert result.matched == ["lethal_dose_query"]
    assert result.message and "988" in result.message


# --- 3. accepted false alarms ------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "I had chest pain and sweating last year, could it have been my heart?",  # tense
        "My right hand is numb from carpal tunnel",  # one-sided numbness has a benign cause
        "What to do if someone is choking? My friend is asking",  # hypothetical with a person
    ],
)
def test_accepted_false_alarms(question: str) -> None:
    """Known over-triggers. If a rule change makes one of these `routine`, check that it did not
    also lower recall on the corresponding true emergency before updating this list."""
    assert triage(question).level == "emergency"


# --- messages and lexicon ----------------------------------------------------------------------


def test_messages_match_the_kind_of_emergency() -> None:
    assert "911" in triage("I can't breathe").message
    assert "988" in triage("I want to kill myself").message
    assert "Samaritans" in triage("I'm worried my friend is suicidal").message
    assert "poison" in triage("I took too many sleeping pills").message.lower()


def test_only_the_rules_that_set_the_level_are_reported() -> None:
    # The urgent "chest_pain" rule also fires here; it adds nothing to the emergency.
    assert triage("I have chest pain and I'm sweating").matched == ["cardiac"]


def test_typographic_apostrophes_and_case_are_normalised() -> None:
    assert triage("I CAN’T BREATHE").level == "emergency"


def test_lexicon_is_well_formed() -> None:
    names = [rule.name for rule in RULES]
    assert len(names) == len(set(names))
    assert all(rule.all_of and rule.why for rule in RULES)
    covered = {rule for _, rule in EMERGENCIES + URGENT + GENERAL_QUESTIONS_ABOUT_RED_FLAGS}
    covered.add("lethal_dose_query")
    assert covered == set(names), "every rule needs at least one positive example in this file"
