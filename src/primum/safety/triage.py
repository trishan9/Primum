"""Emergency triage: a transparent rule lexicon that runs before anything else in the pipeline.

This is a safety rail, not a research contribution. It is rules only, on purpose: every
decision can be traced to a named rule a clinician can read and correct, there is no model to
drift or to be confidently wrong, and it costs microseconds. It is not a symptom checker and
makes no claim to be complete; the lexicon covers the classic red-flag presentations a
question-answering system must not answer as if they were homework.

Levels
------
- `emergency` — a red-flag presentation described as happening to someone. The pipeline stops
  and shows an emergency-services message; nothing clinical is answered.
- `urgent`    — the answer is still generated, under a notice to seek care promptly.
- `routine`   — everything else.

How a level is decided
----------------------
1. Symptom rules. Each rule is a conjunction of patterns ("chest pain" AND one accompanying
   feature such as radiation or sweating). Patterns match anywhere in the question, so the
   features may be spread over several sentences.
2. Framing. "Why does my chest hurt and my arm feel numb?" and "What causes chest pain that
   radiates to the arm?" contain the same symptoms; only the first is someone to send to an
   emergency department. A question is *general* when it is phrased as a request for
   knowledge and nothing in it refers to a particular person or the present moment.

   | red-flag rule matched | framing                              | level       |
   |-----------------------|--------------------------------------|-------------|
   | emergency rule        | refers to a person / now, or unclear | `emergency` |
   | emergency rule        | general                              | `urgent`    |
   | urgent rule           | refers to a person / now, or unclear | `urgent`    |
   | urgent rule           | general                              | `routine`   |

   Two rules skip this step because their wording is itself the risk: a stated intent to die
   is an `emergency` however it is phrased, and a lethal-dose question is always `urgent`.

Precision versus recall
-----------------------
The two errors are not symmetric. A missed emergency can cost a life; a false alarm costs one
unanswered question and a line of advice the reader can ignore. The rules therefore favour
recall wherever a specific person is involved:

- Unclear framing counts as personal. "crushing chest pain left arm sweating" has no pronoun
  and no question word, and is treated as an emergency.
- Any first-person or "he is / she has" reference outranks a general-sounding opening:
  "What should I do, my chest hurts and I'm sweating" is an emergency.
- A general framing never removes the warning, it only removes the hard stop: an emergency
  rule in a general question still yields `urgent`, whose message says to call emergency
  services if this is happening now. So a misjudged framing still puts the right advice on
  screen.
- Tense is not modelled. "I had chest pain and sweating last year" triggers like a current
  episode. Telling history from the present needs more than a lexicon, and cheap heuristics
  ("years ago" => not urgent) fail on exactly the people at highest risk: "crushing chest pain
  down my arm. I had a heart attack years ago."
- Negation is honoured only in its unambiguous form — a negation word directly followed by a
  symptom noun ("no chest pain or shortness of breath", "without fever"). Anything looser
  risks deleting a real symptom ("my chest pain is not going away and I'm sweating"; "no
  pulse", where the "no" is the symptom).

Precision is protected where it is free: single common symptoms are not emergencies on their
own (chest pain needs an accompanying feature; fever needs a young infant), phrases that only
look alarming are excluded ("can't breathe through my nose", "collapsed lung"), and
knowledge-seeking uses of "I" ("what should I know about ...", "I'm a nursing student") are
not taken as a description of the asker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from primum.types import TriageResult

Level = Literal["emergency", "urgent", "routine"]
Advice = Literal["general", "poison", "crisis"]


@dataclass(frozen=True)
class Rule:
    """One red-flag presentation. It fires when every pattern in `all_of` matches."""

    name: str
    level: Literal["emergency", "urgent"]  # before framing is taken into account
    why: str
    all_of: tuple[re.Pattern[str], ...]
    advice: Advice = "general"
    # True when the wording is itself the risk, whoever it is about ("I want to end my life",
    # "what is a lethal dose of ..."): the framing step is skipped.
    any_framing: bool = False


def _any(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{a})" for a in alternatives))


# --- building blocks ---------------------------------------------------------------------------
# Patterns run on lower-cased text with straightened apostrophes and single spaces.

_GAP = r"[^.?!]{0,40}"  # "within the same sentence, close by"
_POSS = r"(?:my|his|her|their|our|the)"
_CANT = r"(?:can'?t|cannot|can not|couldn'?t|unable to)"
_WONT = r"(?:won'?t|will not|doesn'?t|does not|isn'?t|is not|not)"

_CANT_BREATHE = (
    rf"\b{_CANT} (?:breathe|breath|catch {_POSS} breath|get (?:any |enough )?air)\b"
    rf"(?! (?:through|out of|from) {_POSS} nose)"  # a blocked nose is not respiratory distress
)
_BREATHLESS = (
    r"\bshort(?:ness)? of breath\b",
    r"\bbreathless\w*\b",
    r"\bout of breath\b",
    r"\b(?:trouble|difficulty|hard|hard time|struggling|struggle) (?:to )?breath\w*\b",
    _CANT_BREATHE,
)
_FEVER = _any(
    r"\bfever\w*\b",
    r"\bfebrile\b",
    r"\bburning up\b",
    r"\b(?:high|raised|running a|has a|have a|got a|with a) temperature\b",
    r"\btemperature (?:of |is |was |at )?(?:3[89]|4[0-2]|10[0-7])\b",
)
_VOMITING = (r"\bvomit\w*\b", r"\bthrowing up\b", r"\bthrew up\b")

_CHEST_PAIN = _any(
    r"\bchest (?:pains?|pressure|tightness|discomfort|heaviness)\b",
    rf"\b(?:pain|pressure|tightness|heaviness|discomfort|squeezing|crushing|ache) (?:in|on|across) {_POSS} chest\b",
    r"\bchest (?:hurts?|aches?|(?:is|feels) (?:really |very |so )?(?:tight|heavy|hurting|sore))\b",
    r"\b(?:crushing|squeezing|tight|heavy) chest\b",
)
_CARDIAC_FEATURE = _any(
    rf"\b(?:radiat\w+|spread\w*|shoot\w*|travel\w*|mov\w+|go(?:es|ing)) (?:down|up|to|into|through|across)\b{_GAP}\b(?:arms?|jaw|neck|shoulders?|back)\b",
    r"\b(?:arm|jaw)\b[^.?!]{0,20}\b(?:pain|hurts?|aches?|numb\w*|tingl\w+|heavy)\b",
    r"\b(?:pain|numbness|tingling|ache|heaviness)\b[^.?!]{0,20}\b(?:arm|jaw)\b",
    r"\bsweat\w*\b",
    r"\bclammy\b",
    r"\bnause\w*\b",
    r"\b(?:crushing|squeezing)\b",
    r"\b(?:faint\w*|light-?headed\w*|dizz\w+|pass(?:ed|ing) out)\b",
    *_VOMITING,
    *_BREATHLESS,
)
_STATED_EMERGENCY = _any(
    r"\b(?:having|going into) (?:a |an )?(?:heart attack|stroke|cardiac arrest|anaphyla\w+)\b",
)
_STROKE_SIGN = _any(
    r"\bfac(?:e|ial)\b[^.?!]{0,30}\bdroop\w*\b",
    r"\bdroop\w*\b[^.?!]{0,30}\b(?:face|mouth|smile)\b",
    r"\bslurr\w*\b",
    rf"\b(?:weak\w*|numb\w*|paraly[sz]\w+|limp|{_CANT} (?:move|feel|lift))\b{_GAP}\b(?:one|left|right) (?:side|arm|leg|hand)\b",
    rf"\b(?:one|left|right) (?:side|arm|leg|hand)\b{_GAP}\b(?:weak\w*|numb\w*|paraly[sz]\w+|limp|{_WONT} move)\b",
    r"\bworst headache\b",
    r"\bthunderclap\b",
    r"\bsudden\w* (?:severe|intense|excruciating|terrible|bad) headache\b",
    rf"\bsudden\w*\b{_GAP}\b(?:loss of (?:vision|sight)|lost (?:{_POSS} )?(?:vision|sight)|blind\w*|{_CANT} (?:see|speak|talk)|trouble (?:speaking|talking|seeing)|confus\w+)\b",
)
_AIRWAY = _any(
    _CANT_BREATHE,
    r"\b(?:struggling|fighting|gasping) (?:to breathe|for (?:air|breath))\b",
    r"\bchoking\b",
    r"\b(?:turning|turned|going|gone) blue\b",
    r"\b(?:lips|face|fingers) (?:are|is|look|looks) blue\b",
    r"\bblue lips\b",
    # A swelling tongue or a throat that is closing threatens the airway whatever the cause. A
    # throat that is merely "swollen" does not qualify: that is how people describe a sore throat.
    r"\bthroat\b[^.?!]{0,25}\b(?:clos(?:ing|ed|es)|swelling (?:up|shut))\b",
    r"\btongue\b[^.?!]{0,25}\b(?:swell\w*|swollen)\b",
    r"\b(?:swell\w*|swollen)\b[^.?!]{0,10}\btongue\b",
)
_ALLERGIC_SIGN = _any(
    r"\b(?:swell\w*|swollen|puff\w*)\b[^.?!]{0,25}\b(?:tongue|lips?|throat|face|eyes?)\b",
    r"\b(?:tongue|lips?|throat|face|eyes?)\b[^.?!]{0,25}\b(?:swell\w*|swollen|puff\w*)\b",
    r"\bhives\b",
    r"\bwheez\w*\b",
    r"\b(?:rash|itch\w*) all over\b",
    *_BREATHLESS,
)
_ALLERGEN = _any(
    r"\b(?:sting|stings|stung)\b",
    r"\b(?:bee|wasp|hornet)\b",
    r"\b(?:peanuts?|nuts?|shellfish|shrimp|prawns?)\b",
    r"\ballerg\w+\b",
    r"\bepi-?pen\b",
    r"\bafter (?:eating|taking|the injection|the vaccine|a new (?:medication|medicine|antibiotic))\b",
)
_UNRESPONSIVE = _any(
    r"\b(?:unconscious|unresponsive)\b",
    rf"\b{_WONT} (?:wake|waking)\b",
    # "not responding", "not responding to me / to her name" — but not "to antibiotics".
    rf"\b{_WONT} respond\w*\b(?! to (?!me\b|us\b|voice|pain|touch|anything|anyone|{_POSS} (?:voice|name|touch)))",
    rf"\b{_CANT} wake (?:him|her|them|{_POSS} \w+)\b",
    r"\b(?:not|isn'?t|stopped|no longer) breathing\b",
    r"\bno pulse\b",
    r"\bcollapsed\b(?! (?:lung|vein|disc|trachea|vertebra))",
)
_SEIZURE = _any(
    r"\bhaving an? (?:seizure|fit|convulsion)\b",
    r"\b(?:seizing|convulsing)\b",
    rf"\b(?:seizure|convulsion|fit)\b{_GAP}\b(?:won'?t stop|not stopping|still going|(?:more|longer) than (?:5|five|10|ten) min\w*)\b",
)
_SEVERE_BLEEDING = _any(
    rf"\b(?:nose)?bleed\w*\b{_GAP}\b{_WONT} stop\w*\b",
    rf"\b{_CANT} stop\b[^.?!]{{0,20}}\bbleed\w*\b",
    r"\b(?:heavy|severe|profuse|massive|uncontrolled|uncontrollable) bleeding\b",
    r"\bbleeding (?:heavily|profusely|badly|a lot|everywhere)\b",
    r"\b(?:a lot of|lots of|so much) blood\b",
    r"\b(?:spurting|gushing|pouring)\b[^.?!]{0,15}\bblood\b",
    r"\bblood\b[^.?!]{0,15}\b(?:spurting|gushing|pouring)\b",
    r"\b(?:cough\w*|vomit\w*|throwing|threw) (?:up )?blood\b",
    rf"\bblood in (?:{_POSS} )?vomit\b",
)
# An overdose is most often reported as a quantity ("a bottle of sleeping pills", "a handful of my
# blood pressure pills", "20 paracetamol") rather than with the word "overdose". A quantity
# counts when it is a container, a handful or a count of ten or more: no common tablet has a
# single adult dose of ten, while "I took 2 ibuprofen" must stay routine. Up to two words may
# name the medicine ("sleeping", "blood pressure"), but not a dosage unit, so "25 mg of my
# medication" is a dose rather than a count.
_MEDICINE = (
    r"(?:pills?|tablets?|capsules?|caplets?|meds|medications?|medicines?"
    r"|paracetamol|acetaminophen|tylenol|panadol|ibuprofen|advil|motrin|nurofen|aspirin)"
)
_MANY = (
    r"(?:(?:a|an|one|two|three|half (?:a|the)|\d+) (?:bottles?|box(?:es)?|packs?|packets?|strips?"
    r"|sheets?|blister packs?|tubs?|jars?|containers?)"
    r"|a handful|handfuls|a (?:lot|bunch)|lots|loads|dozens"
    r"|[1-9]\d+|ten|eleven|twelve|(?:thir|four|fif|six|seven|eigh|nine)teen|(?:twen|thir|for|fif)ty)"
)
_MANY_DOSES = (
    rf"\b{_MANY}\b(?: of)?(?: {_POSS})?(?: (?!(?:mg|mcg|ug|g|grams?|ml|units?|iu)\b)[a-z-]+){{0,2}}"
    rf" {_MEDICINE}\b"
)
_OVERDOSE = _any(
    r"\boverdos\w+\b",
    rf"\b(?:took|taken|swallowed|gave|given)\b{_GAP}\b(?:too (?:many|much)|double (?:the )?dose|(?:whole|entire|full) (?:bottle|box|pack\w*)|all (?:of )?{_POSS} (?:pills|tablets|medication|medicine|meds))\b",
    rf"\b(?:took|taken|swallowed|gave|given|ate|eaten|ingested|downed|popped)\b{_GAP}{_MANY_DOSES}",
    rf"\b(?:swallowed|drank|drunk|ate|eaten|ingested)\b{_GAP}\b(?:bleach|detergent|laundry pod|drain cleaner|antifreeze|rat poison|pesticide|weed ?killer|batter(?:y|ies)|magnets?)\b",
    r"\bpoisoned\b",
)
_SUICIDAL_INTENT = _any(
    r"\bkill(?:ing)? myself\b",
    r"\bend(?:ing)? (?:my (?:own )?life|it all)\b",
    r"\btak(?:e|ing) my (?:own )?life\b",
    r"\bi (?:just |really |only )?want to die\b",
    r"\bi (?:do not|don'?t) want to (?:live|be alive|be here|wake up)\b",
    r"\bbetter off dead\b",
    r"\bno reason to (?:live|go on)\b",
    r"\b(?:i am|i'?m|i feel|i'?ve been|i have been) (?:so |very |really |feeling )?suicidal\b",
    # "hurt myself" alone is usually an accident ("I hurt myself lifting"); with intent it is not.
    r"\b(?:want|going|plan\w*|thinking (?:of|about)|thoughts? (?:of|about)|urges?|tempted) (?:to )?(?:hurt\w*|harm\w*|cut\w*) myself\b",
)
_SELF_HARM_MENTION = _any(
    r"\bsuicid\w+\b",
    r"\bself-? ?harm\w*\b",
    r"\b(?:kill|hurt|harm)(?:ing)? (?:himself|herself|themselves|themself)\b",
)
_LETHAL_DOSE = _any(
    r"\b(?:lethal|fatal|deadly) (?:dose|amount|overdose)\b",
    rf"\bhow (?:much|many)\b{_GAP}\b(?:to die|(?:would|will|to|does it take to) kill|is (?:fatal|lethal|deadly))\b",
)
# Fever in the first three months of life needs same-day assessment whatever else is going on.
_UNDER_THREE_MONTHS = (
    r"(?:\d{1,2}[- ]?days?"
    r"|(?:[1-9]|1[0-2]|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[- ]?(?:weeks?|wks?)"
    r"|(?:[12]|one|two)[- ]?(?:months?|mos?))"
)
_YOUNG_INFANT = _any(
    r"\bnewborn\b",
    r"\bneonate\b",
    rf"\b{_UNDER_THREE_MONTHS}[- ]old\b",
    # "my baby is 6 weeks and ..." — an age without "old" counts only next to a word for a child,
    # otherwise "a fever for 2 weeks" would read as an infant.
    rf"\b(?:baby|infant|son|daughter|child)(?:'s| is)? (?:only |just |about |almost )?{_UNDER_THREE_MONTHS}\b",
)
_INFANT = _any(r"\b(?:baby|babies|infant|newborn)\b", r"\b(?:[3-9]|1[01])[- ]?months?[- ]old\b")
_STIFF_NECK = _any(
    r"\bstiff neck\b", r"\bneck stiffness\b", r"\bneck (?:is |feels )?(?:very |really |so )?stiff\b"
)
_MENINGEAL_FEATURE = _any(
    _FEVER.pattern,
    r"\brash\b",
    r"\blight (?:hurts|sensitiv\w+)\b",
    r"\bsensitiv\w+ to light\b",
    r"\bphotophobi\w+\b",
    r"\bconfus\w+\b",
    r"\bsevere headache\b",
)
_HEAD_TRAUMA = _any(
    rf"\b(?:hit|banged|bumped|struck|knocked) (?:{_POSS} )?head\b",
    r"\bhead (?:injury|trauma)\b",
    rf"\b(?:fell|fall|fallen)\b{_GAP}\bhead\b",
    r"\bconcuss\w+\b",
)
_HEAD_INJURY_SIGN = _any(
    *_VOMITING,
    r"\bconfus\w+\b",
    r"\b(?:passed|knocked|blacked) out\b",
    r"\bunconscious\b",
    r"\b(?:drowsy|very sleepy)\b",
    r"\bseizure\b",
    r"\b(?:blurr\w+|double) vision\b",
    r"\bclear fluid\b",
)
_FAINTED = _any(r"\b(?:passed|blacked) out\b", r"\bfaint(?:ed|ing)\b", r"\blost consciousness\b")
_HIGH_OR_LASTING_FEVER = _any(
    rf"\bfever\b{_GAP}\b(?:for|lasting|lasted|over|more than) (?:[3-9]|\d\d|three|four|five|six|seven|several|many) days\b",
    r"\b4[0-2](?:\.\d)? ?(?:°|degrees? ?)?c(?:elsius)?\b",
    r"\b10[4-7](?:\.\d)? ?(?:°|degrees? ?)?f(?:ahrenheit)?\b",
)
_BLOOD_IN_STOOL_OR_URINE = _any(
    rf"\bblood in (?:{_POSS} )?(?:stools?|poo|poop|urine|pee|wee)\b",
    r"\b(?:black|tarry) stools?\b",
    r"\b(?:peeing|pooping|urinating|passing) blood\b",
)
_PREGNANT = _any(r"\bpregnan\w+\b", r"\b\d{1,2} weeks along\b")
_PREGNANCY_WARNING = _any(
    r"\bbleed\w*\b",
    r"\bspotting\b",
    r"\bsevere (?:\w+ )?(?:pain|headache|cramp\w*)\b",
    r"\b(?:not|stopped|isn'?t|less) (?:moving|kicking)\b",
    r"\bwaters? (?:broke|have broken)\b",
    r"\bseizure\b",
    r"\b(?:blurr\w+|double) vision\b",
)
_SEVERE_PAIN = _any(r"\b(?:severe|unbearable|excruciating|agoni[sz]ing|worst) (?:\w+ )?pain\b")

# --- the lexicon -------------------------------------------------------------------------------

RULES: tuple[Rule, ...] = (
    Rule("suicidal_intent", "emergency", "Stated intent to die or to harm oneself.",
         (_SUICIDAL_INTENT,), advice="crisis", any_framing=True),
    Rule("self_harm", "emergency", "Suicide or self-harm concerning a particular person.",
         (_SELF_HARM_MENTION,), advice="crisis"),
    Rule("cardiac", "emergency",
         "Chest pain with radiation, sweating, breathlessness, nausea or faintness: possible acute coronary syndrome.",
         (_CHEST_PAIN, _CARDIAC_FEATURE)),
    Rule("stated_emergency", "emergency",
         "The asker names the emergency themselves ('I think I'm having a heart attack').",
         (_STATED_EMERGENCY,)),
    Rule("stroke", "emergency",
         "Any FAST sign (facial droop, one-sided weakness or numbness, slurred speech), thunderclap headache or sudden loss of vision or speech.",
         (_STROKE_SIGN,)),
    Rule("airway", "emergency", "Cannot breathe, choking, cyanosis or a closing throat.", (_AIRWAY,)),
    Rule("anaphylaxis", "emergency",
         "Swelling, hives, wheeze or breathlessness together with a likely allergen.",
         (_ALLERGIC_SIGN, _ALLERGEN)),
    Rule("unresponsive", "emergency", "Unconscious, cannot be woken, not breathing, or collapsed.",
         (_UNRESPONSIVE,)),
    Rule("seizure", "emergency", "A seizure in progress or one that is not stopping.", (_SEIZURE,)),
    Rule("severe_bleeding", "emergency",
         "Bleeding that is heavy or will not stop; coughing or vomiting blood.", (_SEVERE_BLEEDING,)),
    Rule("overdose_or_poisoning", "emergency",
         "Too much of a medicine, or a caustic or toxic substance swallowed.",
         (_OVERDOSE,), advice="poison"),
    Rule("infant_fever", "emergency", "Fever in an infant under three months old.",
         (_YOUNG_INFANT, _FEVER)),
    Rule("meningitis", "emergency",
         "Stiff neck with fever, rash, photophobia, confusion or severe headache.",
         (_STIFF_NECK, _MENINGEAL_FEATURE)),
    Rule("head_injury", "emergency",
         "Head trauma followed by vomiting, confusion, drowsiness, loss of consciousness or visual change.",
         (_HEAD_TRAUMA, _HEAD_INJURY_SIGN)),
    Rule("lethal_dose_query", "urgent",
         "Asking how much of a substance is fatal is a recognised self-harm risk signal, however it is framed.",
         (_LETHAL_DOSE,), advice="crisis", any_framing=True),
    Rule("chest_pain", "urgent", "Chest pain without other features still needs prompt assessment.",
         (_CHEST_PAIN,)),
    Rule("breathlessness", "urgent", "Shortness of breath.", (_any(*_BREATHLESS),)),
    Rule("fainting", "urgent", "Loss of consciousness, now recovered.", (_FAINTED,)),
    Rule("infant_fever_age_unstated", "urgent",
         "Fever in a baby whose age is not given or is three months or more.", (_INFANT, _FEVER)),
    Rule("high_or_lasting_fever", "urgent",
         "Fever of 40 °C / 104 °F or above, or lasting three days or more.", (_HIGH_OR_LASTING_FEVER,)),
    Rule("blood_in_stool_or_urine", "urgent", "Visible blood in stool or urine.",
         (_BLOOD_IN_STOOL_OR_URINE,)),
    Rule("pregnancy_warning", "urgent",
         "Bleeding, severe pain, reduced movements, visual change or seizure in pregnancy.",
         (_PREGNANT, _PREGNANCY_WARNING)),
    Rule("severe_pain", "urgent", "Pain described as severe or the worst ever.", (_SEVERE_PAIN,)),
)  # fmt: skip

# --- framing -----------------------------------------------------------------------------------

# A particular person: the asker ("I", "my", "our baby") or someone being described ("he is",
# "she has", "dad just collapsed"). Lower-casing makes three abbreviations look like pronouns,
# so "i" is not taken from "i.e." / "i.v.", "me" not from "ME/CFS", and "im" only counts with a
# continuation that "IM" (intramuscular) never has.
_SOMEONE = (
    r"(?:he|she|they|dad|father|mom|mum|mother|husband|wife|partner|boyfriend|girlfriend|son"
    r"|daughter|brother|sister|grandma|grandpa|grandmother|grandfather|friend|baby|toddler|kid|child)"
)
_PERSON = _any(
    r"\b(?:i(?!\.\w)|ive|me(?!/)|my|mine|myself|our)\b",
    r"\bim (?:\w+ing|not|so|really|very|in|scared|worried|dizzy|sick|pregnant)\b",
    r"\b(?:he|she|they)(?:'s|'re)\b",
    rf"\b{_SOMEONE} (?:is|are|has|have|was|were|just|keeps?|having|can'?t|cannot|won'?t|isn'?t"
    r"|doesn'?t|feels?|looks?|seems?|says?|said|started|stopped|took|swallowed|fell|collapsed|hit)\b",
)
# The present moment, or asking what to do about a situation rather than what is true in general.
_HAPPENING = _any(
    r"\b(?:right|just|happening) now\b",
    r"\bat (?:the|this) moment\b",
    r"\b(?:\d+|a few|a couple of|five|ten|fifteen|twenty|thirty) min(?:ute)?s? ago\b",
    r"\bwhat (?:to do|(?:do|should|can) we do)\b(?! (?:if|when|in case|about|for)\b)",
    r"\bis this (?:serious|dangerous|an emergency|life[- ]threatening)\b",
)
# First-person phrases that are about wanting knowledge, not about having symptoms. They are
# removed before looking for a person, so "what should I know about stroke signs?" stays general
# while "I'm a student and my chest hurts" does not.
_KNOWLEDGE_SEEKING = _any(
    r"\b(?:should|do|can|could|may|might|must|would) i (?:need to )?(?:know|learn|understand|ask)\b",
    r"\bi (?:want|need|wish|would like|'?d like) to (?:know|learn|understand|ask)\b",
    r"\bi(?:'?m| am| was)(?: just)? (?:curious|wondering|studying|reading|researching|writing|revising|learning|asking)\b",
    r"\bi (?:read|heard|saw|learned|learnt)\b",
    r"\b(?:tell|teach|show|remind) me\b",
    r"\bgive me (?:an? |some |more )?(?:information|info|overview|summary|details|explanation|list|examples?)\b",
    r"\bhelp me (?:understand|learn|study|revise)\b",  # a bare "help me" is a plea, not a question
    r"\bexplain (?:to )?me\b",
    r"\bi(?:'?m| am) an? (?:\w+ )?(?:student|nurse|doctor|physician|paramedic|researcher|resident|clinician|pharmacist)\b",
    r"\b(?:for )?my (?:exam|class|course|homework|assignment|essay|thesis|research|studies|paper|textbook|lecture|notes|professor|teacher)\b",
)
# A request for knowledge: a clause that opens with a question word or auxiliary, a textbook
# phrase, or one of the knowledge-seeking phrases above. Negated auxiliaries are excluded so that
# "Can't breathe" and "Does not wake up" are not read as questions. "Have/has/had/did/was/were"
# also open symptom reports with the subject dropped ("Had a seizure, won't wake up"), so they
# count only when the clause actually ends in a question mark ("Have prescriptions changed?").
_CLAUSE_START = r"(?:^|[.?!,;:] )"
_KNOWLEDGE_REQUEST = _any(
    rf"{_CLAUSE_START}(?:what|why|how|when|which|who)\b",
    rf"{_CLAUSE_START}(?:is|are|can|does|do|should|could|will|would)(?!'| not)\b",
    rf"{_CLAUSE_START}(?:have|has|had|did|was|were)(?!'| not)\b[^.?!]*\?",
    _KNOWLEDGE_SEEKING.pattern,
    r"\b(?:causes?|symptoms?|signs?|definition|meaning) of\b",
    r"\b(?:treatments?|first aid|guidelines?|management) (?:for|of)\b",
    r"\b(?:in general|generally|typically|usually|commonly)\b",
    r"\b(?:risk factors?|difference between|differential diagnosis|pathophysiology|mechanisms?|epidemiology|prevalence|incidence)\b",
    r"\b(?:explain|describe|understand)\b",
)

# Negation in its one unambiguous form: a negation word directly followed by a symptom noun,
# optionally continued with "or"/"nor". Deliberately not continued across "and" or commas
# ("no fever, sweating a lot" most likely affirms the sweating).
_SYMPTOM_NOUN = (
    r"(?:chest (?:pain|pressure|tightness)|shortness of breath|(?:trouble|difficulty) breathing"
    r"|breathlessness|pain|fever|temperature|bleeding|sweating|nausea|vomiting|swelling|rash"
    r"|hives|headache|weakness|numbness|dizziness|fainting|stiff neck|slurred speech|seizures?|wheezing)"
)
_NEGATED_SYMPTOMS = re.compile(
    r"\b(?:no|without|denies|not having|no (?:signs? |history )of"
    r"|(?:don'?t|do not|doesn'?t|does not|didn'?t|did not|haven'?t|hasn'?t|never) (?:have|has|had|having|got))"
    rf" (?:any )?{_SYMPTOM_NOUN}(?:,? (?:or|nor) (?:any )?{_SYMPTOM_NOUN})*\b"
)

# --- messages ----------------------------------------------------------------------------------

_ACTION: dict[Advice, str] = {
    "general": (
        "Call your local emergency number now (911 in the US and Canada, 999 in the UK, 112 in "
        "the EU) or go to the nearest emergency department."
    ),
    "poison": (
        "Call your local emergency number or poison centre now (911 or Poison Help "
        "1-800-222-1222 in the US, 999 or NHS 111 in the UK). Do not wait for symptoms to appear."
    ),
    "crisis": (
        "If you or someone you know is thinking about suicide or self-harm, you do not have to "
        "face it alone: call or text 988 (Suicide & Crisis Lifeline, US and Canada) or call "
        "Samaritans on 116 123 (UK and Ireland). If there is immediate danger, call your local "
        "emergency number."
    ),
}
_EMERGENCY_LEAD: dict[Advice, str] = {
    "general": "This may be a medical emergency. ",
    "poison": "This may be a poisoning or overdose. ",
    "crisis": "",
}
_NOT_ANSWERED = (
    "Primum is a research prototype and cannot assess an emergency, so it has not answered "
    "this question."
)
_GENERAL_QUESTION_LEAD = (
    "This reads as a general question, so it is answered below. If it describes something "
    "happening to you or someone with you right now, do not wait for the answer. "
)
_URGENT = (
    "These symptoms can need prompt medical attention. Contact a doctor or an urgent-care "
    "service today, and call your local emergency number if they are severe, sudden or getting "
    "worse. The answer below is general information, not a diagnosis."
)
_ADVICE_PRIORITY: tuple[Advice, ...] = ("crisis", "poison", "general")
_LEVEL_RANK: dict[Level, int] = {"routine": 0, "urgent": 1, "emergency": 2}


def triage(question: str) -> TriageResult:
    """Classify a question as `emergency`, `urgent` or `routine` (see the module docstring).

    `matched` names the rules that set the level, in lexicon order; `message` is the text to
    show the user (None for `routine`).
    """
    text = " ".join(question.lower().replace("’", "'").replace("‘", "'").split())
    affirmed = _NEGATED_SYMPTOMS.sub(" ", text)
    general = _is_general(text)

    fired = [
        (rule, _level_given_framing(rule, general))
        for rule in RULES
        if all(pattern.search(affirmed) for pattern in rule.all_of)
    ]
    level = max((lvl for _, lvl in fired), key=_LEVEL_RANK.__getitem__, default="routine")
    if level == "routine":
        return TriageResult(level="routine")
    # Only the rules at the final level are reported: "cardiac" explains an emergency, and the
    # weaker "chest_pain" rule that fired alongside it adds nothing.
    deciding = [rule for rule, lvl in fired if lvl == level]
    return TriageResult(
        level=level, matched=[rule.name for rule in deciding], message=_message(level, deciding)
    )


def _is_general(text: str) -> bool:
    """A request for knowledge that refers to no particular person and not to the present."""
    if _HAPPENING.search(text) or _PERSON.search(_KNOWLEDGE_SEEKING.sub(" ", text)):
        return False
    return _KNOWLEDGE_REQUEST.search(text) is not None


def _level_given_framing(rule: Rule, general: bool) -> Level:
    """A general framing lowers a rule by one level; it never silences an emergency rule."""
    if rule.any_framing or not general:
        return rule.level
    return "urgent" if rule.level == "emergency" else "routine"


def _message(level: Level, deciding: list[Rule]) -> str:
    advice = min((rule.advice for rule in deciding), key=_ADVICE_PRIORITY.index)
    if level == "emergency":
        return f"{_EMERGENCY_LEAD[advice]}{_ACTION[advice]} {_NOT_ANSWERED}"
    if advice == "crisis":
        return f"{_ACTION['crisis']} The answer below is general information."
    if any(rule.level == "emergency" for rule in deciding):
        return f"{_GENERAL_QUESTION_LEAD}{_ACTION[advice]}"
    return _URGENT
