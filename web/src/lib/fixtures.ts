/**
 * Development fixtures for the Ask view.
 *
 * Loaded only when NEXT_PUBLIC_FIXTURES=1, and the view is bannered "Fixture data. Not real
 * output." for as long as they are in use. Every passage, claim and number below is invented
 * to exercise the interface; none of it comes from the pipeline, a model, or a real source.
 */

import type { ClaimVerdict, ConfidenceEvent, Decision, Hit, Mode, PipelineEvent, PipelineResult, TriageResult } from "./types.ts";

export const FIXTURE_SCENARIOS = [
  { id: "answered", label: "Answer released" },
  { id: "withheld", label: "Answer withheld" },
  { id: "emergency", label: "Emergency triage" },
  { id: "urgent", label: "Urgent, answered with a warning" },
  { id: "uncalibrated", label: "No deployment bundle" },
  { id: "benchmark", label: "Benchmark question" },
  { id: "no_api_key", label: "Error: no API key" },
  { id: "budget", label: "Error: budget spent" },
  { id: "no_index", label: "Error: no index" },
  { id: "internal", label: "Error: internal" },
] as const;

export type FixtureScenario = (typeof FIXTURE_SCENARIOS)[number]["id"];

const NOTE = "Fixture text written for interface development.";

function hit(
  rank: number,
  source: Hit["passage"]["source"],
  title: string,
  body: string,
  url: string | null,
  components: Record<string, number>,
): Hit {
  return {
    passage: {
      passage_id: `fixture-${rank}`,
      doc_id: `fixture-${String(rank).padStart(4, "0")}`,
      title: `Fixture passage: ${title}`,
      text: `${NOTE} ${body}`,
      source,
      url,
      meta: { fixture: true },
    },
    score: components.rrf ?? 0,
    rank,
    retriever: "hybrid-ft",
    components,
  };
}

const S = {
  adenoma: "Recurrence of adenoma was less frequent in the aspirin group than in the placebo group.",
  cohort:
    "In this invented cohort summary, people who took aspirin regularly for ten years or longer had fewer colorectal cancers than people who did not.",
  early: "No difference was seen during the first five years of use.",
  dose: "The summary did not assess doses above 325 milligrams a day.",
  bleeding: "Daily aspirin use was associated with more major bleeding events, and the excess was larger in adults over seventy.",
  clinician: "People are advised to talk with a clinician before taking any medicine regularly to lower cancer risk.",
};

const HITS: Hit[] = [
  hit(
    1,
    "pubmed",
    "low-dose aspirin and adenoma recurrence",
    `In this invented trial summary, adults with a previous adenoma took low-dose aspirin or placebo for three years. ${S.adenoma} Gastrointestinal bleeding was more frequent in the aspirin group. The authors of this invented summary note that the benefit appeared only after several years of regular use.`,
    null,
    { bm25: 14.2, bm25_rank: 1, dense: 0.71, dense_rank: 2, rrf: 0.0325 },
  ),
  hit(
    2,
    "medlineplus",
    "colorectal cancer overview",
    `Colorectal cancer begins in the colon or the rectum. Risk rises with age, with a family history of the disease, and with certain inherited conditions. Screening can find polyps before they become cancer. ${S.clinician}`,
    "https://medlineplus.gov/colorectalcancer.html",
    { bm25: 9.8, bm25_rank: 4, dense: 0.74, dense_rank: 1, rrf: 0.032 },
  ),
  hit(
    3,
    "pubmed",
    "long-term follow-up of aspirin users",
    `${S.cohort} ${S.early} ${S.dose}`,
    null,
    { bm25: 12.6, bm25_rank: 2, dense: 0.66, dense_rank: 3, rrf: 0.032 },
  ),
  hit(
    4,
    "pubmed",
    "bleeding risk with daily aspirin",
    `${S.bleeding} The invented summary concludes that the decision to take aspirin should weigh bleeding risk against any expected benefit.`,
    null,
    { bm25: 10.9, bm25_rank: 3, dense: 0.58, dense_rank: 6, rrf: 0.031 },
  ),
  hit(
    5,
    "nfcorpus",
    "dietary fibre and bowel habits",
    "Higher fibre intake was associated with more regular bowel habits in this invented survey. The survey did not measure cancer outcomes.",
    null,
    { bm25: 3.1, dense: 0.61, dense_rank: 4, rrf: 0.0156 },
  ),
];

const RELEASED_TEXT = `Regular low-dose aspirin is associated with fewer colorectal adenomas and, after long-term use, fewer colorectal cancers [1][3]. The benefit takes years to appear: no difference was seen in the first five years [3]. Aspirin also raises the risk of bleeding, **especially in adults over seventy** [1, 4].

The passages do not say who should take aspirin for this purpose, so that decision belongs with a clinician [2].`;

const RELEASED_VERDICTS: ClaimVerdict[] = [
  {
    claim: { text: "Low-dose aspirin was associated with less frequent adenoma recurrence.", cited: [1] },
    label: "supported",
    p_entail: 0.96,
    p_neutral: 0.03,
    p_contra: 0.01,
    best_hit: 1,
    best_sentence: S.adenoma,
  },
  {
    claim: { text: "Fewer colorectal cancers were seen after ten or more years of regular aspirin use.", cited: [3] },
    label: "supported",
    p_entail: 0.93,
    p_neutral: 0.06,
    p_contra: 0.01,
    best_hit: 3,
    best_sentence: S.cohort,
  },
  {
    claim: { text: "Aspirin increases bleeding risk, more so in adults over seventy.", cited: [1, 4] },
    label: "supported",
    p_entail: 0.91,
    p_neutral: 0.08,
    p_contra: 0.01,
    best_hit: 4,
    best_sentence: S.bleeding,
  },
  {
    claim: { text: "The passages do not identify who should take aspirin to lower cancer risk.", cited: [2] },
    label: "unsupported",
    p_entail: 0.38,
    p_neutral: 0.57,
    p_contra: 0.05,
    best_hit: 2,
    best_sentence: S.clinician,
  },
];

const WITHHELD_TEXT = `Yes. Aspirin prevents colorectal cancer, and the protection begins within the first year of use [3]. Higher doses give more protection [3]. Bleeding is the main harm of taking it daily [4].`;

const WITHHELD_VERDICTS: ClaimVerdict[] = [
  {
    claim: { text: "Protection against colorectal cancer begins within the first year of aspirin use.", cited: [3] },
    label: "contradicted",
    p_entail: 0.03,
    p_neutral: 0.09,
    p_contra: 0.88,
    best_hit: 3,
    best_sentence: S.early,
  },
  {
    claim: { text: "Higher doses of aspirin give more protection.", cited: [3] },
    label: "unsupported",
    p_entail: 0.06,
    p_neutral: 0.71,
    p_contra: 0.23,
    best_hit: 3,
    best_sentence: S.dose,
  },
  {
    claim: { text: "Aspirin prevents colorectal cancer.", cited: [3] },
    label: "unsupported",
    p_entail: 0.41,
    p_neutral: 0.52,
    p_contra: 0.07,
    best_hit: 3,
    best_sentence: S.cohort,
  },
  {
    claim: { text: "Bleeding is the main harm of daily aspirin.", cited: [4] },
    label: "supported",
    p_entail: 0.74,
    p_neutral: 0.24,
    p_contra: 0.02,
    best_hit: 4,
    best_sentence: S.bleeding,
  },
];

/** Invented thresholds, one per risk level, so that changing alpha changes the decision. */
const FIXTURE_THRESHOLDS: Record<string, number | null> = { "0.05": null, "0.10": 0.78, "0.15": 0.66, "0.20": 0.55 };
const DELTA = 0.1;

function confidenceFor(p: number, alpha: number, verdicts: ClaimVerdict[], extra: Partial<ConfidenceEvent> = {}): ConfidenceEvent {
  const threshold = FIXTURE_THRESHOLDS[alpha.toFixed(2)] ?? null;
  const supported = verdicts.filter((v) => v.label === "supported").length;
  const entail = verdicts.map((v) => v.p_entail);
  return {
    features: {
      ret_top: 0.74,
      ret_mean: 0.66,
      ret_margin: 0.03,
      ret_agree: 0.8,
      nli_mean_entail: entail.reduce((a, b) => a + b, 0) / Math.max(1, entail.length),
      nli_min_entail: Math.min(...entail),
      nli_max_contra: Math.max(...verdicts.map((v) => v.p_contra)),
      frac_supported: supported / Math.max(1, verdicts.length),
      n_claims: verdicts.length,
      verbal_conf: p > 0.7 ? 0.85 : 0.9,
    },
    p_correct: p,
    alpha,
    delta: DELTA,
    threshold,
    answered: threshold !== null && p >= threshold,
    // As the server reports a chat answer: scored by the bundle, but outside the setting its
    // thresholds were certified on (the benchmark scenario below overrides this).
    uncalibrated: false,
    guarantee_applies: false,
    ...extra,
  };
}

interface Timed {
  wait: number;
  event: PipelineEvent;
}

function stage(name: string, ms: number): Timed[] {
  return [
    { wait: 60, event: { type: "stage", data: { name, status: "start" } } },
    { wait: Math.min(ms, 700), event: { type: "stage", data: { name, status: "done", ms } } },
  ];
}

function splice(pair: Timed[], between: Timed[]): Timed[] {
  return [pair[0] as Timed, ...between, pair[1] as Timed];
}

function tokens(text: string): Timed[] {
  return (text.match(/\S+\s*/g) ?? []).map((piece) => ({ wait: 28, event: { type: "token", data: { text: piece } } }));
}

const ROUTINE: TriageResult = { level: "routine", matched: [], message: null };

function result(
  question: string,
  mode: Mode,
  triage: TriageResult,
  hits: Hit[],
  text: string,
  verdicts: ClaimVerdict[],
  confidence: ConfidenceEvent | null,
  timings: Record<string, number>,
  decision: Decision | null = null,
  gold: Decision | null = null,
): PipelineResult {
  return {
    trace_id: "fixture",
    question,
    mode,
    triage,
    concepts: [],
    hits,
    answer_text: text,
    decision,
    verdicts,
    confidence,
    usage: { input_tokens: 0, output_tokens: 0, cache_read_tokens: 0, cost_usd: 0, cached: true },
    timings_ms: timings,
    retriever: hits.length ? "hybrid-ft" : "",
    model: "fixture (no model was called)",
    gold_label: gold,
  };
}

const CONCEPTS = {
  concepts: [
    {
      surface: "aspirin",
      start: 0,
      end: 7,
      preferred: "Aspirin",
      mesh_ids: ["D001241"],
      synonyms: ["acetylsalicylic acid"],
      url: null,
    },
    {
      surface: "colorectal cancer",
      start: 0,
      end: 17,
      preferred: "Colorectal Neoplasms",
      mesh_ids: ["D015179"],
      synonyms: ["colon cancer", "rectal cancer"],
      url: "https://medlineplus.gov/colorectalcancer.html",
    },
  ],
};

function script(scenario: FixtureScenario, question: string, alpha: number): Timed[] {
  const head: Timed[] = [
    ...splice(stage("triage", 1.4), [{ wait: 20, event: { type: "triage", data: ROUTINE } }]),
    ...splice(stage("concepts", 3.2), [{ wait: 20, event: { type: "concepts", data: CONCEPTS } }]),
  ];
  const retrieval = splice(stage("retrieve", 184), [
    { wait: 20, event: { type: "retrieval", data: { retriever: "hybrid-ft", hits: HITS } } },
  ]);
  const timings = { triage: 1.4, concepts: 3.2, retrieve: 184, generate: 2310, verify: 412, confidence: 6.1 };

  const full = (text: string, verdicts: ClaimVerdict[], confidence: ConfidenceEvent, triage: TriageResult = ROUTINE): Timed[] => [
    ...(triage === ROUTINE
      ? head
      : [
          ...splice(stage("triage", 1.4), [{ wait: 20, event: { type: "triage", data: triage } } as Timed]),
          ...splice(stage("concepts", 3.2), [{ wait: 20, event: { type: "concepts", data: CONCEPTS } } as Timed]),
        ]),
    ...retrieval,
    ...splice(stage("generate", 2310), [...tokens(text), { wait: 40, event: { type: "answer", data: { text } } }]),
    ...splice(stage("verify", 412), [{ wait: 20, event: { type: "verdicts", data: { verdicts } } }]),
    ...splice(stage("confidence", 6.1), [{ wait: 20, event: { type: "confidence", data: confidence } }]),
    { wait: 40, event: { type: "done", data: result(question, "chat", triage, HITS, text, verdicts, confidence, timings) } },
  ];

  switch (scenario) {
    case "answered":
      return full(RELEASED_TEXT, RELEASED_VERDICTS, confidenceFor(0.86, alpha, RELEASED_VERDICTS));
    case "withheld":
      return full(WITHHELD_TEXT, WITHHELD_VERDICTS, confidenceFor(0.61, alpha, WITHHELD_VERDICTS));
    case "urgent":
      return full(RELEASED_TEXT, RELEASED_VERDICTS, confidenceFor(0.86, alpha, RELEASED_VERDICTS), {
        level: "urgent",
        matched: ["blood in stool"],
        message: "Blood in the stool should be assessed by a clinician soon.",
      });
    case "uncalibrated":
      return full(RELEASED_TEXT, RELEASED_VERDICTS, {
        ...confidenceFor(0.5, alpha, RELEASED_VERDICTS),
        p_correct: null,
        threshold: null,
        answered: true,
        uncalibrated: true,
      });
    case "emergency": {
      const triage: TriageResult = {
        level: "emergency",
        matched: ["chest pain with radiation to the arm", "chest pain with sweating"],
        message: "Chest pain that spreads to the arm and comes with sweating can be a heart attack.",
      };
      return [
        ...splice(stage("triage", 1.4), [{ wait: 20, event: { type: "triage", data: triage } }]),
        { wait: 40, event: { type: "done", data: result(question, "chat", triage, [], "", [], null, { triage: 1.4 }) } },
      ];
    }
    case "benchmark": {
      const text = "Long-term regular aspirin use was associated with fewer colorectal cancers in the retrieved passages.";
      const verdicts = RELEASED_VERDICTS.slice(0, 2);
      const confidence = confidenceFor(0.84, alpha, verdicts, { prediction_set: ["yes", "maybe"] });
      confidence.features = { ...confidence.features, sc_agreement: 0.8, sc_entropy: 0.46, verbal_margin: 0.55 };
      confidence.guarantee_applies = confidence.threshold !== null;
      return [
        ...head,
        ...retrieval,
        ...splice(stage("generate", 3120), [{ wait: 40, event: { type: "answer", data: { text, decision: "yes" } } }]),
        ...splice(stage("verify", 230), [{ wait: 20, event: { type: "verdicts", data: { verdicts } } }]),
        ...splice(stage("confidence", 5.4), [{ wait: 20, event: { type: "confidence", data: confidence } }]),
        {
          wait: 40,
          event: {
            type: "done",
            data: result(question, "benchmark", ROUTINE, HITS, text, verdicts, confidence, { ...timings, generate: 3120, verify: 230 }, "yes", "yes"),
          },
        },
      ];
    }
    case "no_index":
      return [
        ...head,
        { wait: 60, event: { type: "stage", data: { name: "retrieve", status: "start" } } },
        {
          wait: 200,
          event: { type: "error", data: { stage: "retrieve", kind: "no_index", message: "No index found for corpus 'kb'." } },
        },
      ];
    case "no_api_key":
    case "budget":
    case "internal": {
      const message =
        scenario === "no_api_key"
          ? "ANTHROPIC_API_KEY is not configured."
          : scenario === "budget"
            ? "Cumulative spend has reached the configured budget."
            : "RuntimeError: fixture failure raised inside the generator.";
      return [
        ...head,
        ...retrieval,
        { wait: 60, event: { type: "stage", data: { name: "generate", status: "start" } } },
        { wait: 300, event: { type: "error", data: { stage: "generate", kind: scenario, message } } },
      ];
    }
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

/** Replays a canned event sequence with small delays, in place of `POST /api/ask`. */
export async function* replayFixture(
  scenario: FixtureScenario,
  request: { question: string; alpha: number },
  signal?: AbortSignal,
): AsyncGenerator<PipelineEvent> {
  for (const step of script(scenario, request.question, request.alpha)) {
    await sleep(step.wait, signal);
    yield step.event;
  }
}
