/**
 * Data contracts shared with the API. These mirror `src/primum/types.py` field for field;
 * when that file changes, change this one.
 */

export type Source = "pubmed" | "medlineplus" | "nfcorpus" | "pdf";
export type Decision = "yes" | "no" | "maybe";
export type VerdictLabel = "supported" | "unsupported" | "contradicted";
export type Mode = "chat" | "benchmark";

export interface Passage {
  passage_id: string;
  doc_id: string;
  title: string;
  text: string;
  source: Source;
  url: string | null;
  meta: Record<string, unknown>;
}

export interface Hit {
  passage: Passage;
  score: number;
  /** 1-based */
  rank: number;
  retriever: string;
  /** Per-stage scores and ranks: bm25, dense, rrf, rerank, bm25_rank, dense_rank. */
  components: Record<string, number>;
}

export interface Claim {
  text: string;
  /** 1-based indices into the hit list */
  cited: number[];
}

export interface ClaimVerdict {
  claim: Claim;
  label: VerdictLabel;
  p_entail: number;
  p_neutral: number;
  p_contra: number;
  /** 1-based index of the passage that best supports it */
  best_hit: number | null;
  best_sentence: string | null;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cost_usd: number;
  cached: boolean;
}

export interface ConfidenceReport {
  features: Record<string, number>;
  /** calibrated probability that the answer is correct; null when no deployment bundle exists */
  p_correct: number | null;
  /** target selective risk */
  alpha: number;
  /** tolerated failure probability of the guarantee */
  delta: number;
  /** lambda-hat; null means no threshold certifies this alpha */
  threshold: number | null;
  /** p_correct >= threshold */
  answered: boolean;
  /** conformal set, benchmark mode only */
  prediction_set?: Decision[] | null;
}

/**
 * What the report may be read as. The `confidence` event carries these flags, and the server
 * stores them on the trace's report too.
 */
export interface ConfidenceFlags {
  /** no deployment bundle exists, so the answer is released unscored (SPEC 3.7) */
  uncalibrated?: boolean;
  /**
   * The certificate P(risk <= alpha) >= 1 - delta covers this answer. True only for a benchmark
   * question answered by the pipeline the thresholds were certified on (same generator and
   * retriever). For a chat question the threshold is an operating point, not a guarantee.
   */
  guarantee_applies?: boolean;
}

export interface ConfidenceEvent extends ConfidenceReport, ConfidenceFlags {}

export interface TriageResult {
  level: "emergency" | "urgent" | "routine";
  matched: string[];
  message: string | null;
}

export interface Concept {
  surface: string;
  start: number;
  end: number;
  preferred: string;
  mesh_ids: string[];
  synonyms: string[];
  url: string | null;
}

export interface PipelineResult {
  trace_id: string;
  question: string;
  mode: Mode;
  triage: TriageResult;
  concepts: Concept[];
  hits: Hit[];
  answer_text: string;
  decision: Decision | null;
  verdicts: ClaimVerdict[];
  confidence: ConfidenceEvent | null;
  usage: Usage;
  timings_ms: Record<string, number>;
  retriever: string;
  model: string;
  gold_label: Decision | null;
}

/* ---------- Pipeline events (SPEC 3.7) ---------- */

export const STAGES = ["triage", "concepts", "retrieve", "generate", "verify", "confidence"] as const;
export type StageName = (typeof STAGES)[number];

export type ErrorKind = "no_api_key" | "budget" | "no_index" | "internal";

export interface StageEvent {
  name: string;
  status: "start" | "done";
  ms?: number | null;
}

export interface PipelineErrorData {
  stage: string;
  message: string;
  kind: ErrorKind;
}

export type PipelineEvent =
  | { type: "stage"; data: StageEvent }
  | { type: "triage"; data: TriageResult }
  /** `available` is false when the concept dictionary (the `kb` corpus) is not built: nothing was linked. */
  | { type: "concepts"; data: { concepts: Concept[]; available?: boolean } }
  | { type: "retrieval"; data: { retriever: string; hits: Hit[] } }
  | { type: "token"; data: { text: string } }
  | { type: "answer"; data: { text: string; decision?: Decision | null; probabilities?: Partial<Record<Decision, number>> } }
  | { type: "verdicts"; data: { verdicts: ClaimVerdict[] } }
  | { type: "confidence"; data: ConfidenceEvent }
  | { type: "done"; data: PipelineResult }
  | { type: "error"; data: PipelineErrorData };

/* ---------- Request and response bodies (SPEC 3.8) ---------- */

export interface AskRequest {
  question: string;
  mode?: Mode;
  retriever?: string;
  alpha?: number;
  benchmark_qid?: string;
}

export interface RetrieveRequest {
  query: string;
  k?: number;
  retrievers: string[];
  corpus?: string;
}

export interface RetrieveResponse {
  results: Record<string, Hit[]>;
  concepts: Concept[];
  expanded_query: string;
  /** false when the concept dictionary (the `kb` corpus) is not built, so no term could be linked or expanded */
  concepts_available?: boolean;
  /** the corpus searched (the request's, or the server's chat corpus) */
  corpus: string;
}

export interface BenchmarkQuestion {
  qid: string;
  question: string;
  label?: Decision | null;
}

/**
 * `/api/health` and `/api/meta` are described in prose only, so every field is optional
 * and the interface reads them defensively.
 */
export interface Health {
  status?: string;
  indexes?: Record<string, unknown> | string[] | boolean;
  finetuned_encoder?: boolean;
  deployment_bundle?: boolean;
  api_key?: boolean;
}

export interface Meta {
  retrievers?: string[];
  default_retriever?: string;
  alphas?: number[];
  alpha?: number;
  delta?: number;
  models?: Record<string, string>;
  /** passages per corpus; null for a corpus that has not been built */
  corpora?: Record<string, number | null>;
  spend?: { total_usd?: number; calls?: number; cached_calls?: number };
  budget_usd?: number;
}

/**
 * One generator run of the experiments, as `/api/experiments` lists it. Every field is read by
 * the server from the run's results files (`meta.models.llm`, `meta.n`,
 * `meta.self_consistency_samples`, `meta.conditions`, `summary.pubmedqa.<condition>.n_failed`);
 * null where the files do not record it.
 */
export interface RunInfo {
  run: string;
  generator: string | null;
  /** questions attempted */
  n: number | null;
  self_consistency_samples: number | null;
  conditions: string[] | null;
  /** failed generations per condition, excluded from RQ2 and RQ3 */
  n_failed: Record<string, number> | null;
  available: string[];
  /** set when another run of the same generator covers all of this run's questions (counted from the records) */
  subset_of?: RunSubset | null;
}

export interface RunSubset {
  run: string;
  /** this run's questions are the first ones of that run, in dataset order */
  prefix: boolean;
  answered_in_both: number;
  /** questions answered in both whose first-sample decision agrees */
  first_sample_matches: number;
  /** questions that failed here but were answered in that run */
  failed_only_here: number;
}

export interface ExperimentsIndex {
  run?: string;
  available: string[];
  files: Record<string, { updated?: string }>;
  runs?: RunInfo[];
}

/** Spec defaults, used only to offer controls before `/api/meta` answers. */
export const SPEC_RETRIEVERS = [
  "bm25",
  "dense-base",
  "dense-biomed",
  "dense-ft",
  "hybrid-base",
  "hybrid-ft",
  "hybrid-ft+rerank",
] as const;
export const SPEC_ALPHAS = [0.05, 0.1, 0.15, 0.2] as const;
export const SPEC_DEFAULT_ALPHA = 0.1;
export const SPEC_DEFAULT_RETRIEVER = "hybrid-ft";
