/**
 * Derivations for one Ask run, from the pipeline's own result (`PipelineResult`) or events:
 * where the time went, how the claims were judged, and how the passages scored.
 */

import { fmtAlpha, fmtMs, fmtPct, fmtProb } from "../format.ts";
import { STAGES, type ClaimVerdict, type Hit, type VerdictLabel } from "../types.ts";

export interface StageSpan {
  stage: string;
  /** ms from the start of the run */
  start: number;
  end: number;
  ms: number;
  share: number;
}

/**
 * The stages as a waterfall: they run one after another, so each starts where the previous ended.
 * Stages are taken in pipeline order; any other timed key follows in the order given. A stage the
 * run skipped (no timing) is left out, not drawn as zero.
 */
export function stageWaterfall(timings: Readonly<Record<string, number>> | null | undefined, order: readonly string[] = STAGES): { spans: StageSpan[]; total: number } {
  if (!timings) return { spans: [], total: 0 };
  const keys = [...order.filter((k) => k in timings), ...Object.keys(timings).filter((k) => !order.includes(k))];
  const valid = keys.filter((k) => {
    const ms = timings[k];
    return typeof ms === "number" && Number.isFinite(ms) && ms >= 0;
  });
  const total = valid.reduce((sum, k) => sum + (timings[k] as number), 0);
  let cursor = 0;
  const spans = valid.map((stage) => {
    const ms = timings[stage] as number;
    const span = { stage, start: cursor, end: cursor + ms, ms, share: total > 0 ? ms / total : 0 };
    cursor += ms;
    return span;
  });
  return { spans, total };
}

export const VERDICT_ORDER: readonly VerdictLabel[] = ["supported", "unsupported", "contradicted"];

/** Claims per verdict, every verdict listed (a zero is a finding), in fixed order. */
export function claimVerdictCounts(verdicts: readonly ClaimVerdict[] | null | undefined): { label: VerdictLabel; count: number }[] {
  return VERDICT_ORDER.map((label) => ({ label, count: (verdicts ?? []).filter((v) => v.label === label).length }));
}

export interface PassageScore {
  rank: number;
  passageId: string;
  title: string;
  score: number;
  retriever: string;
  /** per-stage scores and ranks as the retriever reported them (bm25, dense, rrf, rerank, …) */
  components: Record<string, number>;
}

/** The hits in rank order with their scores. Scores are only comparable within one retriever. */
export function passageScores(hits: readonly Hit[] | null | undefined): PassageScore[] {
  return [...(hits ?? [])]
    .filter((h) => Number.isFinite(h.score))
    .sort((a, b) => a.rank - b.rank)
    .map((h) => ({ rank: h.rank, passageId: h.passage.passage_id, title: h.passage.title, score: h.score, retriever: h.retriever, components: h.components ?? {} }));
}

/* ---------- Confidence gauge ---------- */

export interface GaugeReading {
  /** calibrated P(correct), clamped to [0, 1]; null when the run has no calibrated confidence */
  pointer: number | null;
  /** the certified threshold, clamped to [0, 1]; null when no threshold certifies this alpha */
  gate: number | null;
  /** the run's decision as the server reported it; null before the confidence event */
  released: boolean | null;
  /** pointer minus gate (positive clears it); null when either is missing */
  margin: number | null;
  /** start of the cleared zone: the gate, or null when every value is withheld (no threshold) */
  clearedFrom: number | null;
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/**
 * Where the confidence reading sits against the gate. Values come from the confidence event;
 * the decision is the server's `answered`, never re-derived here.
 */
export function gaugeReading(p: number | null | undefined, threshold: number | null | undefined, answered: boolean | null | undefined): GaugeReading {
  const pointer = finite(p) ? clamp01(p) : null;
  const gate = finite(threshold) ? clamp01(threshold) : null;
  return {
    pointer,
    gate,
    released: typeof answered === "boolean" ? answered : null,
    margin: pointer !== null && gate !== null ? pointer - gate : null,
    clearedFrom: gate,
  };
}

/** A value in [0, 1] as an angle on a half-circle gauge (d3-shape convention: 0 at twelve o'clock, clockwise). */
export function gaugeAngle(value: number): number {
  return -Math.PI / 2 + clamp01(value) * Math.PI;
}

/** The gauge's one-line reading, from the numbers on it. */
export function gaugeTakeaway(reading: GaugeReading, alpha: number): string {
  const a = fmtAlpha(alpha);
  if (reading.pointer === null) return "No calibrated confidence for this answer.";
  const p = fmtProb(reading.pointer);
  if (reading.gate === null) return `Confidence ${p}; no threshold certifies α = ${a}, so every answer is withheld at this level.`;
  const g = fmtProb(reading.gate);
  const margin = fmtProb(Math.abs(reading.margin ?? 0));
  const m = reading.margin ?? 0;
  if (reading.released === null) return `Confidence ${p} against the threshold ${g} for α = ${a}.`;
  if (reading.released) {
    return m >= 0
      ? `Confidence ${p} clears the threshold ${g} for α = ${a} by ${margin}; released.`
      : `Confidence ${p} is below the threshold ${g} for α = ${a}, but the server released the answer.`;
  }
  if (m >= 0) return `Confidence ${p} is at or above the threshold ${g} for α = ${a}, but the server withheld the answer.`;
  return `Confidence ${p} is ${margin} short of the threshold ${g} for α = ${a}; withheld.`;
}

/* ---------- Timing ---------- */

/** "Generation took 2.31 s of 2.92 s (79%)": the stage that dominated the run. */
export function waterfallTakeaway({ spans, total }: { spans: readonly StageSpan[]; total: number }, label: (stage: string) => string = (s) => s): string {
  if (spans.length === 0 || total <= 0) return "No stage timings were reported.";
  const longest = spans.reduce((a, b) => (b.ms > a.ms ? b : a));
  return `${label(longest.stage)} took ${fmtMs(longest.ms)} of ${fmtMs(total)} (${fmtPct(longest.share)}); ${spans.length} ${spans.length === 1 ? "stage" : "stages"} timed.`;
}

/* ---------- Claim verdicts ---------- */

export interface VerdictShare {
  label: VerdictLabel;
  count: number;
  share: number;
}

/** Verdict counts with their share of all claims, fixed order, zeros kept. */
export function verdictShares(verdicts: readonly ClaimVerdict[] | null | undefined): { rows: VerdictShare[]; total: number } {
  const counts = claimVerdictCounts(verdicts);
  const total = counts.reduce((sum, c) => sum + c.count, 0);
  return { rows: counts.map((c) => ({ ...c, share: total > 0 ? c.count / total : 0 })), total };
}

export function verdictTakeaway({ rows, total }: { rows: readonly VerdictShare[]; total: number }): string {
  if (total === 0) return "The answer made no checkable claim.";
  const n = (label: VerdictLabel) => rows.find((r) => r.label === label)?.count ?? 0;
  const contra = n("contradicted");
  return `${n("supported")} of ${total} ${total === 1 ? "claim" : "claims"} supported by the cited passages, ${n("unsupported")} unsupported${contra ? `, ${contra} contradicted` : ", none contradicted"}.`;
}

/* ---------- Passage score profile ---------- */

export const SCORE_COLUMNS: readonly { key: string; label: string; rankKey?: string }[] = [
  { key: "bm25", label: "BM25", rankKey: "bm25_rank" },
  { key: "dense", label: "Dense", rankKey: "dense_rank" },
  { key: "rrf", label: "RRF" },
  { key: "rerank", label: "Rerank" },
];

export interface ScoreCell {
  key: string;
  value: number | null;
  rank: number | null;
  /**
   * Whether the passage was in this stage's list. For a stage that reports ranks (BM25, dense) a
   * missing rank means it was not in that retriever's candidate list, even when a score is given.
   */
  inList: boolean;
  /** bar start in [0, 1]: the column's zero (0 unless the column has negative values) or the value */
  start: number | null;
  /** bar length in [0, 1] measured from zero on the column's own scale; null when the value is missing */
  length: number | null;
}

export interface ScoreMatrix {
  /** `zero`: where 0 sits on the column's [min, max] scale (0 unless some value is negative) */
  columns: { key: string; label: string; max: number; min: number; zero: number; ranked: boolean }[];
  /** in the order the hits were given, which is how the sources panel numbers them */
  rows: { index: number; passageId: string; title: string; cells: ScoreCell[] }[];
}

/**
 * Each passage's per-stage scores, one column per stage that reported any. Scores are on
 * different scales per stage (BM25 is unbounded, dense is a cosine, RRF a sum of 1/(k + rank)),
 * so every column is drawn on its own scale from zero to its largest value and is never
 * compared with another column. Bars run from zero, so a column with negative values (a
 * cross-encoder logit can be) diverges either side of its zero. A missing rank means the passage
 * was not in that retriever's candidate list; the backend still reports a score for it.
 */
export function scoreMatrix(hits: readonly Hit[] | null | undefined): ScoreMatrix {
  const list = hits ?? [];
  const columns = SCORE_COLUMNS.filter((c) => list.some((h) => finite(h.components?.[c.key]))).map((c) => {
    const values = list.map((h) => h.components?.[c.key]).filter(finite);
    const max = Math.max(0, ...values);
    const min = Math.min(0, ...values);
    return { key: c.key, label: c.label, max, min, zero: max - min > 0 ? -min / (max - min) : 0, ranked: !!c.rankKey };
  });
  const rows = list.map((h, i) => ({
    index: i + 1,
    passageId: h.passage.passage_id,
    title: h.passage.title,
    cells: columns.map((c) => {
      const raw = h.components?.[c.key];
      const value = finite(raw) ? raw : null;
      const spec = SCORE_COLUMNS.find((s) => s.key === c.key);
      const rankRaw = spec?.rankKey ? h.components?.[spec.rankKey] : undefined;
      const span = c.max - c.min;
      const rank = finite(rankRaw) ? rankRaw : null;
      const pos = value === null ? null : span > 0 ? (value - c.min) / span : 0;
      return {
        key: c.key,
        value,
        rank,
        inList: c.ranked ? rank !== null : value !== null,
        start: pos === null ? null : Math.min(pos, c.zero),
        length: pos === null ? null : Math.abs(pos - c.zero),
      };
    }),
  }));
  return { columns, rows };
}

/** Which passage leads the final score, and how many came from only one of the two searches. */
export function scoreMatrixTakeaway(matrix: ScoreMatrix): string {
  if (matrix.rows.length === 0 || matrix.columns.length === 0) return "No passage scores were reported.";
  const final = matrix.columns.at(-1)!;
  const col = matrix.columns.indexOf(final);
  let best = matrix.rows[0]!;
  for (const row of matrix.rows) if ((row.cells[col]?.value ?? -Infinity) > (best.cells[col]?.value ?? -Infinity)) best = row;
  const ties = matrix.rows.filter((r) => r.cells[col]?.value === best.cells[col]?.value).length;
  const lead = ties > 1 ? `${ties} passages tie for the top ${final.label} score` : `Passage ${best.index} has the top ${final.label} score`;
  const bm = matrix.columns.findIndex((c) => c.key === "bm25");
  const de = matrix.columns.findIndex((c) => c.key === "dense");
  if (bm === -1 || de === -1) return `${lead}.`;
  // Membership is the rank, not the score: the backend reports both scores for every hit.
  const single = matrix.rows.filter((r) => (r.cells[bm]?.rank === null) !== (r.cells[de]?.rank === null)).length;
  return `${lead}; ${single} of ${matrix.rows.length} ${matrix.rows.length === 1 ? "passage was" : "passages were"} found by only one of BM25 and dense search.`;
}
