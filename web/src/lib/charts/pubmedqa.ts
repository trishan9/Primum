/**
 * PubMedQA end-to-end runs (`results/<run>/pubmedqa_runs.json`): answer accuracy by evidence
 * condition, paired differences between conditions, the confusion of decisions against gold
 * labels, and the stated confidence of right and wrong answers.
 */

import { fmtInt, fmtPct } from "../format.ts";
import { at, equalBinIndex, equalBins, isRecord, num, readEstimate, summarise, type Bin, type Estimate, type IntervalInfo, type Summary, INTERVAL_LABEL } from "./common.ts";

/** Conditions in the order the method builds them up: no evidence, retrieved, gold abstract. */
export const CONDITION_ORDER = ["closed_book", "rag", "oracle"] as const;

export const CONDITION_LABEL: Record<string, string> = {
  closed_book: "Closed book",
  rag: "With retrieval",
  oracle: "Gold abstract",
};

export const conditionLabel = (condition: string): string => CONDITION_LABEL[condition] ?? condition.replace(/_/g, " ");

/**
 * `run_pubmedqa.py` writes accuracy and macro-F1 with `bootstrap_estimate` (95% percentile
 * bootstrap over questions) and paired differences the same way over paired questions. The file
 * does not restate it in `meta`, so the label is fixed here; it is a label, not a number.
 */
export const PUBMEDQA_INTERVAL: IntervalInfo = {
  kind: "bootstrap",
  text: "95% percentile bootstrap over questions",
  label: INTERVAL_LABEL.bootstrap,
};

const byConditionOrder = (a: string, b: string) => {
  const ia = CONDITION_ORDER.indexOf(a as (typeof CONDITION_ORDER)[number]);
  const ib = CONDITION_ORDER.indexOf(b as (typeof CONDITION_ORDER)[number]);
  return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib) || a.localeCompare(b);
};

export interface ConditionAccuracy {
  condition: string;
  label: string;
  /** questions scored (failed generations excluded) */
  n: number | null;
  nFailed: number | null;
  accuracy: Estimate | null;
  macroF1: Estimate | null;
  supportedClaimRate: number | null;
  /** share of questions whose gold abstract was among the retrieved passages; absent closed-book */
  goldInHitsRate: number | null;
}

/** One row per condition the run answered under, in method order. */
export function accuracyByCondition(data: unknown): ConditionAccuracy[] {
  const summary = at(data, "summary", "pubmedqa");
  if (!isRecord(summary)) return [];
  return Object.keys(summary)
    .sort(byConditionOrder)
    .flatMap((condition) => {
      const entry = summary[condition];
      if (!isRecord(entry)) return [];
      return [
        {
          condition,
          label: conditionLabel(condition),
          n: num(entry.n),
          nFailed: num(entry.n_failed),
          accuracy: readEstimate(entry.accuracy),
          macroF1: readEstimate(entry.macro_f1),
          supportedClaimRate: num(entry.supported_claim_rate),
          goldInHitsRate: num(entry.gold_in_hits_rate),
        },
      ];
    });
}

export interface ConditionDiff {
  key: string;
  /** the condition whose accuracy is first in the difference (a minus b) */
  a: string;
  b: string;
  diff: Estimate | null;
  p: number | null;
  nPairs: number | null;
}

/** `paired_tests: {"rag_vs_closed_book": {accuracy_diff, p_value, n_pairs}}`. */
export function pairedConditionDiffs(data: unknown): ConditionDiff[] {
  const tests = at(data, "paired_tests");
  if (!isRecord(tests)) return [];
  return Object.entries(tests).flatMap(([key, entry]) => {
    const match = key.match(/^(.+)_vs_(.+)$/);
    if (!match || !isRecord(entry)) return [];
    return [{ key, a: match[1] as string, b: match[2] as string, diff: readEstimate(entry.accuracy_diff), p: num(entry.p_value), nPairs: num(entry.n_pairs) }];
  });
}

/* ---------- Per-question records ---------- */

export const DECISIONS = ["yes", "no", "maybe"] as const;
export type Decision = (typeof DECISIONS)[number];

interface RecordRow {
  condition: string;
  gold: string | null;
  decision: string | null;
  correct: boolean | null;
  verbalConf: number | null;
  failed: boolean;
}

function readRecords(data: unknown): RecordRow[] {
  const records = at(data, "records");
  if (!Array.isArray(records)) return [];
  return records.flatMap((r) => {
    if (!isRecord(r) || typeof r.condition !== "string") return [];
    const features = isRecord(r.features) ? r.features : {};
    return [
      {
        condition: r.condition,
        gold: typeof r.gold === "string" ? r.gold : null,
        decision: typeof r.decision === "string" ? r.decision : null,
        correct: typeof r.correct === "boolean" ? r.correct : null,
        verbalConf: num(features.verbal_conf),
        failed: "error" in r || typeof r.decision !== "string",
      },
    ];
  });
}

export interface ConfusionCell {
  gold: Decision;
  decision: Decision;
  count: number;
  /** count over all scored questions with this gold label; null when there are none */
  shareOfGold: number | null;
}

export interface Confusion {
  condition: string;
  label: string;
  /** scored questions (failed generations excluded) */
  n: number;
  failed: number;
  /** gold-major order: yes→yes, yes→no, yes→maybe, no→yes, … */
  cells: ConfusionCell[];
  /** questions per gold label, the row totals */
  goldTotals: Record<Decision, number>;
}

/** Gold label against the generator's decision, counted from the per-question records. */
export function confusionByCondition(data: unknown): Confusion[] {
  const rows = readRecords(data);
  const conditions = [...new Set(rows.map((r) => r.condition))].sort(byConditionOrder);
  return conditions.map((condition) => {
    const mine = rows.filter((r) => r.condition === condition);
    const scored = mine.filter((r) => !r.failed && DECISIONS.includes(r.gold as Decision) && DECISIONS.includes(r.decision as Decision));
    const goldTotals = Object.fromEntries(DECISIONS.map((g) => [g, scored.filter((r) => r.gold === g).length])) as Record<Decision, number>;
    const cells = DECISIONS.flatMap((gold) =>
      DECISIONS.map((decision) => {
        const count = scored.filter((r) => r.gold === gold && r.decision === decision).length;
        return { gold, decision, count, shareOfGold: goldTotals[gold] > 0 ? count / goldTotals[gold] : null };
      }),
    );
    return { condition, label: conditionLabel(condition), n: scored.length, failed: mine.length - scored.length, cells, goldTotals };
  });
}

export interface ConfidenceHistogram {
  condition: string;
  /** one entry per bin: answers that were right and wrong whose stated confidence fell in it */
  bins: (Bin & { correct: number; wrong: number })[];
  nCorrect: number;
  nWrong: number;
  /** scored answers with no stated confidence */
  missing: number;
  /** one bin per distinct stated value (lo = hi = the value) rather than equal-width bins */
  distinct: boolean;
}

/** Stated confidences with at most this many distinct values get one bin per value. */
export const MAX_DISTINCT_BINS = 12;

/**
 * Bins for stated confidences. Stated values cluster on a few round numbers, so narrow equal
 * bins make a comb of alternating empty and full bins. At most 12 distinct values: one bin per
 * value. Otherwise `count` equal-width bins over [0, 1] (10, so each is 0.1 wide).
 */
export function confidenceBins(values: readonly number[], count = 10): { bins: Bin[]; distinct: boolean } {
  const finiteValues = values.filter((v) => Number.isFinite(v) && v >= 0 && v <= 1);
  const levels = [...new Set(finiteValues)].sort((a, b) => a - b);
  if (levels.length > 0 && levels.length <= MAX_DISTINCT_BINS) {
    return { bins: levels.map((v) => ({ lo: v, hi: v, count: finiteValues.filter((x) => x === v).length })), distinct: true };
  }
  return { bins: equalBins(finiteValues, [0, 1], count), distinct: false };
}

/**
 * The generator's stated (verbal) confidence for right and wrong answers, in equal-width bins
 * over [0, 1]. Counts, not densities: the two groups differ in size, and the chart says so.
 */
export function verbalConfidenceHistogram(data: unknown, condition: string, binCount = 10): ConfidenceHistogram | null {
  const rows = readRecords(data).filter((r) => r.condition === condition && !r.failed && r.correct !== null);
  if (rows.length === 0) return null;
  const withConf = rows.filter((r) => r.verbalConf !== null);
  const right = withConf.filter((r) => r.correct).map((r) => r.verbalConf as number);
  const wrong = withConf.filter((r) => !r.correct).map((r) => r.verbalConf as number);
  const { bins, distinct } = confidenceBins([...right, ...wrong], binCount);
  const inBin = (b: Bin, i: number, v: number) => (distinct ? v === b.lo : v >= 0 && v <= 1 && equalBinIndex(v, [0, 1], bins.length) === i);
  return {
    condition,
    bins: bins.map((b, i) => {
      const correct = right.filter((v) => inBin(b, i, v)).length;
      const w = wrong.filter((v) => inBin(b, i, v)).length;
      return { lo: b.lo, hi: b.hi, count: correct + w, correct, wrong: w };
    }),
    nCorrect: right.length,
    nWrong: wrong.length,
    missing: rows.length - withConf.length,
    distinct,
  };
}

/** The conditions present in the per-question records, in method order. */
export function recordConditions(data: unknown): string[] {
  return [...new Set(readRecords(data).map((r) => r.condition))].sort(byConditionOrder);
}

/* ---------- Takeaway ---------- */

const points = (d: number) => `${d >= 0 ? "+" : "\u2212"}${(Math.abs(d) * 100).toFixed(1)} points`;

/**
 * One sentence for the accuracy chart, from the summary and the paired tests: each paired
 * difference with its bootstrap CI, or, for a single-condition run, its one accuracy.
 */
export function accuracyTakeaway(rows: readonly ConditionAccuracy[], diffs: readonly ConditionDiff[]): string | null {
  const scored = rows.filter((r) => r.accuracy);
  if (scored.length === 0) return null;
  if (scored.length === 1) {
    const r = scored[0] as ConditionAccuracy & { accuracy: Estimate };
    const ci = r.accuracy.lo !== null && r.accuracy.hi !== null ? ` (95% CI ${fmtPct(r.accuracy.lo, 1)} to ${fmtPct(r.accuracy.hi, 1)})` : "";
    return `One condition was run, ${r.label.toLowerCase()}: ${fmtPct(r.accuracy.value, 1)} of ${fmtInt(r.n)} answers correct${ci}.`;
  }
  const byCondition = new Map(scored.map((r) => [r.condition, r]));
  const parts = diffs.flatMap((d) => {
    const a = byCondition.get(d.a);
    const b = byCondition.get(d.b);
    if (!a?.accuracy || !b?.accuracy || !d.diff) return [];
    const ci = d.diff.lo !== null && d.diff.hi !== null ? `, 95% CI ${points(d.diff.lo).replace(" points", "")} to ${points(d.diff.hi)}` : "";
    return [`${a.label} vs ${b.label.toLowerCase()}: ${fmtPct(a.accuracy.value, 1)} against ${fmtPct(b.accuracy.value, 1)} (${points(d.diff.value)}${ci}${d.nPairs !== null ? `, ${fmtInt(d.nPairs)} paired questions` : ""})`];
  });
  if (parts.length > 0) return `${parts.join("; ")}.`;
  return scored.map((r) => `${r.label}: ${fmtPct((r.accuracy as Estimate).value, 1)}`).join(", ") + ".";
}

/* ---------- Confusion and confidence readings ---------- */

const quote = (label: string) => `“${label}”`;

/**
 * One sentence for a confusion matrix: overall agreement, then the gold label the generator
 * matched least often and what it said instead. Built from the counts, never typed in.
 */
export function confusionTakeaway(c: Confusion): string | null {
  if (c.n === 0) return null;
  const agree = c.cells.filter((cell) => cell.gold === cell.decision).reduce((s, cell) => s + cell.count, 0);
  const head = `${fmtPct(agree / c.n, 1)} of ${fmtInt(c.n)} scored answers match the gold label`;
  const rows = DECISIONS.filter((g) => c.goldTotals[g] > 0).map((gold) => {
    const own = c.cells.filter((cell) => cell.gold === gold);
    const hit = own.find((cell) => cell.decision === gold) as ConfusionCell;
    const miss = own.filter((cell) => cell.decision !== gold).sort((a, b) => b.count - a.count)[0] as ConfusionCell;
    return { gold, hit, miss };
  });
  if (rows.length < 2) return `${head}.`;
  const worst = [...rows].sort((a, b) => (a.hit.shareOfGold ?? 0) - (b.hit.shareOfGold ?? 0))[0] as (typeof rows)[number];
  const missPart = worst.miss.count > 0 ? `; it most often said ${quote(worst.miss.decision)} instead (${fmtPct(worst.miss.shareOfGold, 0)})` : "";
  return `${head}. Gold ${quote(worst.gold)} is matched least often: ${fmtPct(worst.hit.shareOfGold, 0)} of ${fmtInt(c.goldTotals[worst.gold])}${missPart}.`;
}

export interface ConfidenceSummaries {
  condition: string;
  correct: Summary | null;
  wrong: Summary | null;
}

/** Five-number summaries of the stated confidence of right and of wrong answers. */
export function verbalConfidenceSummaries(data: unknown, condition: string): ConfidenceSummaries {
  const rows = readRecords(data).filter((r) => r.condition === condition && !r.failed && r.correct !== null && r.verbalConf !== null);
  return {
    condition,
    correct: summarise(rows.filter((r) => r.correct).map((r) => r.verbalConf as number)),
    wrong: summarise(rows.filter((r) => !r.correct).map((r) => r.verbalConf as number)),
  };
}

export type HistogramScale = "count" | "share";

/**
 * The histogram as drawn: counts, or each group's bins as a share of that group, so a group of
 * 300 wrong answers can be compared in shape with 700 right ones.
 */
export function histogramValues(h: ConfidenceHistogram, scale: HistogramScale): { lo: number; hi: number; correct: number; wrong: number }[] {
  return h.bins.map((b) => ({
    lo: b.lo,
    hi: b.hi,
    correct: scale === "count" ? b.correct : h.nCorrect > 0 ? b.correct / h.nCorrect : 0,
    wrong: scale === "count" ? b.wrong : h.nWrong > 0 ? b.wrong / h.nWrong : 0,
  }));
}

/** "Right answers: median stated confidence 0.80 (IQR 0.72 to 0.85); wrong answers: 0.60 (…)." */
export function confidenceTakeaway(s: ConfidenceSummaries): string | null {
  const part = (label: string, x: Summary | null) =>
    x ? `${label} (${fmtInt(x.n)}) state a median confidence of ${x.median.toFixed(2)}, middle half ${x.q1.toFixed(2)} to ${x.q3.toFixed(2)}` : null;
  const parts = [part("Right answers", s.correct), part("wrong answers", s.wrong)].filter(Boolean);
  if (parts.length === 0) return null;
  let tail = "";
  if (s.correct && s.wrong) {
    const overlap = Math.min(s.correct.q3, s.wrong.q3) - Math.max(s.correct.q1, s.wrong.q1);
    tail = overlap >= 0 ? ", so the middle halves overlap" : ", so the middle halves do not overlap";
  }
  return `${parts.join("; ")}${tail}.`;
}
