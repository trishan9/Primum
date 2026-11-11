/**
 * RQ2 confidence (`results/<run>/rq2_confidence.json`).
 *
 * Layout: `estimators: {<calibrator>: {<estimator>: {auroc, aurc, e_aurc, ece, brier: {mean, ci}}}}`,
 * `ablation: {<feature set>: {…same metrics}}`, `comparisons: {vs_<reference>: {<model>: {brier_diff,
 * brier_p, auroc_diff, auroc_better_share, n_questions}}}`, `reliability: {<estimator>: [{lo, hi, count,
 * confidence, accuracy}]}`, `risk_coverage: {<estimator>: {coverage: [], risk: []}}`.
 *
 * Estimator names say what they are: `stated:` is the generator's own stated confidence, `signal:`
 * one feature used as a score, `logistic:` / `mlp:` a learned combination.
 */

import { fmtInt, fmtPct } from "../format.ts";
import { metricDirection } from "../results.ts";
import { at, hasInterval, intervalInfo, isRecord, num, readEstimate, type Estimate, type IntervalInfo } from "./common.ts";

export type EstimatorKind = "stated" | "signal" | "learned" | "other";

export function estimatorKind(name: string): EstimatorKind {
  if (name.startsWith("stated:")) return "stated";
  if (name.startsWith("signal:")) return "signal";
  if (/^(logistic|mlp|gbm|learned)\b/i.test(name)) return "learned";
  return "other";
}

/** "signal:verbal_conf" → "verbal_conf"; "logistic:deploy" stays as written. */
export function estimatorShortName(name: string): string {
  return name.replace(/^(stated|signal):/, "");
}

/**
 * `signal:X` is the same score as `stated:X` (the stated confidence used as a lone signal), so
 * when both are present the signal copy is dropped: drawn twice, it would let the reference look
 * beaten by itself and inflate "n others".
 */
export function dropDuplicateSignals<T extends { name: string }>(rows: readonly T[]): T[] {
  const stated = new Set(rows.filter((r) => estimatorKind(r.name) === "stated").map((r) => estimatorShortName(r.name)));
  return rows.filter((r) => !(estimatorKind(r.name) === "signal" && stated.has(estimatorShortName(r.name))));
}

/** RQ2 interval descriptions from `meta.intervals`, by block. */
export function rq2Interval(data: unknown, block: "estimators" | "ablation" | "auroc_diff" | "brier_diff" | "reliability" | "risk_coverage"): IntervalInfo {
  return intervalInfo(data, block);
}

export function calibrators(data: unknown): string[] {
  const block = at(data, "estimators");
  return isRecord(block) ? Object.keys(block).filter((k) => isRecord(block[k])) : [];
}

/** `meta.main_calibrator`, or the first calibrator in the file. */
export function mainCalibrator(data: unknown): string | null {
  const named = at(data, "meta", "main_calibrator");
  const all = calibrators(data);
  return typeof named === "string" && all.includes(named) ? named : (all[0] ?? null);
}

export interface RankedEstimator {
  name: string;
  kind: EstimatorKind;
  estimate: Estimate;
}

/**
 * Every estimator under one calibrator, best first by the metric's own direction (AUROC higher,
 * Brier/ECE/AURC lower). Ties keep file order.
 */
export function estimatorRanking(data: unknown, calibrator: string, metric = "auroc"): RankedEstimator[] {
  const block = at(data, "estimators", calibrator);
  if (!isRecord(block)) return [];
  const direction = metricDirection(metric) ?? "higher";
  const rows = Object.entries(block).flatMap(([name, entry]) => {
    const estimate = readEstimate(at(entry, metric));
    return estimate ? [{ name, kind: estimatorKind(name), estimate }] : [];
  });
  return dropDuplicateSignals(rows).sort((a, b) => (direction === "higher" ? b.estimate.value - a.estimate.value : a.estimate.value - b.estimate.value));
}

export interface RiskCoverageSeries {
  name: string;
  kind: EstimatorKind;
  points: { coverage: number; risk: number }[];
}

/** Mean risk-coverage curves, one per estimator, sorted by coverage. */
export function riskCoverageSeries(data: unknown): RiskCoverageSeries[] {
  const block = at(data, "risk_coverage");
  if (!isRecord(block)) return [];
  return dropDuplicateSignals(Object.entries(block).flatMap(([name, entry]) => {
    const coverage = at(entry, "coverage");
    const risk = at(entry, "risk");
    if (!Array.isArray(coverage) || !Array.isArray(risk)) return [];
    const points = coverage
      .map((c, i) => ({ coverage: num(c), risk: num(risk[i]) }))
      .filter((p): p is { coverage: number; risk: number } => p.coverage !== null && p.risk !== null)
      .sort((a, b) => a.coverage - b.coverage);
    return points.length > 0 ? [{ name, kind: estimatorKind(name), points }] : [];
  }));
}

/** The error rate of answering everything: risk at full coverage, from the first curve that reaches it. */
export function riskAtFullCoverage(series: readonly RiskCoverageSeries[]): number | null {
  for (const s of series) {
    const last = s.points[s.points.length - 1];
    if (last && Math.abs(last.coverage - 1) < 1e-9) return last.risk;
  }
  return null;
}

export interface ReliabilityBin {
  lo: number;
  hi: number;
  count: number;
  /** mean predicted probability in the bin */
  confidence: number;
  /** observed share correct in the bin */
  accuracy: number;
}

export interface ReliabilitySeries {
  name: string;
  kind: EstimatorKind;
  bins: ReliabilityBin[];
  /** predictions over all bins (pooled over repeats) */
  total: number;
}

/** Reliability bins per estimator; empty bins are left out. */
export function reliabilityBins(data: unknown): ReliabilitySeries[] {
  const block = at(data, "reliability");
  if (!isRecord(block)) return [];
  return dropDuplicateSignals(Object.entries(block).flatMap(([name, list]) => {
    if (!Array.isArray(list)) return [];
    const bins = list.flatMap((b) => {
      const lo = num(at(b, "lo"));
      const hi = num(at(b, "hi"));
      const count = num(at(b, "count"));
      const confidence = num(at(b, "confidence"));
      const accuracy = num(at(b, "accuracy"));
      return lo !== null && hi !== null && count !== null && count > 0 && confidence !== null && accuracy !== null ? [{ lo, hi, count, confidence, accuracy }] : [];
    });
    return bins.length > 0 ? [{ name, kind: estimatorKind(name), bins, total: bins.reduce((s, b) => s + b.count, 0) }] : [];
  }));
}

/** One estimator's metric under one calibrator, as the file reports it (mean over splits and its range). */
export function estimatorMetric(data: unknown, calibrator: string | null, name: string, metric: string): Estimate | null {
  return calibrator ? readEstimate(at(data, "estimators", calibrator, name, metric)) : null;
}

export interface AblationRow {
  name: string;
  estimate: Estimate;
  /**
   * This feature set's mean minus the full model's mean. A difference of means: the file has no
   * interval for the difference itself, so none is drawn for it.
   */
  deltaFromAll: number | null;
}

/** The feature-group ablation for one metric; the reference row is the one named "all features". */
export function ablation(data: unknown, metric = "auroc"): { reference: string | null; rows: AblationRow[] } {
  const block = at(data, "ablation");
  if (!isRecord(block)) return { reference: null, rows: [] };
  const reference = Object.keys(block).find((k) => /^all/i.test(k)) ?? null;
  const base = reference ? readEstimate(at(block, reference, metric)) : null;
  const rows = Object.entries(block).flatMap(([name, entry]) => {
    const estimate = readEstimate(at(entry, metric));
    return estimate ? [{ name, estimate, deltaFromAll: base && name !== reference ? estimate.value - base.value : null }] : [];
  });
  return { reference, rows };
}

/* ---------- Learned models against the stated confidence ---------- */

export interface ComparisonRow {
  model: string;
  /** model minus reference; negative is better */
  brier: Estimate | null;
  brierP: number | null;
  /** model minus reference; positive is better */
  auroc: Estimate | null;
  aurocBetterShare: number | null;
}

export interface LearnedVsStated {
  reference: string;
  rows: ComparisonRow[];
  /** learned models whose AUROC-difference interval lies entirely above zero */
  aurocBetter: string[];
  /** … entirely below zero */
  aurocWorse: string[];
  /** learned models whose Brier-difference interval lies entirely below zero (lower is better) */
  brierBetter: string[];
  brierWorse: string[];
  /** models with no interval for the AUROC difference, so no reading is made */
  aurocNoInterval: string[];
}

/**
 * The comparison RQ2 turns on, read from `comparisons.vs_stated:*`: does any learned model beat
 * the generator's own stated confidence? A model "beats" it only when the interval of the
 * difference excludes zero in its favour; overlapping intervals are reported as no difference.
 */
export function learnedVsStated(data: unknown): LearnedVsStated | null {
  const block = at(data, "comparisons");
  if (!isRecord(block)) return null;
  const key = Object.keys(block).find((k) => /^vs[_-]stated/i.test(k)) ?? null;
  if (!key || !isRecord(block[key])) return null;
  const rows: ComparisonRow[] = Object.entries(block[key] as Record<string, unknown>).flatMap(([model, entry]) =>
    isRecord(entry)
      ? [{ model, brier: readEstimate(entry.brier_diff), brierP: num(entry.brier_p), auroc: readEstimate(entry.auroc_diff), aurocBetterShare: num(entry.auroc_better_share) }]
      : [],
  );
  const names = (pred: (r: ComparisonRow) => boolean) => rows.filter(pred).map((r) => r.model);
  return {
    reference: key.replace(/^vs[_-]/, ""),
    rows,
    aurocBetter: names((r) => hasInterval(r.auroc) && r.auroc.lo > 0),
    aurocWorse: names((r) => hasInterval(r.auroc) && r.auroc.hi < 0),
    brierBetter: names((r) => hasInterval(r.brier) && r.brier.hi < 0),
    brierWorse: names((r) => hasInterval(r.brier) && r.brier.lo > 0),
    aurocNoInterval: names((r) => !hasInterval(r.auroc)),
  };
}

/** The stated confidence's own metric under the main calibrator (the KPI the page leads RQ2 with). */
export function statedConfidence(data: unknown, metric = "auroc"): { name: string; calibrator: string; estimate: Estimate } | null {
  const calibrator = mainCalibrator(data);
  if (!calibrator) return null;
  const block = at(data, "estimators", calibrator);
  if (!isRecord(block)) return null;
  const comparisonRef = learnedVsStated(data)?.reference;
  const name = (comparisonRef && comparisonRef in block ? comparisonRef : Object.keys(block).find((k) => estimatorKind(k) === "stated")) ?? null;
  if (!name) return null;
  const estimate = readEstimate(at(block, name, metric));
  return estimate ? { name, calibrator, estimate } : null;
}

/* ---------- Labels and takeaways for the RQ2 figures ---------- */

/** A display name: the stated confidence in words, a signal by its feature, a learned model as written. */
export function estimatorLabel(name: string): string {
  if (estimatorKind(name) === "stated") return `stated confidence (${estimatorShortName(name)})`;
  return estimatorShortName(name);
}

const fixed3 = (v: number) => v.toFixed(3);
/** A signed difference with a true minus sign: "+0.006", "−0.070". */
export function signed(v: number, digits = 3): string {
  const text = Math.abs(v).toFixed(digits);
  if (Number(text) === 0) return text;
  return `${v > 0 ? "+" : "\u2212"}${text}`;
}

const list = (names: readonly string[]) =>
  names.length <= 1 ? (names[0] ?? "") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;

/** The learned estimators a figure can set against the stated confidence, in file order. */
export function learnedNames(names: readonly string[]): string[] {
  return names.filter((n) => estimatorKind(n) === "learned");
}

/**
 * The AUROC ranking in one sentence: where the stated confidence sits, which estimator has the
 * highest mean, and whether any learned model's difference interval clears zero.
 */
export function rankingTakeaway(ranking: readonly RankedEstimator[], comparison: LearnedVsStated | null, diffIntervalLabel = "central 95% range across splits"): string | null {
  const stated = ranking.find((r) => r.kind === "stated");
  if (!stated) return null;
  const others = ranking.filter((r) => r !== stated);
  // Compared at the printed precision, so a tie in the third decimal is not called "higher".
  const higher = others.filter((r) => Number(fixed3(r.estimate.value)) > Number(fixed3(stated.estimate.value)));
  const range = hasInterval(stated.estimate) ? ` (${fixed3(stated.estimate.lo)} to ${fixed3(stated.estimate.hi)} across splits)` : "";
  const head = `Stated confidence has mean AUROC ${fixed3(stated.estimate.value)}${range}`;
  const rank =
    higher.length === 0
      ? `; no other estimator has a higher mean`
      : `; ${fmtInt(higher.length)} of ${fmtInt(others.length)} others ${higher.length === 1 ? "has" : "have"} a higher mean (${higher.map((r) => `${estimatorShortName(r.name)} ${fixed3(r.estimate.value)}`).join(", ")})`;
  if (!comparison || comparison.rows.length === 0) return `${head}${rank}.`;
  const verdict =
    comparison.aurocBetter.length > 0
      ? `${list(comparison.aurocBetter)} ${comparison.aurocBetter.length === 1 ? "has" : "have"} an AUROC difference whose ${diffIntervalLabel} lies entirely above zero.`
      : `no learned model's AUROC difference from it has its ${diffIntervalLabel} entirely above zero.`;
  return `${head}${rank}${higher.length === 0 ? ", and" : ", but"} ${verdict}`;
}

/** The Brier comparison in one sentence, read from where each difference interval sits against zero. */
export function brierTakeaway(comparison: LearnedVsStated | null): string | null {
  if (!comparison || comparison.rows.length === 0) return null;
  const withInterval = comparison.rows.filter((r) => hasInterval(r.brier));
  if (withInterval.length === 0) return null;
  const better =
    comparison.brierBetter.length > 0
      ? `${list(comparison.brierBetter)} ${comparison.brierBetter.length === 1 ? "has" : "have"} a lower Brier loss than stated confidence, with the interval entirely below zero.`
      : `No learned model has a Brier difference interval entirely below zero, so none is shown to give better probabilities than stated confidence.`;
  const worse =
    comparison.brierWorse.length > 0
      ? ` ${list(comparison.brierWorse)} ${comparison.brierWorse.length === 1 ? "is" : "are"} worse: ${comparison.brierWorse.length === 1 ? "its interval lies" : "their intervals lie"} entirely above zero.`
      : "";
  return `${better}${worse}`;
}

/** "without verbal" → "verbal". */
export function ablationGroup(name: string): string {
  return name.replace(/^without[\s_-]+/i, "");
}

/** The ablation in one sentence: which dropped group moves mean AUROC most, and how far the rest move it. */
export function ablationTakeaway(result: { reference: string | null; rows: readonly AblationRow[] }): string | null {
  const ref = result.rows.find((r) => r.name === result.reference);
  const dropped = result.rows.filter((r) => r.deltaFromAll !== null) as (AblationRow & { deltaFromAll: number })[];
  if (!ref || dropped.length === 0) return null;
  const sorted = [...dropped].sort((a, b) => Math.abs(b.deltaFromAll) - Math.abs(a.deltaFromAll));
  const top = sorted[0] as AblationRow & { deltaFromAll: number };
  const direction = top.deltaFromAll < 0 ? "lowers" : "raises";
  const head = `Dropping the ${ablationGroup(top.name)} group ${direction} mean AUROC the most, by ${fixed3(Math.abs(top.deltaFromAll))} (${fixed3(ref.estimate.value)} to ${fixed3(top.estimate.value)})`;
  const rest = sorted.slice(1);
  if (rest.length === 0) return `${head}.`;
  const largest = Math.max(...rest.map((r) => Math.abs(r.deltaFromAll)));
  return `${head}; dropping any other group moves it by ${fixed3(largest)} or less.`;
}

/** Selective risk at the smallest grid coverage at or above `coverage` (no interpolation). */
export function riskAtCoverage(series: RiskCoverageSeries, coverage: number): { coverage: number; risk: number } | null {
  return series.points.find((p) => p.coverage >= coverage - 1e-9) ?? null;
}

/** Two curves compared at one coverage, against the risk of answering everything. */
export function riskCoverageTakeaway(all: readonly RiskCoverageSeries[], reference: string, other: string | null, coverage = 0.5): string | null {
  const ref = all.find((s) => s.name === reference);
  if (!ref) return null;
  const a = riskAtCoverage(ref, coverage);
  if (!a) return null;
  const o = other ? all.find((s) => s.name === other) : undefined;
  const b = o ? riskAtCoverage(o, coverage) : null;
  const full = riskAtFullCoverage(all);
  const share = fmtPct(a.coverage);
  const pair = b && o ? `, ${estimatorLabel(o.name)} ${fmtPct(b.risk, 1)}` : "";
  const everything = full !== null ? `; answering every question, ${fmtPct(full, 1)} are wrong` : "";
  return `Releasing only the most confident ${share} of answers, ${fmtPct(a.risk, 1)} are wrong ranked by ${estimatorLabel(ref.name)}${pair}${everything}.`;
}

/** The mean gap between predicted and observed accuracy over bins, weighted by predictions (a pooled ECE). */
export function weightedGap(series: ReliabilitySeries): number | null {
  if (series.total <= 0) return null;
  return series.bins.reduce((sum, b) => sum + b.count * Math.abs(b.accuracy - b.confidence), 0) / series.total;
}

/** Reliability in one sentence: each named curve's weighted gap, and where most predictions fall. */
export function reliabilityTakeaway(
  all: readonly ReliabilitySeries[],
  names: readonly string[],
  /** the file's per-split ECE for an estimator (mean over splits, with its range), to set beside the pooled figure */
  fileEce: (name: string) => Estimate | null = () => null,
): string | null {
  const picked = names.flatMap((n) => all.filter((s) => s.name === n));
  if (picked.length === 0) return null;
  const gaps = picked.flatMap((s) => {
    const gap = weightedGap(s);
    return gap === null ? [] : [`${estimatorLabel(s.name)} ${(gap * 100).toFixed(1)} points`];
  });
  const first = picked[0] as ReliabilitySeries;
  const busiest = [...first.bins].sort((a, b) => b.count - a.count)[0];
  const busy = busiest ? ` The busiest bin for ${estimatorLabel(first.name)} is ${fmtPct(busiest.lo)} to ${fmtPct(busiest.hi)}, with ${fmtPct(busiest.count / first.total)} of predictions.` : "";
  const eces = picked.map((s) => fileEce(s.name));
  let file = "";
  if (eces.every((e): e is Estimate => e !== null)) {
    const withRange = eces.filter(hasInterval);
    const overlap =
      withRange.length === eces.length && eces.length > 1
        ? Math.min(...withRange.map((e) => e.hi)) >= Math.max(...withRange.map((e) => e.lo))
          ? ", ranges overlapping"
          : ", ranges not overlapping"
        : "";
    file = ` (the file's per-split ECE averages ${eces.map((e) => fixed3(e.value)).join(" and ")}${overlap})`;
  }
  return `Mean distance from the diagonal, weighted by predictions and pooled over all repeats: ${gaps.join("; ")}${file}.${busy}`;
}
