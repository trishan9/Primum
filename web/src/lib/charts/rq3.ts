/**
 * RQ3 certified abstention (`results/<run>/rq3_conformal.json`).
 *
 * Layout: `selective: {<method>: {<alpha>: LevelRecord}}`, `single_signal: {<signal>: {<alpha>: …}}`,
 * `min_selected_sensitivity: {<setting>: {<alpha>: …}}` where a level record is `{violation_rate,
 * violation_ci, coverage: {mean, ci}, risk, certified_rate, threshold, n_repeats}`;
 * `prediction_sets: {<method>: {set_coverage|mean_set_size|empty_set_rate: {<alpha>: {mean, ci}}}}`;
 * `calibration: {p_correct, correct, thresholds: {<alpha>: number|null}, heldout: {p_correct, correct}}`.
 */

import { fmtAlpha, fmtPct } from "../format.ts";
import { at, equalBins, intervalInfo, isRecord, num, readEstimate, type Bin, type Estimate, type IntervalInfo } from "./common.ts";

export type MethodGroup = "selective" | "single_signal" | "min_selected_sensitivity";

export interface LevelPoint {
  alpha: number;
  /** share of repeated splits whose test risk exceeded α, with its Clopper-Pearson CI */
  violation: Estimate | null;
  /** mean share of held-out questions answered, with the central 95% range across splits */
  coverage: Estimate | null;
  risk: Estimate | null;
  threshold: Estimate | null;
  certifiedRate: number | null;
  nRepeats: number | null;
  /** no split certified a threshold: everything was withheld, so a zero violation rate is trivial */
  trivial: boolean;
}

export interface MethodSeries {
  method: string;
  group: MethodGroup;
  points: LevelPoint[];
}

const parseAlpha = (key: string): number | null => {
  const value = Number.parseFloat(key);
  return Number.isFinite(value) && value > 0 && value < 1 && /^0?\.\d+$/.test(key.trim()) ? value : null;
};

function readLevel(alpha: number, entry: unknown): LevelPoint | null {
  if (!isRecord(entry)) return null;
  const rate = num(entry.violation_rate);
  const ci = entry.violation_ci;
  const violation: Estimate | null =
    rate === null ? null : Array.isArray(ci) && num(ci[0]) !== null && num(ci[1]) !== null ? { value: rate, lo: ci[0] as number, hi: ci[1] as number } : { value: rate, lo: null, hi: null };
  const coverage = readEstimate(entry.coverage);
  const certifiedRate = num(entry.certified_rate);
  return {
    alpha,
    violation,
    coverage,
    risk: readEstimate(entry.risk),
    threshold: readEstimate(entry.threshold),
    certifiedRate,
    nRepeats: num(entry.n_repeats),
    trivial: certifiedRate !== null ? certifiedRate === 0 : coverage !== null && coverage.value === 0,
  };
}

function readMethods(data: unknown, group: MethodGroup): MethodSeries[] {
  const block = at(data, group);
  if (!isRecord(block)) return [];
  return Object.entries(block).flatMap(([method, levels]) => {
    if (!isRecord(levels)) return [];
    const points = Object.entries(levels)
      .flatMap(([key, entry]) => {
        const alpha = parseAlpha(key);
        const point = alpha === null ? null : readLevel(alpha, entry);
        return point ? [point] : [];
      })
      .sort((a, b) => a.alpha - b.alpha);
    return points.length > 0 ? [{ method, group, points }] : [];
  });
}

/** Every method's records across α: the selective methods first, then single signals and sensitivity settings. */
export function guaranteeSeries(data: unknown, groups: readonly MethodGroup[] = ["selective", "single_signal", "min_selected_sensitivity"]): MethodSeries[] {
  return groups.flatMap((g) => readMethods(data, g));
}

/** The α levels any method reports, ascending. */
export function alphaLevels(series: readonly MethodSeries[]): number[] {
  return [...new Set(series.flatMap((s) => s.points.map((p) => p.alpha)))].sort((a, b) => a - b);
}

/** δ, the tolerated violation rate: `meta.delta`, else `calibration.delta`. */
export function deltaOf(data: unknown): number | null {
  return num(at(data, "meta", "delta")) ?? num(at(data, "calibration", "delta"));
}

export function rq3Interval(data: unknown, which: "violation_ci" | "coverage" | "set metrics"): IntervalInfo {
  return intervalInfo(data, which);
}

/** Readings of one method against δ, for takeaways: worst non-trivial violation and the coverage range. */
export function methodReading(series: MethodSeries, delta: number | null) {
  const live = series.points.filter((p) => !p.trivial);
  const violations = live.flatMap((p) => (p.violation ? [p.violation.value] : []));
  const coverage = live.flatMap((p) => (p.coverage ? [p.coverage.value] : []));
  const worst = violations.length > 0 ? Math.max(...violations) : null;
  return {
    trivialAlphas: series.points.filter((p) => p.trivial).map((p) => p.alpha),
    worstViolation: worst,
    withinDelta: worst !== null && delta !== null ? worst <= delta : null,
    coverageMin: coverage.length > 0 ? Math.min(...coverage) : null,
    coverageMax: coverage.length > 0 ? Math.max(...coverage) : null,
  };
}

/* ---------- Prediction sets ---------- */

export interface SetPoint {
  alpha: number;
  setCoverage: Estimate | null;
  meanSetSize: Estimate | null;
  emptySetRate: Estimate | null;
}

export function predictionSets(data: unknown): { method: string; points: SetPoint[] }[] {
  const block = at(data, "prediction_sets");
  if (!isRecord(block)) return [];
  return Object.entries(block).flatMap(([method, metrics]) => {
    if (!isRecord(metrics)) return [];
    const alphas = new Set<number>();
    for (const levels of Object.values(metrics)) if (isRecord(levels)) for (const k of Object.keys(levels)) if (parseAlpha(k) !== null) alphas.add(parseAlpha(k) as number);
    const keyOf = (levels: unknown, alpha: number) => (isRecord(levels) ? Object.keys(levels).find((k) => parseAlpha(k) === alpha) : undefined);
    const read = (metric: string, alpha: number) => {
      const levels = metrics[metric];
      const key = keyOf(levels, alpha);
      return key ? readEstimate(at(levels, key)) : null;
    };
    const points = [...alphas]
      .sort((a, b) => a - b)
      .map((alpha) => ({ alpha, setCoverage: read("set_coverage", alpha), meanSetSize: read("mean_set_size", alpha), emptySetRate: read("empty_set_rate", alpha) }));
    return points.length > 0 ? [{ method, points }] : [];
  });
}

/* ---------- The deployed model's scores ---------- */

export interface ScoreHistogram {
  set: "calibration" | "heldout";
  bins: (Bin & { correct: number; wrong: number })[];
  n: number;
}

export interface DeployedScores {
  histograms: ScoreHistogram[];
  /** the certified threshold per α; null where none certified */
  thresholds: { alpha: number; threshold: number | null }[];
}

function histogramOf(set: ScoreHistogram["set"], pCorrect: unknown, correct: unknown, binCount: number): ScoreHistogram | null {
  if (!Array.isArray(pCorrect) || !Array.isArray(correct)) return null;
  const right: number[] = [];
  const wrong: number[] = [];
  pCorrect.forEach((p, i) => {
    const value = num(p);
    const c = correct[i];
    if (value === null || (c !== 0 && c !== 1 && c !== true && c !== false)) return;
    (c === 1 || c === true ? right : wrong).push(value);
  });
  if (right.length + wrong.length === 0) return null;
  const r = equalBins(right, [0, 1], binCount);
  const w = equalBins(wrong, [0, 1], binCount);
  return { set, n: right.length + wrong.length, bins: r.map((b, i) => ({ lo: b.lo, hi: b.hi, count: b.count + (w[i]?.count ?? 0), correct: b.count, wrong: w[i]?.count ?? 0 })) };
}

/**
 * The deployed model's calibrated P(correct) for right and wrong answers: on the calibration rows
 * the thresholds were certified on (in-sample for the threshold) and on held-out rows.
 */
export function deployedScores(data: unknown, binCount = 20): DeployedScores | null {
  const cal = at(data, "calibration");
  if (!isRecord(cal)) return null;
  const histograms = [histogramOf("calibration", cal.p_correct, cal.correct, binCount), histogramOf("heldout", at(cal, "heldout", "p_correct"), at(cal, "heldout", "correct"), binCount)].filter(
    (h): h is ScoreHistogram => h !== null,
  );
  const t = cal.thresholds;
  const thresholds = isRecord(t)
    ? Object.entries(t)
        .flatMap(([key, value]) => {
          const alpha = parseAlpha(key);
          return alpha === null ? [] : [{ alpha, threshold: num(value) }];
        })
        .sort((a, b) => a.alpha - b.alpha)
    : [];
  return histograms.length > 0 ? { histograms, thresholds } : null;
}

/* ---------- Runs side by side (descriptive) ---------- */

export interface RunSeries {
  run: string;
  label: string;
  points: LevelPoint[];
}

/**
 * One method across several runs (generators), for a side-by-side picture. Descriptive only: each
 * run's splits are drawn from its own questions and no test pairs them.
 */
export function compareRuns(runs: readonly { run: string; label: string; data: unknown }[], method = "certified"): RunSeries[] {
  return runs.flatMap(({ run, label, data }) => {
    const series = readMethods(data, "selective").find((s) => s.method === method);
    return series ? [{ run, label, points: series.points }] : [];
  });
}

/* ---------- Labels and takeaways for the RQ3 figures ---------- */

const METHOD_LABEL: Record<string, string> = {
  certified: "Certified (binomial test)",
  certified_hb: "Certified (Hoeffding-Bentkus)",
  plug_in: "Plug-in (no correction)",
  lac: "LAC",
  aps: "APS",
};

/** A method's name in words, from the file's key; unknown keys are shown as written. */
export function methodLabel(method: string): string {
  return METHOD_LABEL[method] ?? method;
}

/**
 * The colour domain for the selective methods: the system's procedure first and the naive rule
 * second, so they keep the first two hues whatever else the file holds; then the rest in file order.
 */
export function selectiveDomain(series: readonly MethodSeries[]): string[] {
  const names = series.filter((s) => s.group === "selective").map((s) => s.method);
  const lead = ["certified", "plug_in"].filter((n) => names.includes(n));
  return [...lead, ...names.filter((n) => !lead.includes(n))];
}

/**
 * Violations among the splits that certified a threshold, as counts. `violation_rate` is over all
 * splits (a split that certifies nothing withholds everything and cannot violate), which is what
 * the guarantee bounds; this conditional view shows how few splits a level's rate rests on.
 */
export function certifiedSplitViolations(point: LevelPoint): { violations: number; certified: number; rate: number | null } | null {
  if (point.nRepeats === null || point.certifiedRate === null || !point.violation) return null;
  const certified = Math.round(point.certifiedRate * point.nRepeats);
  const violations = Math.round(point.violation.value * point.nRepeats);
  return { violations, certified, rate: certified > 0 ? violations / certified : null };
}

const alphaList = (alphas: readonly number[]) => alphas.map(fmtAlpha).join(", ");

/** Violation rates in one sentence: the certified method's worst non-trivial level against δ, and the plug-in rule's range. */
export function violationTakeaway(series: readonly MethodSeries[], delta: number | null): string | null {
  const certified = series.find((s) => s.method === "certified");
  if (!certified) return null;
  const r = methodReading(certified, delta);
  const parts: string[] = [];
  if (r.worstViolation !== null) {
    const against = delta !== null ? `, ${r.withinDelta ? "within" : "above"} δ = ${fmtAlpha(delta)}` : "";
    const n = certified.points.find((p) => p.nRepeats !== null)?.nRepeats ?? null;
    const scope = n !== null ? `Across all ${n} splits` : "Across all splits";
    parts.push(
      `${scope} (a split that certifies nothing withholds everything and counts as no violation), the certified rule's test risk exceeded α in at most ${fmtPct(r.worstViolation, 1)} of splits${against}`,
    );
  }
  const plug = series.find((s) => s.method === "plug_in");
  const plugRates = plug ? plug.points.flatMap((p) => (p.violation ? [p.violation.value] : [])) : [];
  if (plugRates.length > 0) {
    const lo = Math.min(...plugRates);
    const hi = Math.max(...plugRates);
    parts.push(`the plug-in rule did so in ${lo === hi ? fmtPct(lo, 1) : `${fmtPct(lo, 1)} to ${fmtPct(hi, 1)}`}`);
  }
  if (parts.length === 0) return null;
  const trivial = r.trivialAlphas.length > 0 ? ` At α = ${alphaList(r.trivialAlphas)} nothing was certified, so its zero rate is trivial.` : "";
  return `${parts.join("; ")}.${trivial}`;
}

/** Coverage in one sentence: the certified rule's coverage and certification rate at each end of α, then the plug-in rule's. */
export function coverageTakeaway(series: readonly MethodSeries[]): string | null {
  const certified = series.find((s) => s.method === "certified");
  const live = certified ? certified.points.filter((p) => p.coverage) : [];
  if (live.length === 0) return null;
  const last = live[live.length - 1] as LevelPoint & { coverage: Estimate };
  const certifiedIn = last.certifiedRate !== null ? `, where a threshold was certified in ${fmtPct(last.certifiedRate, 1)} of splits` : "";
  const head = `At α = ${fmtAlpha(last.alpha)} the certified rule answers ${fmtPct(last.coverage.value, 1)} of held-out questions on average${certifiedIn}`;
  const plug = series.find((s) => s.method === "plug_in")?.points.find((p) => p.alpha === last.alpha && p.coverage);
  const tail = plug?.coverage ? `; the plug-in rule answers ${fmtPct(plug.coverage.value, 1)}, without the guarantee` : "";
  return `${head}${tail}. Lower coverage is the price of the certificate.`;
}

/** Runs side by side in one sentence, at the largest α every run reports. Descriptive only. */
export function compareTakeaway(runs: readonly RunSeries[], delta: number | null): string | null {
  if (runs.length < 2) return null;
  const common = runs
    .map((r) => new Set(r.points.filter((p) => p.coverage).map((p) => p.alpha)))
    .reduce((a, b) => new Set([...a].filter((x) => b.has(x))));
  const alpha = Math.max(...common);
  if (!Number.isFinite(alpha)) return null;
  const coverage = runs.map((r) => {
    const p = r.points.find((q) => q.alpha === alpha);
    return `${fmtPct(p?.coverage?.value, 1)} (${r.label})`;
  });
  const within =
    delta === null
      ? ""
      : runs.every((r) => methodReading({ method: "certified", group: "selective", points: r.points }, delta).withinDelta !== false)
        ? ` Every run keeps its violation rate over all splits, within δ = ${fmtAlpha(delta)}.`
        : ` Not every run keeps its violation rate, over all splits, within δ = ${fmtAlpha(delta)}.`;
  return `At α = ${fmtAlpha(alpha)} the certified rule answers ${coverage.join(" and ")} of held-out questions on average.${within} Each run is split on its own questions, so this is a description, not a test.`;
}

/** Prediction sets in one sentence per method: the largest shortfall below the 1 − α target and the range of set sizes. */
export function predictionSetTakeaway(sets: readonly { method: string; points: SetPoint[] }[]): string | null {
  const parts = sets.flatMap(({ method, points }) => {
    const covered = points.filter((p) => p.setCoverage);
    if (covered.length === 0) return [];
    const gaps = covered.map((p) => ({ alpha: p.alpha, gap: 1 - p.alpha - (p.setCoverage as Estimate).value }));
    const worst = gaps.reduce((a, b) => (b.gap > a.gap ? b : a));
    const coverageText =
      worst.gap > 0.0005
        ? `falls short of 1 − α by at most ${(worst.gap * 100).toFixed(1)} points (α = ${fmtAlpha(worst.alpha)})`
        : `meets or exceeds 1 − α at every level`;
    const sizes = points.flatMap((p) => (p.meanSetSize ? [p.meanSetSize.value] : []));
    const sizeText = sizes.length > 0 ? `, with ${Math.min(...sizes).toFixed(2)} to ${Math.max(...sizes).toFixed(2)} labels per set` : "";
    return [`${methodLabel(method)} ${coverageText}${sizeText}`];
  });
  return parts.length > 0 ? `${parts.join("; ")}.` : null;
}
