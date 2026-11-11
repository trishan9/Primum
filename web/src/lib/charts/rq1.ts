/**
 * RQ1 retrieval (`results/rq1_retrieval.json`) and encoder fine-tuning
 * (`results/encoder_training.json`).
 *
 * Layout: `{<dataset>: {<system>: {<metric>: {mean, ci}}}, paired_tests: {<dataset>: {vs_<baseline>:
 * {<system>: {<metric>_diff, <metric>_p, <metric>_p_holm}}}}, per_query: {<dataset>: {qids, <system>:
 * {<metric>: number[]}}}}`.
 */

import { at, intervalInfo, isRecord, num, readEstimate, summarise, type Estimate, type IntervalInfo, type Summary } from "./common.ts";

const NOT_DATASETS = new Set(["meta", "paired_tests", "per_query"]);

/** RQ1 intervals: `meta.inference.ci` ("95% percentile bootstrap over queries"). */
export function rq1Interval(data: unknown): IntervalInfo {
  return intervalInfo(data, "ci");
}

/** Datasets with a systems-by-metrics block, in file order (nfcorpus, pubmedqa). */
export function retrievalDatasets(data: unknown): string[] {
  if (!isRecord(data)) return [];
  return Object.keys(data).filter((key) => !NOT_DATASETS.has(key) && isRecord(data[key]) && Object.values(data[key]).some(isRecord));
}

/** Systems for a dataset, in `meta.systems` order where the file gives one. */
export function retrievalSystems(data: unknown, dataset: string): string[] {
  const block = at(data, dataset);
  if (!isRecord(block)) return [];
  const present = Object.keys(block).filter((k) => isRecord(block[k]));
  const order = at(data, "meta", "systems");
  if (!Array.isArray(order)) return present;
  const ordered = order.filter((s): s is string => typeof s === "string" && present.includes(s));
  return [...ordered, ...present.filter((s) => !ordered.includes(s))];
}

/** Metrics reported for a dataset, in the order of its first system. */
export function retrievalMetrics(data: unknown, dataset: string): string[] {
  const systems = retrievalSystems(data, dataset);
  const first = systems[0] ? at(data, dataset, systems[0]) : null;
  return isRecord(first) ? Object.keys(first).filter((m) => readEstimate(first[m]) !== null) : [];
}

export interface SystemEstimate {
  system: string;
  estimate: Estimate;
}

/** One metric for every system on a dataset; systems without the metric are left out. */
export function metricBySystem(data: unknown, dataset: string, metric: string): SystemEstimate[] {
  return retrievalSystems(data, dataset).flatMap((system) => {
    const estimate = readEstimate(at(data, dataset, system, metric));
    return estimate ? [{ system, estimate }] : [];
  });
}

export interface RecallCurve {
  system: string;
  points: { k: number; estimate: Estimate }[];
}

/** `recall@k` for every k the dataset reports, per system, sorted by k. */
export function recallAtK(data: unknown, dataset: string): RecallCurve[] {
  const ks = retrievalMetrics(data, dataset)
    .map((m) => ({ metric: m, k: Number(m.match(/^recall@(\d+)$/i)?.[1]) }))
    .filter((m) => Number.isFinite(m.k))
    .sort((a, b) => a.k - b.k);
  if (ks.length === 0) return [];
  return retrievalSystems(data, dataset).flatMap((system) => {
    const points = ks.flatMap(({ metric, k }) => {
      const estimate = readEstimate(at(data, dataset, system, metric));
      return estimate ? [{ k, estimate }] : [];
    });
    return points.length > 0 ? [{ system, points }] : [];
  });
}

export interface PerQuery {
  system: string;
  values: number[];
  summary: Summary;
  /** share of queries scoring exactly zero (no relevant passage in the cut-off) */
  zeroShare: number;
}

/** Per-query values of one metric, per system, with a five-number summary. */
export function perQueryDistribution(data: unknown, dataset: string, metric: string): PerQuery[] {
  const block = at(data, "per_query", dataset);
  if (!isRecord(block)) return [];
  const order = retrievalSystems(data, dataset);
  const systems = [...order, ...Object.keys(block).filter((k) => k !== "qids" && !order.includes(k))];
  return systems.flatMap((system) => {
    const raw = at(block, system, metric);
    if (!Array.isArray(raw)) return [];
    const values = raw.map(num).filter((v): v is number => v !== null);
    const summary = summarise(values);
    if (!summary) return [];
    return [{ system, values, summary, zeroShare: values.filter((v) => v === 0).length / values.length }];
  });
}

export interface PairedDiff {
  system: string;
  metric: string;
  diff: Estimate;
  p: number | null;
  /** Holm-adjusted p, when the file has it */
  pHolm: number | null;
}

/** Baselines a dataset is tested against (`vs_bm25` → "bm25"). */
export function pairedBaselines(data: unknown, dataset: string): string[] {
  const block = at(data, "paired_tests", dataset);
  return isRecord(block) ? Object.keys(block).map((k) => k.replace(/^vs[_-]/, "")) : [];
}

/** System minus baseline, per metric, for every system the file tests against that baseline. */
export function pairedDiffs(data: unknown, dataset: string, baseline: string): PairedDiff[] {
  const block = at(data, "paired_tests", dataset, `vs_${baseline}`);
  if (!isRecord(block)) return [];
  return Object.entries(block).flatMap(([system, entry]) => {
    if (!isRecord(entry)) return [];
    return Object.keys(entry)
      .filter((k) => k.endsWith("_diff"))
      .flatMap((key) => {
        const metric = key.slice(0, -"_diff".length);
        const diff = readEstimate(entry[key]);
        return diff ? [{ system, metric, diff, p: num(entry[`${metric}_p`]), pHolm: num(entry[`${metric}_p_holm`]) }] : [];
      });
  });
}

/* ---------- Encoder fine-tuning ---------- */

export interface LossSeries {
  points: { step: number; loss: number }[];
  /** `loss_at_chance`: the loss of an encoder that ranks at random, as the file records it */
  chance: number | null;
  steps: number | null;
  wallTimeS: number | null;
}

/** The logged training loss, unsmoothed, in step order. */
export function lossSeries(training: unknown): LossSeries | null {
  const curve = at(training, "loss_curve");
  if (!Array.isArray(curve)) return null;
  const points = curve
    .flatMap((p) => {
      const step = num(at(p, "step"));
      const loss = num(at(p, "loss"));
      return step !== null && loss !== null ? [{ step, loss }] : [];
    })
    .sort((a, b) => a.step - b.step);
  if (points.length === 0) return null;
  return { points, chance: num(at(training, "loss_at_chance")), steps: num(at(training, "steps")), wallTimeS: num(at(training, "wall_time_s")) };
}

export interface BeforeAfter {
  metric: string;
  before: number;
  after: number;
}

/** Dev-set retrieval metrics before and after fine-tuning (`dev.before` / `dev.after`). */
export function devBeforeAfter(training: unknown): { rows: BeforeAfter[]; nQueries: number | null } {
  const before = at(training, "dev", "before");
  const after = at(training, "dev", "after");
  if (!isRecord(before) || !isRecord(after)) return { rows: [], nQueries: null };
  const rows = Object.keys(before)
    .filter((k) => !k.startsWith("n_") && k !== "pool_size")
    .flatMap((metric) => {
      const b = num(before[metric]);
      const a = num(after[metric]);
      return b !== null && a !== null ? [{ metric, before: b, after: a }] : [];
    });
  return { rows, nQueries: num(before.n_queries) };
}

/* ---------- Chart helpers and readings ---------- */

const pct1 = (v: number) => `${(v * 100).toFixed(1)}%`;
const signed = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
const ciText = (e: Estimate) => (e.lo !== null && e.hi !== null ? ` (CI ${pct1(e.lo)} to ${pct1(e.hi)})` : "");

/**
 * The default retriever the run was configured with (`meta.config.default_retriever`), when the
 * file records it and it is one of the systems.
 */
export function deployedRetriever(data: unknown): string | null {
  const name = at(data, "meta", "config", "default_retriever");
  return typeof name === "string" ? name : null;
}

/**
 * An axis domain covering every value and interval end, widened out to multiples of `step` and
 * kept inside `clamp`. For dot plots, whose marks are positions, not lengths, so they need not start at zero.
 */
export function paddedDomain(values: readonly number[], step: number, clamp: [number, number] = [-Infinity, Infinity]): [number, number] {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return [Math.max(clamp[0], 0), Math.min(clamp[1], 1)];
  const lo = Math.floor(Math.min(...finite) / step + 1e-9) * step;
  let hi = Math.ceil(Math.max(...finite) / step - 1e-9) * step;
  if (hi <= lo) hi = lo + step;
  return [Number(Math.max(clamp[0], lo).toPrecision(12)), Number(Math.min(clamp[1], hi).toPrecision(12))];
}

/** Every value and interval end of a list of estimates, for `paddedDomain`. */
export function estimateExtent(estimates: readonly Estimate[]): number[] {
  return estimates.flatMap((e) => [e.value, e.lo ?? e.value, e.hi ?? e.value]);
}

/**
 * "At k = 1, hybrid-ft+rerank finds the gold abstract for 85.9% (CI …) against 77.2% for bm25;
 * at k = 20 every retriever is between 89.9% and 97.8%."
 */
export function recallTakeaway(curves: readonly RecallCurve[], baseline = "bm25"): string | null {
  const ks = [...new Set(curves.flatMap((c) => c.points.map((p) => p.k)))].sort((a, b) => a - b);
  if (ks.length === 0) return null;
  const first = ks[0] as number;
  const last = ks[ks.length - 1] as number;
  const atK = (k: number) => curves.flatMap((c) => c.points.filter((p) => p.k === k).map((p) => ({ system: c.system, e: p.estimate })));
  const lead = [...atK(first)].sort((a, b) => b.e.value - a.e.value)[0];
  if (!lead) return null;
  const base = atK(first).find((r) => r.system === baseline);
  const verb = first === 1 ? "ranks the gold abstract first" : `places the gold abstract in its top ${first}`;
  let s = `At k = ${first}, ${lead.system} ${verb} for ${pct1(lead.e.value)} of questions${ciText(lead.e)}`;
  if (base && base.system !== lead.system) s += `, against ${pct1(base.e.value)} for ${baseline}`;
  if (last !== first) {
    const values = atK(last).map((r) => r.e.value);
    s += `; by k = ${last} every retriever is between ${pct1(Math.min(...values))} and ${pct1(Math.max(...values))}`;
  }
  return `${s}.`;
}

/** Interval position of a paired difference against zero. */
export function diffSide(d: Estimate): "above" | "below" | "spans" | "unknown" {
  if (d.lo === null || d.hi === null) return "unknown";
  if (d.lo > 0) return "above";
  if (d.hi < 0) return "below";
  return "spans";
}

/** "Against bm25 on nDCG@10, 3 of 7 retrievers have an interval wholly above zero, 1 below, 3 span it." */
export function pairedTakeaway(diffs: readonly PairedDiff[], baseline: string, metricName: string): string | null {
  if (diffs.length === 0) return null;
  const count = (side: ReturnType<typeof diffSide>) => diffs.filter((d) => diffSide(d.diff) === side).length;
  const above = count("above");
  const below = count("below");
  const spans = count("spans");
  const parts = [`${above} wholly above zero`, `${below} wholly below`, `${spans} spanning zero`];
  const best = [...diffs].sort((a, b) => b.diff.value - a.diff.value)[0] as PairedDiff;
  const holm = best.pHolm !== null ? `, Holm-adjusted p ${best.pHolm < 0.001 ? "< 0.001" : `= ${best.pHolm.toFixed(3)}`}` : "";
  // The whiskers are unadjusted CIs; the tests are Holm-adjusted, so say both.
  const withHolm = diffs.filter((d) => d.pHolm !== null);
  const adjusted =
    withHolm.length > 0
      ? ` After Holm adjustment, ${withHolm.filter((d) => (d.pHolm as number) < 0.05).length} of ${withHolm.length} differ from ${baseline} at p < 0.05.`
      : "";
  return `Against ${baseline} on ${metricName}, of ${diffs.length} retrievers the unadjusted 95% CI of the difference lies ${parts.join(", ")}.${adjusted} Largest gain: ${best.system}, ${signed(best.diff.value)}${holm}.`;
}

/**
 * True when every per-query value is 0 or 1 (recall@k with one gold abstract per question). A
 * strip and box of such values collapses to two lines and says nothing the mean does not; the
 * per-query figure offers only metrics that are not binary.
 */
export function isBinaryMetric(values: readonly number[]): boolean {
  return values.length > 0 && values.every((v) => v === 0 || v === 1);
}

/**
 * A deterministic spread in [-1, 1] for the i-th point of a strip plot (a golden-ratio sequence),
 * so the same file always draws the same picture. Position only; it carries no data.
 */
export function stripJitter(i: number): number {
  const x = (i * 0.6180339887498949) % 1;
  return x * 2 - 1;
}

/** "Median nDCG@10 per query runs from 0.17 (dense-biomed) to 0.31 (hybrid-ft+rerank); 18% to 31% of queries score zero." */
export function perQueryTakeaway(rows: readonly PerQuery[], metricName: string): string | null {
  if (rows.length === 0) return null;
  const byMedian = [...rows].sort((a, b) => a.summary.median - b.summary.median);
  const low = byMedian[0] as PerQuery;
  const high = byMedian[byMedian.length - 1] as PerQuery;
  const zeros = rows.map((r) => r.zeroShare);
  const n = rows[0]?.summary.n ?? 0;
  const median =
    low.summary.median === high.summary.median
      ? `Median ${metricName} per query is ${low.summary.median.toFixed(2)} for ${rows.length === 1 ? low.system : "every retriever"}`
      : `Median ${metricName} per query runs from ${low.summary.median.toFixed(2)} (${low.system}) to ${high.summary.median.toFixed(2)} (${high.system})`;
  const zeroText =
    Math.min(...zeros) === Math.max(...zeros) ? `${pct1(zeros[0] as number)}` : `between ${pct1(Math.min(...zeros))} and ${pct1(Math.max(...zeros))}`;
  return `${median}, over ${n} queries; ${zeroText} of queries score exactly zero, depending on the retriever.`;
}

export interface LossSummary {
  /** mean loss over the first and the last `share` of logged points */
  start: number;
  end: number;
  /** logged points in each window */
  window: number;
}

/** Mean of the first and last `share` of the logged losses: a fair start and end for a noisy, unsmoothed curve. */
export function lossSummary(series: LossSeries, share = 0.1): LossSummary | null {
  const n = series.points.length;
  if (n === 0) return null;
  const window = Math.max(1, Math.round(n * share));
  const avg = (pts: { loss: number }[]) => pts.reduce((s, p) => s + p.loss, 0) / pts.length;
  return { start: avg(series.points.slice(0, window)), end: avg(series.points.slice(n - window)), window };
}

/** "Mean loss fell from 1.12 over the first 22 logged steps to 0.45 over the last 22 (chance 3.81); dev MRR@10 0.971 → 0.982." */
export function lossTakeaway(series: LossSeries, dev: { rows: BeforeAfter[]; nQueries: number | null }): string | null {
  const s = lossSummary(series);
  if (!s) return null;
  const verb = s.end < s.start ? "fell" : s.end > s.start ? "rose" : "stayed";
  let text = `Mean loss ${verb} from ${s.start.toFixed(2)} over the first ${s.window} logged points to ${s.end.toFixed(2)} over the last ${s.window}`;
  if (series.chance !== null) text += ` (an encoder ranking at random would sit near ${series.chance.toFixed(2)})`;
  const mrr = dev.rows.find((r) => /^mrr/i.test(r.metric));
  if (mrr) text += `. On ${dev.nQueries !== null ? `${dev.nQueries} ` : ""}held-out dev queries, ${mrr.metric.replace(/^mrr/i, "MRR")} went from ${mrr.before.toFixed(3)} to ${mrr.after.toFixed(3)}`;
  return `${text}.`;
}
