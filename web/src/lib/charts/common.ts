/**
 * Shared pieces for the chart derivations in this folder. Every module here is a pure function
 * of a parsed results file (or an API response): no React, no DOM, no typed-in numbers. A value
 * the input does not contain comes back as null, and the chart leaves it out or prints a dash.
 */

import { max, mean, min, quantileSorted } from "d3-array";

import { isRecord, readStat } from "../results.ts";

export { isRecord };

/** A point estimate and, when the file has one, its interval. */
export interface Estimate {
  value: number;
  lo: number | null;
  hi: number | null;
}

export const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);

/** `{mean, ci: [lo, hi]}`, a bare number, or null. */
export function readEstimate(value: unknown): Estimate | null {
  return readStat(value);
}

/** Walks `obj[k0][k1]…`, returning undefined as soon as a step is not an object. */
export function at(obj: unknown, ...path: string[]): unknown {
  let node: unknown = obj;
  for (const key of path) {
    if (!isRecord(node)) return undefined;
    node = node[key];
  }
  return node;
}

export function hasInterval(e: Estimate | null | undefined): e is Estimate & { lo: number; hi: number } {
  return !!e && e.lo !== null && e.hi !== null;
}

/* ---------- What an interval is ---------- */

/**
 * The kinds of interval the results files carry. They mean different things, so a chart must
 * name the kind it draws: a bootstrap CI is uncertainty about an estimate; a split range is how
 * much one random split varies (not uncertainty of the mean); Clopper-Pearson is the exact
 * binomial CI on a rate counted over repeats.
 */
export type IntervalKind = "bootstrap" | "split-range" | "clopper-pearson" | "pooled" | "unknown";

/** Short labels for legends and footers. */
export const INTERVAL_LABEL: Record<IntervalKind, string> = {
  bootstrap: "95% bootstrap CI",
  "split-range": "central 95% range across splits",
  "clopper-pearson": "95% Clopper-Pearson CI",
  pooled: "pooled over repeats, no interval",
  unknown: "95% interval (method not recorded)",
};

/** Reads the kind from the description a results file writes in `meta.intervals` / `meta.inference`. */
export function intervalKindOf(text: string | null | undefined): IntervalKind {
  if (!text) return "unknown";
  if (/clopper/i.test(text)) return "clopper-pearson";
  if (/bootstrap/i.test(text)) return "bootstrap";
  if (/range of the per-repeat|split variab|across splits|over repeats and central/i.test(text)) return "split-range";
  if (/pooled/i.test(text)) return "pooled";
  return "unknown";
}

/**
 * `meta.intervals` with compound keys split: `{"estimators, ablation, auroc_diff": "…"}` becomes
 * one entry per name. `meta.inference.ci` (RQ1) is returned under the key "ci".
 */
export function intervalNotes(data: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  const intervals = at(data, "meta", "intervals");
  if (isRecord(intervals)) {
    for (const [key, text] of Object.entries(intervals)) {
      if (typeof text !== "string") continue;
      for (const part of key.split(",")) out[part.trim()] = text;
    }
  }
  const ci = at(data, "meta", "inference", "ci");
  if (typeof ci === "string") out.ci = ci;
  return out;
}

export interface IntervalInfo {
  kind: IntervalKind;
  /** The file's own words, when it has them. */
  text: string | null;
  label: string;
}

export function intervalInfo(data: unknown, key: string): IntervalInfo {
  const text = intervalNotes(data)[key] ?? null;
  const kind = intervalKindOf(text);
  return { kind, text, label: INTERVAL_LABEL[kind] };
}

/* ---------- Distributions ---------- */

export interface Bin {
  lo: number;
  hi: number;
  count: number;
}

/**
 * Equal-width bins over a fixed domain. A value equal to the upper edge falls in the last bin
 * (d3's bins are half-open), so a confidence of exactly 1 is counted, not dropped.
 */
export function equalBins(values: readonly number[], domain: [number, number], count: number): Bin[] {
  const [d0, d1] = domain;
  const edge = (i: number) => Number((d0 + ((d1 - d0) * i) / count).toPrecision(12));
  const bins: Bin[] = Array.from({ length: count }, (_, i) => ({ lo: edge(i), hi: edge(i + 1), count: 0 }));
  for (const v of values) {
    if (!Number.isFinite(v) || v < d0 || v > d1) continue;
    (bins[equalBinIndex(v, domain, count)] as Bin).count += 1;
  }
  return bins;
}

/** The bin `equalBins` puts `v` in (the caller checks `v` is inside the domain). */
export function equalBinIndex(v: number, [d0, d1]: [number, number], count: number): number {
  // Stated confidences are round numbers (0.6, 0.8); the small tolerance keeps 0.6 in [0.6, 0.7)
  // instead of letting float error drop it into [0.5, 0.6).
  return Math.min(count - 1, Math.max(0, Math.floor(((v - d0) / (d1 - d0)) * count + 1e-9)));
}

export interface Summary {
  n: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
}

/** Five-number summary and mean; null for an empty sample. */
export function summarise(values: readonly number[]): Summary | null {
  const sorted = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  return {
    n: sorted.length,
    min: min(sorted) as number,
    q1: quantileSorted(sorted, 0.25) as number,
    median: quantileSorted(sorted, 0.5) as number,
    q3: quantileSorted(sorted, 0.75) as number,
    max: max(sorted) as number,
    mean: mean(sorted) as number,
  };
}

/* ---------- Tables ---------- */

/**
 * Columns whose value is the same in every row (all-zero triage counts, `n_failed` = 0): they
 * carry no comparison, so a table drops them and names them in its footer. A table of one row
 * keeps everything. A column missing from some rows is not constant.
 */
export function constantColumns(table: { metrics: readonly string[]; rows: readonly { cells: Record<string, { value: number; lo: number | null; hi: number | null }> }[] }): string[] {
  if (table.rows.length < 2) return [];
  return table.metrics.filter((m) => {
    const cells = table.rows.map((r) => r.cells[m]);
    if (cells.some((c) => c === undefined)) return false;
    const first = cells[0] as { value: number; lo: number | null; hi: number | null };
    return cells.every((c) => c!.value === first.value && c!.lo === first.lo && c!.hi === first.hi);
  });
}

/**
 * The row with the best point estimate, but only when its interval clears the runner-up's (no
 * overlap); otherwise null, since the data does not single out a best row. Rows without an
 * interval never clear anything.
 */
export function clearLeader<T extends { name: string; estimate: Estimate }>(rows: readonly T[], direction: "higher" | "lower"): T | null {
  if (rows.length < 2) return null;
  const sorted = [...rows].sort((a, b) => (direction === "higher" ? b.estimate.value - a.estimate.value : a.estimate.value - b.estimate.value));
  const [best, second] = sorted as [T, T];
  if (!hasInterval(best.estimate) || !hasInterval(second.estimate)) return null;
  return direction === "higher" ? (best.estimate.lo > second.estimate.hi ? best : null) : best.estimate.hi < second.estimate.lo ? best : null;
}
