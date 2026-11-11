/**
 * Small readers that pull the headline figures out of the results files. Pure functions over the
 * parsed JSON: every figure the page prints is computed here from the file at runtime, and a
 * figure that cannot be found comes back as null (printed as a dash), never as a guess.
 */

import { findLevelRecords, findMetricTables, isRecord, metricDirection, readProvenance, readStat, type LevelRecord, type MetricTable, type Stat } from "@/lib/results";

const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);

export function metaOf(data: unknown): Record<string, unknown> {
  return isRecord(data) && isRecord(data.meta) ? data.meta : {};
}

/* ---------- RQ3 ---------- */

/** A level where no split certified a threshold abstains on everything: its zero violation rate is trivial. */
export function neverCertified(r: LevelRecord): boolean {
  return r.certifiedRate !== null ? r.certifiedRate === 0 : r.coverage !== null && r.coverage.value === 0;
}

export interface Rq3Summary {
  records: LevelRecord[];
  alphas: number[];
  delta: number | null;
  /** Where δ came from, for the caption. */
  deltaSource: "the results file" | "the server configuration" | null;
  /** The α the deployed configuration certifies at, when the file records it. */
  configAlpha: number | null;
  scope: string | null;
}

export function readRq3(data: unknown, serverDelta: number | null): Rq3Summary {
  const records = findLevelRecords(data);
  const fileDelta = readProvenance(data).delta;
  const meta = metaOf(data);
  const config = isRecord(meta.config) ? meta.config : {};
  const alphas = [...new Set(records.map((r) => r.alpha))].sort((a, b) => a - b);
  const configAlpha = num(config.alpha);
  return {
    records,
    alphas,
    delta: fileDelta ?? serverDelta,
    deltaSource: fileDelta !== null ? "the results file" : serverDelta !== null ? "the server configuration" : null,
    configAlpha: configAlpha !== null && alphas.includes(configAlpha) ? configAlpha : null,
    scope: typeof meta.guarantee === "string" ? meta.guarantee : null,
  };
}

export function certifiedAt(records: readonly LevelRecord[], alpha: number | null): LevelRecord | null {
  if (alpha === null) return null;
  return records.find((r) => r.method === "certified" && r.alpha === alpha) ?? null;
}

/* ---------- Best rows of a metric ---------- */

export interface Best {
  row: string;
  stat: Stat;
  table: MetricTable;
}

/** The best row for a metric across tables, by the metric's known good direction. */
function bestOf(tables: readonly MetricTable[], metric: string): Best | null {
  const direction = metricDirection(metric);
  if (!direction) return null;
  let best: Best | null = null;
  for (const table of tables) {
    for (const row of table.rows) {
      const stat = row.cells[metric];
      if (!stat) continue;
      if (!best || (direction === "higher" ? stat.value > best.stat.value : stat.value < best.stat.value)) best = { row: row.name, stat, table };
    }
  }
  return best;
}

/** RQ1: the best nDCG@10 on the first dataset that reports it (NFCorpus in the current files). */
export function readBestRetrieval(data: unknown): (Best & { metric: string; dataset: string }) | null {
  const tables = findMetricTables(data).filter((t) => t.path.length === 1);
  for (const table of tables) {
    const metric = table.metrics.find((m) => /^ndcg@?_?10$/i.test(m)) ?? null;
    if (!metric) continue;
    const best = bestOf([table], metric);
    if (best) return { ...best, metric, dataset: table.path[0] ?? "" };
  }
  return null;
}


/** RQ2 meta: the accuracy of the answers the confidence models are trained on, and where they came from. */
export function readScoredAccuracy(data: unknown): { accuracy: number; correct: number | null; n: number | null; source: string | null } | null {
  const meta = metaOf(data);
  const accuracy = num(meta.accuracy);
  if (accuracy === null) return null;
  return {
    accuracy,
    correct: num(meta.n_correct),
    n: num(meta.n),
    source: typeof meta.source === "string" ? meta.source : null,
  };
}

/* ---------- RQ2 comparisons ---------- */

export interface Comparison {
  reference: string;
  model: string;
  brier: Stat | null;
  brierP: number | null;
  auroc: Stat | null;
  aurocBetterShare: number | null;
}

/** `comparisons: {vs_<reference>: {<model>: {brier_diff, brier_p, auroc_diff, auroc_better_share}}}`. */
export function readComparisons(data: unknown): Comparison[] {
  if (!isRecord(data) || !isRecord(data.comparisons)) return [];
  const out: Comparison[] = [];
  for (const [key, block] of Object.entries(data.comparisons)) {
    if (!isRecord(block)) continue;
    const reference = key.replace(/^vs[_-]/, "");
    for (const [model, entry] of Object.entries(block)) {
      if (!isRecord(entry)) continue;
      out.push({
        reference,
        model,
        brier: readStat(entry.brier_diff),
        brierP: num(entry.brier_p),
        auroc: readStat(entry.auroc_diff),
        aurocBetterShare: num(entry.auroc_better_share),
      });
    }
  }
  return out;
}

/* ---------- Grouping tables ---------- */

export interface TableGroup {
  /** The shared first key of the path, or "" for tables at the top of the file. */
  key: string;
  tables: MetricTable[];
}

/**
 * Tables that are variants of one another (the same estimators under four calibrators, the same
 * retrievers on two datasets) are grouped by the first key of their path so they can share one
 * card with a switch, instead of stacking four near-identical cards.
 */
export function groupTables(tables: readonly MetricTable[]): TableGroup[] {
  const groups: TableGroup[] = [];
  for (const table of tables) {
    const key = table.path.length > 1 ? (table.path[0] ?? "") : "";
    const group = groups.find((g) => g.key === key);
    if (group) group.tables.push(table);
    else groups.push({ key, tables: [table] });
  }
  return groups;
}
