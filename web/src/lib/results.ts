/**
 * Readers for the experiment results files served by `/api/experiments/{name}`.
 *
 * SPEC 3.9 fixes what each file contains but not its exact layout, and the files are written
 * by code that evolves. So these readers look for structure instead of assuming one layout:
 * a point estimate with an interval, a table of systems by metrics, a curve, a record per
 * (method, alpha). Whatever they do not recognise is still shown on the page as raw JSON, so
 * a layout change can hide a chart but can never put a wrong number on screen.
 */

import { fmtInt } from "./format.ts";

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

export interface Stat {
  value: number;
  lo: number | null;
  hi: number | null;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);

const POINT_KEYS = ["mean", "value", "estimate", "point", "rate", "median"];
const LO_KEYS = ["lo", "low", "lower", "ci_low", "ci_lo", "ci_lower", "q025", "p2.5"];
const HI_KEYS = ["hi", "high", "upper", "ci_high", "ci_hi", "ci_upper", "q975", "p97.5"];
const PAIR_KEYS = ["ci", "ci95", "ci_95", "interval", "interval95", "ci95_interval"];
const STAT_EXTRA = ["std", "sd", "se", "sem", "n", "n_boot", "level", "count"];

/** True for an object that is one estimate with its interval, e.g. `{mean, ci: [lo, hi]}`. */
function isStatObject(value: unknown): boolean {
  if (!isRecord(value)) return false;
  const keys = Object.keys(value);
  if (!keys.some((k) => POINT_KEYS.includes(k))) return false;
  const known = [...POINT_KEYS, ...LO_KEYS, ...HI_KEYS, ...PAIR_KEYS, ...STAT_EXTRA];
  return keys.every((k) => known.includes(k));
}

export function readStat(value: unknown): Stat | null {
  const direct = num(value);
  if (direct !== null) return { value: direct, lo: null, hi: null };
  if (Array.isArray(value) && value.length === 3 && value.every((v) => num(v) !== null)) {
    // [estimate, lo, hi]
    return { value: value[0] as number, lo: value[1] as number, hi: value[2] as number };
  }
  if (!isRecord(value) || !isStatObject(value)) return null;
  const pointKey = POINT_KEYS.find((k) => num(value[k]) !== null);
  if (!pointKey) return null;
  let lo: number | null = null;
  let hi: number | null = null;
  for (const key of PAIR_KEYS) {
    const pair = value[key];
    if (Array.isArray(pair) && pair.length === 2 && num(pair[0]) !== null && num(pair[1]) !== null) {
      lo = pair[0] as number;
      hi = pair[1] as number;
      break;
    }
  }
  if (lo === null) {
    const loKey = LO_KEYS.find((k) => num(value[k]) !== null);
    const hiKey = HI_KEYS.find((k) => num(value[k]) !== null);
    if (loKey && hiKey) {
      lo = value[loKey] as number;
      hi = value[hiKey] as number;
    }
  }
  return { value: value[pointKey] as number, lo, hi };
}

/* ---------- Tables: systems by metrics ---------- */

export interface MetricTable {
  /** Where in the file the table was found, e.g. "nfcorpus". */
  path: string[];
  metrics: string[];
  rows: { name: string; cells: Record<string, Stat> }[];
}

const NAME_KEYS = ["system", "retriever", "method", "model", "name", "signal", "label", "condition", "feature_set", "calibrator"];

/** The metric cells of one row. Nested objects of stats are flattened one level: `tests.vs_bm25`. */
function readCells(entry: Record<string, unknown>): Record<string, Stat> {
  const cells: Record<string, Stat> = {};
  for (const [key, value] of Object.entries(entry)) {
    if (NAME_KEYS.includes(key)) continue;
    const stat = readStat(value);
    if (stat) {
      cells[key] = stat;
    } else if (isRecord(value)) {
      for (const [inner, innerValue] of Object.entries(value)) {
        const innerStat = readStat(innerValue);
        if (innerStat) cells[`${key}.${inner}`] = innerStat;
      }
    }
  }
  return cells;
}

function tableFromEntries(path: string[], entries: [string, Record<string, unknown>][], minRows = 2): MetricTable | null {
  if (entries.length < minRows) return null;
  const rows = entries.map(([name, entry]) => ({ name, cells: readCells(entry) })).filter((row) => Object.keys(row.cells).length > 0);
  if (rows.length < minRows) return null;
  // Keep metrics that at least half the rows report, in order of first appearance.
  const counts = new Map<string, number>();
  for (const row of rows) for (const key of Object.keys(row.cells)) counts.set(key, (counts.get(key) ?? 0) + 1);
  const metrics = [...counts.entries()].filter(([, count]) => count >= rows.length / 2).map(([key]) => key);
  if (metrics.length === 0) return null;
  return { path, metrics, rows };
}

const SKIP_KEYS = ["meta", "config", "per_item", "items", "records", "per_query", "seeds"];

/**
 * Finds every "name → metrics" table in a results file, wherever it is nested. A table needs
 * `minRows` rows (2 by default, so a lone object is not mistaken for one); pass 1 where the
 * caller knows the node is a table, e.g. a run's summary that has a single condition.
 */
export function findMetricTables(json: unknown, maxDepth = 5, minRows = 2): MetricTable[] {
  const tables: MetricTable[] = [];

  const visit = (node: unknown, path: string[], depth: number) => {
    if (depth > maxDepth) return;
    if (Array.isArray(node)) {
      const objects = node.filter(isRecord);
      if (objects.length >= 2 && objects.length === node.length) {
        const nameKey = NAME_KEYS.find((k) => objects.every((o) => typeof o[k] === "string"));
        // Rows that also carry an alpha are per-level records, read by findLevelRecords instead.
        if (nameKey && !objects.some((o) => "alpha" in o)) {
          const table = tableFromEntries(path, objects.map((o) => [o[nameKey] as string, o]), minRows);
          if (table) tables.push(table);
        }
      }
      return;
    }
    if (!isRecord(node) || isStatObject(node)) return;

    const entries = Object.entries(node).filter(([key]) => !SKIP_KEYS.includes(key));
    const objectEntries = entries.filter((entry): entry is [string, Record<string, unknown>] => isRecord(entry[1]) && !isStatObject(entry[1]));
    // A node is a table when most of its children are rows: objects whose own values are estimates.
    const rowLike = objectEntries.filter(([, value]) => Object.values(value).some((v) => readStat(v) !== null));
    if (objectEntries.length >= minRows && rowLike.length === objectEntries.length && !objectEntries.some(([key]) => isAlphaKey(key))) {
      const table = tableFromEntries(path, objectEntries, minRows);
      if (table) {
        tables.push(table);
        return;
      }
    }
    for (const [key, value] of entries) visit(value, [...path, key], depth + 1);
  };

  visit(json, [], 0);
  return tables;
}

/* ---------- Curves ---------- */

export interface Curve {
  name: string;
  points: { x: number; y: number; n: number | null }[];
}

function readPoints(value: unknown, xKeys: string[], yKeys: string[], nKeys: string[]): Curve["points"] | null {
  if (Array.isArray(value)) {
    const points: Curve["points"] = [];
    for (const item of value) {
      if (Array.isArray(item) && item.length >= 2 && num(item[0]) !== null && num(item[1]) !== null) {
        points.push({ x: item[0] as number, y: item[1] as number, n: num(item[2]) });
        continue;
      }
      if (!isRecord(item)) return null;
      const xKey = xKeys.find((k) => num(item[k]) !== null);
      const yKey = yKeys.find((k) => num(item[k]) !== null);
      if (!xKey || !yKey) continue; // an empty bin
      const nKey = nKeys.find((k) => num(item[k]) !== null);
      points.push({ x: item[xKey] as number, y: item[yKey] as number, n: nKey ? (item[nKey] as number) : null });
    }
    return points.length >= 2 ? points : null;
  }
  if (isRecord(value)) {
    const xKey = xKeys.find((k) => Array.isArray(value[k]));
    const yKey = yKeys.find((k) => Array.isArray(value[k]));
    if (!xKey || !yKey) return null;
    const xs = value[xKey] as unknown[];
    const ys = value[yKey] as unknown[];
    const nKey = nKeys.find((k) => Array.isArray(value[k]));
    const ns = nKey ? (value[nKey] as unknown[]) : [];
    const points: Curve["points"] = [];
    for (let i = 0; i < Math.min(xs.length, ys.length); i++) {
      const x = num(xs[i]);
      const y = num(ys[i]);
      if (x !== null && y !== null) points.push({ x, y, n: num(ns[i]) });
    }
    return points.length >= 2 ? points : null;
  }
  return null;
}

function findCurves(json: unknown, keyPattern: RegExp, xKeys: string[], yKeys: string[], nKeys: string[]): Curve[] {
  const curves: Curve[] = [];
  const visit = (node: unknown, depth: number) => {
    if (depth > 4 || !isRecord(node)) return;
    for (const [key, value] of Object.entries(node)) {
      if (SKIP_KEYS.includes(key)) continue;
      if (keyPattern.test(key)) {
        const single = readPoints(value, xKeys, yKeys, nKeys);
        if (single) {
          curves.push({ name: key, points: single });
        } else if (isRecord(value)) {
          for (const [name, inner] of Object.entries(value)) {
            const points = readPoints(inner, xKeys, yKeys, nKeys);
            if (points) curves.push({ name, points });
          }
        }
      } else {
        visit(value, depth + 1);
      }
    }
  };
  visit(json, 0);
  return curves;
}

const CONF_KEYS = ["confidence", "conf", "mean_conf", "mean_confidence", "avg_conf", "p_mean", "mean_predicted", "predicted", "mean_p", "bin_conf"];
const ACC_KEYS = ["accuracy", "acc", "frac_correct", "frac_pos", "fraction_correct", "mean_correct", "observed", "empirical", "bin_acc"];
const COUNT_KEYS = ["count", "n", "size", "counts", "bin_count"];

/** Reliability diagrams: mean confidence against observed accuracy, per bin, per model. */
export function findReliability(json: unknown): Curve[] {
  return findCurves(json, /reliab|calibration_curve/i, CONF_KEYS, ACC_KEYS, COUNT_KEYS).map((curve) => ({
    ...curve,
    points: [...curve.points].sort((a, b) => a.x - b.x),
  }));
}

/** Risk-coverage curves: selective risk as a function of the share of questions answered. */
export function findRiskCoverage(json: unknown): Curve[] {
  return findCurves(json, /risk[_-]?cov|rc[_-]?curve/i, ["coverage", "cov", "coverages"], ["risk", "selective_risk", "risks"], COUNT_KEYS).map(
    (curve) => ({ ...curve, points: [...curve.points].sort((a, b) => a.x - b.x) }),
  );
}

/** A training loss curve: `{steps: [...], loss: [...]}`, `[{step, loss}]`, or a bare list of losses. */
export function findLossCurve(json: unknown): Curve | null {
  if (!isRecord(json)) return null;
  const columns = readPoints(json, ["steps", "step"], ["loss", "losses", "loss_curve"], []);
  if (columns) return { name: "loss", points: columns };
  for (const key of ["loss_curve", "losses", "loss", "curve", "history", "log"]) {
    const value = json[key];
    if (Array.isArray(value) && value.length >= 2 && value.every((v) => num(v) !== null)) {
      return { name: "loss", points: (value as number[]).map((y, i) => ({ x: i + 1, y, n: null })) };
    }
    const points = readPoints(value, ["step", "steps", "iteration"], ["loss", "train_loss", "losses"], []);
    if (points) return { name: "loss", points };
  }
  return null;
}

/* ---------- Records per (method, alpha) ---------- */

export interface LevelRecord {
  method: string;
  alpha: number;
  violation: Stat | null;
  coverage: Stat | null;
  risk: Stat | null;
  /** Share of repeats in which any threshold was certified (`certified_rate`), when reported. */
  certifiedRate: number | null;
}

function parseAlpha(key: string): number | null {
  const match = key.match(/^(?:alpha|a|α)?[_=\s]*(0?\.\d+)$/i);
  if (!match) return null;
  const value = Number.parseFloat(match[1] as string);
  return value > 0 && value < 1 ? value : null;
}

function isAlphaKey(key: string): boolean {
  return parseAlpha(key) !== null;
}

const STRUCTURAL = ["results", "methods", "by_alpha", "alphas", "per_alpha", "summary", "selective", "certified", "baselines"];

function pick(node: Record<string, unknown>, patterns: RegExp[]): Stat | null {
  for (const pattern of patterns) {
    for (const [key, value] of Object.entries(node)) {
      if (!pattern.test(key)) continue;
      const stat = readStat(value);
      if (!stat) continue;
      // A point estimate may keep its interval in sibling keys: violation_rate + violation_ci.
      if (stat.lo === null) {
        const stem = key.replace(/_?(rate|mean)$/i, "");
        for (const [otherKey, other] of Object.entries(node)) {
          if (otherKey === key || !otherKey.startsWith(stem) || !/ci|interval|clopper/i.test(otherKey)) continue;
          if (Array.isArray(other) && other.length === 2 && num(other[0]) !== null && num(other[1]) !== null) {
            return { value: stat.value, lo: other[0] as number, hi: other[1] as number };
          }
        }
      }
      return stat;
    }
  }
  return null;
}

/**
 * Finds every record that reports a violation rate or a coverage at a named risk level.
 * The level comes from an `alpha` field or from a key on the path such as "0.10";
 * the method name comes from a `method` field or from the rest of the path.
 */
export function findLevelRecords(json: unknown): LevelRecord[] {
  const records: LevelRecord[] = [];

  const visit = (node: unknown, path: string[], depth: number) => {
    if (depth > 6) return;
    if (Array.isArray(node)) {
      node.forEach((item) => visit(item, path, depth + 1));
      return;
    }
    if (!isRecord(node) || isStatObject(node)) return;

    const violation = pick(node, [/^violation(_rate)?$/i, /violation.*rate/i, /^p_violation$/i]);
    const coverage = pick(node, [/^(mean_|test_)?coverage(_mean)?$/i]);
    const risk = pick(node, [/^(mean_|test_)?(selective_)?risk(_mean)?$/i]);
    const alpha = num(node.alpha) ?? path.map(parseAlpha).find((a): a is number => a !== null) ?? null;

    if (alpha !== null && (violation || coverage)) {
      const named = NAME_KEYS.map((k) => node[k]).find((v): v is string => typeof v === "string");
      const fromPath = path.filter((key) => !isAlphaKey(key) && !STRUCTURAL.includes(key));
      const certifiedRate = num(node.certified_rate);
      records.push({ method: named ?? (fromPath.join(" / ") || "certified"), alpha, violation, coverage, risk, certifiedRate });
      return;
    }
    for (const [key, value] of Object.entries(node)) {
      if (SKIP_KEYS.includes(key)) continue;
      visit(value, [...path, key], depth + 1);
    }
  };

  visit(json, [], 0);
  return records.sort((a, b) => a.alpha - b.alpha);
}

/* ---------- Provenance ---------- */

export interface Provenance {
  created: string | null;
  n: number | null;
  delta: number | null;
  models: string[];
}

export function readProvenance(json: unknown): Provenance {
  const meta = isRecord(json) && isRecord(json.meta) ? json.meta : {};
  const config = isRecord(meta.config) ? meta.config : isRecord(meta.config_snapshot) ? meta.config_snapshot : {};
  const top = isRecord(json) ? json : {};
  const models: string[] = [];
  const modelSource = meta.model_ids ?? meta.models ?? meta.model;
  if (typeof modelSource === "string") models.push(modelSource);
  else if (Array.isArray(modelSource)) models.push(...modelSource.filter((m): m is string => typeof m === "string"));
  else if (isRecord(modelSource)) {
    // Named entries keep their names: `llm_effort: low` means something, a bare `low` does not.
    for (const [key, value] of Object.entries(modelSource)) {
      if (typeof value === "string") models.push(`${key}: ${value}`);
      else if (Array.isArray(value) && value.length > 0 && value.every((v) => typeof v === "string")) models.push(`${key}: ${value.join(", ")}`);
    }
  }
  return {
    created: typeof meta.created === "string" ? meta.created : null,
    n: num(meta.n) ?? num(top.n),
    delta: num(top.delta) ?? num(meta.delta) ?? num(config.delta),
    models,
  };
}

/* ---------- Generator runs ---------- */

/** What the run picker needs from `/api/experiments` `runs` (see `RunInfo` in types.ts). */
export interface RunLike {
  run: string;
  generator: string | null;
  n: number | null;
  self_consistency_samples: number | null;
}

/** `claude-haiku-5-5` reads "Haiku 5.5"; any other id is shown as written. */
export function generatorName(model: string | null): string {
  if (!model) return "Unknown generator";
  const m = model.match(/^claude-([a-z]+)-(\d+)-(\d+)$/i);
  if (!m) return model;
  const [, family = "", major, minor] = m;
  return `${family.charAt(0).toUpperCase()}${family.slice(1).toLowerCase()} ${major}.${minor}`;
}

/**
 * The picker label of a run: its generator's name, plus its samples per question when another
 * run shares that generator (the only thing that tells two such runs apart at a glance).
 */
export function runLabel(info: RunLike, all: readonly RunLike[]): string {
  const name = generatorName(info.generator);
  const shared = all.filter((other) => other.generator === info.generator).length > 1;
  const k = info.self_consistency_samples;
  if (!shared || k === null) return name;
  // Runs of one generator over different question sets also say how many questions they cover.
  const differentN = info.n !== null && all.some((other) => other.generator === info.generator && other.n !== info.n);
  return `${name}, ${k} sample${k === 1 ? "" : "s"}${differentN ? `, ${fmtInt(info.n as number)} questions` : ""}`;
}

/** Where a run's files sit under `results/`: the main run at the top, the others in `<run>/`. */
export function runPathPrefix(run: string): string {
  return run === "main" ? "" : `${run}/`;
}

/** Fields of RQ2's `meta` that say which features or feature sets a run could not measure. */
export function readUnmeasured(json: unknown): { features: string[]; featureSets: string[] } {
  const meta = isRecord(json) && isRecord(json.meta) ? json.meta : {};
  const strings = (value: unknown) => (Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : []);
  return { features: strings(meta.unavailable_features), featureSets: strings(meta.skipped_feature_sets) };
}

/* ---------- Names ---------- */

const METRIC_NAMES: [RegExp, string][] = [
  [/^ndcg@?_?(\d+)$/i, "nDCG@$1"],
  [/^recall@?_?(\d+)$/i, "Recall@$1"],
  [/^mrr@?_?(\d+)$/i, "MRR@$1"],
  [/^map$/i, "MAP"],
  [/^auroc$/i, "AUROC"],
  [/^e[_-]?aurc$/i, "E-AURC"],
  [/^aurc$/i, "AURC"],
  [/^ece$/i, "ECE"],
  [/^brier$/i, "Brier score"],
  [/^accuracy$/i, "Accuracy"],
  [/^macro[_-]?f1$/i, "Macro-F1"],
  [/^coverage$/i, "Coverage"],
  [/^mean_set_size$/i, "Mean set size"],
];

export function metricLabel(key: string): string {
  for (const [pattern, label] of METRIC_NAMES) if (pattern.test(key)) return key.replace(pattern, label);
  const pValue = key.match(/^(?:tests?\.)?p[_-]?(?:value[_-]?)?(?:vs[_-]?)?(.+)$/i);
  if (pValue && /p[_-]?(value|vs)/i.test(key)) return `p vs ${pValue[1]}`;
  return key.replace(/_/g, " ");
}

/** Which direction is good for a metric, when that is known. Used only to mark the best row. */
export function metricDirection(key: string): "higher" | "lower" | null {
  // p-values and counts have no better direction; a `_diff` keeps its metric's direction.
  if (isProbabilityLike(key) || /^n(_|$)/i.test(key)) return null;
  if (/^(ndcg|recall|mrr|map|auroc|accuracy|macro[_-]?f1|coverage)/i.test(key)) return "higher";
  if (/^(e[_-]?aurc|aurc|ece|brier|risk|violation)/i.test(key)) return "lower";
  return null;
}

/** A p-value column: `p_vs_bm25`, `tests.*`, `p_value`, `ndcg@10_p`, `ndcg@10_p_holm`, `brier_p`. */
export function isProbabilityLike(key: string): boolean {
  return /^(p[_-]|tests?\.)/i.test(key) || /(^|_)p(_holm|_value)?$/i.test(key);
}

/**
 * A table whose columns are risk levels (`{"set_coverage": {"0.05": ..., "0.10": ...}}`) turned
 * so each row is one level and each column one quantity; other tables are returned unchanged.
 */
export function levelsAsRows(table: MetricTable): MetricTable {
  if (!table.metrics.every(isAlphaKey)) return table;
  return {
    path: table.path,
    metrics: table.rows.map((row) => row.name),
    rows: table.metrics.map((level) => ({
      name: `α = ${level}`,
      cells: Object.fromEntries(table.rows.flatMap((row) => (row.cells[level] ? [[row.name, row.cells[level]]] : []))),
    })),
  };
}
