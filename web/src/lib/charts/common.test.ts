import assert from "node:assert/strict";
import { test } from "node:test";

import { at, clearLeader, constantColumns, equalBins, intervalInfo, intervalKindOf, intervalNotes, summarise } from "./common.ts";

test("at walks nested objects and stops at non-objects", () => {
  assert.equal(at({ a: { b: 2 } }, "a", "b"), 2);
  assert.equal(at({ a: 1 }, "a", "b"), undefined);
  assert.equal(at(null, "a"), undefined);
});

test("interval kinds are read from the file's own words", () => {
  assert.equal(intervalKindOf("95% percentile bootstrap over queries"), "bootstrap");
  assert.equal(intervalKindOf("95% Clopper-Pearson over repeats (split Monte-Carlo error)"), "clopper-pearson");
  assert.equal(intervalKindOf("mean over repeats and central 95% range of the per-repeat values (split variability)"), "split-range");
  assert.equal(intervalKindOf("test predictions pooled over all repeats"), "pooled");
  assert.equal(intervalKindOf(null), "unknown");
});

test("compound interval keys are split per block", () => {
  const data = { meta: { intervals: { "estimators, ablation, auroc_diff": "central 95% range of the per-repeat values", brier_diff: "95% percentile bootstrap" } } };
  const notes = intervalNotes(data);
  assert.equal(notes.ablation, "central 95% range of the per-repeat values");
  assert.equal(intervalInfo(data, "brier_diff").kind, "bootstrap");
  assert.equal(intervalInfo(data, "auroc_diff").label, "central 95% range across splits");
  assert.equal(intervalInfo(data, "nothing").kind, "unknown");
  assert.equal(intervalInfo({ meta: { inference: { ci: "95% percentile bootstrap over queries" } } }, "ci").kind, "bootstrap");
});

test("equalBins counts the upper edge in the last bin and ignores values outside", () => {
  const bins = equalBins([0, 0.05, 0.5, 0.99, 1, 1.2, Number.NaN], [0, 1], 10);
  assert.equal(bins.length, 10);
  assert.equal(bins[0]?.count, 2);
  assert.equal(bins[5]?.count, 1);
  assert.equal(bins[9]?.count, 2);
  assert.equal(bins.reduce((s, b) => s + b.count, 0), 5);
  assert.equal(bins[9]?.hi, 1);
});

test("summarise gives quartiles and mean", () => {
  const s = summarise([4, 1, 3, 2, 5]);
  assert.deepEqual(s, { n: 5, min: 1, q1: 2, median: 3, q3: 4, max: 5, mean: 3 });
  assert.equal(summarise([]), null);
});

test("constant columns are the ones every row shares", () => {
  const cell = (value: number, lo: number | null = null, hi: number | null = null) => ({ value, lo, hi });
  type Cell = { value: number; lo: number | null; hi: number | null };
  const table: { metrics: string[]; rows: { cells: Record<string, Cell> }[] } = {
    metrics: ["accuracy", "n_failed", "triage.urgent", "n"],
    rows: [
      { cells: { accuracy: cell(0.5, 0.4, 0.6), n_failed: cell(0), "triage.urgent": cell(0), n: cell(1000) } },
      { cells: { accuracy: cell(0.7, 0.6, 0.8), n_failed: cell(0), "triage.urgent": cell(0) } },
    ],
  };
  assert.deepEqual(constantColumns(table), ["n_failed", "triage.urgent"]);
  assert.deepEqual(constantColumns({ metrics: ["a"], rows: [table.rows[0]!] }), []);
});

test("a leader is marked only when its interval clears the runner-up's", () => {
  const row = (name: string, value: number, lo: number, hi: number) => ({ name, estimate: { value, lo, hi } });
  assert.equal(clearLeader([row("a", 0.741, 0.69, 0.79), row("b", 0.735, 0.68, 0.78)], "higher"), null);
  assert.equal(clearLeader([row("a", 0.9, 0.85, 0.95), row("b", 0.7, 0.65, 0.8)], "higher")?.name, "a");
  assert.equal(clearLeader([row("a", 0.1, 0.05, 0.12), row("b", 0.3, 0.2, 0.4)], "lower")?.name, "a");
  assert.equal(clearLeader([row("a", 0.9, 0.85, 0.95)], "higher"), null);
});
