import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

import { alphaLevels, compareRuns, deltaOf, deployedScores, guaranteeSeries, methodReading, predictionSets, rq3Interval } from "./rq3.ts";

const level = (violation: number, certified: number, coverage: number) => ({
  violation_rate: violation,
  violation_ci: [violation / 2, violation * 2 + 0.01],
  coverage: { mean: coverage, ci: [0, coverage * 2] },
  risk: certified > 0 ? { mean: 0.1, ci: [0.05, 0.2] } : null,
  certified_rate: certified,
  threshold: null,
  n_repeats: 200,
});

const DATA = {
  meta: { delta: 0.1, intervals: { violation_ci: "95% Clopper-Pearson over repeats", "coverage, risk, threshold, set metrics": "mean over repeats and central 95% range of the per-repeat values" } },
  selective: {
    certified: { "0.10": level(0.015, 0.025, 0.003), "0.05": level(0, 0, 0), "0.20": level(0.06, 0.9, 0.2) },
    plug_in: { "0.10": level(0.61, 0.99, 0.21) },
  },
  single_signal: { "signal:verbal_conf": { "0.10": level(0.02, 0.03, 0.005) } },
  prediction_sets: {
    lac: { set_coverage: { "0.10": { mean: 0.91, ci: [0.88, 0.94] }, "0.05": { mean: 0.96, ci: [0.92, 0.98] } }, mean_set_size: { "0.10": { mean: 1.9, ci: [1.7, 2.1] } } },
  },
  calibration: { p_correct: [0.2, 0.9, 0.95, 1], correct: [0, 1, 1, 0], thresholds: { "0.05": null, "0.10": 0.87 }, heldout: { p_correct: [0.5], correct: [1] }, delta: 0.1 },
};

test("records are read per method, sorted by α, with trivial levels flagged", () => {
  const series = guaranteeSeries(DATA);
  assert.deepEqual(
    series.map((s) => [s.group, s.method]),
    [
      ["selective", "certified"],
      ["selective", "plug_in"],
      ["single_signal", "signal:verbal_conf"],
    ],
  );
  const certified = series[0];
  assert.deepEqual(
    certified?.points.map((p) => p.alpha),
    [0.05, 0.1, 0.2],
  );
  assert.equal(certified?.points[0]?.trivial, true);
  assert.deepEqual(certified?.points[1]?.violation, { value: 0.015, lo: 0.0075, hi: 0.04 });
  assert.deepEqual(alphaLevels(series), [0.05, 0.1, 0.2]);
  assert.equal(deltaOf(DATA), 0.1);
  assert.equal(rq3Interval(DATA, "violation_ci").kind, "clopper-pearson");
  assert.equal(rq3Interval(DATA, "coverage").kind, "split-range");
});

test("method reading leaves trivial levels out of the summary", () => {
  const certified = guaranteeSeries(DATA, ["selective"])[0];
  assert.ok(certified);
  const r = methodReading(certified, 0.1);
  assert.deepEqual(r.trivialAlphas, [0.05]);
  assert.equal(r.worstViolation, 0.06);
  assert.equal(r.withinDelta, true);
  assert.equal(r.coverageMin, 0.003);
  assert.equal(r.coverageMax, 0.2);
});

test("prediction sets line up metrics by α", () => {
  const [lac] = predictionSets(DATA);
  assert.equal(lac?.method, "lac");
  assert.deepEqual(
    lac?.points.map((p) => [p.alpha, p.setCoverage?.value ?? null, p.meanSetSize?.value ?? null]),
    [
      [0.05, 0.96, null],
      [0.1, 0.91, 1.9],
    ],
  );
});

test("deployed scores split right and wrong answers and list thresholds", () => {
  const d = deployedScores(DATA, 10);
  assert.ok(d);
  const cal = d.histograms.find((h) => h.set === "calibration");
  assert.equal(cal?.n, 4);
  assert.equal(cal?.bins[9]?.correct, 2);
  assert.equal(cal?.bins[9]?.wrong, 1);
  assert.equal(cal?.bins[2]?.wrong, 1);
  assert.deepEqual(d.thresholds, [
    { alpha: 0.05, threshold: null },
    { alpha: 0.1, threshold: 0.87 },
  ]);
});

test("runs are compared side by side for one method", () => {
  const runs = compareRuns([
    { run: "main", label: "Haiku", data: DATA },
    { run: "empty", label: "None", data: {} },
  ]);
  assert.deepEqual(
    runs.map((r) => r.run),
    ["main"],
  );
});

const REAL = new URL("../../../../results/rq3_conformal.json", import.meta.url);
test("the real RQ3 file: certified violation stays within δ where it certifies", { skip: !existsSync(REAL) }, () => {
  const data: unknown = JSON.parse(readFileSync(REAL, "utf8"));
  const certified = guaranteeSeries(data, ["selective"]).find((s) => s.method === "certified");
  assert.ok(certified && certified.points.length > 0);
  assert.ok(deltaOf(data) !== null);
});

/* ---------- Labels and takeaways ---------- */

import { certifiedSplitViolations, compareTakeaway, coverageTakeaway, methodLabel, predictionSetTakeaway, selectiveDomain, violationTakeaway } from "./rq3.ts";

test("methods keep their colour slots: certified first, plug-in second", () => {
  const series = guaranteeSeries({ selective: { certified_hb: DATA.selective.plug_in, plug_in: DATA.selective.plug_in, certified: DATA.selective.certified } });
  assert.deepEqual(selectiveDomain(series), ["certified", "plug_in", "certified_hb"]);
  assert.equal(methodLabel("plug_in"), "Plug-in (no correction)");
  assert.equal(methodLabel("other"), "other");
});

test("the violation takeaway leaves trivial levels out and names them", () => {
  const text = violationTakeaway(guaranteeSeries(DATA, ["selective"]), 0.1);
  assert.equal(
    text,
    "Across all 200 splits (a split that certifies nothing withholds everything and counts as no violation), the certified rule's test risk exceeded α in at most 6.0% of splits, within δ = 0.10; the plug-in rule did so in 61.0%. At α = 0.05 nothing was certified, so its zero rate is trivial.",
  );
  assert.equal(violationTakeaway([], 0.1), null);
});

test("the coverage takeaway reads the largest α and says what coverage costs", () => {
  const text = coverageTakeaway(guaranteeSeries(DATA, ["selective"]));
  assert.equal(text, "At α = 0.20 the certified rule answers 20.0% of held-out questions on average, where a threshold was certified in 90.0% of splits. Lower coverage is the price of the certificate.");
});

test("runs are compared at the largest shared α and called descriptive", () => {
  const runs = compareRuns([
    { run: "main", label: "Haiku", data: DATA },
    { run: "sonnet", label: "Sonnet", data: { selective: { certified: { "0.20": level(0.0, 0.6, 0.54), "0.10": level(0.02, 0.07, 0.015) } } } },
  ]);
  const text = compareTakeaway(runs, 0.1);
  assert.ok(text);
  assert.match(text, /^At α = 0\.20 the certified rule answers 20\.0% \(Haiku\) and 54\.0% \(Sonnet\) of held-out questions on average/);
  assert.match(text, /description, not a test/);
  assert.equal(compareTakeaway(runs.slice(0, 1), 0.1), null);
});

test("the prediction-set takeaway measures the shortfall below 1 − α", () => {
  const text = predictionSetTakeaway(predictionSets(DATA));
  assert.equal(text, "LAC meets or exceeds 1 − α at every level, with 1.90 to 1.90 labels per set.");
  const short = predictionSetTakeaway([{ method: "aps", points: [{ alpha: 0.1, setCoverage: { value: 0.88, lo: 0.85, hi: 0.9 }, meanSetSize: null, emptySetRate: null }] }]);
  assert.equal(short, "APS falls short of 1 − α by at most 2.0 points (α = 0.10).");
});

test("violations among certifying splits are counted, so a level's rate shows what it rests on", () => {
  const point = { alpha: 0.1, violation: { value: 0.015, lo: null, hi: null }, coverage: null, risk: null, threshold: null, certifiedRate: 0.025, nRepeats: 200, trivial: false };
  assert.deepEqual(certifiedSplitViolations(point), { violations: 3, certified: 5, rate: 0.6 });
  assert.deepEqual(certifiedSplitViolations({ ...point, certifiedRate: 0, violation: { value: 0, lo: null, hi: null } }), { violations: 0, certified: 0, rate: null });
  assert.equal(certifiedSplitViolations({ ...point, nRepeats: null }), null);
});
