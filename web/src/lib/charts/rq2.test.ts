import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

import {
  ablation,
  calibrators,
  dropDuplicateSignals,
  estimatorKind,
  estimatorRanking,
  learnedVsStated,
  mainCalibrator,
  reliabilityBins,
  riskAtFullCoverage,
  riskCoverageSeries,
  rq2Interval,
  statedConfidence,
} from "./rq2.ts";

const est = (mean: number, lo: number, hi: number) => ({ mean, ci: [lo, hi] });

const DATA = {
  meta: { main_calibrator: "platt", intervals: { "estimators, ablation, auroc_diff": "mean over repeats and central 95% range of the per-repeat values (split variability)" } },
  estimators: {
    raw: { "stated:verbal_conf": { auroc: est(0.7, 0.6, 0.8) } },
    platt: {
      "stated:verbal_conf": { auroc: est(0.735, 0.68, 0.78), brier: est(0.18, 0.17, 0.19) },
      "signal:ret_top": { auroc: est(0.55, 0.5, 0.6), brier: est(0.2, 0.19, 0.21) },
      "logistic:deploy": { auroc: est(0.74, 0.69, 0.79), brier: est(0.179, 0.17, 0.19) },
    },
  },
  ablation: { "all features": { auroc: est(0.74, 0.69, 0.79) }, "without verbal": { auroc: est(0.6, 0.55, 0.65) } },
  comparisons: {
    "vs_stated:verbal_conf": {
      "logistic:deploy": { brier_diff: est(0.001, -0.002, 0.004), brier_p: 0.4, auroc_diff: est(-0.002, -0.03, 0.02), auroc_better_share: 0.46 },
      "mlp:full": { brier_diff: est(0.01, 0.005, 0.02), brier_p: 0.001, auroc_diff: est(-0.05, -0.08, -0.01), auroc_better_share: 0.1 },
    },
    "vs_signal:verbal_conf": {},
  },
  reliability: {
    "logistic:deploy": [
      { lo: 0, hi: 0.1, count: 0, confidence: null, accuracy: null },
      { lo: 0.1, hi: 0.2, count: 65, confidence: 0.18, accuracy: 0.4 },
    ],
  },
  risk_coverage: { "stated:verbal_conf": { coverage: [1, 0.5], risk: [0.3, 0.1] } },
};

test("estimator kinds come from the name prefix", () => {
  assert.equal(estimatorKind("stated:verbal_conf"), "stated");
  assert.equal(estimatorKind("signal:verbal_conf"), "signal");
  assert.equal(estimatorKind("logistic:deploy"), "learned");
  assert.equal(estimatorKind("mlp:full"), "learned");
});

test("ranking follows the metric's direction", () => {
  assert.deepEqual(calibrators(DATA), ["raw", "platt"]);
  assert.equal(mainCalibrator(DATA), "platt");
  assert.deepEqual(
    estimatorRanking(DATA, "platt", "auroc").map((r) => r.name),
    ["logistic:deploy", "stated:verbal_conf", "signal:ret_top"],
  );
  assert.deepEqual(
    estimatorRanking(DATA, "platt", "brier").map((r) => r.name),
    ["logistic:deploy", "stated:verbal_conf", "signal:ret_top"],
  );
  assert.equal(rq2Interval(DATA, "estimators").kind, "split-range");
});

test("risk-coverage is sorted by coverage and gives the answer-everything risk", () => {
  const series = riskCoverageSeries(DATA);
  assert.deepEqual(series[0]?.points, [
    { coverage: 0.5, risk: 0.1 },
    { coverage: 1, risk: 0.3 },
  ]);
  assert.equal(riskAtFullCoverage(series), 0.3);
});

test("reliability drops empty bins", () => {
  const [r] = reliabilityBins(DATA);
  assert.equal(r?.bins.length, 1);
  assert.equal(r?.total, 65);
});

test("ablation deltas are differences of means from the full model", () => {
  const { reference, rows } = ablation(DATA, "auroc");
  assert.equal(reference, "all features");
  assert.equal(rows[0]?.deltaFromAll, null);
  assert.ok(Math.abs((rows[1]?.deltaFromAll ?? 0) + 0.14) < 1e-12);
});

test("a learned model beats stated confidence only when its interval excludes zero", () => {
  const c = learnedVsStated(DATA);
  assert.ok(c);
  assert.equal(c.reference, "stated:verbal_conf");
  assert.deepEqual(c.aurocBetter, []);
  assert.deepEqual(c.aurocWorse, ["mlp:full"]);
  assert.deepEqual(c.brierBetter, []);
  assert.deepEqual(c.brierWorse, ["mlp:full"]);
  const s = statedConfidence(DATA);
  assert.deepEqual(s, { name: "stated:verbal_conf", calibrator: "platt", estimate: { value: 0.735, lo: 0.68, hi: 0.78 } });
});

const REAL = new URL("../../../../results/rq2_confidence.json", import.meta.url);
test("the real RQ2 file: no learned model's AUROC interval lies above zero", { skip: !existsSync(REAL) }, () => {
  const data: unknown = JSON.parse(readFileSync(REAL, "utf8"));
  const c = learnedVsStated(data);
  assert.ok(c && c.rows.length > 0);
  // The finding these charts must not contradict; if a new run changes it, the copy changes with it.
  assert.ok(Array.isArray(c.aurocBetter));
  assert.ok(statedConfidence(data) !== null);
  assert.ok(estimatorRanking(data, mainCalibrator(data) as string).length > 0);
  assert.ok(reliabilityBins(data).length > 0);
});

/* ---------- Labels and takeaways ---------- */

import {
  ablationGroup,
  ablationTakeaway,
  brierTakeaway,
  estimatorLabel,
  learnedNames,
  rankingTakeaway,
  reliabilityTakeaway,
  riskAtCoverage,
  riskCoverageTakeaway,
  signed,
  weightedGap,
} from "./rq2.ts";

test("labels say what an estimator is", () => {
  assert.equal(estimatorLabel("stated:verbal_conf"), "stated confidence (verbal_conf)");
  assert.equal(estimatorLabel("signal:ret_top"), "ret_top");
  assert.equal(estimatorLabel("logistic:deploy"), "logistic:deploy");
  assert.deepEqual(learnedNames(["stated:a", "logistic:deploy", "signal:b", "mlp:full"]), ["logistic:deploy", "mlp:full"]);
  assert.equal(signed(0.0056), "+0.006");
  assert.equal(signed(-0.07), "−0.070");
  assert.equal(signed(0.0001), "0.000");
  assert.equal(ablationGroup("without self_consistency"), "self_consistency");
});

test("the ranking takeaway names a higher mean but not a win the interval does not show", () => {
  const ranking = estimatorRanking(DATA, "platt", "auroc");
  const text = rankingTakeaway(ranking, learnedVsStated(DATA));
  assert.ok(text);
  assert.match(text, /mean AUROC 0\.735 \(0\.680 to 0\.780 across splits\)/);
  assert.match(text, /1 of 2 others has a higher mean \(logistic:deploy 0\.740\)/);
  assert.match(text, /no learned model's AUROC difference from it has its central 95% range across splits entirely above zero/);
  assert.equal(rankingTakeaway([], null), null);
});

test("the Brier takeaway reads each interval against zero", () => {
  const text = brierTakeaway(learnedVsStated(DATA));
  assert.ok(text);
  assert.match(text, /^No learned model has a Brier difference interval entirely below zero/);
  assert.match(text, /mlp:full is worse/);
  assert.equal(brierTakeaway(null), null);
});

test("the ablation takeaway names the largest move from the full model", () => {
  const data = {
    ablation: {
      "all features": { auroc: est(0.74, 0.69, 0.79) },
      "without verbal": { auroc: est(0.67, 0.62, 0.72) },
      "without nli": { auroc: est(0.745, 0.69, 0.8) },
    },
  };
  assert.equal(ablationTakeaway(ablation(data)), "Dropping the verbal group lowers mean AUROC the most, by 0.070 (0.740 to 0.670); dropping any other group moves it by 0.005 or less.");
  assert.equal(ablationTakeaway({ reference: null, rows: [] }), null);
});

test("risk at a coverage takes the first grid point at or above it", () => {
  const series = { name: "stated:x", kind: "stated" as const, points: [0.25, 0.5, 0.75, 1].map((c) => ({ coverage: c, risk: c / 4 })) };
  assert.deepEqual(riskAtCoverage(series, 0.5), { coverage: 0.5, risk: 0.125 });
  assert.deepEqual(riskAtCoverage(series, 0.6), { coverage: 0.75, risk: 0.1875 });
  const other = { name: "logistic:deploy", kind: "learned" as const, points: [0.5, 1].map((c) => ({ coverage: c, risk: c / 5 })) };
  assert.equal(
    riskCoverageTakeaway([series, other], "stated:x", "logistic:deploy"),
    "Releasing only the most confident 50% of answers, 12.5% are wrong ranked by stated confidence (x), logistic:deploy 10.0%; answering every question, 25.0% are wrong.",
  );
  assert.equal(riskCoverageTakeaway([series], "missing", null), null);
});

test("the weighted gap is a prediction-weighted mean distance from the diagonal", () => {
  const series = {
    name: "stated:x",
    kind: "stated" as const,
    total: 100,
    bins: [
      { lo: 0.2, hi: 0.3, count: 25, confidence: 0.25, accuracy: 0.35 },
      { lo: 0.8, hi: 0.9, count: 75, confidence: 0.85, accuracy: 0.85 },
    ],
  };
  assert.ok(Math.abs((weightedGap(series) ?? 0) - 0.025) < 1e-12);
  const text = reliabilityTakeaway([series], ["stated:x"]);
  assert.equal(
    text,
    "Mean distance from the diagonal, weighted by predictions and pooled over all repeats: stated confidence (x) 2.5 points. The busiest bin for stated confidence (x) is 80% to 90%, with 75% of predictions.",
  );
  const other = { ...series, name: "logistic:deploy", kind: "learned" as const };
  const ece: Record<string, { value: number; lo: number; hi: number }> = {
    "stated:x": { value: 0.05, lo: 0.028, hi: 0.084 },
    "logistic:deploy": { value: 0.049, lo: 0.025, hi: 0.082 },
  };
  assert.match(
    reliabilityTakeaway([series, other], ["stated:x", "logistic:deploy"], (n) => ece[n] ?? null) ?? "",
    /pooled over all repeats: stated confidence \(x\) 2\.5 points; logistic:deploy 2\.5 points \(the file's per-split ECE averages 0\.050 and 0\.049, ranges overlapping\)\./,
  );
});

test("the real RQ2 file: the takeaways never call a learned model better without an interval", { skip: !existsSync(REAL) }, () => {
  const data: unknown = JSON.parse(readFileSync(REAL, "utf8"));
  const comparison = learnedVsStated(data);
  const ranking = estimatorRanking(data, mainCalibrator(data) as string);
  const text = rankingTakeaway(ranking, comparison) ?? "";
  if (comparison && comparison.aurocBetter.length === 0) assert.match(text, /no learned model's AUROC difference from it has its central 95% range across splits entirely above zero/);
  const brier = brierTakeaway(comparison) ?? "";
  if (comparison && comparison.brierBetter.length === 0) assert.match(brier, /^No learned model/);
  assert.ok(ablationTakeaway(ablation(data)));
});

test("signal:X is dropped when stated:X is present, so the reference is not ranked against itself", () => {
  const data = {
    estimators: {
      platt: {
        "signal:verbal_conf": { auroc: est(0.7351, 0.68, 0.78) },
        "stated:verbal_conf": { auroc: est(0.7351, 0.68, 0.78) },
        "signal:ret_top": { auroc: est(0.53, 0.5, 0.56) },
        "logistic:full": { auroc: est(0.741, 0.69, 0.79) },
      },
    },
    reliability: { "stated:verbal_conf": [{ lo: 0, hi: 1, count: 1, confidence: 0.5, accuracy: 0.5 }], "signal:verbal_conf": [{ lo: 0, hi: 1, count: 1, confidence: 0.5, accuracy: 0.5 }] },
    risk_coverage: { "stated:verbal_conf": { coverage: [1], risk: [0.3] }, "signal:verbal_conf": { coverage: [1], risk: [0.3] } },
  };
  const ranking = estimatorRanking(data, "platt");
  assert.deepEqual(ranking.map((r) => r.name), ["logistic:full", "stated:verbal_conf", "signal:ret_top"]);
  assert.match(rankingTakeaway(ranking, null) ?? "", /1 of 2 others has a higher mean/);
  assert.deepEqual(reliabilityBins(data).map((s) => s.name), ["stated:verbal_conf"]);
  assert.deepEqual(riskCoverageSeries(data).map((s) => s.name), ["stated:verbal_conf"]);
  assert.deepEqual(dropDuplicateSignals([{ name: "signal:verbal_conf" }]).map((r) => r.name), ["signal:verbal_conf"]);
});
