import assert from "node:assert/strict";
import { test } from "node:test";

import {
  findLevelRecords,
  findLossCurve,
  findMetricTables,
  findReliability,
  findRiskCoverage,
  generatorName,
  isProbabilityLike,
  levelsAsRows,
  metricDirection,
  metricLabel,
  readProvenance,
  readStat,
  readUnmeasured,
  runLabel,
  runPathPrefix,
} from "./results.ts";

test("readStat accepts numbers, mean with ci pair, and lo/hi fields", () => {
  assert.deepEqual(readStat(0.5), { value: 0.5, lo: null, hi: null });
  assert.deepEqual(readStat({ mean: 0.5, ci: [0.4, 0.6] }), { value: 0.5, lo: 0.4, hi: 0.6 });
  assert.deepEqual(readStat({ value: 0.5, ci_low: 0.4, ci_high: 0.6, n: 20 }), { value: 0.5, lo: 0.4, hi: 0.6 });
  assert.equal(readStat({ mean: 0.5, something_else: 1 }), null);
  assert.equal(readStat("0.5"), null);
});

test("finds a nested systems-by-metrics table and ignores meta", () => {
  const tables = findMetricTables({
    meta: { created: "2026-11-10", config: { a: { x: 1 }, b: { x: 2 } } },
    nfcorpus: {
      bm25: { "ndcg@10": { mean: 0.3, ci: [0.28, 0.32] }, "recall@100": 0.25 },
      "dense-ft": { "ndcg@10": { mean: 0.35, ci: [0.33, 0.37] }, "recall@100": 0.3, tests: { p_vs_bm25: 0.002 } },
    },
  });
  assert.equal(tables.length, 1);
  assert.deepEqual(tables[0]?.path, ["nfcorpus"]);
  assert.deepEqual(tables[0]?.metrics, ["ndcg@10", "recall@100", "tests.p_vs_bm25"]);
  assert.equal(tables[0]?.rows[1]?.cells["ndcg@10"]?.hi, 0.37);
});

test("finds a table given as an array of named rows", () => {
  const tables = findMetricTables({
    models: [
      { name: "logistic", auroc: { mean: 0.7, lo: 0.65, hi: 0.75 }, ece: 0.05 },
      { name: "mlp", auroc: { mean: 0.72, lo: 0.67, hi: 0.77 }, ece: 0.06 },
    ],
  });
  assert.equal(tables.length, 1);
  assert.deepEqual(tables[0]?.rows.map((r) => r.name), ["logistic", "mlp"]);
});

test("a single system's metrics are not mistaken for a table of systems", () => {
  const tables = findMetricTables({ only: { "ndcg@10": { mean: 0.3, lo: 0.2, hi: 0.4 }, "mrr@10": { mean: 0.5, lo: 0.4, hi: 0.6 } } });
  assert.equal(tables.length, 0);
});

test("a one-condition run summary is a one-row table when the caller allows it", () => {
  const summary = {
    pubmedqa: {
      rag: { n: 996, n_failed: 4, accuracy: { mean: 0.8, ci: [0.77, 0.82] }, macro_f1: { mean: 0.68, ci: [0.64, 0.71] } },
    },
  };
  assert.equal(findMetricTables(summary).length, 0);
  const tables = findMetricTables(summary, 5, 1);
  assert.equal(tables.length, 1);
  assert.deepEqual(tables[0]?.path, ["pubmedqa"]);
  assert.deepEqual(tables[0]?.rows.map((r) => r.name), ["rag"]);
  assert.equal(tables[0]?.rows[0]?.cells.accuracy?.value, 0.8);
});

test("reads reliability bins in row and column layouts", () => {
  const rows = findReliability({
    reliability: { mlp: [{ confidence: 0.2, accuracy: 0.25, count: 40 }, { confidence: 0.8, accuracy: 0.75, count: 60 }] },
  });
  assert.deepEqual(rows[0], { name: "mlp", points: [{ x: 0.2, y: 0.25, n: 40 }, { x: 0.8, y: 0.75, n: 60 }] });
  const columns = findReliability({ reliability_bins: { mean_conf: [0.1, 0.9], frac_correct: [0.2, 0.8], counts: [5, 7] } });
  assert.equal(columns[0]?.points[1]?.n, 7);
});

test("reads risk-coverage curves and sorts them by coverage", () => {
  const curves = findRiskCoverage({ risk_coverage: { mlp: { coverage: [1, 0.5, 0.1], risk: [0.3, 0.2, 0.05] } } });
  assert.deepEqual(curves[0]?.points.map((p) => p.x), [0.1, 0.5, 1]);
});

test("level records come from alpha keys on the path or an alpha field", () => {
  const nested = findLevelRecords({
    methods: {
      certified: { "0.05": { violation_rate: 0.04, violation_ci: [0.02, 0.08], coverage: 0.3 }, "0.10": { violation_rate: 0.07, coverage: 0.5 } },
      plugin: { "0.05": { violation_rate: 0.4, coverage: 0.45 } },
    },
  });
  assert.equal(nested.length, 3);
  assert.deepEqual(nested[0], {
    method: "certified",
    alpha: 0.05,
    violation: { value: 0.04, lo: 0.02, hi: 0.08 },
    coverage: { value: 0.3, lo: null, hi: null },
    risk: null,
    certifiedRate: null,
  });
  const flat = findLevelRecords({ rows: [{ method: "certified", alpha: 0.2, violation_rate: { mean: 0.05, ci: [0.03, 0.09] }, mean_coverage: 0.8, certified_rate: 0.41 }] });
  assert.equal(flat[0]?.certifiedRate, 0.41);
  assert.equal(flat[0]?.method, "certified");
  assert.equal(flat[0]?.violation?.hi, 0.09);
  assert.equal(flat[0]?.coverage?.value, 0.8);
});

test("loss curves are read from steps and loss columns or a bare list", () => {
  assert.equal(findLossCurve({ steps: [10, 20], loss: [2.1, 1.4] })?.points[1]?.x, 20);
  assert.equal(findLossCurve({ loss_curve: [3, 2, 1] })?.points.length, 3);
  assert.equal(findLossCurve({ nothing: 1 }), null);
});

test("provenance and names", () => {
  assert.deepEqual(readProvenance({ meta: { created: "2026-11-10", n: 1000, config: { delta: 0.1 }, model_ids: { llm: "claude-haiku-5-5" } } }), {
    created: "2026-11-10",
    n: 1000,
    delta: 0.1,
    models: ["llm: claude-haiku-5-5"],
  });
  assert.equal(metricLabel("ndcg@10"), "nDCG@10");
  assert.equal(metricLabel("e_aurc"), "E-AURC");
  assert.equal(metricDirection("ece"), "lower");
  assert.equal(metricDirection("recall@100"), "higher");
  assert.equal(metricDirection("p_vs_bm25"), null);
  assert.equal(metricDirection("ndcg@10_p_holm"), null);
  assert.equal(metricDirection("brier_diff"), "lower");
  assert.equal(metricDirection("n_questions"), null);
  assert.ok(isProbabilityLike("ndcg@10_p") && isProbabilityLike("brier_p") && isProbabilityLike("p_value"));
  assert.ok(!isProbabilityLike("top1_accuracy") && !isProbabilityLike("mrr@10"));
});

test("a table keyed by risk level is turned to one row per level", () => {
  const [table] = findMetricTables({
    prediction_sets: {
      lac: {
        set_coverage: { "0.05": { mean: 0.96, ci: [0.92, 0.98] }, "0.10": { mean: 0.91, ci: [0.88, 0.94] } },
        mean_set_size: { "0.05": { mean: 2.4, ci: [2.1, 2.5] }, "0.10": { mean: 1.9, ci: [1.8, 2.1] } },
      },
    },
  });
  assert.ok(table);
  const turned = levelsAsRows(table);
  assert.deepEqual(turned.metrics, ["set_coverage", "mean_set_size"]);
  assert.deepEqual(turned.rows.map((r) => r.name), ["α = 0.05", "α = 0.10"]);
  assert.equal(turned.rows[1]?.cells["mean_set_size"]?.value, 1.9);
  const plain = { path: [], metrics: ["auroc"], rows: [{ name: "a", cells: {} }] };
  assert.equal(levelsAsRows(plain), plain);
});

test("run labels name the generator, and the sample design only when a generator has several runs", () => {
  assert.equal(generatorName("claude-haiku-5-5"), "Haiku 5.5");
  assert.equal(generatorName("some-other-model"), "some-other-model");
  assert.equal(generatorName(null), "Unknown generator");
  const runs = [
    { run: "main", generator: "claude-haiku-5-5", n: 10, self_consistency_samples: 5 },
    { run: "b", generator: "claude-sonnet-5-5", n: 10, self_consistency_samples: 1 },
    { run: "c", generator: "claude-sonnet-5-5", n: 3, self_consistency_samples: 5 },
  ];
  assert.deepEqual(
    runs.map((r) => runLabel(r, runs)),
    ["Haiku 5.5", "Sonnet 5.5, 1 sample, 10 questions", "Sonnet 5.5, 5 samples, 3 questions"],
  );
  assert.equal(runPathPrefix("main"), "");
  assert.equal(runPathPrefix("sonnet"), "sonnet/");
});

test("readUnmeasured lists the features and feature sets RQ2 dropped", () => {
  assert.deepEqual(readUnmeasured({ meta: { unavailable_features: ["sc_agreement"], skipped_feature_sets: ["full"] } }), {
    features: ["sc_agreement"],
    featureSets: ["full"],
  });
  assert.deepEqual(readUnmeasured({ meta: {} }), { features: [], featureSets: [] });
});
