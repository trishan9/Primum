import assert from "node:assert/strict";
import { test } from "node:test";

import {
  devBeforeAfter,
  lossSeries,
  metricBySystem,
  pairedBaselines,
  pairedDiffs,
  perQueryDistribution,
  recallAtK,
  retrievalDatasets,
  retrievalMetrics,
  retrievalSystems,
  rq1Interval,
} from "./rq1.ts";

const est = (mean: number, lo: number, hi: number) => ({ mean, ci: [lo, hi] });

const DATA = {
  meta: { systems: ["bm25", "dense-ft", "hybrid-ft"], inference: { ci: "95% percentile bootstrap over queries" } },
  pubmedqa: {
    "hybrid-ft": { "recall@20": est(0.97, 0.96, 0.98), "recall@1": est(0.8, 0.78, 0.82), "mrr@10": est(0.85, 0.83, 0.87) },
    bm25: { "recall@1": est(0.77, 0.75, 0.8), "recall@20": est(0.95, 0.94, 0.97), "mrr@10": est(0.83, 0.81, 0.85) },
  },
  nfcorpus: { bm25: { "ndcg@10": est(0.32, 0.29, 0.35) } },
  paired_tests: {
    pubmedqa: { "vs_bm25": { "hybrid-ft": { "recall@1_diff": est(0.03, 0.01, 0.05), "recall@1_p": 0.002, "recall@1_p_holm": 0.01 } } },
  },
  per_query: { pubmedqa: { qids: ["a", "b", "c", "d"], bm25: { "recall@1": [1, 0, 1, 1] }, "hybrid-ft": { "recall@1": [1, 1, 1, 0] } } },
};

test("datasets, systems and metrics follow the file", () => {
  assert.deepEqual(retrievalDatasets(DATA), ["pubmedqa", "nfcorpus"]);
  assert.deepEqual(retrievalSystems(DATA, "pubmedqa"), ["bm25", "hybrid-ft"]);
  assert.deepEqual(retrievalMetrics(DATA, "pubmedqa"), ["recall@1", "recall@20", "mrr@10"]);
  assert.equal(rq1Interval(DATA).kind, "bootstrap");
});

test("metric by system keeps the interval", () => {
  assert.deepEqual(metricBySystem(DATA, "nfcorpus", "ndcg@10"), [{ system: "bm25", estimate: { value: 0.32, lo: 0.29, hi: 0.35 } }]);
  assert.deepEqual(metricBySystem(DATA, "nfcorpus", "mrr@10"), []);
});

test("recall@k is sorted by k per system", () => {
  const curves = recallAtK(DATA, "pubmedqa");
  assert.deepEqual(
    curves.map((c) => [c.system, c.points.map((p) => p.k)]),
    [
      ["bm25", [1, 20]],
      ["hybrid-ft", [1, 20]],
    ],
  );
  assert.deepEqual(recallAtK(DATA, "nfcorpus"), []);
});

test("per-query distributions summarise each system", () => {
  const [bm25] = perQueryDistribution(DATA, "pubmedqa", "recall@1");
  assert.equal(bm25?.system, "bm25");
  assert.equal(bm25?.summary.mean, 0.75);
  assert.equal(bm25?.zeroShare, 0.25);
  assert.equal(perQueryDistribution(DATA, "nfcorpus", "ndcg@10").length, 0);
});

test("paired differences against a baseline", () => {
  assert.deepEqual(pairedBaselines(DATA, "pubmedqa"), ["bm25"]);
  assert.deepEqual(pairedDiffs(DATA, "pubmedqa", "bm25"), [{ system: "hybrid-ft", metric: "recall@1", diff: { value: 0.03, lo: 0.01, hi: 0.05 }, p: 0.002, pHolm: 0.01 }]);
});

test("loss series is sorted and unsmoothed; dev metrics pair before and after", () => {
  const training = {
    loss_curve: [
      { step: 16, loss: 1.1 },
      { step: 8, loss: 1.3 },
      { step: 24, loss: "x" },
    ],
    loss_at_chance: 3.8,
    dev: { before: { n_queries: 500, pool_size: 651, "mrr@10": 0.97 }, after: { n_queries: 500, pool_size: 651, "mrr@10": 0.98 } },
  };
  const loss = lossSeries(training);
  assert.deepEqual(loss?.points, [
    { step: 8, loss: 1.3 },
    { step: 16, loss: 1.1 },
  ]);
  assert.equal(loss?.chance, 3.8);
  assert.deepEqual(devBeforeAfter(training), { rows: [{ metric: "mrr@10", before: 0.97, after: 0.98 }], nQueries: 500 });
  assert.equal(lossSeries({}), null);
});

/* ---------- readings and helpers ---------- */

import {
  deployedRetriever,
  diffSide,
  estimateExtent,
  lossSummary,
  lossTakeaway,
  paddedDomain,
  pairedTakeaway,
  perQueryTakeaway,
  recallTakeaway,
  isBinaryMetric,
  stripJitter,
} from "./rq1.ts";

test("a metric whose per-query values are all 0 or 1 is binary", () => {
  assert.equal(isBinaryMetric([0, 1, 1, 0]), true);
  assert.equal(isBinaryMetric([0, 0.5, 1]), false);
  assert.equal(isBinaryMetric([]), false);
});

test("deployed retriever is read from meta.config", () => {
  assert.equal(deployedRetriever({ meta: { config: { default_retriever: "hybrid-ft" } } }), "hybrid-ft");
  assert.equal(deployedRetriever({}), null);
});

test("padded domain widens to the step and respects the clamp", () => {
  assert.deepEqual(paddedDomain([0.671, 0.986], 0.1, [0, 1]), [0.6, 1]);
  assert.deepEqual(paddedDomain([-0.02, 0.05], 0.05), [-0.05, 0.05]);
  assert.deepEqual(paddedDomain([0.3, 0.3], 0.1), [0.3, 0.4]);
  assert.deepEqual(paddedDomain([], 0.1, [0, 1]), [0, 1]);
  assert.deepEqual(estimateExtent([{ value: 0.5, lo: 0.4, hi: 0.6 }, { value: 0.7, lo: null, hi: null }]), [0.5, 0.4, 0.6, 0.7, 0.7, 0.7]);
});

test("recall takeaway names the leader at the smallest k, the baseline, and the spread at the largest", () => {
  const text = recallTakeaway(recallAtK(DATA, "pubmedqa"));
  assert.equal(
    text,
    "At k = 1, hybrid-ft ranks the gold abstract first for 80.0% of questions (CI 78.0% to 82.0%), against 77.0% for bm25; by k = 20 every retriever is between 95.0% and 97.0%.",
  );
  assert.equal(recallTakeaway([]), null);
  const fromFive = recallAtK(DATA, "pubmedqa").map((c) => ({ ...c, points: c.points.filter((p) => p.k >= 5) }));
  assert.match(recallTakeaway(fromFive) ?? "", /^At k = 20, hybrid-ft places the gold abstract in its top 20 for/);
});

test("diff side reads the interval against zero", () => {
  assert.equal(diffSide({ value: 0.1, lo: 0.01, hi: 0.2 }), "above");
  assert.equal(diffSide({ value: -0.1, lo: -0.2, hi: -0.01 }), "below");
  assert.equal(diffSide({ value: 0.01, lo: -0.02, hi: 0.03 }), "spans");
  assert.equal(diffSide({ value: 0.01, lo: null, hi: null }), "unknown");
});

test("paired takeaway counts interval sides and names the largest gain with its Holm p", () => {
  const text = pairedTakeaway(pairedDiffs(DATA, "pubmedqa", "bm25"), "bm25", "Recall@1");
  assert.equal(
    text,
    "Against bm25 on Recall@1, of 1 retrievers the unadjusted 95% CI of the difference lies 1 wholly above zero, 0 wholly below, 0 spanning zero. After Holm adjustment, 1 of 1 differ from bm25 at p < 0.05. Largest gain: hybrid-ft, +0.030, Holm-adjusted p = 0.010.",
  );
  assert.equal(pairedTakeaway([], "bm25", "x"), null);
});

test("strip jitter is deterministic and stays in [-1, 1]", () => {
  const xs = Array.from({ length: 500 }, (_, i) => stripJitter(i));
  assert.ok(xs.every((x) => x >= -1 && x <= 1));
  assert.equal(stripJitter(7), stripJitter(7));
  assert.equal(stripJitter(0), -1);
});

test("per-query takeaway reports the median range and the share of zeros", () => {
  const rows = perQueryDistribution(DATA, "pubmedqa", "recall@1");
  assert.equal(perQueryTakeaway(rows, "Recall@1"), "Median Recall@1 per query is 1.00 for every retriever, over 4 queries; 25.0% of queries score exactly zero, depending on the retriever.");
  assert.equal(perQueryTakeaway([], "x"), null);
});

test("loss summary averages the first and last tenth; takeaway adds chance and dev MRR", () => {
  const points = Array.from({ length: 20 }, (_, i) => ({ step: (i + 1) * 8, loss: 2 - i * 0.1 }));
  const series = { points, chance: 3.8, steps: 160, wallTimeS: 10 };
  const s = lossSummary(series);
  assert.ok(s);
  assert.equal(s.window, 2);
  assert.ok(Math.abs(s.start - 1.95) < 1e-9);
  assert.ok(Math.abs(s.end - 0.15) < 1e-9);
  const text = lossTakeaway(series, { rows: [{ metric: "mrr@10", before: 0.9, after: 0.95 }], nQueries: 500 });
  assert.equal(
    text,
    "Mean loss fell from 1.95 over the first 2 logged points to 0.15 over the last 2 (an encoder ranking at random would sit near 3.80). On 500 held-out dev queries, MRR@10 went from 0.900 to 0.950.",
  );
  assert.equal(lossSummary({ points: [], chance: null, steps: null, wallTimeS: null }), null);
});
