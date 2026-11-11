import assert from "node:assert/strict";
import { test } from "node:test";

import {
  accuracyByCondition,
  accuracyTakeaway,
  confidenceBins,
  confusionByCondition,
  pairedConditionDiffs,
  recordConditions,
  verbalConfidenceHistogram,
} from "./pubmedqa.ts";

test("stated confidences bin per distinct value up to 12 values, else in tenths", () => {
  const few = confidenceBins([0.6, 0.8, 0.8, 0.4]);
  assert.equal(few.distinct, true);
  assert.deepEqual(few.bins.map((b) => [b.lo, b.count]), [
    [0.4, 1],
    [0.6, 1],
    [0.8, 2],
  ]);
  const many = confidenceBins(Array.from({ length: 13 }, (_, i) => 0.3 + i * 0.05));
  assert.equal(many.distinct, false);
  assert.equal(many.bins.length, 10);
  assert.equal(many.bins[3]?.count, 2); // 0.30 and 0.35
  assert.equal(many.bins.reduce((s, b) => s + b.count, 0), 13);
});

const rec = (condition: string, gold: string, decision: string | null, conf: number | null, extra: Record<string, unknown> = {}) => ({
  qid: `${condition}-${gold}-${decision}-${conf}`,
  condition,
  gold,
  decision,
  correct: decision === gold,
  features: conf === null ? {} : { verbal_conf: conf },
  ...extra,
});

const DATA = {
  summary: {
    pubmedqa: {
      rag: { n: 4, n_failed: 1, accuracy: { mean: 0.75, ci: [0.5, 1] }, macro_f1: { mean: 0.6, ci: [0.4, 0.8] }, supported_claim_rate: 0.3, gold_in_hits_rate: 0.9 },
      closed_book: { n: 4, n_failed: 0, accuracy: { mean: 0.5, ci: [0.25, 0.75] }, supported_claim_rate: 0 },
    },
  },
  paired_tests: { rag_vs_closed_book: { accuracy_diff: { mean: 0.25, ci: [0.1, 0.4] }, p_value: 0.01, n_pairs: 4 }, junk: 3 },
  records: [
    rec("rag", "yes", "yes", 0.9),
    rec("rag", "yes", "no", 0.6),
    rec("rag", "no", "no", 1),
    rec("rag", "maybe", "maybe", null),
    { qid: "x", condition: "rag", gold: "yes", error: "LLMInvalidOutput" },
    rec("closed_book", "yes", "yes", 0.8),
  ],
};

test("accuracy by condition comes back in method order with its intervals", () => {
  const rows = accuracyByCondition(DATA);
  assert.deepEqual(
    rows.map((r) => r.condition),
    ["closed_book", "rag"],
  );
  const rag = rows[1];
  assert.deepEqual(rag?.accuracy, { value: 0.75, lo: 0.5, hi: 1 });
  assert.equal(rag?.nFailed, 1);
  assert.equal(rag?.label, "With retrieval");
  assert.equal(rows[0]?.macroF1, null);
  assert.equal(rows[0]?.goldInHitsRate, null);
  assert.deepEqual(accuracyByCondition({}), []);
});

test("paired differences parse a and b from the key", () => {
  const diffs = pairedConditionDiffs(DATA);
  assert.equal(diffs.length, 1);
  assert.deepEqual(diffs[0], { key: "rag_vs_closed_book", a: "rag", b: "closed_book", diff: { value: 0.25, lo: 0.1, hi: 0.4 }, p: 0.01, nPairs: 4 });
});

test("confusion counts gold against decision and excludes failed generations", () => {
  const [closed, rag] = confusionByCondition(DATA);
  assert.equal(closed?.n, 1);
  assert.equal(rag?.n, 4);
  assert.equal(rag?.failed, 1);
  assert.equal(rag?.goldTotals.yes, 2);
  const yesNo = rag?.cells.find((c) => c.gold === "yes" && c.decision === "no");
  assert.equal(yesNo?.count, 1);
  assert.equal(yesNo?.shareOfGold, 0.5);
  assert.equal(rag?.cells.length, 9);
  assert.equal(rag?.cells.reduce((s, c) => s + c.count, 0), 4);
});

test("verbal confidence histogram splits right and wrong answers", () => {
  const h = verbalConfidenceHistogram(DATA, "rag", 10);
  assert.ok(h);
  assert.equal(h.nCorrect, 2);
  assert.equal(h.nWrong, 1);
  assert.equal(h.missing, 1);
  // Three distinct stated values, so one bin per value.
  assert.equal(h.distinct, true);
  assert.deepEqual(h.bins.map((b) => [b.lo, b.correct, b.wrong]), [
    [0.6, 0, 1],
    [0.9, 1, 0],
    [1, 1, 0],
  ]);
  assert.equal(verbalConfidenceHistogram(DATA, "oracle"), null);
  assert.deepEqual(recordConditions(DATA), ["closed_book", "rag"]);
});

test("the takeaway is templated from the paired tests", () => {
  const text = accuracyTakeaway(accuracyByCondition(DATA), pairedConditionDiffs(DATA));
  assert.equal(text, "With retrieval vs closed book: 75.0% against 50.0% (+25.0 points, 95% CI +10.0 to +40.0 points, 4 paired questions).");
  const single = accuracyTakeaway(accuracyByCondition({ summary: { pubmedqa: { rag: DATA.summary.pubmedqa.rag } } }), []);
  assert.equal(single, "One condition was run, with retrieval: 75.0% of 4 answers correct (95% CI 50.0% to 100.0%).");
  assert.equal(accuracyTakeaway([], []), null);
});

import { confidenceTakeaway, confusionTakeaway, histogramValues, verbalConfidenceSummaries } from "./pubmedqa.ts";

test("confusion takeaway: overall agreement and the least-matched gold label", () => {
  const data = {
    records: [
      rec("rag", "yes", "yes", 0.9),
      rec("rag", "yes", "yes", 0.9),
      rec("rag", "no", "no", 0.8),
      rec("rag", "maybe", "yes", 0.7),
      rec("rag", "maybe", "yes", 0.6),
      rec("rag", "maybe", "maybe", 0.5),
    ],
  };
  const c = confusionByCondition(data)[0];
  assert.ok(c);
  assert.equal(
    confusionTakeaway(c),
    "66.7% of 6 scored answers match the gold label. Gold “maybe” is matched least often: 33% of 3; it most often said “yes” instead (67%).",
  );
});

test("confidence summaries split right and wrong answers; takeaway says whether middle halves overlap", () => {
  const s = verbalConfidenceSummaries(DATA, "rag");
  assert.equal(s.correct?.n, 2);
  assert.equal(s.wrong?.n, 1);
  assert.equal(s.correct?.median, 0.95);
  assert.equal(
    confidenceTakeaway(s),
    "Right answers (2) state a median confidence of 0.95, middle half 0.93 to 0.97; wrong answers (1) state a median confidence of 0.60, middle half 0.60 to 0.60, so the middle halves do not overlap.",
  );
  assert.equal(confidenceTakeaway({ condition: "x", correct: null, wrong: null }), null);
});

test("histogram values as counts or as a share of each group", () => {
  const h = verbalConfidenceHistogram(DATA, "rag", 10);
  assert.ok(h);
  const counts = histogramValues(h, "count");
  const shares = histogramValues(h, "share");
  assert.equal(counts.reduce((s, b) => s + b.correct, 0), h.nCorrect);
  assert.ok(Math.abs(shares.reduce((s, b) => s + b.correct, 0) - 1) < 1e-9);
  assert.ok(Math.abs(shares.reduce((s, b) => s + b.wrong, 0) - 1) < 1e-9);
});
