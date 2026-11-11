import assert from "node:assert/strict";
import { test } from "node:test";

import type { ClaimVerdict, Hit } from "../types.ts";
import {
  claimVerdictCounts,
  gaugeAngle,
  gaugeReading,
  gaugeTakeaway,
  passageScores,
  scoreMatrix,
  scoreMatrixTakeaway,
  stageWaterfall,
  verdictShares,
  verdictTakeaway,
  waterfallTakeaway,
} from "./ask.ts";

test("stages are laid end to end in pipeline order", () => {
  const { spans, total } = stageWaterfall({ generate: 2000, triage: 1, retrieve: 199, extra: 10, verify: Number.NaN });
  assert.deepEqual(
    spans.map((s) => [s.stage, s.start, s.end]),
    [
      ["triage", 0, 1],
      ["retrieve", 1, 200],
      ["generate", 200, 2200],
      ["extra", 2200, 2210],
    ],
  );
  assert.equal(total, 2210);
  assert.ok(Math.abs((spans[2]?.share ?? 0) - 2000 / 2210) < 1e-12);
  assert.deepEqual(stageWaterfall(null), { spans: [], total: 0 });
});

const verdict = (label: ClaimVerdict["label"]): ClaimVerdict => ({
  claim: { text: label, cited: [] },
  label,
  p_entail: 0,
  p_neutral: 0,
  p_contra: 0,
  best_hit: null,
  best_sentence: null,
});

test("verdict counts list every label, zeros included", () => {
  assert.deepEqual(claimVerdictCounts([verdict("supported"), verdict("supported"), verdict("contradicted")]), [
    { label: "supported", count: 2 },
    { label: "unsupported", count: 0 },
    { label: "contradicted", count: 1 },
  ]);
});

const hit = (rank: number, score: number): Hit => ({
  passage: { passage_id: `p${rank}`, doc_id: "d", title: `T${rank}`, text: "", source: "pubmed", url: null, meta: {} },
  score,
  rank,
  retriever: "hybrid-ft",
  components: { rrf: score },
});

test("passage scores come back in rank order", () => {
  assert.deepEqual(
    passageScores([hit(2, 0.5), hit(1, 0.9)]).map((p) => [p.rank, p.score]),
    [
      [1, 0.9],
      [2, 0.5],
    ],
  );
});

test("gauge reading keeps the server's decision and the margin to the gate", () => {
  const r = gaugeReading(0.61, 0.78, false);
  assert.equal(r.pointer, 0.61);
  assert.equal(r.gate, 0.78);
  assert.equal(r.released, false);
  assert.ok(Math.abs((r.margin ?? 0) + 0.17) < 1e-12);
  assert.equal(gaugeTakeaway(r, 0.1), "Confidence 0.61 is 0.17 short of the threshold 0.78 for α = 0.10; withheld.");
  assert.equal(gaugeTakeaway(gaugeReading(0.86, 0.78, true), 0.1), "Confidence 0.86 clears the threshold 0.78 for α = 0.10 by 0.08; released.");
});

test("gauge without a threshold or a confidence says so", () => {
  const none = gaugeReading(0.86, null, false);
  assert.equal(none.gate, null);
  assert.equal(none.clearedFrom, null);
  assert.match(gaugeTakeaway(none, 0.05), /no threshold certifies α = 0\.05/);
  const empty = gaugeReading(null, 0.7, undefined);
  assert.equal(empty.pointer, null);
  assert.equal(empty.released, null);
  assert.equal(gaugeTakeaway(empty, 0.1), "No calibrated confidence for this answer.");
  assert.equal(gaugeReading(1.4, -0.2, true).pointer, 1);
  assert.equal(gaugeReading(1.4, -0.2, true).gate, 0);
});

test("gauge angles span the upper half-circle", () => {
  assert.equal(gaugeAngle(0), -Math.PI / 2);
  assert.equal(gaugeAngle(0.5), 0);
  assert.equal(gaugeAngle(1), Math.PI / 2);
  assert.equal(gaugeAngle(2), Math.PI / 2);
});

test("waterfall takeaway names the longest stage", () => {
  const w = stageWaterfall({ triage: 1.4, retrieve: 184, generate: 2310, verify: 412 });
  assert.equal(waterfallTakeaway(w, (s) => s.toUpperCase()), "GENERATE took 2.31 s of 2.91 s (79%); 4 stages timed.");
  assert.equal(waterfallTakeaway({ spans: [], total: 0 }), "No stage timings were reported.");
});

test("verdict shares sum to one and the takeaway reads the counts", () => {
  const s = verdictShares([verdict("supported"), verdict("unsupported"), verdict("unsupported"), verdict("contradicted")]);
  assert.equal(s.total, 4);
  assert.deepEqual(s.rows.map((r) => r.share), [0.25, 0.5, 0.25]);
  assert.equal(verdictTakeaway(s), "1 of 4 claims supported by the cited passages, 2 unsupported, 1 contradicted.");
  assert.equal(verdictTakeaway(verdictShares([])), "The answer made no checkable claim.");
  assert.equal(verdictTakeaway(verdictShares([verdict("supported")])), "1 of 1 claim supported by the cited passages, 0 unsupported, none contradicted.");
});

const hitWith = (rank: number, components: Record<string, number>): Hit => ({ ...hit(rank, components.rrf ?? 0), components });

test("score matrix scales each column from zero on its own and keeps gaps as gaps", () => {
  const m = scoreMatrix([
    hitWith(1, { bm25: 14, bm25_rank: 1, dense: 0.7, dense_rank: 2, rrf: 0.032 }),
    hitWith(2, { dense: 0.35, dense_rank: 1, rrf: 0.016 }),
  ]);
  assert.deepEqual(m.columns.map((c) => [c.key, c.min, c.max]), [
    ["bm25", 0, 14],
    ["dense", 0, 0.7],
    ["rrf", 0, 0.032],
  ]);
  assert.deepEqual(m.rows[0]?.cells.map((c) => [c.value, c.rank, c.length]), [
    [14, 1, 1],
    [0.7, 2, 1],
    [0.032, null, 1],
  ]);
  assert.deepEqual(m.rows[1]?.cells.map((c) => c.length), [null, 0.5, 0.5]);
  assert.equal(m.rows[1]?.index, 2);
  assert.equal(scoreMatrixTakeaway(m), "Passage 1 has the top RRF score; 1 of 2 passages were found by only one of BM25 and dense search.");
  assert.deepEqual(scoreMatrix(null), { columns: [], rows: [] });
  assert.equal(scoreMatrixTakeaway(scoreMatrix([])), "No passage scores were reported.");
});

test("score matrix reports ties instead of picking one", () => {
  const m = scoreMatrix([hitWith(1, { rrf: 0.03 }), hitWith(2, { rrf: 0.03 })]);
  assert.equal(scoreMatrixTakeaway(m), "2 passages tie for the top RRF score.");
});

test("score matrix counts single-retriever passages by rank, since the backend reports both scores", () => {
  const m = scoreMatrix([
    hitWith(1, { bm25: 14, bm25_rank: 1, dense: 0.7, dense_rank: 2, rrf: 0.032 }),
    hitWith(2, { bm25: 3.1, dense: 0.61, dense_rank: 4, rrf: 0.0156 }),
    hitWith(3, { bm25: 9, bm25_rank: 2, dense: 0.2, rrf: 0.016 }),
  ]);
  assert.deepEqual(m.rows.map((r) => r.cells.map((c) => c.inList)), [
    [true, true, true],
    [false, true, true],
    [true, false, true],
  ]);
  assert.equal(scoreMatrixTakeaway(m), "Passage 1 has the top RRF score; 2 of 3 passages were found by only one of BM25 and dense search.");
});

test("score matrix bars measure distance from zero when a column has negative values", () => {
  const m = scoreMatrix([hitWith(1, { rerank: 3, rrf: 0.03 }), hitWith(2, { rerank: -1, rrf: 0.02 }), hitWith(3, { rerank: 0, rrf: 0.01 })]);
  const rerank = m.columns.findIndex((c) => c.key === "rerank");
  assert.equal(m.columns[rerank]?.zero, 0.25);
  assert.deepEqual(m.rows.map((r) => [r.cells[rerank]?.start, r.cells[rerank]?.length]), [
    [0.25, 0.75],
    [0, 0.25],
    [0.25, 0],
  ]);
});

test("gauge takeaway never claims a shortfall or decision the server did not report", () => {
  assert.equal(gaugeTakeaway(gaugeReading(0.7, 0.78, undefined), 0.1), "Confidence 0.70 against the threshold 0.78 for α = 0.10.");
  assert.equal(
    gaugeTakeaway(gaugeReading(0.9, 0.78, false), 0.1),
    "Confidence 0.90 is at or above the threshold 0.78 for α = 0.10, but the server withheld the answer.",
  );
});
