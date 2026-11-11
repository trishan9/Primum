import assert from "node:assert/strict";
import { test } from "node:test";

import type { Hit } from "../types.ts";
import {
  bumpSegments,
  bumpTakeaway,
  overlapMatrix,
  overlapPairs,
  overlapTakeaway,
  rankBump,
  scoreProfiles,
  scoreTakeaway,
  widestDisagreement,
} from "./retrieval.ts";

const hit = (id: string, rank: number, retriever: string, score = 1 / rank): Hit => ({
  passage: { passage_id: id, doc_id: id, title: id.toUpperCase(), text: "", source: "pubmed", url: null, meta: {} },
  score,
  rank,
  retriever,
  components: {},
});

const RESULTS = {
  bm25: [hit("a", 1, "bm25"), hit("b", 2, "bm25"), hit("c", 3, "bm25")],
  dense: [hit("b", 1, "dense"), hit("d", 2, "dense"), hit("a", 3, "dense")],
};

test("rank bump lists every passage with its rank per retriever", () => {
  const series = rankBump(RESULTS, ["bm25", "dense"]);
  assert.deepEqual(
    series.map((s) => [s.passageId, s.ranks.map((r) => r.rank), s.foundBy]),
    [
      ["a", [1, 3], 2],
      ["b", [2, 1], 2],
      ["d", [null, 2], 1],
      ["c", [3, null], 1],
    ],
  );
});

test("rank bump cuts each list to the top k and ranks by position", () => {
  // The server's `rank` field is ignored in favour of the position, which the columns print.
  const shifted = { x: [hit("a", 7, "x"), hit("b", 9, "x"), hit("c", 11, "x")] };
  const series = rankBump(shifted, ["x"], 2);
  assert.deepEqual(
    series.map((s) => [s.passageId, s.ranks[0]?.rank]),
    [
      ["a", 1],
      ["b", 2],
    ],
  );
});

test("rank bump skips retrievers missing from the response", () => {
  const series = rankBump(RESULTS, ["bm25", "ghost", "dense"]);
  assert.deepEqual(series[0]?.ranks.map((r) => r.retriever), ["bm25", "dense"]);
});

test("segments join only neighbouring columns that both returned the passage", () => {
  const three = { ...RESULTS, third: [hit("a", 1, "third")] };
  const segments = bumpSegments(rankBump(three, ["bm25", "dense", "third"]));
  assert.deepEqual(
    segments.map((s) => `${s.passageId}:${s.from.retriever}${s.from.rank}->${s.to.retriever}${s.to.rank}`),
    ["a:bm251->dense3", "a:dense3->third1", "b:bm252->dense1"],
  );
});

test("widest disagreement needs two placements and a real spread", () => {
  const wide = widestDisagreement(rankBump(RESULTS, ["bm25", "dense"]));
  assert.deepEqual(wide && [wide.passageId, wide.best.retriever, wide.worst.rank, wide.spread], ["a", "bm25", 3, 2]);
  const same = { x: [hit("a", 1, "x")], y: [hit("a", 1, "y")] };
  assert.equal(widestDisagreement(rankBump(same, ["x", "y"])), null);
});

test("bump takeaway counts agreement from the series", () => {
  const text = bumpTakeaway(rankBump(RESULTS, ["bm25", "dense"]), 2, 3);
  assert.match(text, /^4 distinct passages across 2 top-3 lists: 2 in every list, 2 in only one\./);
  assert.match(text, /bm25 ranks 1 and dense ranks 3/);
  assert.equal(bumpTakeaway([], 2, 3), "No retriever returned a passage.");
});

test("overlap is symmetric with a full diagonal", () => {
  const m = overlapMatrix(RESULTS, ["bm25", "dense"]);
  const get = (a: string, b: string) => m.find((o) => o.a === a && o.b === b);
  assert.equal(get("bm25", "bm25")?.jaccard, 1);
  assert.equal(get("bm25", "dense")?.shared, 2);
  assert.equal(get("dense", "bm25")?.jaccard, 2 / 4);
});

test("overlap respects k and pairs are unordered, diagonal-free", () => {
  const m = overlapMatrix(RESULTS, ["bm25", "dense"], 1);
  assert.equal(m.find((o) => o.a === "bm25" && o.b === "dense")?.shared, 0);
  const pairs = overlapPairs(overlapMatrix(RESULTS, ["bm25", "dense"]), ["bm25", "dense"]);
  assert.deepEqual(
    pairs.map((p) => [p.a, p.b, p.shared]),
    [["bm25", "dense", 2]],
  );
});

test("overlap takeaway names the most and least alike pairs", () => {
  const three = { ...RESULTS, z: [hit("q", 1, "z"), hit("a", 2, "z")] };
  const order = ["bm25", "dense", "z"];
  const text = overlapTakeaway(overlapPairs(overlapMatrix(three, order), order), 3);
  assert.equal(text, "Most alike: bm25 and dense, sharing 2 of their top 3. Least alike: bm25 and z, sharing 1.");
  assert.equal(overlapTakeaway([], 3), "Choose at least two retrievers to compare their lists.");
});

test("score profiles find the steepest fall as a share of the list's own range", () => {
  const scored = {
    bm25: [hit("a", 1, "bm25", 22), hit("b", 2, "bm25", 21), hit("c", 3, "bm25", 16), hit("d", 4, "bm25", 14)],
    rrf: [hit("a", 1, "rrf", 0.033), hit("b", 2, "rrf", 0.032), hit("c", 3, "rrf", 0.031), hit("d", 4, "rrf", Number.NaN)],
  };
  const [bm25, rrf] = scoreProfiles(scored, ["bm25", "rrf"]);
  assert.deepEqual(bm25?.steepest, { afterRank: 2, drop: 5, share: 5 / 8 });
  assert.equal(bm25?.min, 14);
  assert.equal(rrf?.points.length, 3, "a non-finite score is left out");
  assert.equal(scoreTakeaway([bm25!, rrf!]).startsWith("bm25 has the sharpest step: 63% of its score range falls between ranks 2 and 3."), true);
  assert.match(scoreTakeaway([bm25!, rrf!]), /rrf falls most evenly \(largest step 50% of its range\)/);
});

test("score takeaway with nothing to describe", () => {
  assert.equal(scoreTakeaway(scoreProfiles({ x: [hit("a", 1, "x")] })), "Too few scored passages to describe how the scores fall.");
});
