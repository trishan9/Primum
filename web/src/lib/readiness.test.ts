import assert from "node:assert/strict";
import { test } from "node:test";

import { needsFinetunedEncoder, readinessGaps } from "./readiness.ts";

test("retrievers built on the fine-tuned encoder are recognised", () => {
  for (const name of ["dense-ft", "hybrid-ft", "hybrid-ft+rerank", "dense-ft+concepts"]) assert.ok(needsFinetunedEncoder(name), name);
  for (const name of ["bm25", "dense-base", "hybrid-base", "dense-biomed", "bm25+concepts"]) assert.ok(!needsFinetunedEncoder(name), name);
});

test("an untrained encoder is a gap when the retriever in use needs it", () => {
  // What a fresh installation reports: a BM25 index for kb, no fine-tuned encoder.
  const health = { api_key: true, finetuned_encoder: false, indexes: { kb: ["bm25"] }, deployment_bundle: true };
  assert.deepEqual(readinessGaps(health, "hybrid-ft"), ["encoder"]);
  assert.deepEqual(readinessGaps(health, "bm25"), []);
  assert.deepEqual(readinessGaps(health, null), []);
  assert.deepEqual(readinessGaps({ ...health, finetuned_encoder: true }, "hybrid-ft"), []);
});

test("missing key, indexes and bundle are each reported", () => {
  const health = { api_key: false, finetuned_encoder: true, indexes: {}, deployment_bundle: false };
  assert.deepEqual(readinessGaps(health, "bm25"), ["api_key", "index", "bundle"]);
});
