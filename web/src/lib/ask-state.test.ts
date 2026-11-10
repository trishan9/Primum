import assert from "node:assert/strict";
import { test } from "node:test";

import { askReducer, initialAskState, outcomeOf, type AskState } from "./ask-state.ts";
import type { ConfidenceEvent, PipelineEvent } from "./types.ts";

const request = { question: "q?", alpha: 0.1, mode: "chat" as const, retriever: null, benchmarkQid: null };

function run(events: PipelineEvent[], end = true): AskState {
  let state = askReducer(initialAskState, { type: "start", request, origin: "live" });
  for (const event of events) state = askReducer(state, { type: "event", event });
  return end ? askReducer(state, { type: "end" }) : state;
}

const confidence = (answered: boolean): ConfidenceEvent => ({
  features: {},
  p_correct: answered ? 0.9 : 0.4,
  alpha: 0.1,
  delta: 0.1,
  threshold: 0.7,
  answered,
});

test("tokens accumulate and stages record latency", () => {
  const state = run(
    [
      { type: "stage", data: { name: "generate", status: "start" } },
      { type: "token", data: { text: "Hel" } },
      { type: "token", data: { text: "lo" } },
      { type: "stage", data: { name: "generate", status: "done", ms: 812 } },
    ],
    false,
  );
  assert.equal(state.streamed, "Hello");
  assert.deepEqual(state.stages.generate, { status: "done", ms: 812 });
  assert.equal(outcomeOf(state), "pending");
});

test("the concepts event records whether the concept dictionary exists", () => {
  const missing = run([{ type: "concepts", data: { concepts: [], available: false } }], false);
  assert.equal(missing.conceptsAvailable, false);
  const present = run([{ type: "concepts", data: { concepts: [], available: true } }], false);
  assert.equal(present.conceptsAvailable, true);
  const unstated = run([{ type: "concepts", data: { concepts: [] } }], false);
  assert.equal(unstated.conceptsAvailable, null);
});

test("an unanswered confidence report is an abstention, not an error", () => {
  const state = run([
    { type: "answer", data: { text: "draft" } },
    { type: "confidence", data: confidence(false) },
  ]);
  assert.equal(state.phase, "done");
  assert.equal(outcomeOf(state), "withheld");
  assert.equal(state.answer?.text, "draft");
});

test("emergency triage ends the run without an answer", () => {
  const state = run([
    { type: "stage", data: { name: "triage", status: "done", ms: 1 } },
    { type: "triage", data: { level: "emergency", matched: ["chest pain"], message: "Call now." } },
  ]);
  assert.equal(state.phase, "done");
  assert.equal(outcomeOf(state), "emergency");
  assert.equal(state.stages.retrieve.status, "skipped");
});

test("an error event marks its stage and keeps earlier output", () => {
  const state = run([
    { type: "retrieval", data: { retriever: "bm25", hits: [] } },
    { type: "stage", data: { name: "generate", status: "start" } },
    { type: "error", data: { stage: "generate", kind: "no_api_key", message: "no key" } },
  ]);
  assert.equal(outcomeOf(state), "failed");
  assert.equal(state.error?.kind, "no_api_key");
  assert.equal(state.stages.generate.status, "error");
  assert.equal(state.retriever, "bm25");
});

test("a stream that closes early is reported as a failure", () => {
  const state = run([{ type: "stage", data: { name: "retrieve", status: "start" } }]);
  assert.equal(state.phase, "error");
  assert.equal(state.error?.stage, "retrieve");
});

test("events after a cancel are ignored", () => {
  let state = askReducer(initialAskState, { type: "start", request, origin: "live" });
  state = askReducer(state, { type: "cancel" });
  state = askReducer(state, { type: "event", event: { type: "token", data: { text: "late" } } });
  assert.equal(state.phase, "idle");
  assert.equal(state.streamed, "");
});
