import type {
  ClaimVerdict,
  Concept,
  ConfidenceEvent,
  Decision,
  Hit,
  Mode,
  PipelineEvent,
  PipelineResult,
  StageName,
  TriageResult,
} from "./types.ts";
import { STAGES } from "./types.ts";

export type StageStatus = "idle" | "running" | "done" | "error" | "skipped";

export interface StageState {
  status: StageStatus;
  ms: number | null;
}

export interface AskRequestState {
  question: string;
  alpha: number;
  mode: Mode;
  retriever: string | null;
  benchmarkQid: string | null;
}

export interface AskError {
  /** A pipeline kind from SPEC 3.7, or a transport kind from the API client. */
  kind: string;
  message: string;
  stage: string | null;
}

export interface AskState {
  phase: "idle" | "running" | "done" | "error";
  /** Where the reading on screen came from. */
  origin: "live" | "fixture" | "trace";
  request: AskRequestState | null;
  stages: Record<StageName, StageState>;
  triage: TriageResult | null;
  concepts: Concept[] | null;
  /** False when the server reported that no concept dictionary exists; null when it did not say. */
  conceptsAvailable: boolean | null;
  retriever: string | null;
  hits: Hit[] | null;
  /** Tokens received so far, before the `answer` event replaces them with the final text. */
  streamed: string;
  answer: { text: string; decision: Decision | null } | null;
  verdicts: ClaimVerdict[] | null;
  confidence: ConfidenceEvent | null;
  result: PipelineResult | null;
  error: AskError | null;
}

export type AskAction =
  | { type: "start"; request: AskRequestState; origin: "live" | "fixture" }
  | { type: "event"; event: PipelineEvent }
  | { type: "fail"; error: AskError }
  | { type: "end" }
  | { type: "cancel" }
  | { type: "trace"; result: PipelineResult }
  | { type: "reset" };

function idleStages(): Record<StageName, StageState> {
  return Object.fromEntries(STAGES.map((name) => [name, { status: "idle", ms: null }])) as Record<
    StageName,
    StageState
  >;
}

export const initialAskState: AskState = {
  phase: "idle",
  origin: "live",
  request: null,
  stages: idleStages(),
  triage: null,
  concepts: null,
  conceptsAvailable: null,
  retriever: null,
  hits: null,
  streamed: "",
  answer: null,
  verdicts: null,
  confidence: null,
  result: null,
  error: null,
};

function isStage(name: string): name is StageName {
  return (STAGES as readonly string[]).includes(name);
}

/** Stages that never ran stop being "idle" once the run is over, so the rail tells the truth. */
function settleStages(stages: Record<StageName, StageState>, failed: string | null): Record<StageName, StageState> {
  const next = { ...stages };
  for (const name of STAGES) {
    const stage = next[name];
    if (name === failed) next[name] = { ...stage, status: "error" };
    else if (stage.status === "running") next[name] = { ...stage, status: failed ? "error" : "skipped" };
    else if (stage.status === "idle") next[name] = { ...stage, status: "skipped" };
  }
  return next;
}

function applyEvent(state: AskState, event: PipelineEvent): AskState {
  switch (event.type) {
    case "stage": {
      const { name, status, ms } = event.data;
      if (!isStage(name)) return state;
      const previous = state.stages[name];
      const next: StageState =
        status === "start"
          ? { status: "running", ms: null }
          : { status: "done", ms: typeof ms === "number" ? ms : previous.ms };
      return { ...state, stages: { ...state.stages, [name]: next } };
    }
    case "triage":
      return { ...state, triage: event.data };
    case "concepts":
      return {
        ...state,
        concepts: event.data.concepts ?? [],
        conceptsAvailable: typeof event.data.available === "boolean" ? event.data.available : null,
      };
    case "retrieval":
      return { ...state, retriever: event.data.retriever ?? null, hits: event.data.hits ?? [] };
    case "token":
      return { ...state, streamed: state.streamed + (event.data.text ?? "") };
    case "answer":
      return { ...state, answer: { text: event.data.text ?? "", decision: event.data.decision ?? null } };
    case "verdicts":
      return { ...state, verdicts: event.data.verdicts ?? [] };
    case "confidence":
      return { ...state, confidence: event.data };
    case "done": {
      const result = event.data;
      const stages = { ...state.stages };
      for (const name of STAGES) {
        const ms = result.timings_ms?.[name];
        const stage = stages[name];
        if (typeof ms === "number" && stage.ms === null) stages[name] = { status: "done", ms };
      }
      return {
        ...state,
        phase: "done",
        result,
        stages: settleStages(stages, null),
        triage: state.triage ?? result.triage ?? null,
        concepts: state.concepts ?? result.concepts ?? [],
        hits: state.hits ?? result.hits ?? [],
        retriever: state.retriever ?? (result.retriever || null),
        answer: state.answer ?? { text: result.answer_text ?? "", decision: result.decision ?? null },
        verdicts: state.verdicts ?? result.verdicts ?? [],
        // The stored report carries the same flags as the event (`uncalibrated`,
        // `guarantee_applies`); it is the fallback only for a stream that skipped the event.
        confidence: state.confidence ?? result.confidence ?? null,
      };
    }
    case "error": {
      const { stage, message, kind } = event.data;
      return {
        ...state,
        phase: "error",
        error: { kind: kind ?? "internal", message: message ?? "", stage: stage ?? null },
        stages: settleStages(state.stages, stage ?? null),
      };
    }
  }
}

export function askReducer(state: AskState, action: AskAction): AskState {
  switch (action.type) {
    case "start":
      return { ...initialAskState, stages: idleStages(), phase: "running", origin: action.origin, request: action.request };
    case "event":
      // Events that arrive after the run has been cancelled or replaced are ignored.
      return state.phase === "running" ? applyEvent(state, action.event) : state;
    case "fail":
      return {
        ...state,
        phase: "error",
        error: action.error,
        stages: settleStages(state.stages, action.error.stage),
      };
    case "end": {
      if (state.phase !== "running") return state;
      const complete = state.triage?.level === "emergency" || state.confidence !== null;
      if (complete) return { ...state, phase: "done", stages: settleStages(state.stages, null) };
      return {
        ...state,
        phase: "error",
        error: {
          kind: "internal",
          message: "The event stream closed before the pipeline reported a result.",
          stage: STAGES.find((name) => state.stages[name].status === "running") ?? null,
        },
        stages: settleStages(state.stages, STAGES.find((name) => state.stages[name].status === "running") ?? null),
      };
    }
    case "cancel":
      return state.phase === "running" ? { ...initialAskState, stages: idleStages() } : state;
    case "trace": {
      const result = action.result;
      const stages = idleStages();
      for (const name of STAGES) {
        const ms = result.timings_ms?.[name];
        if (typeof ms === "number") stages[name] = { status: "done", ms };
      }
      return {
        ...initialAskState,
        phase: "done",
        origin: "trace",
        request: {
          question: result.question,
          alpha: result.confidence?.alpha ?? Number.NaN,
          mode: result.mode,
          retriever: result.retriever || null,
          benchmarkQid: null,
        },
        stages: settleStages(stages, null),
        triage: result.triage,
        concepts: result.concepts ?? [],
        retriever: result.retriever || null,
        hits: result.hits ?? [],
        answer: { text: result.answer_text ?? "", decision: result.decision ?? null },
        verdicts: result.verdicts ?? [],
        confidence: result.confidence,
        result,
      };
    }
    case "reset":
      return { ...initialAskState, stages: idleStages() };
  }
}

/**
 * What the reading amounts to.
 * `withheld` is a decision the system made, not a failure: it found evidence and drafted
 * an answer, then declined to show it because it could not certify it at the chosen risk level.
 */
export type Outcome = "idle" | "pending" | "emergency" | "answered" | "withheld" | "failed";

export function outcomeOf(state: AskState): Outcome {
  if (state.phase === "idle") return "idle";
  if (state.triage?.level === "emergency") return "emergency";
  if (state.phase === "error") return "failed";
  if (state.confidence) return state.confidence.answered ? "answered" : "withheld";
  return "pending";
}
