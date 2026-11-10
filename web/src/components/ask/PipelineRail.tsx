import type { AskState, StageStatus } from "@/lib/ask-state";
import { fmtMs, fmtProb, fmtUsd } from "@/lib/format";
import { STAGES, type StageName } from "@/lib/types";

import { STAGE_COPY } from "./copy";

function StageMark({ status }: { status: StageStatus }) {
  if (status === "running") {
    return (
      <span className="relative flex size-3 items-center justify-center" aria-hidden="true">
        <span className="absolute inset-0 rounded-full border-[1.5px] border-control" />
        <span
          className="absolute inset-0 rounded-full border-[1.5px] border-transparent border-t-ink"
          style={{ animation: "stage-sweep 0.9s linear infinite" }}
        />
      </span>
    );
  }
  if (status === "done") return <span aria-hidden="true" className="size-3 rounded-full bg-ink" />;
  if (status === "error") {
    return (
      <svg aria-hidden="true" width="12" height="12" viewBox="0 0 12 12" className="text-contradicted">
        <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
      </svg>
    );
  }
  if (status === "skipped") return <span aria-hidden="true" className="h-px w-3 bg-control" />;
  return <span aria-hidden="true" className="size-3 rounded-full border-[1.5px] border-control" />;
}

const STATUS_WORD: Record<StageStatus, string> = {
  idle: "not started",
  running: "running",
  done: "done",
  error: "failed",
  skipped: "not run",
};

/** One line saying what the stage produced, in the stage's own terms. */
function stageDetail(name: StageName, state: AskState): string {
  const status = state.stages[name].status;
  if (status === "idle") return STAGE_COPY[name].idle;
  if (status === "skipped") return "Not run";
  if (status === "error") return "Failed here";
  switch (name) {
    case "triage":
      if (!state.triage) return "Checking";
      return state.triage.level === "emergency"
        ? "Emergency: answering stopped"
        : state.triage.level === "urgent"
          ? "Urgent: answered with a warning"
          : "Routine";
    case "concepts":
      if (state.conceptsAvailable === false) return "Not available: no concept dictionary is built";
      if (!state.concepts) return "Linking";
      return state.concepts.length === 1 ? "1 concept recognised" : `${state.concepts.length} concepts recognised`;
    case "retrieve":
      if (!state.hits) return "Searching";
      return `${state.hits.length} ${state.hits.length === 1 ? "passage" : "passages"}${state.retriever ? `, ${state.retriever}` : ""}`;
    case "generate": {
      const text = state.answer?.text ?? state.streamed;
      const words = text.trim() ? text.trim().split(/\s+/).length : 0;
      if (status === "running") return words ? `Writing, ${words} words` : "Waiting for the model";
      return `${words} words`;
    }
    case "verify": {
      if (!state.verdicts) return "Checking claims";
      const supported = state.verdicts.filter((v) => v.label === "supported").length;
      return `${supported} of ${state.verdicts.length} claims supported`;
    }
    case "confidence":
      if (!state.confidence) return "Estimating";
      return typeof state.confidence.threshold === "number"
        ? `${fmtProb(state.confidence.p_correct)} against ${fmtProb(state.confidence.threshold)}`
        : `${fmtProb(state.confidence.p_correct)}, no threshold`;
  }
}

/**
 * The pipeline as it runs: six stages in the order they execute, each with what it produced
 * and how long it took. The connecting line is there because this is a sequence.
 */
export function PipelineRail({ state }: { state: AskState }) {
  const timed = STAGES.map((name) => state.stages[name].ms).filter((ms): ms is number => ms !== null);
  const total = timed.reduce((sum, ms) => sum + ms, 0);
  const usage = state.result?.usage;

  return (
    <section aria-labelledby="pipeline-heading">
      <h2 id="pipeline-heading" className="text-[0.8125rem] font-medium text-muted">
        Pipeline
      </h2>

      <ol className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lg:block lg:space-y-0">
        {STAGES.map((name, i) => {
          const stage = state.stages[name];
          // A concepts stage without a dictionary ran but could link nothing: drawn as not run.
          const status: StageStatus = name === "concepts" && state.conceptsAvailable === false && stage.status === "done" ? "skipped" : stage.status;
          return (
            <li key={name} className="relative flex gap-3 lg:pb-5 lg:last:pb-0">
              {i < STAGES.length - 1 ? (
                <span aria-hidden="true" className="absolute bottom-0 left-[5.5px] top-4 hidden w-px bg-line lg:block" />
              ) : null}
              <span className="relative z-10 mt-1 flex size-3 shrink-0 items-center justify-center bg-ground">
                <StageMark status={status} />
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className={`text-sm ${status === "idle" || status === "skipped" ? "text-muted" : "font-medium text-ink"}`}>
                    {STAGE_COPY[name].label}
                  </span>
                  <span className="num shrink-0 text-xs text-muted">
                    {stage.ms !== null ? fmtMs(stage.ms) : null}
                    <span className="sr-only">, {STATUS_WORD[status]}</span>
                  </span>
                </div>
                <p className="mt-0.5 text-xs leading-snug text-muted">{stageDetail(name, state)}</p>
              </div>
            </li>
          );
        })}
      </ol>

      {timed.length > 0 && state.phase !== "running" ? (
        <dl className="num mt-4 space-y-1 border-t border-line pt-3 text-xs text-muted">
          <div className="flex justify-between gap-2">
            <dt>Total</dt>
            <dd className="text-ink">{fmtMs(total)}</dd>
          </div>
          {state.result?.model ? (
            <div className="flex justify-between gap-2">
              <dt>Generator</dt>
              <dd className="truncate text-ink">{state.result.model}</dd>
            </div>
          ) : null}
          {usage ? (
            <div className="flex justify-between gap-2">
              <dt>LLM cost</dt>
              <dd className="text-ink">{usage.cached ? "cached, $0.00" : fmtUsd(usage.cost_usd)}</dd>
            </div>
          ) : null}
        </dl>
      ) : null}
    </section>
  );
}
