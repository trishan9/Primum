import { useMemo, type ReactNode } from "react";

import { Badge, StatusDot } from "@/components/ui/badge";
import { cn } from "@/components/ui/cn";
import {
  CheckGlyph,
  ChevronGlyph,
  ClockGlyph,
  CrossGlyph,
  ExternalGlyph,
  GaugeGlyph,
  LinkNodesGlyph,
  PenGlyph,
  SearchGlyph,
  ShieldGlyph,
} from "@/components/ui/glyphs";
import type { AskState, StageStatus } from "@/lib/ask-state";
import { fmtMs, fmtProb, fmtUsd } from "@/lib/format";
import { STAGES, type StageName } from "@/lib/types";

import { STAGE_COPY } from "./copy";
import { StageWaterfall } from "./StageWaterfall";

const STAGE_GLYPH: Record<StageName, (props: { size?: number; className?: string }) => ReactNode> = {
  triage: ShieldGlyph,
  concepts: LinkNodesGlyph,
  retrieve: SearchGlyph,
  generate: PenGlyph,
  verify: CheckGlyph,
  confidence: GaugeGlyph,
};

const STATUS_WORD: Record<StageStatus, string> = {
  idle: "Waiting",
  running: "Running",
  done: "Done",
  error: "Failed",
  skipped: "Not run",
};

/** One line saying what the stage produced, in the stage's own terms. */
function stageDetail(name: StageName, state: AskState, status: StageStatus): string {
  if (status === "idle") return STAGE_COPY[name].idle;
  if (status === "skipped")
    return name === "concepts" && state.conceptsAvailable === false
      ? "Not available: no concept dictionary is built"
      : "Not run";
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
      if (!state.concepts) return "Linking";
      return state.concepts.length === 1 ? "1 concept recognised" : `${state.concepts.length} concepts recognised`;
    case "retrieve":
      if (!state.hits) return "Searching";
      return `${state.hits.length} ${state.hits.length === 1 ? "passage" : "passages"}${state.retriever ? `, ranked by ${state.retriever}` : ""}`;
    case "generate": {
      const text = state.answer?.text ?? state.streamed;
      const words = text.trim() ? text.trim().split(/\s+/).length : 0;
      if (status === "running") return words ? `Writing, ${words} words` : "Waiting for the model";
      return `${words} words drafted`;
    }
    case "verify": {
      if (!state.verdicts) return "Checking claims";
      const supported = state.verdicts.filter((v) => v.label === "supported").length;
      return `${supported} of ${state.verdicts.length} claims supported`;
    }
    case "confidence":
      if (!state.confidence) return "Estimating";
      return typeof state.confidence.threshold === "number"
        ? `${fmtProb(state.confidence.p_correct)} against a threshold of ${fmtProb(state.confidence.threshold)}`
        : `${fmtProb(state.confidence.p_correct)}, no threshold`;
  }
}

/** A concepts stage without a dictionary ran but could link nothing: drawn as not run. */
function effectiveStatus(name: StageName, state: AskState): StageStatus {
  const stage = state.stages[name];
  return name === "concepts" && state.conceptsAvailable === false && stage.status === "done" ? "skipped" : stage.status;
}

/** The step's tool glyph in a round well whose tone says the status; a sweeping ring while it runs. */
function StepIcon({ name, status }: { name: StageName; status: StageStatus }) {
  const Glyph = STAGE_GLYPH[name];
  return (
    <span
      aria-hidden="true"
      className={cn(
        "relative z-10 flex size-7 shrink-0 items-center justify-center rounded-full",
        status === "done" && "bg-raised text-ink ring-1 ring-line",
        status === "running" && "bg-accent-soft text-accent-ink",
        status === "error" && "bg-contradicted-bg text-contradicted-ink",
        (status === "idle" || status === "skipped") && "border border-dashed border-frame bg-surface text-faint",
      )}
    >
      {status === "error" ? <CrossGlyph size={14} /> : <Glyph size={14} />}
      {status === "running" ? (
        <span
          className="absolute -inset-[3px] rounded-full border-[1.5px] border-transparent border-t-accent"
          style={{ animation: "stage-sweep 0.9s linear infinite" }}
        />
      ) : null}
    </span>
  );
}

function StepStatus({ status, ms }: { status: StageStatus; ms: number | null }) {
  if (status === "running") {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-accent-ink">
        <StatusDot tone="busy" />
        Running
      </span>
    );
  }
  if (status === "done" && ms !== null) {
    return (
      <span className="num inline-flex items-center gap-1 text-xs text-muted">
        <ClockGlyph size={12} />
        {fmtMs(ms)}
        <span className="sr-only">, done</span>
      </span>
    );
  }
  return (
    <span className={cn("text-xs", status === "error" ? "font-medium text-contradicted-ink" : "text-muted")}>
      {STATUS_WORD[status]}
    </span>
  );
}

interface TraceStepProps {
  name: StageName;
  status: StageStatus;
  ms: number | null;
  detail: string;
  last: boolean;
  /** What the step found, beyond its one-line result. When present the step can be expanded. */
  children?: ReactNode;
}

/** One tool call: glyph, name, one-line result, timing. Expands when it has more to show. */
function TraceStep({ name, status, ms, detail, last, children }: TraceStepProps) {
  const row = (
    <span className="flex min-w-0 flex-1 items-start gap-3">
      <StepIcon name={name} status={status} />
      <span className="min-w-0 flex-1 pt-0.5">
        <span className="flex flex-wrap items-baseline justify-between gap-x-3">
          <span
            className={cn("text-sm", status === "idle" || status === "skipped" ? "text-muted" : "font-medium text-ink")}
          >
            {STAGE_COPY[name].label}
          </span>
          <StepStatus status={status} ms={ms} />
        </span>
        <span className="mt-0.5 block text-[0.8125rem] leading-snug text-muted">{detail}</span>
      </span>
    </span>
  );

  return (
    <li className="relative pb-3 last:pb-0">
      {!last ? <span aria-hidden="true" className="absolute bottom-0 left-[13.5px] top-8 w-px bg-line" /> : null}
      {children ? (
        <details className="group/step">
          <summary className="-mx-2 flex min-h-11 cursor-pointer items-start gap-1 rounded-control px-2 py-1 hover:bg-raised sm:min-h-9 sm:py-0.5">
            {row}
            <ChevronGlyph
              size={14}
              className="mt-2 -rotate-90 text-muted transition-transform duration-(--dur) group-open/step:rotate-0"
            />
            <span className="sr-only">Show details</span>
          </summary>
          <div className="ml-10 mt-2">{children}</div>
        </details>
      ) : (
        <div className="py-0.5">{row}</div>
      )}
    </li>
  );
}

function ConceptChips({ state }: { state: AskState }) {
  if (!state.concepts?.length) return null;
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Concepts recognised">
      {state.concepts.map((concept, i) => {
        const mesh = concept.mesh_ids[0];
        const body = (
          <>
            <span className="text-ink">{concept.preferred}</span>
            {mesh ? <span className="num font-mono text-[0.6875rem] text-muted">{mesh}</span> : null}
          </>
        );
        const chip = "inline-flex min-h-7 items-center gap-1.5 rounded-chip border border-line bg-surface px-2 text-xs";
        return (
          <li key={`${mesh ?? concept.preferred}-${i}`}>
            {concept.url ? (
              <a
                href={concept.url}
                target="_blank"
                rel="noreferrer noopener"
                className={cn(chip, "hover:border-control")}
              >
                {body}
                <ExternalGlyph size={11} className="text-muted" />
                <span className="sr-only">(opens MedlinePlus in a new tab)</span>
              </a>
            ) : (
              <span className={chip}>{body}</span>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function stepExtra(name: StageName, state: AskState): ReactNode {
  if (name === "concepts" && state.concepts?.length) return <ConceptChips state={state} />;
  if (name === "triage" && state.triage && state.triage.matched.length > 0) {
    return (
      <p className="text-[0.8125rem] text-muted">
        {state.triage.matched.length === 1 ? "Rule matched: " : "Rules matched: "}
        <span className="text-ink">{state.triage.matched.join("; ")}</span>
      </p>
    );
  }
  return null;
}

/** Six short segments, one per step: done in ink, running in the accent, not yet run as a groove. */
function StepProgress({ state }: { state: AskState }) {
  return (
    <span aria-hidden="true" className="flex items-center gap-[3px] max-sm:hidden">
      {STAGES.map((name) => {
        const status = effectiveStatus(name, state);
        return (
          <span
            key={name}
            className={cn(
              "h-1.5 w-3 rounded-full",
              status === "done" && "bg-muted",
              status === "running" && "bg-accent",
              status === "error" && "bg-contradicted",
              (status === "idle" || status === "skipped") && "bg-line",
            )}
          />
        );
      })}
    </span>
  );
}

/**
 * The pipeline as an agent trace: six tool calls in the order they run, each with its result and
 * timing. While it runs it is one compact strip (the running step, its result so far, and the
 * progress), so the streaming answer stays on screen under it; it expands on request. Once the
 * reply is in it folds to one summary line (a failure card names the stage that stopped).
 */
export function AgentTrace({ state }: { state: AskState }) {
  const running = state.phase === "running";
  const timed = STAGES.map((name) => state.stages[name].ms).filter((ms): ms is number => ms !== null);
  const total = timed.reduce((sum, ms) => sum + ms, 0);
  const ran = STAGES.filter((name) => state.stages[name].status === "done").length;
  const current = STAGES.find((name) => state.stages[name].status === "running") ?? null;
  const step = current ? STAGES.indexOf(current) + 1 : ran;
  const usage = state.result?.usage;
  // The run's own timings when it finished; otherwise the stages that reported one (a failed run).
  const timings = useMemo(() => {
    if (state.result?.timings_ms) return state.result.timings_ms;
    const done = STAGES.flatMap((name) => (state.stages[name].ms === null ? [] : [[name, state.stages[name].ms] as const]));
    return done.length > 0 ? Object.fromEntries(done) : null;
  }, [state.result, state.stages]);

  return (
    <details className="group/trace rounded-card border border-line bg-surface">
      <summary className="flex min-h-12 cursor-pointer items-center gap-2.5 rounded-card px-3.5 py-2 hover:bg-raised/60">
        {running && current ? (
          <StepIcon name={current} status="running" />
        ) : (
          <ChevronGlyph
            size={14}
            className="-rotate-90 text-muted transition-transform duration-(--dur) group-open/trace:rotate-0"
          />
        )}
        <span className="min-w-0 flex-1">
          <span className="flex min-w-0 items-baseline gap-2">
            <span className="shrink-0 text-sm font-medium text-ink">
              {running && current ? STAGE_COPY[current].label : running ? "Working" : "Agent trace"}
            </span>
            <span className="min-w-0 truncate text-meta text-muted">
              {running && current
                ? stageDetail(current, state, "running")
                : running
                  ? "Starting"
                  : `${ran} of ${STAGES.length} steps ran`}
            </span>
          </span>
        </span>
        {running ? (
          <span className="flex shrink-0 items-center gap-2.5">
            <StepProgress state={state} />
            <span className="num text-xs text-muted">
              {step} of {STAGES.length}
            </span>
            <ChevronGlyph
              size={14}
              className="-rotate-90 text-muted transition-transform duration-(--dur) group-open/trace:rotate-0"
            />
          </span>
        ) : timed.length > 0 ? (
          <Badge tone="outline" className="num">
            <ClockGlyph size={12} />
            {fmtMs(total)}
          </Badge>
        ) : null}
      </summary>

      <div className="border-t border-rule px-3.5 pb-3.5 pt-3.5">
        <ol aria-label="Pipeline steps">
          {STAGES.map((name, i) => {
            const status = effectiveStatus(name, state);
            const extra = stepExtra(name, state);
            return (
              <TraceStep
                key={name}
                name={name}
                status={status}
                ms={state.stages[name].ms}
                detail={stageDetail(name, state, status)}
                last={i === STAGES.length - 1}
              >
                {extra}
              </TraceStep>
            );
          })}
        </ol>

        {!running && timings ? (
          <StageWaterfall
            timings={timings}
            source={state.result?.timings_ms ? "done event, timings_ms" : "stage events"}
            className="mt-3.5"
          />
        ) : null}

        {!running && (state.result?.model || usage) ? (
          <dl className="num mt-3.5 flex flex-wrap gap-x-6 gap-y-1 border-t border-rule pt-3 text-xs text-muted">
            {state.result?.model ? (
              <div className="flex min-w-0 gap-1.5">
                <dt>Generator</dt>
                <dd className="truncate font-mono text-ink">{state.result.model}</dd>
              </div>
            ) : null}
            {usage ? (
              <div className="flex gap-1.5">
                <dt>LLM cost</dt>
                <dd className="text-ink">{usage.cached ? "cached, $0.00" : fmtUsd(usage.cost_usd)}</dd>
              </div>
            ) : null}
          </dl>
        ) : null}
      </div>
    </details>
  );
}

/** Before any question: the six steps every answer goes through, as a quiet preview of the trace. */
export function PipelinePreview() {
  return (
    <ol aria-label="What happens to a question" className="grid gap-x-6 gap-y-4 sm:grid-cols-2 lg:grid-cols-3">
      {STAGES.map((name) => {
        const Glyph = STAGE_GLYPH[name];
        return (
          <li key={name} className="flex items-start gap-3">
            <span
              aria-hidden="true"
              className="flex size-7 shrink-0 items-center justify-center rounded-full bg-raised text-muted"
            >
              <Glyph size={14} />
            </span>
            <span className="min-w-0 pt-0.5">
              <span className="block text-sm font-medium text-ink">{STAGE_COPY[name].label}</span>
              <span className="block text-[0.8125rem] leading-snug text-muted">{STAGE_COPY[name].idle}</span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}
