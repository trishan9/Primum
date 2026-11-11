"use client";

import type { ReactNode } from "react";

import { StatusDot } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import type { AskState, Outcome } from "@/lib/ask-state";
import { fmtAlpha } from "@/lib/format";
import { STAGES } from "@/lib/types";

import { AgentTrace } from "./AgentTrace";
import { AnswerText } from "./AnswerText";
import { Claims } from "./Claims";
import { CopyButton } from "./CopyButton";
import { STAGE_COPY } from "./copy";
import { Emergency, Failure, Released, UrgentNotice, Withheld } from "./Outcomes";
import { AssistantMessage, MessageMeta, UserMessage } from "./Thread";

/** The line beside Primum's name: what the reply amounts to, or what it is doing now. */
function replyStatus(outcome: Outcome, state: AskState): ReactNode {
  switch (outcome) {
    case "pending": {
      const stage = STAGES.find((name) => state.stages[name].status === "running");
      return (
        <span className="inline-flex items-center gap-1.5">
          <StatusDot tone="busy" />
          {stage ? STAGE_COPY[stage].label : "Working"}
        </span>
      );
    }
    case "answered":
      return state.confidence?.uncalibrated ? "Answer shown without a guarantee" : "Answer released";
    case "withheld":
      return "Answer withheld";
    case "emergency":
      return "Stopped at safety triage";
    case "failed":
      return "Stopped with an error";
    default:
      return null;
  }
}

/**
 * The answer prose under a small heading, with a copy action once it is final. The streaming text
 * is not a live region: the reply's status line announces the outcome once, so a screen reader is
 * not read every token.
 */
function AnswerSection({ title, text, final, children }: { title: string; text: string; final: boolean; children: ReactNode }) {
  return (
    <section aria-labelledby="answer-heading">
      <div className="flex items-center justify-between gap-3">
        <h3 id="answer-heading" className="text-sm font-semibold text-ink">
          {title}
        </h3>
        {final && text.trim() ? <CopyButton text={text} label="Copy" /> : null}
      </div>
      <div className="mt-1.5">{children}</div>
    </section>
  );
}

/** While the pipeline runs: the decision is not in yet, and the draft below is not certified. */
function PendingDecision({ state }: { state: AskState }) {
  const runningStage = STAGES.find((name) => state.stages[name].status === "running") ?? null;
  return (
    <section aria-labelledby="decision-heading" className="flex items-start gap-3 rounded-card bg-raised/70 px-4 py-3">
      <span aria-hidden="true" className="relative mt-0.5 flex size-5 shrink-0 items-center justify-center">
        <span className="absolute inset-0 rounded-full border-[1.5px] border-line" />
        <span
          className="absolute inset-0 rounded-full border-[1.5px] border-transparent border-t-accent"
          style={{ animation: "stage-sweep 0.9s linear infinite" }}
        />
      </span>
      <div className="min-w-0">
        <h3 id="decision-heading" className="text-sm font-semibold text-ink">
          Not yet certified
          {runningStage ? (
            <span className="font-normal text-muted"> Now: {STAGE_COPY[runningStage].label.toLowerCase()}.</span>
          ) : null}
        </h3>
        <p className="mt-0.5 max-w-[70ch] text-meta leading-relaxed text-muted">
          The draft is shown as it is written. It is released or withheld once every claim has been checked and the
          confidence estimate is in.
        </p>
      </div>
    </section>
  );
}

/** A withheld answer's draft, one click away and labelled as uncertified. */
function WithheldDraft({ children }: { children: ReactNode }) {
  return (
    <Disclosure variant="dashed">
      <Disclosure.Summary size="md" meta="not certified">
        <span className="group-open/disclosure:hidden">Show the withheld draft</span>
        <span className="hidden group-open/disclosure:inline">Hide the withheld draft</span>
      </Disclosure.Summary>
      <Disclosure.Body className="space-y-5 text-ink">
        <p className="rounded-control bg-raised px-3 py-2 text-sm leading-relaxed text-ink">
          This draft failed the release test. It may be wrong, and it is shown only so you can see what was held back
          and why.
        </p>
        {children}
      </Disclosure.Body>
    </Disclosure>
  );
}

/** The reader's question, with the settings it was asked under. */
export function Question({ state }: { state: AskState }) {
  const request = state.request;
  if (!request) return null;
  return (
    <UserMessage
      meta={
        <>
          {state.origin !== "live" ? (
            <MessageMeta>
              {state.origin === "trace" ? "Stored reading" : "Fixture reading: a canned aspirin answer, whatever was asked"}
            </MessageMeta>
          ) : null}
          {Number.isFinite(request.alpha) ? <MessageMeta>α = {fmtAlpha(request.alpha)}</MessageMeta> : null}
          {request.retriever ? <MessageMeta>{request.retriever}</MessageMeta> : null}
          {request.mode === "benchmark" ? (
            <MessageMeta>
              Benchmark mode
              {request.benchmarkQid ? (
                <>
                  , PubMedQA <span className="ml-1 font-mono">{request.benchmarkQid}</span>
                </>
              ) : null}
            </MessageMeta>
          ) : null}
        </>
      }
    >
      {request.question}
    </UserMessage>
  );
}

interface ReplyProps {
  state: AskState;
  outcome: Outcome;
  alphaOptions: readonly number[];
  onRetry: () => void;
  onAskAgain: (alpha: number) => void;
  /** Hovering or focusing a citation or claim lights its passage in the sources. */
  onInspect: (passage: number | null, sentence: string | null) => void;
  onOpenPassage: (passage: number) => void;
}

/**
 * Primum's turn: the agent trace, then one outcome. The decision block carries `data-decision`, so
 * the view can bring it on screen once the run settles.
 */
export function Reply({ state, outcome, alphaOptions, onRetry, onAskAgain, onInspect, onOpenPassage }: ReplyProps) {
  const running = state.phase === "running";
  const answerText = state.answer?.text ?? state.streamed;
  const passageCount = state.hits?.length ?? 0;
  const generating = state.stages.generate.status === "running";
  const gold = state.result?.gold_label;

  const answer =
    answerText.trim() || generating ? (
      <AnswerText
        text={answerText}
        passageCount={passageCount}
        streaming={generating && !state.answer}
        onCite={(n) => onInspect(n, null)}
        onOpenPassage={onOpenPassage}
      />
    ) : null;
  const claims = state.verdicts ? (
    <Claims verdicts={state.verdicts} onInspect={onInspect} onOpenPassage={onOpenPassage} />
  ) : null;

  return (
    <AssistantMessage status={replyStatus(outcome, state)} busy={running}>
      <AgentTrace state={state} />

      {state.triage?.level === "urgent" ? <UrgentNotice triage={state.triage} /> : null}

      {outcome === "emergency" && state.triage ? (
        <div data-decision="">
          <Emergency triage={state.triage} />
        </div>
      ) : null}

      {outcome === "failed" && state.error ? (
        <div data-decision="">
          <Failure error={state.error} foundPassages={passageCount} onRetry={onRetry} />
        </div>
      ) : null}

      {outcome === "answered" && state.confidence ? (
        <>
          <AnswerSection
            title={state.confidence.uncalibrated ? "Answer, not certified" : "Answer"}
            text={answerText}
            final
          >
            {answer}
          </AnswerSection>
          <div data-decision="">
            <Released confidence={state.confidence} decision={state.answer?.decision ?? null} gold={gold} />
          </div>
          {claims}
        </>
      ) : null}

      {outcome === "withheld" && state.confidence ? (
        <>
          <div data-decision="">
            <Withheld
              confidence={state.confidence}
              decision={state.answer?.decision ?? null}
              gold={gold}
              passageCount={passageCount}
              verdicts={state.verdicts ?? []}
              alphaOptions={alphaOptions}
              busy={running}
              onAskAgain={onAskAgain}
            />
          </div>
          <WithheldDraft>
            {answer}
            {claims}
          </WithheldDraft>
        </>
      ) : null}

      {outcome === "pending" ? (
        <>
          <PendingDecision state={state} />
          {answer ? (
            <AnswerSection title="Draft" text={answerText} final={false}>
              {answer}
            </AnswerSection>
          ) : null}
          {claims}
        </>
      ) : null}
    </AssistantMessage>
  );
}
