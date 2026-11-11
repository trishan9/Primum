"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { useApi } from "@/components/shell/ApiProvider";
import { cn } from "@/components/ui/cn";
import { ApiError, ask, getTrace, getTraces, type TraceSummary } from "@/lib/api";
import { askReducer, initialAskState, outcomeOf, type AskRequestState, type Outcome } from "@/lib/ask-state";
import { citedIndices } from "@/lib/markdown";
import { scrollBehavior } from "@/lib/motion";
import {
  SPEC_ALPHAS,
  SPEC_DEFAULT_ALPHA,
  SPEC_RETRIEVERS,
  type BenchmarkQuestion,
  type PipelineEvent,
} from "@/lib/types";

import { AskEmpty } from "./AskEmpty";
import { Composer, type ComposerValue } from "./Composer";
import { Evidence } from "./Evidence";
import { FixtureBar } from "./FixtureBar";
import { Question, Reply } from "./Reply";
import { Thread } from "./Thread";
import { useDockHeight } from "./useDockHeight";

const FIXTURES = process.env.NEXT_PUBLIC_FIXTURES === "1";

type FixtureModule = typeof import("@/lib/fixtures");

/** The composer as the reader left it. `alpha` is null until they choose one: the server's default then applies. */
type ComposerDraft = Omit<ComposerValue, "alpha"> & { alpha: number | null };

/** Read out once per change of outcome, so a screen reader hears the decision and not every token. */
const ANNOUNCE: Record<Outcome, string> = {
  idle: "",
  pending: "Working on the question.",
  emergency: "Stopped: this may be an emergency.",
  answered: "Answer released.",
  withheld: "Answer withheld.",
  failed: "The pipeline stopped with an error.",
};

/**
 * The Ask view: holds the run (reducer, stream, stop) and lays out three slots: the thread (or the
 * empty state), the sources column, and the docked composer.
 */
export function AskView() {
  const { meta, reachability } = useApi();
  const [state, dispatch] = useReducer(askReducer, initialAskState);
  const [draft, setDraft] = useState<ComposerDraft>({ question: "", alpha: null, retriever: "", benchmark: null });
  const [active, setActive] = useState<{ passage: number | null; sentence: string | null }>({
    passage: null,
    sentence: null,
  });
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [fixtures, setFixtures] = useState<FixtureModule | null>(null);
  const [scenario, setScenario] = useState<string>("answered");
  const controller = useRef<AbortController | null>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const threadRef = useRef<HTMLDivElement>(null);
  const dockRef = useRef<HTMLDivElement>(null);
  const flash = useRef<number | null>(null);
  /** The settings of the last question sent, so "Ask again" repeats it after the composer is cleared. */
  const lastRun = useRef<ComposerValue | null>(null);

  useDockHeight(dockRef);

  const alphaOptions = meta?.alphas?.length ? meta.alphas : SPEC_ALPHAS;
  const retrieverOptions = meta?.retrievers?.length ? meta.retrievers : SPEC_RETRIEVERS;
  // Derived, not synced: the reader's choice, else the server's default, else the spec's.
  const composer: ComposerValue = { ...draft, alpha: draft.alpha ?? meta?.alpha ?? SPEC_DEFAULT_ALPHA };
  const outcome = outcomeOf(state);
  const running = state.phase === "running";
  const showWorkspace = state.phase !== "idle";

  useEffect(() => {
    if (!FIXTURES) return;
    let alive = true;
    void import("@/lib/fixtures").then((loaded) => {
      if (alive) setFixtures(loaded);
    });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (FIXTURES || reachability !== "up" || state.phase !== "idle") return;
    const abort = new AbortController();
    getTraces(5, abort.signal)
      .then(setTraces)
      .catch(() => setTraces([]));
    return () => abort.abort();
  }, [reachability, state.phase]);

  useEffect(() => () => controller.current?.abort(), []);

  // When a run settles, bring its decision on screen once: it may have landed under the composer.
  useEffect(() => {
    if (outcome === "idle" || outcome === "pending") return;
    const frame = window.requestAnimationFrame(() =>
      document.querySelector("[data-decision]")?.scrollIntoView({ block: "nearest", behavior: scrollBehavior() }),
    );
    return () => window.cancelAnimationFrame(frame);
  }, [outcome]);

  const run = useCallback(
    async (value: ComposerValue) => {
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      setActive({ passage: null, sentence: null });
      lastRun.current = value;

      const request: AskRequestState = {
        question: value.question.trim(),
        alpha: value.alpha,
        mode: value.benchmark ? "benchmark" : "chat",
        retriever: value.retriever || null,
        benchmarkQid: value.benchmark?.qid ?? null,
      };
      dispatch({ type: "start", request, origin: FIXTURES ? "fixture" : "live" });
      window.requestAnimationFrame(() =>
        threadRef.current?.scrollIntoView({ block: "start", behavior: scrollBehavior() }),
      );

      try {
        let events: AsyncGenerator<PipelineEvent>;
        if (FIXTURES) {
          const replay = fixtures ?? (await import("@/lib/fixtures"));
          const chosen = replay.FIXTURE_SCENARIOS.find((s) => s.id === scenario)?.id ?? "answered";
          events = replay.replayFixture(value.benchmark ? "benchmark" : chosen, request, abort.signal);
        } else {
          events = ask(
            {
              question: request.question,
              alpha: request.alpha,
              mode: request.mode,
              ...(request.retriever ? { retriever: request.retriever } : {}),
              ...(request.benchmarkQid ? { benchmark_qid: request.benchmarkQid } : {}),
            },
            abort.signal,
          );
        }
        for await (const event of events) {
          if (abort.signal.aborted) return;
          dispatch({ type: "event", event });
        }
        if (!abort.signal.aborted) dispatch({ type: "end" });
      } catch (error) {
        if (abort.signal.aborted) return;
        dispatch({
          type: "fail",
          error:
            error instanceof ApiError
              ? { kind: error.kind, message: error.message, stage: null }
              : {
                  kind: "unreachable",
                  message: "The connection to the API dropped while the answer was being produced.",
                  stage: null,
                },
        });
      }
    },
    [fixtures, scenario],
  );

  const stop = () => {
    controller.current?.abort();
    dispatch({ type: "cancel" });
  };

  const openPassage = useCallback((n: number) => {
    document.getElementById(`passage-${n}`)?.scrollIntoView({ block: "nearest", behavior: scrollBehavior() });
    setActive({ passage: n, sentence: null });
    if (flash.current) window.clearTimeout(flash.current);
    flash.current = window.setTimeout(() => setActive({ passage: null, sentence: null }), 2400);
  }, []);

  const inspect = useCallback((passage: number | null, sentence: string | null) => {
    if (flash.current) window.clearTimeout(flash.current);
    setActive({ passage, sentence });
  }, []);

  const loadTrace = async (id: string) => {
    try {
      const result = await getTrace(id);
      dispatch({ type: "trace", result });
      setDraft((value) => ({ ...value, question: result.question, benchmark: null }));
    } catch {
      setTraces((list) => list.filter((t) => t.trace_id !== id));
    }
  };

  const fill = (question: string, benchmark: BenchmarkQuestion | null) => {
    setDraft((value) => ({ ...value, question, benchmark }));
    textarea.current?.focus();
  };

  const answerText = state.answer?.text ?? state.streamed;
  // A passage counts as cited if the text marks it or a claim names it (benchmark answers cite only through claims).
  const cited = useMemo(() => {
    const all = new Set(citedIndices(answerText));
    for (const verdict of state.verdicts ?? []) for (const n of verdict.claim.cited) all.add(n);
    return [...all];
  }, [answerText, state.verdicts]);

  return (
    <div className="-mb-20 flex min-h-[calc(100dvh-var(--topbar-h)-1.5rem)] flex-col lg:min-h-[calc(100dvh-var(--topbar-h)-1.75rem-var(--sheet-inset))]">
      {FIXTURES ? (
        <FixtureBar scenario={scenario} scenarios={fixtures?.FIXTURE_SCENARIOS ?? []} onScenarioChange={setScenario} />
      ) : null}

      {/* Once a run starts the sources column is reserved, so the thread never moves sideways mid-stream. */}
      <div className={cn("flex-1", showWorkspace && "xl:grid xl:grid-cols-[minmax(0,1fr)_24rem] xl:items-start xl:gap-8")}>
        <div className={cn("mx-auto w-full max-w-[46rem]", showWorkspace && "xl:max-w-none")}>
          {!showWorkspace ? (
            <AskEmpty
              live={!FIXTURES}
              retriever={draft.retriever}
              traces={traces}
              onFill={fill}
              onOpenTrace={(id) => void loadTrace(id)}
            />
          ) : (
            <div ref={threadRef} className="scroll-mt-20 pb-6">
              <h1 className="sr-only">Ask</h1>
              <p role="status" className="sr-only">
                {ANNOUNCE[outcome]}
              </p>
              <Thread>
                <Question state={state} />
                <Reply
                  state={state}
                  outcome={outcome}
                  alphaOptions={alphaOptions}
                  onRetry={() => void run(lastRun.current ?? composer)}
                  onAskAgain={(alpha) => {
                    const base = lastRun.current ?? composer;
                    setDraft((value) => ({ ...value, alpha }));
                    void run({ ...base, alpha, question: state.request?.question ?? base.question });
                  }}
                  onInspect={inspect}
                  onOpenPassage={openPassage}
                />
              </Thread>
            </div>
          )}
        </div>

        {showWorkspace ? (
          <div
            className={cn(
              "mx-auto mb-6 w-full max-w-[46rem] xl:sticky xl:top-[calc(var(--topbar-h)+1rem)] xl:mb-0 xl:max-w-none",
              outcome === "emergency" && "max-xl:hidden",
            )}
          >
            <Evidence
              className="xl:max-h-[calc(100dvh-var(--topbar-h)-2rem)]"
              hits={state.hits}
              retriever={state.retriever}
              searching={state.stages.retrieve.status === "running"}
              emptyNote={outcome === "emergency" ? "Nothing was retrieved: Primum stopped at safety triage." : undefined}
              citedIndices={cited}
              citedLabel={outcome === "answered" ? "Cited" : "Cited in draft"}
              activePassage={active.passage}
              activeSentence={active.sentence}
            />
          </div>
        ) : null}
      </div>

      {/* Docked composer. Its ground covers only the thread column, so the sources panel stays visible beside it. */}
      <div className="pointer-events-none sticky bottom-0 z-20">
        <div
          ref={dockRef}
          className={cn(
            "pointer-events-auto relative mx-auto w-full max-w-[46rem] bg-ground pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-1",
            // A short fade above the dock says the thread runs on underneath it.
            "before:pointer-events-none before:absolute before:inset-x-0 before:-top-6 before:h-6 before:bg-linear-to-b before:from-transparent before:to-ground before:content-['']",
            showWorkspace && "xl:ml-0 xl:mr-[26rem] xl:w-auto xl:max-w-none",
          )}
        >
          <Composer
            value={composer}
            onChange={(patch) => setDraft((value) => ({ ...value, ...patch }))}
            onSubmit={() => {
              void run(composer);
              setDraft((value) => ({ ...value, question: "", benchmark: null }));
            }}
            compact={showWorkspace}
            onStop={stop}
            running={running}
            alphaOptions={alphaOptions}
            retrieverOptions={retrieverOptions}
            defaultRetriever={meta?.default_retriever}
            textareaRef={textarea}
          />
        </div>
      </div>
    </div>
  );
}
