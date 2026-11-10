"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { ApiDown } from "@/components/shell/ApiDown";
import { useApi } from "@/components/shell/ApiProvider";
import { Button, Code } from "@/components/ui/controls";
import { ExternalGlyph } from "@/components/ui/glyphs";
import { ApiError, ask, getTrace, getTraces, type TraceSummary } from "@/lib/api";
import { askReducer, initialAskState, outcomeOf, type AskRequestState, type Outcome } from "@/lib/ask-state";
import { fmtAlpha } from "@/lib/format";
import { citedIndices } from "@/lib/markdown";
import { readinessGaps, type Gap } from "@/lib/readiness";
import {
  SPEC_ALPHAS,
  SPEC_DEFAULT_ALPHA,
  SPEC_RETRIEVERS,
  STAGES,
  type BenchmarkQuestion,
  type PipelineEvent,
} from "@/lib/types";

import { AnswerText } from "./AnswerText";
import { BenchmarkPicker } from "./BenchmarkPicker";
import { Claims } from "./Claims";
import { Composer, type ComposerValue } from "./Composer";
import { EXAMPLE_QUESTIONS, STAGE_COPY } from "./copy";
import { Evidence } from "./Evidence";
import { Emergency, Failure, Released, UrgentNotice, Withheld } from "./Outcomes";
import { PipelineRail } from "./PipelineRail";

const FIXTURES = process.env.NEXT_PUBLIC_FIXTURES === "1";

type FixtureModule = typeof import("@/lib/fixtures");

/** Read out once per change of outcome, so a screen reader hears the decision and not every token. */
const ANNOUNCE: Record<Outcome, string> = {
  idle: "",
  pending: "Working on the question.",
  emergency: "Stopped: this may be an emergency.",
  answered: "Answer released.",
  withheld: "Answer withheld.",
  failed: "The pipeline stopped with an error.",
};

export function AskView() {
  const { meta, reachability, health } = useApi();
  const [state, dispatch] = useReducer(askReducer, initialAskState);
  const [composer, setComposer] = useState<ComposerValue>({
    question: "",
    alpha: SPEC_DEFAULT_ALPHA,
    retriever: "",
    benchmark: null,
  });
  const [active, setActive] = useState<{ passage: number | null; sentence: string | null }>({ passage: null, sentence: null });
  const [picking, setPicking] = useState(false);
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [fixtures, setFixtures] = useState<FixtureModule | null>(null);
  const [scenario, setScenario] = useState<string>("answered");
  const controller = useRef<AbortController | null>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const readingRef = useRef<HTMLDivElement>(null);
  const flash = useRef<number | null>(null);

  const alphaOptions = meta?.alphas?.length ? meta.alphas : SPEC_ALPHAS;
  const retrieverOptions = meta?.retrievers?.length ? meta.retrievers : SPEC_RETRIEVERS;
  const outcome = outcomeOf(state);
  const running = state.phase === "running";

  // Take the server's default risk level once it is known, unless the reader already chose one.
  const adoptedDefault = useRef(false);
  useEffect(() => {
    if (adoptedDefault.current || typeof meta?.alpha !== "number") return;
    adoptedDefault.current = true;
    const serverAlpha = meta.alpha;
    const timer = window.setTimeout(() => setComposer((value) => ({ ...value, alpha: serverAlpha })), 0);
    return () => window.clearTimeout(timer);
  }, [meta?.alpha]);

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

  const run = useCallback(
    async (value: ComposerValue) => {
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      setActive({ passage: null, sentence: null });

      const request: AskRequestState = {
        question: value.question.trim(),
        alpha: value.alpha,
        mode: value.benchmark ? "benchmark" : "chat",
        retriever: value.retriever || null,
        benchmarkQid: value.benchmark?.qid ?? null,
      };
      dispatch({ type: "start", request, origin: FIXTURES ? "fixture" : "live" });
      window.requestAnimationFrame(() => readingRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }));

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
        if (error instanceof ApiError) {
          dispatch({ type: "fail", error: { kind: error.kind, message: error.message, stage: null } });
        } else {
          dispatch({
            type: "fail",
            error: {
              kind: "unreachable",
              message: "The connection to the API dropped while the answer was being produced.",
              stage: null,
            },
          });
        }
      }
    },
    [fixtures, scenario],
  );

  const closePicker = useCallback(() => setPicking(false), []);

  const stop = () => {
    controller.current?.abort();
    dispatch({ type: "cancel" });
  };

  const openPassage = useCallback((n: number) => {
    const element = document.getElementById(`passage-${n}`);
    if (!element) return;
    element.scrollIntoView({ block: "center", behavior: "smooth" });
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
      setComposer((value) => ({ ...value, question: result.question, benchmark: null }));
    } catch {
      setTraces((list) => list.filter((t) => t.trace_id !== id));
    }
  };

  const answerText = state.answer?.text ?? state.streamed;
  // A passage counts as cited if the text marks it or a claim names it (benchmark answers cite only through claims).
  const cited = useMemo(() => {
    const all = new Set(citedIndices(answerText));
    for (const verdict of state.verdicts ?? []) for (const n of verdict.claim.cited) all.add(n);
    return [...all];
  }, [answerText, state.verdicts]);
  const passageCount = state.hits?.length ?? 0;
  const request = state.request;
  const showWorkspace = state.phase !== "idle";
  const generating = state.stages.generate.status === "running";
  const runningStage = STAGES.find((name) => state.stages[name].status === "running") ?? null;
  const gold = state.result?.gold_label;

  const answerBlock =
    answerText.trim() || generating ? (
      <AnswerText
        text={answerText}
        passageCount={passageCount}
        streaming={generating && !state.answer}
        onCite={(n) => inspect(n, null)}
        onOpenPassage={openPassage}
      />
    ) : null;

  const claimsBlock = state.verdicts ? <Claims verdicts={state.verdicts} onInspect={inspect} onOpenPassage={openPassage} /> : null;

  return (
    <div>
      {FIXTURES ? (
        <div
          role="status"
          className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-card border border-unsupported bg-unsupported-bg px-4 py-2.5 text-sm text-unsupported-ink"
        >
          <p className="font-semibold">Fixture data. Not real output.</p>
          <p className="min-w-0 flex-1 basis-64">
            Every passage, claim and number on this view is invented for interface development. Unset{" "}
            <Code>NEXT_PUBLIC_FIXTURES</Code> to talk to the API.
          </p>
          <label className="flex items-center gap-2 font-medium">
            Scenario
            <select
              value={scenario}
              onChange={(event) => setScenario(event.target.value)}
              className="h-11 rounded-control border border-unsupported bg-surface px-2 text-sm text-ink sm:h-9"
            >
              {(fixtures?.FIXTURE_SCENARIOS ?? []).map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}

      <div className="max-w-[60rem]">
        <h1 className="font-serif text-[1.75rem] font-semibold leading-tight tracking-[-0.015em] text-ink">
          Ask a clinical question
        </h1>
        {!showWorkspace ? (
          <p className="mt-1 max-w-[68ch] text-[0.9375rem] leading-relaxed text-muted">
            Primum answers from retrieved passages, checks each claim against its source, and releases the answer
            only if its confidence clears the threshold for the risk level you set. Otherwise it says so and shows you
            the evidence. The threshold is certified on PubMedQA benchmark questions; for a question of your own it is
            an operating point, not a guarantee.
            {meta?.models?.llm ? (
              <>
                {" "}
                Answers are generated by <span className="font-medium text-ink">{meta.models.llm}</span>, the server&apos;s
                configured generator
                {health?.deployment_bundle === true
                  ? ", and judged by the deployment bundle certified for it; at a risk level where that bundle certified no threshold, every answer is withheld."
                  : health?.deployment_bundle === false
                    ? ". No deployment bundle exists for this generator, so answers are not scored against a certified threshold."
                    : "."}
              </>
            ) : null}
          </p>
        ) : null}
      </div>

      <div className="mt-5">
        <Composer
          value={composer}
          onChange={setComposer}
          onSubmit={() => void run(composer)}
          onStop={stop}
          running={running}
          alphaOptions={alphaOptions}
          retrieverOptions={retrieverOptions}
          defaultRetriever={meta?.default_retriever}
          textareaRef={textarea}
        />
      </div>

      {!showWorkspace ? (
        <div className="mt-8 grid gap-x-10 gap-y-8 lg:grid-cols-[minmax(0,1fr)_17rem]">
          <div className="min-w-0">
            {!FIXTURES && reachability === "down" ? (
              <div className="mb-7">
                <ApiDown what="Questions cannot be answered until it is running." />
              </div>
            ) : null}
            {!FIXTURES && reachability === "up" && health ? (
              <Readiness retriever={composer.retriever || meta?.default_retriever || null} isDefault={!composer.retriever} />
            ) : null}
            <h2 className="text-[0.8125rem] font-medium text-muted">Questions to try</h2>
            <ul className="mt-1.5 divide-y divide-rule border-y border-rule">
              {EXAMPLE_QUESTIONS.map((example) => (
                <li key={example.question}>
                  <button
                    type="button"
                    onClick={() => {
                      setComposer((value) => ({ ...value, question: example.question, benchmark: null }));
                      textarea.current?.focus();
                    }}
                    className="flex min-h-11 w-full flex-wrap items-baseline gap-x-3 rounded-control px-1 py-2 text-left hover:bg-surface"
                  >
                    <span className="font-serif text-[1.0625rem] leading-snug text-ink">{example.question}</span>
                    {example.note ? <span className="text-xs text-muted">{example.note}</span> : null}
                  </button>
                </li>
              ))}
            </ul>
            {!picking ? (
              <Button variant="quiet" className="mt-2" onClick={() => setPicking(true)}>
                Pick a PubMedQA benchmark question
              </Button>
            ) : (
              <BenchmarkPicker
                onClose={closePicker}
                onPick={(item: BenchmarkQuestion) => {
                  setComposer((value) => ({ ...value, question: item.question, benchmark: item }));
                  setPicking(false);
                  textarea.current?.focus();
                }}
              />
            )}

            {traces.length > 0 ? (
              <div className="mt-8">
                <h2 className="text-[0.8125rem] font-medium text-muted">Recent readings on this server</h2>
                <ul className="mt-1.5 divide-y divide-rule border-y border-rule">
                  {traces.map((trace) => (
                    <li key={trace.trace_id}>
                      <button
                        type="button"
                        onClick={() => void loadTrace(trace.trace_id)}
                        className="flex min-h-11 w-full items-baseline justify-between gap-4 rounded-control px-1 py-2 text-left hover:bg-surface"
                      >
                        <span className="min-w-0 truncate font-serif text-[1.0625rem] text-ink">{trace.question}</span>
                        <span className="shrink-0 text-xs text-muted">
                          {trace.answered === null ? "stored" : trace.answered ? "released" : "withheld"}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
          <PipelineRail state={state} />
        </div>
      ) : (
        <div
          ref={readingRef}
          className="mt-8 grid scroll-mt-28 gap-x-8 gap-y-8 border-t border-line pt-6 lg:grid-cols-[12.5rem_minmax(0,1fr)] xl:grid-cols-[12.5rem_minmax(0,1fr)_23.5rem]"
        >
          {/* On a phone the decision comes first; the pipeline follows it. */}
          <div className="order-2 lg:sticky lg:top-28 lg:order-none lg:self-start">
            <PipelineRail state={state} />
          </div>

          <div className="order-1 min-w-0 space-y-6 lg:order-none" aria-busy={running}>
            <p role="status" className="sr-only">
              {ANNOUNCE[outcome]}
            </p>
            {request ? (
              <header>
                <p className="text-xs text-muted">
                  {state.origin === "trace" ? "Stored reading" : state.origin === "fixture" ? "Fixture reading" : "Question"}
                  <span className="num">
                    {Number.isFinite(request.alpha) ? `, α = ${fmtAlpha(request.alpha)}` : ""}
                    {request.mode === "benchmark" ? ", benchmark mode" : ""}
                    {request.benchmarkQid ? `, PubMedQA ${request.benchmarkQid}` : ""}
                  </span>
                </p>
                <h2 className="mt-1 font-serif text-[1.375rem] font-semibold leading-snug tracking-[-0.01em] text-ink">
                  {request.question}
                </h2>
                {state.concepts && state.concepts.length > 0 ? (
                  <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1.5 text-xs text-muted">
                    <span>Concepts recognised:</span>
                    {state.concepts.map((concept, i) => {
                      const mesh = concept.mesh_ids[0];
                      const body = (
                        <>
                          <span className="text-ink">{concept.preferred}</span>
                          {mesh ? <span className="num font-mono text-[0.6875rem]">{mesh}</span> : null}
                        </>
                      );
                      return concept.url ? (
                        <a
                          key={`${concept.preferred}-${i}`}
                          href={concept.url}
                          target="_blank"
                          rel="noreferrer noopener"
                          className="inline-flex h-6 items-center gap-1.5 rounded-chip border border-line bg-surface px-1.5 hover:border-ink"
                        >
                          {body}
                          <ExternalGlyph size={11} />
                          <span className="sr-only">(opens MedlinePlus in a new tab)</span>
                        </a>
                      ) : (
                        <span
                          key={`${concept.preferred}-${i}`}
                          className="inline-flex h-6 items-center gap-1.5 rounded-chip border border-line bg-surface px-1.5"
                        >
                          {body}
                        </span>
                      );
                    })}
                  </p>
                ) : null}
              </header>
            ) : null}

            {state.triage?.level === "urgent" ? <UrgentNotice triage={state.triage} /> : null}

            {outcome === "emergency" && state.triage ? <Emergency triage={state.triage} /> : null}

            {outcome === "failed" && state.error ? (
              <Failure error={state.error} foundPassages={passageCount} onRetry={() => void run(composer)} />
            ) : null}

            {outcome === "answered" && state.confidence ? (
              <>
                <Released confidence={state.confidence} decision={state.answer?.decision ?? null} gold={gold} />
                <section aria-labelledby="answer-heading">
                  <h3 id="answer-heading" className="text-[0.9375rem] font-semibold text-ink">
                    {state.confidence.uncalibrated ? "Answer, not certified" : "Answer"}
                  </h3>
                  <div className="mt-2">{answerBlock}</div>
                </section>
                {claimsBlock}
              </>
            ) : null}

            {outcome === "withheld" && state.confidence ? (
              <>
                <Withheld
                  confidence={state.confidence}
                  decision={state.answer?.decision ?? null}
                  gold={gold}
                  passageCount={passageCount}
                  verdicts={state.verdicts ?? []}
                  alphaOptions={alphaOptions}
                  busy={running}
                  onAskAgain={(alpha) => {
                    const next = { ...composer, alpha, question: request?.question ?? composer.question };
                    setComposer(next);
                    void run(next);
                  }}
                />
                <details className="group rounded-card border border-line bg-surface">
                  <summary className="flex min-h-12 items-center justify-between gap-3 px-4 py-2 text-[0.9375rem] font-semibold text-ink sm:px-5">
                    <span>
                      <span className="group-open:hidden">Show the withheld draft</span>
                      <span className="hidden group-open:inline">Hide the withheld draft</span>
                    </span>
                    <span className="text-xs font-normal text-muted">not certified</span>
                  </summary>
                  <div className="space-y-6 border-t border-line px-4 pb-5 pt-4 sm:px-5">
                    <p className="flex items-start gap-2 rounded-control bg-raised px-3 py-2 text-sm leading-relaxed text-ink">
                      This draft failed the release test. It may be wrong, and it is shown only so you can see what
                      was held back and why.
                    </p>
                    {answerBlock}
                    {claimsBlock}
                  </div>
                </details>
              </>
            ) : null}

            {outcome === "pending" ? (
              <>
                <section aria-labelledby="decision-heading" className="rounded-card border border-line bg-surface p-4 sm:p-5">
                  <h2 id="decision-heading" className="font-serif text-[1.5rem] font-semibold leading-tight tracking-[-0.01em] text-ink">
                    Not yet certified
                  </h2>
                  <p className="mt-1.5 max-w-[70ch] text-sm leading-relaxed text-muted">
                    The draft is shown as it is written. It is released or withheld once every claim has been checked
                    and the confidence estimate is in.
                  </p>
                  {runningStage ? (
                    <p className="mt-3 text-sm font-medium text-ink">Now: {STAGE_COPY[runningStage].label.toLowerCase()}.</p>
                  ) : null}
                </section>
                {answerBlock ? (
                  <section aria-labelledby="answer-heading">
                    <h3 id="answer-heading" className="text-[0.9375rem] font-semibold text-ink">
                      Draft
                    </h3>
                    <div className="mt-2">{answerBlock}</div>
                  </section>
                ) : state.stages.generate.status === "running" ? (
                  <p className="text-sm text-muted">Waiting for the model.</p>
                ) : null}
                {claimsBlock}
              </>
            ) : null}
          </div>

          {outcome === "emergency" ? null : (
            <div className="order-3 min-w-0 lg:order-none lg:col-start-2 xl:col-start-3">
              <Evidence
                hits={state.hits}
                retriever={state.retriever}
                searching={state.stages.retrieve.status === "running"}
                citedIndices={cited}
                citedLabel={
                  outcome === "withheld" ? "Cited in the withheld draft" : outcome === "answered" ? "Cited in the answer" : "Cited in the draft"
                }
                activePassage={active.passage}
                activeSentence={active.sentence}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** What the running server is missing, said before a question fails for that reason. */
function Readiness({ retriever, isDefault }: { retriever: string | null; isDefault: boolean }) {
  const { health, meta } = useApi();
  const model = meta?.models?.llm ?? null;
  if (!health) return null;
  const copy: Record<Gap, React.ReactNode> = {
    api_key: (
      <>
        No API key is configured, so answers cannot be generated. Add <Code>ANTHROPIC_API_KEY</Code> to{" "}
        <Code>.env</Code> and restart the server.
      </>
    ),
    index: (
      <>
        No search index has been built. Run <Code>primum data build kb</Code> and then <Code>primum index build kb</Code>.
      </>
    ),
    encoder: (
      <>
        Questions will fail at retrieval: {isDefault ? "the server's default retriever, " : "the retriever "}
        <Code>{retriever}</Code>, needs the fine-tuned encoder, which has not been trained. Run{" "}
        <Code>primum train encoder</Code>, or choose a retriever that does not, such as <Code>bm25</Code> or{" "}
        <Code>hybrid-base</Code>.
      </>
    ),
    bundle: (
      <>
        No deployment bundle exists for {model ? <Code>{model}</Code> : "the configured generator"}, so answers are shown
        without a certified threshold. A bundle comes from an RQ3 run on that generator&apos;s own records, saved as{" "}
        <Code wrap>artifacts/deployment/{model ?? "<model>"}/bundle.json</Code>.
      </>
    ),
  };
  const gaps = readinessGaps(health, retriever).map((gap) => copy[gap]);
  if (gaps.length === 0) return null;
  return (
    <div className="mb-7 rounded-card border border-dashed border-(--frame) px-4 py-4 sm:px-5">
      <p className="text-sm font-medium text-ink">The server is running, with {gaps.length === 1 ? "one thing" : `${gaps.length} things`} not yet in place</p>
      <ul className="mt-1.5 space-y-1.5 text-sm leading-relaxed text-muted">
        {gaps.map((gap, i) => (
          <li key={i}>{gap}</li>
        ))}
      </ul>
    </div>
  );
}
