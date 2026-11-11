"use client";

import { useMemo, useState, useSyncExternalStore } from "react";

import { ApiDown } from "@/components/shell/ApiDown";
import { useApi } from "@/components/shell/ApiProvider";
import { PageHeader } from "@/components/ui/page-header";
import { StatusDot } from "@/components/ui/badge";
import { Tab, TabPanel, Tabs, TabsList } from "@/components/ui/tabs";
import { getExperimentsIndex, MAIN_RUN } from "@/lib/api";
import { runPathPrefix } from "@/lib/results";

import { Headline } from "./Headline";
import { isQuestionId, QUESTIONS, type QuestionId } from "./questions";
import { readRq3 } from "./readers";
import { ResearchSection } from "./ResearchSection";
import { BenchmarkSection } from "./BenchmarkSection";
import { ConfidenceSection } from "./ConfidenceSection";
import { GuaranteeSection } from "./GuaranteeSection";
import { RetrievalSection } from "./RetrievalSection";
import { RunDetails, RunSwitch } from "./RunPicker";
import { useExperiment, useLoaded, type Loaded } from "./useExperiment";

/* The URL hash names a tab (#rq2), so other pages can link straight to one question. */
function subscribeHash(onChange: () => void) {
  window.addEventListener("hashchange", onChange);
  return () => window.removeEventListener("hashchange", onChange);
}
const readHash = () => window.location.hash.slice(1);
const noHash = () => "";

const fetchIndex = (signal: AbortSignal) => getExperimentsIndex(signal);

const DOT: Record<Loaded<unknown>["status"], { tone: "ok" | "idle" | "busy" | "down"; word: string }> = {
  ready: { tone: "ok", word: "results loaded" },
  missing: { tone: "idle", word: "not run yet" },
  loading: { tone: "busy", word: "loading" },
  down: { tone: "down", word: "API unreachable" },
  error: { tone: "down", word: "could not be read" },
};

export function ExperimentsView() {
  const { reachability, meta } = useApi();
  const [pickedRun, setPickedRun] = useState(MAIN_RUN);
  const [pickedTab, setPickedTab] = useState<QuestionId | null>(null);
  const [pickedAlpha, setPickedAlpha] = useState<number | null>(null);
  const hash = useSyncExternalStore(subscribeHash, readHash, noHash);

  const index = useLoaded(fetchIndex, [] as const);
  const runs = index.status === "ready" ? (index.data.runs ?? []) : [];
  // A run the server no longer lists falls back to the main one.
  const run = runs.some((r) => r.run === pickedRun) ? pickedRun : MAIN_RUN;

  // Every file the page needs is requested at once, so nothing waits on anything else.
  const rq1 = useExperiment("rq1_retrieval");
  const training = useExperiment("encoder_training");
  const rq2 = useExperiment("rq2_confidence", run);
  const rq3 = useExperiment("rq3_conformal", run);
  const loaded: Record<QuestionId, Loaded<unknown>> = { rq1, rq2, rq3 };

  const serverDelta = typeof meta?.delta === "number" ? meta.delta : null;
  const summary = useMemo(() => (rq3.status === "ready" ? readRq3(rq3.data, serverDelta) : null), [rq3, serverDelta]);
  const alphas = summary?.alphas ?? [];
  const alpha = pickedAlpha !== null && alphas.includes(pickedAlpha) ? pickedAlpha : (summary?.configAlpha ?? alphas[0] ?? null);

  const tab: QuestionId = pickedTab ?? (isQuestionId(hash) ? hash : "rq1");
  const selectTab = (value: string) => {
    if (!isQuestionId(value)) return;
    setPickedTab(value);
    window.history.replaceState(null, "", `#${value}`);
  };

  return (
    <div>
      <PageHeader className="items-start">
        <PageHeader.Heading>
          <PageHeader.Title>Experiments</PageHeader.Title>
          <PageHeader.Description>
            Three research questions, each answered by one script and one results file. Every figure is read from those files when the page loads;
            where a file does not exist, the page says so and shows nothing in its place.
          </PageHeader.Description>
        </PageHeader.Heading>
        {runs.length > 1 ? (
          <PageHeader.Actions className="w-full lg:w-auto">
            <RunSwitch runs={runs} run={run} onChange={setPickedRun} />
          </PageHeader.Actions>
        ) : null}
      </PageHeader>
      {runs.length > 0 ? <RunDetails runs={runs} run={run} /> : null}

      {reachability === "down" ? (
        <div className="mt-6">
          <ApiDown
            what="Results files are served by it, so none can be shown."
            aside={
              <>
                <span className="font-medium text-ink">Nothing here is cached or typed in. </span>
                Each figure on this page is drawn from a results file at the moment the page loads. With the API unreachable there is no file to
                read, so the sections below stay empty.
              </>
            }
          />
        </div>
      ) : null}

      <Headline rq1={rq1} rq2={rq2} rq3={rq3} summary={summary} alpha={alpha} onAlpha={setPickedAlpha} />

      <Tabs value={tab} onValueChange={selectTab} className="mt-10">
        <div className="bg-ground lg:sticky lg:top-(--topbar-h) lg:z-20 lg:-mx-2 lg:px-2">
          <TabsList label="Research question">
            {QUESTIONS.map((question) => {
              const status = loaded[question.id].status;
              const dot = DOT[status];
              return (
                <Tab key={question.id} value={question.id}>
                  {/* A loaded file needs no mark; the dot appears only when something is wrong or pending. */}
                  {status !== "ready" ? <StatusDot tone={dot.tone} /> : null}
                  <span className="num hidden text-muted sm:inline">{question.label}</span>
                  <span className="sm:hidden">{question.short}</span>
                  <span className="hidden sm:inline">{question.topic}</span>
                  <span className="sr-only">, {dot.word}</span>
                </Tab>
              );
            })}
          </TabsList>
        </div>
        {QUESTIONS.map((question) => {
          const runForQuestion = question.generatorIndependent ? MAIN_RUN : run;
          const state = loaded[question.id];
          return (
            <TabPanel key={question.id} value={question.id} className="pt-6">
              <ResearchSection
                question={question}
                loaded={state}
                file={`${runPathPrefix(runForQuestion)}${question.file}`}
                otherRun={runForQuestion !== MAIN_RUN}
              >
                {state.status === "ready" ? (
                  question.id === "rq1" ? (
                    <RetrievalSection data={state.data} training={training} />
                  ) : question.id === "rq2" ? (
                    <ConfidenceSection data={state.data} run={run} benchmark={<BenchmarkSection run={run} />} />
                  ) : summary ? (
                    <GuaranteeSection data={state.data} run={run} summary={summary} alpha={alpha} />
                  ) : null
                ) : null}
              </ResearchSection>
            </TabPanel>
          );
        })}
      </Tabs>
    </div>
  );
}
