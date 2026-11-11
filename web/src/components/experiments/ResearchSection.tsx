import type { ReactNode } from "react";

import { Code } from "@/components/ui/code";
import { EmptyBlock } from "@/components/ui/empty-state";
import { SkeletonText } from "@/components/ui/skeleton";
import { readProvenance } from "@/lib/results";

import { Disclosure } from "@/components/ui/disclosure";
import { Card } from "@/components/ui/card";
import { fmtWritten, Provenance } from "./parts";
import type { Question } from "./questions";
import type { Loaded } from "./useExperiment";

interface ResearchSectionProps {
  question: Question;
  loaded: Loaded<unknown>;
  /** The results file's path under results/, without the extension. */
  file: string;
  /** True when another run is picked but this file is not computed for it. */
  otherRun: boolean;
  /** The figures, drawn only when the file is ready. */
  children: ReactNode;
}

/**
 * One research question: its heading and the question in words, a quiet line saying where the
 * numbers come from (expandable to the full provenance), then either the figures or the state
 * that stands in for them (loading, not run, API down, unreadable) inside the same frame.
 */
export function ResearchSection({ question, loaded, file, otherRun, children }: ResearchSectionProps) {
  const written = loaded.status === "ready" ? fmtWritten(readProvenance(loaded.data).created) : null;
  return (
    <section aria-labelledby={`${question.id}-heading`} className="min-w-0">
      <header className="flex flex-col gap-x-8 gap-y-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 max-w-[72ch] flex-1">
          <h2 id={`${question.id}-heading`} className="text-lg font-semibold leading-7 tracking-[-0.01em] text-ink">
            {question.topic}
          </h2>
          <p className="mt-1 text-[0.9375rem] leading-relaxed text-muted">{question.question}</p>
        </div>
        <p className="num flex shrink-0 flex-wrap items-center gap-2 text-xs text-faint sm:pt-1">
          <Code>{question.command}</Code>
          {written ? <span>written {written}</span> : null}
        </p>
      </header>

      {loaded.status === "ready" ? (
        <Disclosure className="mt-2">
          <Disclosure.Summary>Provenance</Disclosure.Summary>
          <Disclosure.Body>
            <Provenance data={loaded.data} file={file}>
              {question.generatorIndependent ? (
                <>
                  <dt className="text-faint">Generator</dt>
                  <dd className="text-ink">None. Retrieval and encoder training use no generator, so this tab reads the main run whichever run is picked.</dd>
                </>
              ) : null}
            </Provenance>
          </Disclosure.Body>
        </Disclosure>
      ) : null}

      <div className="mt-4" aria-busy={loaded.status === "loading"}>
        {loaded.status === "loading" ? (
          <div className="grid gap-4 lg:grid-cols-2">
            {[0, 1].map((i) => (
              <Card as="div" key={i} className="p-5">
                <span className="sr-only">{i === 0 ? `Looking for results/${file}.json.` : ""}</span>
                <SkeletonText lines={2} className="max-w-sm" />
                <div className="mt-6 h-56 rounded-control bg-raised" />
              </Card>
            ))}
          </div>
        ) : null}

        {loaded.status === "missing" && otherRun ? (
          <EmptyBlock title={`${question.label} was not computed for this run`}>
            <p>
              <Code wrap>results/{file}.json</Code> does not exist, so nothing is drawn here. Pick another run above to see {question.label}.
            </p>
          </EmptyBlock>
        ) : null}

        {loaded.status === "missing" && !otherRun ? (
          <EmptyBlock
            title="This experiment has not been run yet"
            aside={
              <>
                <span className="font-medium text-ink">What it will report. </span>
                {question.willShow}
              </>
            }
          >
            <p>
              Run <Code>{question.command}</Code> from the project root. It writes <Code wrap>results/{question.file}.json</Code>, and this section
              draws itself from that file.
            </p>
          </EmptyBlock>
        ) : null}

        {loaded.status === "down" ? (
          <EmptyBlock
            title="Not loaded"
            aside={
              <>
                <span className="font-medium text-ink">What the experiment reports. </span>
                {question.willShow}
              </>
            }
          >
            <p>
              Results are read through the API, which is not answering, so this page cannot tell whether <Code>{question.command}</Code> has been
              run. No figure is shown in its place.
            </p>
          </EmptyBlock>
        ) : null}

        {loaded.status === "error" ? (
          <EmptyBlock title="The results file could not be read">
            <p>{loaded.message}</p>
          </EmptyBlock>
        ) : null}

        {loaded.status === "ready" ? children : null}
      </div>
    </section>
  );
}
