import type { ReactNode } from "react";

import { Card } from "@/components/ui/card";
import { cn } from "@/components/ui/cn";
import {
  CheckGlyph,
  ExternalGlyph,
  GaugeGlyph,
  LinkNodesGlyph,
  PenGlyph,
  SearchGlyph,
  ShieldGlyph,
} from "@/components/ui/glyphs";

import { ANALOGIES, SOURCES, STAGES, type AnalogyName, type StageGlyph } from "./content";

/*
 * The reading surface of the Method page, as small server components composed in the page:
 *
 *   <Doc>
 *     <Doc.Section id="retrieval" title="Retrieval: two searches, one ranking">
 *       <p>…</p>
 *       <Doc.Analogy name="retrieval" />
 *       <Doc.Formula caption="…" label="…">…</Doc.Formula>
 *       <Doc.Built>…</Doc.Built>
 *     </Doc.Section>
 *   </Doc>
 *
 * `Doc` is a size container. When it is wide enough (54rem) for a 36rem prose column and a 16rem
 * margin, each analogy floats into the margin beside the paragraphs that follow it, the way a
 * journal sets a side note; narrower, it sits in the text where it was placed. It is rendered once.
 * Wide blocks (the pipeline, the explainer) clear the margin and take the full width.
 */

function DocRoot({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("@container min-w-0", className)}>{children}</div>;
}

function DocSection({ id, title, children }: { id: string; title: ReactNode; children: ReactNode }) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-title`}
      className={cn(
        "flow-root pt-12 first:pt-0",
        "font-serif text-reading text-ink",
        "[&>p]:max-w-[36rem] [&>ul]:max-w-[36rem] [&>*+*]:mt-[1em]",
        "[&_a:not([data-plain])]:text-link [&_a:not([data-plain])]:underline [&_a:not([data-plain])]:decoration-[0.06em] [&_a:not([data-plain])]:underline-offset-[0.18em] [&_a:not([data-plain]):hover]:decoration-[0.12em]",
      )}
    >
      <h2
        id={`${id}-title`}
        className="max-w-[36rem] text-balance font-serif text-[1.5rem] font-semibold leading-[1.25] tracking-[-0.01em] text-ink sm:text-[1.625rem]"
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

/** The one sentence the page exists to defend, a step up from the reading size. */
function DocLead({ children }: { children: ReactNode }) {
  return <p className="text-[1.3125rem] leading-[1.5] sm:text-[1.4375rem]">{children}</p>;
}

/**
 * A side note: in the margin beside the paragraphs that follow it on wide containers, in the text
 * where it was placed on narrow ones. Compose its parts:
 *
 *   <Doc.Note label="An analogy">
 *     <Doc.Note.Body>…</Doc.Note.Body>
 *     <Doc.Note.Footer>…</Doc.Note.Footer>
 *   </Doc.Note>
 */
function DocNoteRoot({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Card
      as="aside"
      variant="inset"
      aria-label={label}
      className={cn(
        "my-7 max-w-[36rem] font-sans",
        "@min-[54rem]:float-right @min-[54rem]:clear-right @min-[54rem]:mt-2 @min-[54rem]:mb-6 @min-[54rem]:ml-10 @min-[54rem]:w-[16rem] @min-[54rem]:max-w-none",
      )}
    >
      <p className="px-4 pt-3.5 text-meta font-semibold text-ink sm:px-5 @min-[54rem]:px-4">{label}</p>
      {children}
    </Card>
  );
}

function DocNoteBody({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div
      className={cn(
        "px-4 pt-1.5 pb-4 font-serif text-[1.0625rem] leading-[1.5] text-ink sm:px-5 @min-[54rem]:px-4 @min-[54rem]:text-[1rem]",
        className,
      )}
    >
      {children}
    </div>
  );
}

function DocNoteFooter({ children }: { children: ReactNode }) {
  return (
    <div className="border-t border-line px-4 pt-3 pb-3.5 text-meta leading-[1.45] text-muted sm:px-5 @min-[54rem]:px-4">
      {children}
    </div>
  );
}

const DocNote = Object.assign(DocNoteRoot, { Body: DocNoteBody, Footer: DocNoteFooter });

/** One analogy per idea, with the point at which it stops holding. */
function DocAnalogy({ name }: { name: AnalogyName }) {
  const { text, limit } = ANALOGIES[name];
  return (
    <DocNote label="An analogy">
      <DocNote.Body>
        <p>{text}</p>
      </DocNote.Body>
      <DocNote.Footer>
        <p>
          <span className="font-medium text-ink">Where it stops being true: </span>
          {limit}
        </p>
      </DocNote.Footer>
    </DocNote>
  );
}

/** A short list of terms and what they mean, for a margin note. */
export function TermList({ children }: { children: ReactNode }) {
  return <dl className="space-y-2.5 font-sans text-sm leading-snug">{children}</dl>;
}

export function Term({ name, children }: { name: ReactNode; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-2">
      <dt className="font-serif text-[1.0625rem] italic leading-tight text-ink">{name}</dt>
      <dd className="text-muted">{children}</dd>
    </div>
  );
}

/** A displayed formula, set in a well with a plain-words caption and a spoken label. */
function DocFormula({ caption, label, children }: { caption: string; label: string; children: ReactNode }) {
  return (
    <Card as="figure" className="max-w-[36rem] overflow-hidden">
      <figcaption className="border-b border-rule px-4 py-2 font-sans text-xs font-medium text-muted">{caption}</figcaption>
      <p
        role="math"
        aria-label={label}
        className="overflow-x-auto whitespace-nowrap px-4 py-4 font-serif text-[1.25rem] italic leading-normal text-ink sm:px-5"
      >
        {children}
      </p>
    </Card>
  );
}

/** Where the idea lives in the code base, in the interface face. */
function DocBuilt({ children }: { children: ReactNode }) {
  return (
    <div className="max-w-[36rem] rounded-control bg-raised px-4 py-3 font-sans text-sm leading-relaxed text-ink">
      <p>
        <span className="font-semibold">In the code: </span>
        <span className="text-muted">{children}</span>
      </p>
    </div>
  );
}

/** A block that takes the full width of the reading area, clear of any margin note. */
function DocWide({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("clear-both mt-8 font-sans text-base", className)}>{children}</div>;
}

export const Doc = Object.assign(DocRoot, {
  Section: DocSection,
  Lead: DocLead,
  Note: DocNote,
  Analogy: DocAnalogy,
  Formula: DocFormula,
  Built: DocBuilt,
  Wide: DocWide,
});

const STAGE_GLYPH: Record<StageGlyph, (props: { className?: string; size?: number }) => ReactNode> = {
  shield: ShieldGlyph,
  concepts: LinkNodesGlyph,
  search: SearchGlyph,
  pen: PenGlyph,
  check: CheckGlyph,
  gauge: GaugeGlyph,
};

/**
 * The six stages as the agent trace in the Ask view draws them: the same glyphs and names, in
 * order, each with what it does and what it hands on.
 */
export function PipelineStages() {
  return (
    <ol className="grid gap-px overflow-hidden rounded-card border border-line bg-line sm:grid-cols-2 @min-[48rem]:grid-cols-3">
      {STAGES.map((stage, i) => {
        const Glyph = STAGE_GLYPH[stage.glyph];
        return (
          <li key={stage.name} className="flex min-w-0 flex-col bg-surface px-4 py-4 sm:px-5">
            <p className="flex items-center gap-2.5 text-[0.9375rem] font-semibold text-ink">
              <span
                aria-hidden="true"
                className="flex size-7 shrink-0 items-center justify-center rounded-control border border-line text-muted"
              >
                <Glyph size={15} />
              </span>
              <span>
                <span className="sr-only">Stage {i + 1}: </span>
                {stage.name}
              </span>
              <span aria-hidden="true" className="num ml-auto text-xs font-normal text-faint">
                {i + 1} of {STAGES.length}
              </span>
            </p>
            <p className="mt-2.5 text-sm leading-relaxed text-ink">{stage.does}</p>
            <p className="mt-auto pt-3 text-meta text-muted">
              <span className="text-faint">Produces </span>
              {stage.gives}
            </p>
          </li>
        );
      })}
    </ol>
  );
}

/** The limits of the guarantee, each one a row a reader can find again. */
export function LimitList({ children }: { children: ReactNode }) {
  return (
    <Card as="ul" className="max-w-[36rem] divide-y divide-rule overflow-hidden">
      {children}
    </Card>
  );
}

export function LimitItem({ children }: { children: ReactNode }) {
  return <li className="px-4 py-4 text-[1.0625rem] leading-[1.55] sm:px-5">{children}</li>;
}

/** The openly licensed corpora, each a row with its outbound link. */
export function SourceList() {
  return (
    <Card as="ul" className="max-w-[36rem] divide-y divide-rule overflow-hidden font-sans">
      {SOURCES.map((source) => (
        <li key={source.name} className="px-4 py-3.5 sm:px-5">
          <a
            data-plain=""
            href={source.href}
            target="_blank"
            rel="noreferrer noopener"
            className="inline-flex min-h-11 items-center gap-1.5 text-[0.9375rem] font-semibold text-link underline-offset-4 hover:underline sm:min-h-0"
          >
            {source.name}
            <ExternalGlyph size={13} />
            <span className="sr-only">(opens in a new tab)</span>
          </a>
          <p className="mt-0.5 text-sm leading-relaxed text-muted">{source.text.charAt(0).toUpperCase() + source.text.slice(1)}</p>
        </li>
      ))}
    </Card>
  );
}
