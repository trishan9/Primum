"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton, SkeletonText } from "@/components/ui/skeleton";
import { cn } from "@/components/ui/cn";
import { DocumentGlyph, ExternalGlyph } from "@/components/ui/glyphs";
import { fmtScore } from "@/lib/format";
import type { Hit } from "@/lib/types";

import { SOURCE_COPY } from "./copy";
import { ScoreProfile } from "./ScoreProfile";

/** Splits passage text around the sentence a claim was checked against, if it occurs verbatim. */
function splitAround(text: string, sentence: string | null): [string, string, string] | null {
  if (!sentence) return null;
  const needle = sentence.trim();
  if (needle.length < 8) return null;
  let index = text.indexOf(needle);
  let length = needle.length;
  if (index === -1) {
    // The verifier may have normalised whitespace; try a whitespace-tolerant match.
    const pattern = needle.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
    const match = new RegExp(pattern).exec(text);
    if (!match) return null;
    index = match.index;
    length = match[0].length;
  }
  return [text.slice(0, index), text.slice(index, index + length), text.slice(index + length)];
}

const STAGE_SCORES: readonly {
  key: string;
  label: string;
  rankKey?: string;
}[] = [
  { key: "bm25", label: "BM25", rankKey: "bm25_rank" },
  { key: "dense", label: "Dense", rankKey: "dense_rank" },
  { key: "rrf", label: "RRF" },
  { key: "rerank", label: "Rerank" },
];

function PassageId({ hit }: { hit: Hit }) {
  const { passage } = hit;
  if (passage.source === "pubmed" && /^\d+$/.test(passage.doc_id)) return <>PMID {passage.doc_id}</>;
  return <>{passage.doc_id}</>;
}

interface PassageCardProps {
  hit: Hit;
  index: number;
  cited: boolean;
  citedLabel: string;
  active: boolean;
  sentence: string | null;
}

function PassageCard({ hit, index, cited, citedLabel, active, sentence }: PassageCardProps) {
  const [expanded, setExpanded] = useState(false);
  const { passage, components } = hit;
  const source = SOURCE_COPY[passage.source] ?? {
    label: passage.source,
    open: "Open source",
  };
  const parts = active ? splitAround(passage.text, sentence) : null;
  const long = passage.text.length > 320;
  const open = expanded || parts !== null || !long;
  const scores = STAGE_SCORES.filter((s) => typeof components[s.key] === "number");

  return (
    <li>
      <article
        id={`passage-${index}`}
        aria-labelledby={`passage-${index}-title`}
        className={cn(
          "scroll-mt-20 px-4 py-4 transition-[background-color,box-shadow] duration-(--dur) sm:px-5",
          active ? "bg-accent-soft/60 shadow-[inset_0_0_0_1.5px_var(--accent)]" : "",
        )}
      >
        <header className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted">
          <span
            className={cn(
              "num flex size-6 items-center justify-center rounded-full text-xs font-semibold",
              cited ? "bg-accent text-on-accent" : "bg-raised text-ink ring-1 ring-inset ring-line",
            )}
          >
            {index}
            <span className="sr-only">, {cited ? citedLabel.toLowerCase() : "retrieved, not cited"}</span>
          </span>
          <span className="font-medium text-ink">{source.label}</span>
          <span className="num min-w-0 truncate font-mono text-[0.75rem]">
            <PassageId hit={hit} />
          </span>
          {cited ? (
            <Badge tone="accent" className="ml-auto" aria-hidden="true">
              {citedLabel}
            </Badge>
          ) : (
            <span className="ml-auto" aria-hidden="true">
              Not cited
            </span>
          )}
        </header>

        <h3 id={`passage-${index}-title`} className="mt-2 text-sm font-semibold leading-snug text-ink">
          {passage.title || "Untitled passage"}
        </h3>

        <p className={cn("mt-1 font-serif text-[0.9688rem] leading-[1.55] text-ink", !open && "line-clamp-4")}>
          {parts ? (
            <>
              {parts[0]}
              <mark>{parts[1]}</mark>
              {parts[2]}
            </>
          ) : (
            passage.text
          )}
        </p>

        <div className="mt-1.5 flex flex-wrap items-center gap-x-4 text-[0.8125rem]">
          {long ? (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              aria-expanded={open}
              className="min-h-11 text-ink underline decoration-control underline-offset-4 hover:decoration-ink sm:min-h-8"
            >
              {expanded ? "Show less" : "Show the full passage"}
            </button>
          ) : null}
          {passage.url ? (
            <a
              href={passage.url}
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex min-h-11 items-center gap-1 text-link underline decoration-1 underline-offset-4 sm:min-h-8"
            >
              {source.open}
              <ExternalGlyph size={13} />
              <span className="sr-only">(opens in a new tab)</span>
            </a>
          ) : null}
        </div>

        {scores.length > 0 ? (
          <dl className="num mt-2 grid grid-cols-4 overflow-hidden rounded-control bg-raised text-xs">
            {STAGE_SCORES.map((s) => {
              const value = components[s.key];
              const rank = s.rankKey ? components[s.rankKey] : undefined;
              return (
                <div key={s.key} className="px-2.5 py-1.5 [&+&]:border-l [&+&]:border-rule">
                  <dt className="text-muted">{s.label}</dt>
                  <dd className="mt-0.5 text-ink">
                    {typeof value === "number" ? fmtScore(value) : <span className="text-muted">–</span>}
                    {typeof rank === "number" ? <span className="text-muted"> #{rank}</span> : null}
                  </dd>
                </div>
              );
            })}
          </dl>
        ) : null}
      </article>
    </li>
  );
}

interface EvidenceProps {
  hits: Hit[] | null;
  retriever: string | null;
  searching: boolean;
  citedIndices: number[];
  /** How a cited passage is described: the answer may be a draft, or withheld. */
  citedLabel: string;
  activePassage: number | null;
  activeSentence: string | null;
  /** Said in place of the idle hint when nothing will be retrieved (the run stopped before retrieval). */
  emptyNote?: string;
  className?: string;
}

/** The sources panel: every retrieved passage, numbered as the answer cites it, with the scores that ranked it. */
export function Evidence({
  hits,
  retriever,
  searching,
  citedIndices,
  citedLabel,
  activePassage,
  activeSentence,
  emptyNote,
  className,
}: EvidenceProps) {
  const citedCount = hits ? hits.filter((_, i) => citedIndices.includes(i + 1)).length : 0;
  return (
    <Card as="aside" aria-labelledby="evidence-heading" className={cn("flex flex-col", className)}>
      <Card.Header className="pb-3">
        <Card.Heading>
          <Card.Title id="evidence-heading" className="flex items-center gap-2">
            <DocumentGlyph size={15} className="text-muted" />
            Sources
          </Card.Title>
          <Card.Description className="num">
            {emptyNote && !hits?.length
              ? "None retrieved"
              : hits && hits.length > 0
              ? `${hits.length} ${hits.length === 1 ? "passage" : "passages"}${retriever ? `, ranked by ${retriever}` : ""}${citedCount ? `; ${citedCount} cited` : ""}`
              : searching
                ? "Searching the index"
                : "Waiting for retrieval"}
          </Card.Description>
        </Card.Heading>
      </Card.Header>

      {/* Focusable so a keyboard can scroll the list on its own. */}
      <div
        tabIndex={0}
        role="region"
        aria-label="Retrieved passages"
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain rounded-b-card border-t border-rule focus-visible:outline-offset-[-2px]"
        aria-busy={searching && hits === null}
      >
        {hits === null && searching ? (
          <ul className="divide-y divide-rule">
            {[0, 1, 2].map((i) => (
              <li key={i} className="px-4 py-4 sm:px-5">
                <Skeleton className="h-5 w-32" />
                <Skeleton className="mt-3 h-4 w-4/5" />
                <SkeletonText lines={3} className="mt-2.5" />
              </li>
            ))}
          </ul>
        ) : null}

        {hits === null && !searching ? (
          <p className="px-4 py-5 text-sm leading-relaxed text-muted sm:px-5">
            {emptyNote ??
              "Retrieved passages appear here, each with its source, a link to it, and the scores that ranked it."}
          </p>
        ) : null}

        {hits !== null && hits.length === 0 && emptyNote ? (
          <p className="px-4 py-5 text-sm leading-relaxed text-muted sm:px-5">{emptyNote}</p>
        ) : null}

        {hits !== null && hits.length === 0 && !emptyNote ? (
          <p className="px-4 py-5 text-sm leading-relaxed text-muted sm:px-5">
            The index returned no passage for this question.
          </p>
        ) : null}

        {hits && hits.length > 0 ? (
          <ScoreProfile hits={hits} retriever={retriever} cited={citedIndices} className="m-3 sm:m-4" />
        ) : null}

        {hits && hits.length > 0 ? (
          <ol className="divide-y divide-rule border-t border-rule">
            {hits.map((hit, i) => (
              <PassageCard
                key={hit.passage.passage_id}
                hit={hit}
                index={i + 1}
                cited={citedIndices.includes(i + 1)}
                citedLabel={citedLabel}
                active={activePassage === i + 1}
                sentence={activePassage === i + 1 ? activeSentence : null}
              />
            ))}
          </ol>
        ) : null}
      </div>
    </Card>
  );
}
