"use client";

import { useState } from "react";

import { ExternalGlyph } from "@/components/ui/glyphs";
import { fmtScore } from "@/lib/format";
import type { Hit } from "@/lib/types";

import { SOURCE_COPY } from "./copy";

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

const STAGE_SCORES: readonly { key: string; label: string; rankKey?: string }[] = [
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
  const source = SOURCE_COPY[passage.source] ?? { label: passage.source, open: "Open source" };
  const parts = active ? splitAround(passage.text, sentence) : null;
  const long = passage.text.length > 420;
  const open = expanded || parts !== null || !long;
  const scores = STAGE_SCORES.filter((s) => typeof components[s.key] === "number");

  return (
    <article
      id={`passage-${index}`}
      aria-labelledby={`passage-${index}-title`}
      className={`scroll-mt-28 rounded-card border bg-surface p-4 outline-1 -outline-offset-1 transition-[border-color,outline-color] duration-150 ${
        active ? "border-ink outline outline-ink" : "border-line outline-transparent"
      }`}
    >
      <header className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
        <span className="num flex h-6 min-w-6 items-center justify-center rounded-chip bg-primary px-1.5 text-xs font-semibold text-on-primary">
          {index}
        </span>
        <span className="font-medium text-ink">{source.label}</span>
        <span className="num min-w-0 truncate font-mono text-[0.75rem]">
          <PassageId hit={hit} />
        </span>
        <span className="ml-auto">{cited ? citedLabel : "Retrieved, not cited"}</span>
      </header>

      <h3 id={`passage-${index}-title`} className="mt-2.5 font-serif text-[1.0625rem] font-semibold leading-snug text-ink">
        {passage.title || "Untitled passage"}
      </h3>

      <p className={`mt-1.5 font-serif text-[0.9688rem] leading-[1.55] text-ink ${open ? "" : "line-clamp-4"}`}>
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

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[0.8125rem]">
        {long ? (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={open}
            className="min-h-11 text-ink underline decoration-control underline-offset-4 hover:decoration-ink sm:min-h-0"
          >
            {expanded ? "Show less" : "Show the full passage"}
          </button>
        ) : null}
        {passage.url ? (
          <a
            href={passage.url}
            target="_blank"
            rel="noreferrer noopener"
            className="inline-flex min-h-11 items-center gap-1 text-link underline decoration-1 underline-offset-4 sm:min-h-0"
          >
            {source.open}
            <ExternalGlyph size={13} />
            <span className="sr-only">(opens in a new tab)</span>
          </a>
        ) : null}
      </div>

      {scores.length > 0 ? (
        <dl className="num mt-3 grid grid-cols-4 gap-2 border-t border-rule pt-3 text-xs">
          {STAGE_SCORES.map((s) => {
            const value = components[s.key];
            const rank = s.rankKey ? components[s.rankKey] : undefined;
            return (
              <div key={s.key}>
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
}

export function Evidence({ hits, retriever, searching, citedIndices, citedLabel, activePassage, activeSentence }: EvidenceProps) {
  return (
    <section aria-labelledby="evidence-heading">
      <div className="flex items-baseline justify-between gap-3">
        <h2 id="evidence-heading" className="text-[0.8125rem] font-medium text-muted">
          Evidence
        </h2>
        {hits && hits.length > 0 ? (
          <p className="num text-xs text-muted">
            {hits.length} {hits.length === 1 ? "passage" : "passages"}
            {retriever ? `, ranked by ${retriever}` : ""}
          </p>
        ) : null}
      </div>

      <div className="mt-3 space-y-3">
        {hits === null && searching ? (
          <div aria-busy="true" className="space-y-3">
            <p className="text-sm text-muted">Searching the index.</p>
            {[0, 1, 2].map((i) => (
              <div key={i} className="rounded-card border border-line bg-surface p-4">
                <div className="h-6 w-28 rounded-chip bg-raised" />
                <div className="mt-3 h-5 w-4/5 rounded-chip bg-raised" />
                <div className="mt-2 h-16 rounded-chip bg-raised" />
              </div>
            ))}
          </div>
        ) : null}

        {hits === null && !searching ? (
          <div className="rounded-card border border-dashed border-(--frame) px-4 py-5 text-sm leading-relaxed text-muted">
            Retrieved passages appear here, each with its source, a link to it, and the scores that ranked it.
          </div>
        ) : null}

        {hits !== null && hits.length === 0 ? (
          <div className="rounded-card border border-dashed border-(--frame) px-4 py-5 text-sm leading-relaxed text-muted">
            The index returned no passage for this question.
          </div>
        ) : null}

        {hits?.map((hit, i) => (
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
      </div>
    </section>
  );
}
