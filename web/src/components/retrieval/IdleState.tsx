"use client";

import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { ArrowRightGlyph, SearchGlyph } from "@/components/ui/glyphs";

import { EXAMPLES, RETRIEVER_NOTES, familyOf } from "./retrievers";

/** Before the first query: queries to start from, how to read the result, and what each retriever does. */
export function IdleState({ available, onTry }: { available: string[]; onTry: (query: string) => void }) {
  return (
    <div className="grid items-stretch gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <Card className="flex flex-col">
        <Card.Header>
          <Card.Heading>
            <Card.Title>Start from a query</Card.Title>
            <Card.Description>Runs the comparison straight away. Retrieval calls no language model.</Card.Description>
          </Card.Heading>
        </Card.Header>
        <Card.Body>
          <ul className="flex flex-col gap-2">
            {EXAMPLES.map((example) => (
              <li key={example}>
                <button
                  type="button"
                  onClick={() => onTry(example)}
                  className="group flex min-h-12 w-full items-center gap-3 rounded-control border border-line bg-surface px-3 py-2.5 text-left transition-[border-color,background-color] duration-[var(--dur-fast)] hover:border-control hover:bg-ground"
                >
                  <SearchGlyph size={14} className="text-faint group-hover:text-accent" />
                  <span className="min-w-0 flex-1 text-sm leading-snug text-ink">{example}</span>
                  <ArrowRightGlyph
                    size={14}
                    className="text-faint transition-transform duration-[var(--dur-fast)] group-hover:translate-x-0.5 group-hover:text-ink"
                  />
                </button>
              </li>
            ))}
          </ul>
        </Card.Body>
        <div className="mx-4 flex-1 border-t border-rule pb-4 pt-4 sm:mx-5">
          <h3 className="text-meta font-medium text-ink">Reading the result</h3>
          <ul className="mt-3 flex flex-col gap-3 text-sm leading-relaxed text-muted">
            <Key
              mark={
                <svg width="28" height="14" viewBox="0 0 28 14" fill="none" stroke="var(--axis)">
                  <path d="M0 2 C 14 2, 14 12, 28 12" />
                </svg>
              }
              term="A line"
            >
              follows one passage from a ranking to the next. The steeper it runs, the more two retrievers disagree.
            </Key>
            <Key
              mark={
                <svg width="28" height="14" viewBox="0 0 28 14" fill="none" stroke="var(--axis)">
                  <path d="M0 7 H16" />
                  <circle cx="19.5" cy="7" r="2.5" />
                </svg>
              }
              term="A ring"
            >
              marks a passage the next ranking left out, or one it found that the ranking before did not.
            </Key>
            <Key mark={<span className="num rounded-chip border border-line px-1 text-xs text-muted">+3</span>} term="A badge">
              counts how far a passage moved against the column on its left.
            </Key>
          </ul>
        </div>
        <Card.Footer>Each comparison is ranked by the server, on its own index, when you send it.</Card.Footer>
      </Card>

      <Card className="flex flex-col">
        <Card.Header>
          <Card.Heading>
            <Card.Title>Retrievers on this server</Card.Title>
            <Card.Description>Lexical search matches words; dense search matches meaning; fusion merges both.</Card.Description>
          </Card.Heading>
        </Card.Header>
        <Card.Body flush className="flex-1 pb-1 pt-2">
          <dl className="divide-y divide-rule">
            {available.map((name) => (
              <div key={name} className="grid gap-x-4 gap-y-1.5 px-4 py-3 sm:grid-cols-[11rem_minmax(0,1fr)] sm:px-5">
                <dt className="flex flex-wrap items-center gap-2 sm:flex-col sm:items-start sm:gap-1">
                  <span className="font-mono text-[0.8125rem] text-ink">{name}</span>
                  <Badge tone="outline" className="min-h-5">
                    {familyOf(name)}
                  </Badge>
                </dt>
                <dd className="text-sm leading-relaxed text-muted">
                  {RETRIEVER_NOTES[name] ?? "Offered by the server; this page has no description for it."}
                </dd>
              </div>
            ))}
          </dl>
        </Card.Body>
      </Card>
    </div>
  );
}

function Key({ mark, term, children }: { mark: ReactNode; term: string; children: ReactNode }) {
  return (
    <li className="flex items-start gap-3">
      <span aria-hidden="true" className="flex h-[1.4rem] w-9 shrink-0 items-center justify-center">
        {mark}
      </span>
      <p>
        <strong className="font-medium text-ink">{term}</strong> {children}
      </p>
    </li>
  );
}
