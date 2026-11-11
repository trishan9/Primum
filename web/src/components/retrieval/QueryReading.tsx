import { memo } from "react";

import { Card } from "@/components/ui/card";
import { ExternalGlyph, InfoGlyph } from "@/components/ui/glyphs";
import { Stat, StatGroup } from "@/components/ui/stat";
import type { RetrieveResponse } from "@/lib/types";

import type { Agreement } from "./retrievers";

/** How the server read the query: the concepts it linked and what it appended. */
export const QueryReading = memo(function QueryReading({ query, response }: { query: string; response: RetrieveResponse }) {
  const expanded = response.expanded_query ?? "";
  const added = expanded.startsWith(query) ? expanded.slice(query.length).trim() : expanded === query ? "" : null;

  return (
    <Card className="flex flex-col">
      <Card.Header>
        <Card.Heading>
          <Card.Title>How the query was read</Card.Title>
          <Card.Description>Medical terms linked to MeSH, and the synonyms lexical search adds.</Card.Description>
        </Card.Heading>
      </Card.Header>
      <Card.Body className="flex flex-1 flex-col gap-4">
        {response.concepts_available === false ? (
          <p className="flex items-start gap-2 text-sm leading-relaxed text-muted">
            <InfoGlyph size={15} className="mt-0.5 text-faint" />
            <span>
              Concepts are not available: this server has no concept dictionary (it is built with the <code className="font-mono text-[0.8125rem]">kb</code>{" "}
              corpus), so no term was linked and retrievers that expand queries with concepts are not offered.
            </span>
          </p>
        ) : response.concepts.length === 0 ? (
          <p className="text-sm text-muted">No term in the query matched the concept dictionary.</p>
        ) : (
          <ul className="flex flex-wrap gap-2" aria-label="Concepts recognised">
            {response.concepts.map((concept, i) => (
              <li
                key={`${concept.mesh_ids[0] ?? concept.preferred}-${i}`}
                className="flex min-w-0 max-w-full flex-col rounded-control border border-line bg-ground px-3 py-2"
              >
                <span className="flex flex-wrap items-baseline gap-x-2 text-sm">
                  <span className="text-muted">“{concept.surface}”</span>
                  <span aria-hidden="true" className="text-faint">
                    is
                  </span>
                  {concept.url ? (
                    <a
                      href={concept.url}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="inline-flex items-center gap-1 font-medium text-link underline decoration-link/40 underline-offset-4 hover:decoration-link"
                    >
                      {concept.preferred}
                      <ExternalGlyph size={12} />
                      <span className="sr-only"> (opens MedlinePlus in a new tab)</span>
                    </a>
                  ) : (
                    <span className="font-medium text-ink">{concept.preferred}</span>
                  )}
                </span>
                <span className="mt-0.5 text-xs text-muted">
                  MeSH <span className="font-mono">{concept.mesh_ids.join(", ") || "–"}</span>
                </span>
                {concept.synonyms.length > 0 ? (
                  <span className="text-xs text-muted">Also called {concept.synonyms.slice(0, 3).join(", ")}</span>
                ) : null}
              </li>
            ))}
          </ul>
        )}

        <div className="mt-auto rounded-control bg-raised px-3 py-2.5">
          <p className="text-xs font-medium text-muted">Searched as</p>
          {added === "" ? (
            <p className="mt-0.5 text-sm text-ink">
              {query} <span className="text-muted">(nothing added)</span>
            </p>
          ) : (
            <p className="mt-0.5 text-sm leading-relaxed text-ink">
              {added === null ? (
                expanded
              ) : (
                <>
                  {query} <mark className="rounded-[3px] bg-accent-soft px-1 text-accent-ink no-underline">{added}</mark>
                  <span className="mt-1 block text-xs text-muted">Highlighted words were added from the concepts.</span>
                </>
              )}
            </p>
          )}
        </div>
      </Card.Body>
    </Card>
  );
});

/**
 * Where the rankings agree, counted from this response, with the one-line reading of how the
 * passages spread over the lists (templated in `bumpTakeaway`).
 */
export const AgreementStats = memo(function AgreementStats({ agreement, k, summary }: { agreement: Agreement; k: number; summary?: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <StatGroup columns={3} aria-label="Agreement between rankings">
        <Stat>
          <Stat.Label>Distinct passages</Stat.Label>
          <Stat.Value unit={`of ${agreement.rankings * k} slots`}>{agreement.distinct}</Stat.Value>
          <Stat.Note className="mt-auto pt-2">Fewer means the retrievers found the same things.</Stat.Note>
        </Stat>
        <Stat>
          <Stat.Label>In every ranking</Stat.Label>
          <Stat.Value unit={`of top ${k}`}>{agreement.shared}</Stat.Value>
          <Stat.Note className="mt-auto pt-2">Passages all {agreement.rankings} retrievers returned.</Stat.Note>
        </Stat>
        <Stat>
          <Stat.Label>Same first passage</Stat.Label>
          <Stat.Value unit={`of ${agreement.rankings}`}>{agreement.sameTop}</Stat.Value>
          <Stat.Note className="mt-auto pt-2">Rankings that agree on the top result.</Stat.Note>
        </Stat>
      </StatGroup>
      {summary ? <p className="px-1 text-[0.8125rem] leading-[1.35rem] text-muted">{summary}</p> : null}
    </div>
  );
});
