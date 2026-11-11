import { memo, type Ref } from "react";

import { SOURCE_COPY } from "@/components/ask/copy";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { CrossGlyph, ExternalGlyph } from "@/components/ui/glyphs";
import { fmtScore } from "@/lib/format";
import type { Hit } from "@/lib/types";

import { docLabel, scoreKind } from "./retrievers";

export interface PassagePlacement {
  retriever: string;
  rank: number | null;
  score: number | null;
}

/** The selected passage, in full, with where each retriever placed it. */
export const PassageDetail = memo(function PassageDetail({
  hit,
  placements,
  k,
  onClose,
  ref,
}: {
  hit: Hit;
  placements: PassagePlacement[];
  k: number;
  onClose: () => void;
  ref?: Ref<HTMLDivElement>;
}) {
  const { passage } = hit;
  const source = SOURCE_COPY[passage.source];
  return (
    <div ref={ref} className="scroll-mt-20">
    <Card aria-labelledby="passage-detail-title">
      <Card.Header>
        <Card.Heading>
          <p className="flex flex-wrap items-center gap-2 text-xs text-muted">
            <Badge tone="accent">{source?.label ?? passage.source}</Badge>
            {passage.title ? <span className="font-mono">{docLabel(passage.source, passage.doc_id)}</span> : null}
          </p>
          <Card.Title id="passage-detail-title" className="mt-2 font-serif text-[1.25rem] leading-snug">
            {passage.title || docLabel(passage.source, passage.doc_id)}
          </Card.Title>
        </Card.Heading>
        <Card.Actions>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <CrossGlyph size={13} />
            Close
          </Button>
        </Card.Actions>
      </Card.Header>
      <Card.Body className="grid gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,1fr)_19rem]">
        <div className="min-w-0">
          <p className="reading max-w-[68ch] text-ink">{passage.text}</p>
          {passage.url ? (
            <a
              href={passage.url}
              target="_blank"
              rel="noreferrer noopener"
              className="mt-3 inline-flex min-h-11 items-center gap-1.5 text-sm text-link underline decoration-link/40 underline-offset-4 hover:decoration-link sm:min-h-0"
            >
              {source?.open ?? "Open source"}
              <ExternalGlyph size={13} />
              <span className="sr-only">(opens in a new tab)</span>
            </a>
          ) : null}
        </div>
        <div className="self-start rounded-control bg-raised p-3">
          <table className="num w-full text-sm">
            <caption className="pb-2 text-left text-xs font-medium text-muted">Where each retriever placed it</caption>
            <thead className="sr-only">
              <tr>
                <th scope="col">Retriever</th>
                <th scope="col">Rank</th>
                <th scope="col">Score</th>
              </tr>
            </thead>
            <tbody>
              {placements.map((row) => (
                <tr key={row.retriever} className="border-t border-line first:border-t-0">
                  <th scope="row" className="py-2 pr-2 text-left font-mono text-[0.8125rem] font-normal text-ink">
                    {row.retriever}
                  </th>
                  <td className="py-2 pr-3 text-right">
                    {row.rank === null ? (
                      <span className="text-xs text-muted">not in top {k}</span>
                    ) : (
                      <span className="font-semibold text-ink">#{row.rank}</span>
                    )}
                  </td>
                  <td className="w-20 py-2 text-right text-xs text-muted">
                    {row.score === null ? "–" : `${scoreKind(row.retriever)} ${fmtScore(row.score)}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card.Body>
    </Card>
    </div>
  );
});
