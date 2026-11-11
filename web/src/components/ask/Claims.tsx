"use client";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { CheckGlyph, CrossGlyph, RingGlyph } from "@/components/ui/glyphs";
import { fmtProb } from "@/lib/format";
import type { ClaimVerdict, VerdictLabel } from "@/lib/types";

import { VERDICT_COPY } from "./copy";
import { VerdictBar } from "./VerdictBar";

function VerdictGlyph({ label }: { label: VerdictLabel }) {
  if (label === "supported") return <CheckGlyph size={13} />;
  if (label === "contradicted") return <CrossGlyph size={13} />;
  return <RingGlyph size={13} />;
}

function VerdictTag({ label }: { label: VerdictLabel }) {
  return (
    <Badge tone={label}>
      <VerdictGlyph label={label} />
      {VERDICT_COPY[label].label}
    </Badge>
  );
}

/** The three NLI probabilities as one bar that sums to one, with the numbers printed beside it. */
function NliBar({ verdict }: { verdict: ClaimVerdict }) {
  const parts = [
    { key: "entail", value: verdict.p_entail, className: "bg-supported" },
    { key: "neutral", value: verdict.p_neutral, className: "bg-control" },
    { key: "contra", value: verdict.p_contra, className: "bg-contradicted" },
  ];
  return (
    <div className="num flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
      <div className="flex h-1.5 w-24 gap-0.5 overflow-hidden rounded-full" aria-hidden="true">
        {parts.map((part) =>
          part.value > 0.005 ? (
            <span key={part.key} className={part.className} style={{ width: `${part.value * 100}%` }} />
          ) : null,
        )}
      </div>
      <span>
        entail <span className="text-ink">{fmtProb(verdict.p_entail)}</span>
      </span>
      <span>
        neutral <span className="text-ink">{fmtProb(verdict.p_neutral)}</span>
      </span>
      <span>
        contradict <span className="text-ink">{fmtProb(verdict.p_contra)}</span>
      </span>
    </div>
  );
}

interface ClaimsProps {
  verdicts: ClaimVerdict[];
  onInspect: (passage: number | null, sentence: string | null) => void;
  onOpenPassage: (passage: number) => void;
}

/**
 * The claim ledger: every atomic statement in the answer, the verdict of the entailment
 * model, and the sentence of the source it was judged against. Pointing at or focusing a
 * claim marks that sentence inside its passage in the evidence column.
 */
export function Claims({ verdicts, onInspect, onOpenPassage }: ClaimsProps) {
  const counts = {
    supported: verdicts.filter((v) => v.label === "supported").length,
    unsupported: verdicts.filter((v) => v.label === "unsupported").length,
    contradicted: verdicts.filter((v) => v.label === "contradicted").length,
  };

  return (
    <Card aria-labelledby="claims-heading">
      <Card.Header>
        <Card.Heading>
          <Card.Title as="h3" id="claims-heading">
            Claim check
          </Card.Title>
          <Card.Description>Each claim in the answer, checked against the passage it cites</Card.Description>
        </Card.Heading>
        <div className="num flex min-w-0 flex-wrap items-center gap-1.5">
          <Badge tone="supported">
            <CheckGlyph size={12} />
            {counts.supported} supported
          </Badge>
          <Badge tone="unsupported">
            <RingGlyph size={12} />
            {counts.unsupported} unsupported
          </Badge>
          <Badge tone="contradicted">
            <CrossGlyph size={12} />
            {counts.contradicted} contradicted
          </Badge>
        </div>
      </Card.Header>

      <VerdictBar verdicts={verdicts} className="mt-3 px-4 sm:px-5" />

      {verdicts.length === 0 ? (
        <p className="px-4 py-4 text-sm text-muted sm:px-5">The answer made no checkable claim.</p>
      ) : (
        <ol className="mt-3 divide-y divide-rule border-t border-rule">
          {verdicts.map((verdict, i) => {
            const inspect = () => onInspect(verdict.best_hit, verdict.best_sentence);
            return (
              // Not a focus stop itself: focus reaches it through the passage button inside, which bubbles here.
              <li
                key={`${verdict.claim.text}-${i}`}
                onMouseEnter={inspect}
                onMouseLeave={() => onInspect(null, null)}
                onFocus={inspect}
                onBlur={(event) => {
                  if (!event.currentTarget.contains(event.relatedTarget)) onInspect(null, null);
                }}
                className="group px-4 py-3.5 last:rounded-b-card hover:bg-raised/60 focus-within:bg-raised/60 sm:px-5"
              >
                <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:gap-3">
                  <div className="sm:w-[7.5rem] sm:shrink-0 sm:pt-0.5">
                    <VerdictTag label={verdict.label} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="font-serif text-[1.0625rem] leading-snug text-ink">{verdict.claim.text}</p>

                    {verdict.best_sentence ? (
                      <figure className="mt-2 rounded-control bg-raised px-3 py-2">
                        <figcaption className="text-xs text-muted">
                          {VERDICT_COPY[verdict.label].sentence}
                          {verdict.best_hit ? (
                            <>
                              , passage{" "}
                              <button
                                type="button"
                                onClick={() => verdict.best_hit && onOpenPassage(verdict.best_hit)}
                                className="num -my-3 inline-flex min-h-11 items-center px-1 font-medium text-ink underline decoration-control underline-offset-2 hover:decoration-ink sm:my-0 sm:min-h-0"
                              >
                                {verdict.best_hit}
                              </button>
                            </>
                          ) : null}
                        </figcaption>
                        <blockquote className="mt-0.5 line-clamp-3 font-serif text-[0.9375rem] italic leading-normal text-ink group-hover:line-clamp-none group-focus-within:line-clamp-none">
                          {verdict.best_sentence}
                        </blockquote>
                      </figure>
                    ) : (
                      <p className="mt-2 text-xs text-muted">
                        No sentence in the retrieved passages was close to this claim.
                      </p>
                    )}

                    <div className="mt-2.5">
                      <NliBar verdict={verdict} />
                    </div>
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </Card>
  );
}
