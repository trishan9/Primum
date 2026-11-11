"use client";

import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { Stat, StatGroup } from "@/components/ui/stat";
import { CheckGlyph, CrossGlyph } from "@/components/ui/glyphs";
import { Skeleton } from "@/components/ui/skeleton";
import { fmtAlpha, fmtInt, fmtMetric, fmtPct } from "@/lib/format";
import { metricLabel } from "@/lib/results";

import { Picker } from "@/components/ui/picker";
import { learnedVsStated, statedConfidence } from "@/lib/charts/rq2";

import { certifiedAt, neverCertified, readBestRetrieval, readScoredAccuracy, type Rq3Summary } from "./readers";
import type { Loaded } from "./useExperiment";

const DATASET: Record<string, string> = { nfcorpus: "NFCorpus", pubmedqa: "PubMedQA" };

/** A figure that is still loading, missing, or present; never a guess. */
function Figure({ loaded, value, unit, lead, missing }: { loaded: Loaded<unknown>; value: string | null; unit?: ReactNode; lead?: boolean; missing: string }) {
  if (loaded.status === "loading")
    return (
      <dd className="mt-1 flex h-10 items-center max-[30rem]:col-start-2 max-[30rem]:row-span-3 max-[30rem]:row-start-1 max-[30rem]:mt-0">
        <Skeleton className={lead ? "h-8 w-24" : "h-7 w-20"} />
      </dd>
    );
  return (
    <Stat.Value lead={lead} unit={value !== null ? unit : undefined}>
      {value ?? <span className="text-faint" title={missing}>–</span>}
    </Stat.Value>
  );
}

function Why({ loaded, children, missing }: { loaded: Loaded<unknown>; children: ReactNode; missing: string }) {
  if (loaded.status === "loading") return <Stat.Note>Reading the results file</Stat.Note>;
  if (loaded.status !== "ready" || children === null) return <Stat.Note>{missing}</Stat.Note>;
  return <Stat.Note>{children}</Stat.Note>;
}

/**
 * What the learned confidence models did against the stated confidence, read from the intervals
 * in `comparisons`: a model beats it only when its AUROC-difference interval lies above zero.
 */
function LearnedNote({ versus }: { versus: NonNullable<ReturnType<typeof learnedVsStated>> }) {
  const n = versus.rows.length;
  if (n === 0) return null;
  if (versus.aurocBetter.length > 0)
    return (
      <>
        {versus.aurocBetter.length} of {n} learned models beat it (AUROC-difference interval above zero).
      </>
    );
  return (
    <>
      None of {n} learned models beats it (no AUROC-difference interval above zero).
    </>
  );
}

interface HeadlineProps {
  rq1: Loaded<unknown>;
  rq2: Loaded<unknown>;
  rq3: Loaded<unknown>;
  summary: Rq3Summary | null;
  alpha: number | null;
  onAlpha: (alpha: number) => void;
}

/**
 * The headline figures: one row of numbers, each computed from a results file when the page
 * loads, each with a note saying where it comes from. The lead figure answers the page's
 * question: did the certified threshold keep its promise at the chosen α?
 */
export function Headline({ rq1, rq2, rq3, summary, alpha, onAlpha }: HeadlineProps) {
  const record = summary ? certifiedAt(summary.records, alpha) : null;
  const delta = summary?.delta ?? null;
  const trivial = record ? neverCertified(record) : false;
  const violation = record?.violation ?? null;
  const coverage = record?.coverage ?? null;
  const accuracy = rq2.status === "ready" ? readScoredAccuracy(rq2.data) : null;
  const stated = rq2.status === "ready" ? statedConfidence(rq2.data) : null;
  const versus = rq2.status === "ready" ? learnedVsStated(rq2.data) : null;
  const retrieval = rq1.status === "ready" ? readBestRetrieval(rq1.data) : null;
  const condition = accuracy?.source?.match(/condition\s+([\w-]+)/i)?.[1] ?? null;
  // Non-breaking spaces keep "α = 0.10" on one line.
  const atAlpha = alpha !== null ? `at α\u00a0=\u00a0${fmtAlpha(alpha)}` : "at the certified α";
  const rq3Missing = rq3.status === "missing" ? "RQ3 was not computed for this run" : "No certified record at this level";

  return (
    <section aria-labelledby="headline-title" className="mt-8">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          <h2 id="headline-title" className="text-[0.9375rem] font-semibold leading-6 text-ink">
            Headline figures
          </h2>
          <p className="text-[0.8125rem] text-muted">Computed from the results files when this page loaded.</p>
        </div>
        {summary && summary.alphas.length > 1 && alpha !== null ? (
          <Picker
            legend="Target risk α"
            showLabel
            value={fmtAlpha(alpha)}
            options={summary.alphas.map((a) => ({ value: fmtAlpha(a), label: fmtAlpha(a) }))}
            onChange={(v) => onAlpha(Number.parseFloat(v))}
            segmentedFrom="sm"
          />
        ) : null}
      </div>

      <StatGroup columns={6} aria-busy={[rq1, rq2, rq3].some((l) => l.status === "loading")}>
        {/* The lead: did the certified threshold keep its promise? Two columns wide from 640px. */}
        <Stat className="sm:col-span-2">
          <Stat.Label>Violation rate {atAlpha}</Stat.Label>
          <Figure lead loaded={rq3} value={violation ? fmtPct(violation.value, 1) : null} missing={rq3Missing} />
          {rq3.status === "ready" && record ? (
            <dd className="mt-1.5">
              {trivial ? (
                <Badge tone="neutral">Nothing certified, all withheld</Badge>
              ) : violation && delta !== null ? (
                violation.value <= delta ? (
                  <Badge tone="supported">
                    <CheckGlyph size={12} /> Within δ = {fmtAlpha(delta)}
                  </Badge>
                ) : (
                  <Badge tone="contradicted">
                    <CrossGlyph size={12} /> Above δ = {fmtAlpha(delta)}
                  </Badge>
                )
              ) : null}
            </dd>
          ) : null}
          <Why loaded={rq3} missing={rq3Missing}>
            {violation && violation.lo !== null && violation.hi !== null
              ? `Share of splits whose test risk went above α; interval ${fmtPct(violation.lo, 1)} to ${fmtPct(violation.hi, 1)}.`
              : record
                ? "Share of splits whose test risk went above α."
                : null}
          </Why>
        </Stat>

        <Stat>
          <Stat.Label>Coverage {atAlpha}</Stat.Label>
          <Figure loaded={rq3} value={coverage ? fmtPct(coverage.value, 1) : null} missing={rq3Missing} />
          <Why loaded={rq3} missing={rq3Missing}>
            {coverage
              ? `Mean share of held-out questions answered${record?.certifiedRate !== null && record?.certifiedRate !== undefined ? `; a threshold was certified in ${fmtPct(record.certifiedRate, 1)} of splits` : ""}.`
              : null}
          </Why>
        </Stat>

        <Stat>
          <Stat.Label>Answer accuracy{condition ? `, ${condition}` : ""}</Stat.Label>
          <Figure loaded={rq2} value={accuracy ? fmtPct(accuracy.accuracy, 1) : null} missing="RQ2 was not computed for this run" />
          <Why loaded={rq2} missing="RQ2 was not computed for this run">
            {accuracy
              ? `${accuracy.correct !== null && accuracy.n !== null ? `${fmtInt(accuracy.correct)} of ${fmtInt(accuracy.n)} answers correct` : "Share of answers correct"}; the answers RQ2 and RQ3 score.`
              : null}
          </Why>
        </Stat>

        <Stat>
          <Stat.Label>Stated confidence AUROC</Stat.Label>
          <Figure loaded={rq2} value={stated ? fmtMetric(stated.estimate.value) : null} missing="No stated-confidence AUROC in the RQ2 file" />
          <Why loaded={rq2} missing="No stated-confidence AUROC in the RQ2 file">
            {stated ? (
              <>
                The generator&apos;s own confidence{stated.calibrator ? `, ${stated.calibrator}-calibrated` : ""}.{" "}
                {versus ? <LearnedNote versus={versus} /> : null}
              </>
            ) : null}
          </Why>
        </Stat>

        <Stat>
          <Stat.Label>Best {retrieval ? metricLabel(retrieval.metric) : "nDCG@10"}</Stat.Label>
          <Figure loaded={rq1} value={retrieval ? fmtMetric(retrieval.stat.value) : null} missing="No nDCG@10 in the RQ1 file" />
          <Why loaded={rq1} missing="No nDCG@10 in the RQ1 file">
            {retrieval ? (
              <>
                <span className="font-mono text-ink">{retrieval.row}</span> on {DATASET[retrieval.dataset] ?? retrieval.dataset}; retrieval uses no
                generator.
              </>
            ) : null}
          </Why>
        </Stat>
      </StatGroup>
    </section>
  );
}
