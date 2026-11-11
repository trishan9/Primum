"use client";

import { ApiDown } from "@/components/shell/ApiDown";
import { useApi } from "@/components/shell/ApiProvider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Code } from "@/components/ui/code";
import { CheckGlyph, ChevronGlyph, WarningGlyph } from "@/components/ui/glyphs";
import type { AskError } from "@/lib/ask-state";
import { fmtAlpha, fmtPct, fmtProb, fmtUsd } from "@/lib/format";
import type { ClaimVerdict, ConfidenceEvent, Decision, TriageResult } from "@/lib/types";

import { ConfidenceGauge } from "./ConfidenceGauge";
import { FEATURE_COPY, STAGE_COPY } from "./copy";

/* ---------- Confidence ---------- */

function guaranteeSentence(alpha: number, delta: number): string {
  return `At this threshold the error rate among released answers is certified to be at most ${fmtPct(alpha)}, a guarantee that holds with probability at least ${fmtPct(1 - delta)} over the calibration data.`;
}

/**
 * Said instead of the guarantee when the server reports that it does not cover this answer
 * (`guarantee_applies` false or absent): the threshold was certified on benchmark questions
 * answered by one model and retriever, and a chat question, or another configuration, is not
 * drawn from that distribution.
 */
const OPERATING_POINT =
  "That threshold was certified on PubMedQA benchmark questions answered with the deployed model and retriever. This answer comes from outside that setting, so here the threshold is an operating point: it carries no guarantee on the error rate.";

function Signals({ features }: { features: Record<string, number> }) {
  const entries = Object.entries(features);
  if (entries.length === 0) return null;
  return (
    <details className="group border-t border-rule px-4 py-2 sm:px-5">
      <summary className="-mx-2 flex min-h-11 cursor-pointer items-center gap-2 rounded-control px-2 text-[0.8125rem] font-medium text-ink hover:bg-raised sm:min-h-9">
        <ChevronGlyph
          size={14}
          className="-rotate-90 text-muted transition-transform duration-150 group-open:rotate-0"
        />
        Signals behind this estimate ({entries.length})
      </summary>
      <table className="num mb-2 mt-1 w-full max-w-md text-[0.8125rem]">
        <caption className="sr-only">Input signals of the confidence model for this answer</caption>
        <tbody>
          {entries.map(([name, value]) => (
            <tr key={name} className="border-t border-rule first:border-t-0">
              <th scope="row" className="py-1.5 pr-4 text-left font-normal text-muted">
                {FEATURE_COPY[name] ?? name}
              </th>
              <td className="py-1.5 text-right text-ink">{Number.isInteger(value) ? value : value.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

function BenchmarkFacts({
  decision,
  predictionSet,
  gold,
}: {
  decision: Decision | null;
  predictionSet: Decision[] | null | undefined;
  gold: Decision | null | undefined;
}) {
  if (!decision && !predictionSet?.length && !gold) return null;
  return (
    <dl className="mt-4 flex flex-wrap gap-x-8 gap-y-3 rounded-control bg-raised px-3.5 py-3 text-sm">
      {decision ? (
        <div>
          <dt className="text-xs text-muted">Decision</dt>
          <dd className="mt-0.5 text-title font-semibold leading-tight text-ink">{decision}</dd>
        </div>
      ) : null}
      {predictionSet?.length ? (
        <div>
          <dt className="text-xs text-muted">Conformal prediction set</dt>
          <dd className="mt-0.5 font-medium text-ink">{`{ ${predictionSet.join(", ")} }`}</dd>
        </div>
      ) : null}
      {gold ? (
        <div>
          <dt className="text-xs text-muted">Gold label</dt>
          <dd className="mt-0.5 font-medium text-ink">
            {gold}
            {decision ? (decision === gold ? ", matched" : ", missed") : ""}
          </dd>
        </div>
      ) : null}
    </dl>
  );
}

interface DecisionProps {
  confidence: ConfidenceEvent;
  decision: Decision | null;
  gold: Decision | null | undefined;
}

/** The answered state: the reading cleared the certified threshold. */
export function Released({ confidence, decision, gold }: DecisionProps) {
  const { p_correct, threshold, alpha, delta, uncalibrated, guarantee_applies } = confidence;
  const calibrated = !uncalibrated && typeof threshold === "number";

  return (
    <Card variant={calibrated ? "accent" : "default"} aria-labelledby="decision-heading">
      <div className="px-4 pb-4 pt-4 sm:px-5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <h3 id="decision-heading" className="text-[1.0625rem] font-semibold leading-snug text-ink">
            {calibrated ? "Answer released" : "Answer shown without a guarantee"}
          </h3>
          {calibrated ? (
            <Badge tone="accent">
              <CheckGlyph size={12} />
              Cleared the threshold
            </Badge>
          ) : (
            <Badge tone="outline">Not certified</Badge>
          )}
        </div>
        <p className="mt-1.5 max-w-[70ch] text-sm leading-relaxed text-muted">
          {calibrated ? (
            <>
              Calibrated confidence {fmtProb(p_correct)} clears the threshold {fmtProb(threshold)} certified for α&nbsp;=&nbsp;{fmtAlpha(alpha)}. {guarantee_applies ? guaranteeSentence(alpha, delta) : OPERATING_POINT}
            </>
          ) : (
            <>
              No deployment bundle exists yet, so there is no calibrated confidence model and no certified threshold to
              compare against. The answer below is shown as generated. Run <Code>primum exp rq3</Code> to fit the bundle
              and switch the guarantee on.
            </>
          )}
        </p>
        {uncalibrated ? null : (
          <div className="mt-5">
            <ConfidenceGauge p={p_correct} threshold={threshold} alpha={alpha} answered={confidence.answered} />
          </div>
        )}
        <BenchmarkFacts decision={decision} predictionSet={confidence.prediction_set} gold={gold} />
      </div>
      <Signals features={confidence.features} />
    </Card>
  );
}

interface WithheldProps extends DecisionProps {
  passageCount: number;
  verdicts: ClaimVerdict[];
  alphaOptions: readonly number[];
  onAskAgain: (alpha: number) => void;
  busy: boolean;
}

/**
 * Abstention. This is the system working as designed: it states that it is withholding,
 * gives the statistical reason in one sentence, and says what the reader can do next.
 */
export function Withheld({
  confidence,
  decision,
  gold,
  passageCount,
  verdicts,
  alphaOptions,
  onAskAgain,
  busy,
}: WithheldProps) {
  const { p_correct, threshold, alpha, delta, guarantee_applies } = confidence;
  const supported = verdicts.filter((v) => v.label === "supported").length;
  const contradicted = verdicts.filter((v) => v.label === "contradicted").length;
  const unsupported = verdicts.length - supported - contradicted;
  const hasThreshold = typeof threshold === "number";
  const shortfall = hasThreshold && typeof p_correct === "number" ? Math.max(0, threshold - p_correct) : null;
  const looser = alphaOptions.filter((a) => a > alpha + 1e-9);

  return (
    <Card aria-labelledby="decision-heading" className="overflow-hidden">
      <div aria-hidden="true" className="relative h-2 bg-raised">
        <div className="hatch absolute inset-0" />
      </div>
      <div className="px-4 pb-4 pt-4 sm:px-5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <h3 id="decision-heading" className="text-[1.0625rem] font-semibold leading-snug text-ink">
            Answer withheld
          </h3>
          <Badge tone="neutral">
            <span aria-hidden="true" className="hatch inline-block size-3 rounded-[3px]" />
            Below the threshold
          </Badge>
        </div>
        <p className="reading mt-2 max-w-[66ch] !text-[1.0625rem] text-ink">
          {hasThreshold ? (
            <>
              Primum drafted an answer and is not showing it, because its confidence falls below the threshold for the
              risk level you chose. Its calibrated confidence is {fmtProb(p_correct)}, which is {fmtProb(shortfall)}{" "}
              short of the {fmtProb(threshold)} needed at α&nbsp;=&nbsp;{fmtAlpha(alpha)}.
            </>
          ) : (
            <>
              Primum drafted an answer and is not showing it. On the calibration data, no confidence threshold could be
              certified to keep errors at or below {fmtPct(alpha)}, so at α&nbsp;=&nbsp;{fmtAlpha(alpha)} every answer is
              withheld.
            </>
          )}
        </p>
        {verdicts.length > 0 ? (
          <p className="num mt-2 max-w-[70ch] text-sm leading-relaxed text-ink">
            Of the {verdicts.length} {verdicts.length === 1 ? "claim" : "claims"} in the draft, {supported}{" "}
            {supported === 1 ? "was" : "were"} supported by the retrieved passages, {unsupported}{" "}
            {unsupported === 1 ? "was" : "were"} not, and {contradicted} {contradicted === 1 ? "was" : "were"}{" "}
            contradicted.
          </p>
        ) : null}
        <p className="mt-2 max-w-[70ch] text-sm leading-relaxed text-muted">
          {hasThreshold && guarantee_applies
            ? `${guaranteeSentence(alpha, delta)} Withholding is how the system keeps that promise:`
            : hasThreshold
              ? `${OPERATING_POINT} Withholding below it is still the system's rule:`
              : "Withholding is the designed behaviour:"}{" "}
          a confident-sounding answer that might be wrong is worse than no answer.
        </p>

        <div className="mt-5">
          <ConfidenceGauge p={p_correct} threshold={threshold} alpha={alpha} answered={confidence.answered} />
        </div>

        <BenchmarkFacts decision={null} predictionSet={confidence.prediction_set} gold={gold ?? null} />
        {decision ? <span className="sr-only">A decision was drafted and is withheld.</span> : null}
      </div>

      <div className="border-t border-rule px-4 py-4 sm:px-5">
        <h4 className="text-sm font-semibold text-ink">What you can do</h4>
        <ul className="mt-2 space-y-2.5 text-sm leading-relaxed text-ink">
          <li>
            Read the {passageCount === 1 ? "one passage" : `${passageCount} passages`} it retrieved, listed under
            Sources. They are real sources and were not withheld.
          </li>
          {looser.length > 0 ? (
            <li>
              <span>Accept more risk and ask again. A larger α certifies a lower threshold.</span>
              <span className="mt-2 flex flex-wrap gap-2">
                {looser.map((a) => (
                  <Button key={a} size="sm" onClick={() => onAskAgain(a)} disabled={busy}>
                    <span className="num">Ask again at α = {fmtAlpha(a)}</span>
                  </Button>
                ))}
              </span>
            </li>
          ) : (
            <li>This is already the most permissive risk level the system offers.</li>
          )}
          <li>Ask a clinician. This prototype is not a substitute for one.</li>
        </ul>
      </div>

      <Signals features={confidence.features} />
    </Card>
  );
}

/* ---------- Emergency ---------- */

export function Emergency({ triage }: { triage: TriageResult }) {
  return (
    <section
      role="alert"
      aria-labelledby="decision-heading"
      className="rounded-card border border-emergency bg-emergency-bg p-4 text-emergency-ink sm:p-5"
    >
      <div className="flex items-start gap-3">
        <span
          aria-hidden="true"
          className="flex size-9 shrink-0 items-center justify-center rounded-full bg-emergency text-on-primary max-sm:hidden"
        >
          <WarningGlyph size={18} />
        </span>
        <div className="min-w-0">
          <h3 id="decision-heading" className="text-title font-semibold leading-tight tracking-[-0.012em]">
            Call emergency services now
          </h3>
          <p className="reading mt-2 max-w-[62ch] !text-[1.0625rem]">
            This may be an emergency.{" "}
            {triage.message?.trim() ||
              "What you describe can be a sign of a life-threatening condition that needs treatment within minutes."}
          </p>
          <p className="mt-3 max-w-[62ch] text-[0.9375rem] font-semibold leading-relaxed">
            Call your local emergency number: 911 in the United States and Canada, 999 in the United Kingdom, 112 in the
            European Union. Do not wait for an answer from this page.
          </p>
        </div>
      </div>
      <div className="mt-4 border-t border-emergency/30 pt-3.5 text-sm leading-relaxed sm:ml-12">
        <p>
          Primum stopped before retrieving or generating anything, and will not answer this question. The stop comes
          from a fixed list of red-flag rules, not from a model.
        </p>
        {triage.matched.length > 0 ? (
          <p className="mt-2">
            {triage.matched.length === 1 ? "Rule matched: " : "Rules matched: "}
            <span className="font-medium">{triage.matched.join("; ")}</span>
          </p>
        ) : null}
      </div>
    </section>
  );
}

export function UrgentNotice({ triage }: { triage: TriageResult }) {
  return (
    <div
      role="status"
      className="flex items-start gap-2.5 rounded-card bg-unsupported-bg px-4 py-3 text-sm leading-relaxed text-unsupported-ink ring-1 ring-inset ring-unsupported/50"
    >
      <WarningGlyph size={18} className="mt-0.5" />
      <p>
        <span className="font-semibold">This may need prompt medical attention. </span>
        {triage.message?.trim() ||
          "Contact a clinician or an urgent care service today rather than relying on this page."}
        {triage.matched.length > 0 ? ` Rule matched: ${triage.matched.join("; ")}.` : ""}
      </p>
    </div>
  );
}

/* ---------- Errors ---------- */

interface FailureProps {
  error: AskError;
  foundPassages: number;
  onRetry: () => void;
}

/** Each failure names what stopped, why, and the command or setting that fixes it. */
export function Failure({ error, foundPassages, onRetry }: FailureProps) {
  const { meta } = useApi();
  const stageLabel =
    error.stage && error.stage in STAGE_COPY
      ? STAGE_COPY[error.stage as keyof typeof STAGE_COPY].label.toLowerCase()
      : error.stage;

  if (error.kind === "unreachable") {
    return <ApiDown what="The question was not sent." />;
  }

  let title: string;
  let body: React.ReactNode;

  switch (error.kind) {
    case "no_api_key":
      title = "The server has no API key, so no answer was generated";
      body = (
        <>
          <p>
            Generation calls the Claude API, and the server started without a key. Add <Code>ANTHROPIC_API_KEY</Code> to{" "}
            <Code>.env</Code> in the project root, restart <Code>primum serve</Code>, and ask again.
          </p>
          {foundPassages > 0 ? (
            <p className="mt-2">
              Retrieval does not need the key: the {foundPassages} {foundPassages === 1 ? "passage" : "passages"} found
              for this question {foundPassages === 1 ? "is" : "are"} listed under Sources.
            </p>
          ) : null}
        </>
      );
      break;
    case "budget":
      title = "The LLM budget is spent, so no answer was generated";
      body = (
        <>
          <p>
            The server refuses to call the model once recorded spend reaches its ceiling
            {typeof meta?.budget_usd === "number" ? ` of ${fmtUsd(meta.budget_usd)}` : ""}
            {typeof meta?.spend?.total_usd === "number" ? ` (spent so far: ${fmtUsd(meta.spend.total_usd)})` : ""}. This
            is a hard stop by design.
          </p>
          <p className="mt-2">
            Raise <Code>PRIMUM_BUDGET_USD</Code> in <Code>.env</Code> and restart the server, or ask a question that has
            been answered before: cached answers cost nothing. <Code>primum spend</Code> shows the ledger.
          </p>
        </>
      );
      break;
    case "no_index":
      title = "The search index has not been built";
      body = (
        <>
          <p>
            Retrieval has nothing to search yet. Build the corpus and its index from the project root, then ask again:
          </p>
          <p className="mt-2 flex flex-wrap gap-2">
            <Code>primum data build kb</Code>
            <Code>primum index build kb</Code>
          </p>
        </>
      );
      break;
    case "invalid":
      title = "The API did not accept this question";
      body = <p>{error.message || "Questions must be between 3 and 2,000 characters."}</p>;
      break;
    case "bad_response":
      title = "The API answered in a form this page cannot read";
      body = <p>{error.message}</p>;
      break;
    default:
      title = stageLabel ? `The pipeline failed during ${stageLabel}` : "The pipeline failed";
      body = (
        <>
          <p>{error.message || "The server reported an internal error without a message."}</p>
          <p className="mt-2">
            The traceback is in the terminal running <Code>primum serve</Code>.
          </p>
        </>
      );
  }

  return (
    <Card role="alert" aria-labelledby="decision-heading" className="p-4 sm:p-5">
      {/* A setup or server error is not clinical danger: amber glyph, neutral edge. Red stays for emergencies. */}
      <h3 id="decision-heading" className="flex items-start gap-2 text-[1.0625rem] font-semibold leading-snug text-ink">
        <WarningGlyph size={18} className="mt-0.5 text-unsupported-ink" />
        {title}
      </h3>
      <div className="mt-2 max-w-[70ch] text-sm leading-relaxed text-ink">{body}</div>
      <div className="mt-4">
        <Button size="sm" onClick={onRetry}>
          Ask again
        </Button>
      </div>
    </Card>
  );
}
