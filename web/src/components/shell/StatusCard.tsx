"use client";

import type { ReactNode } from "react";

import { StatusDot } from "@/components/ui/badge";
import { cn } from "@/components/ui/cn";
import { RefreshGlyph } from "@/components/ui/glyphs";
import { API_URL } from "@/lib/api";
import { fmtUsd } from "@/lib/format";

import { useApi } from "./ApiProvider";

const FIXTURES = process.env.NEXT_PUBLIC_FIXTURES === "1";

/**
 * A small definition list in the chrome. Compose rows:
 *
 *   <StatusCard title="System">
 *     <StatusCard.Row label="Generator">claude-…</StatusCard.Row>
 *   </StatusCard>
 */
function StatusCardRoot({ title, action, className, children }: { title: string; action?: ReactNode; className?: string; children: ReactNode }) {
  return (
    <section aria-label={title} className={cn("rounded-card border border-line bg-surface/60 p-3", className)}>
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium text-muted">{title}</p>
        {action}
      </div>
      <dl className="mt-2 space-y-1.5">{children}</dl>
    </section>
  );
}

/** One reading. `stacked` puts the value under its label, for values that need the full width (a meter). */
function StatusRow({ label, stacked = false, children }: { label: string; stacked?: boolean; children: ReactNode }) {
  return (
    <div className={stacked ? "min-w-0" : "flex min-w-0 items-baseline justify-between gap-3"}>
      <dt className="shrink-0 text-xs leading-4 text-muted">{label}</dt>
      <dd className={cn("min-w-0 text-[0.8125rem] leading-[1.125rem] text-ink", stacked ? "mt-1" : "text-right")}>{children}</dd>
    </div>
  );
}

export const StatusCard = Object.assign(StatusCardRoot, { Row: StatusRow });

const API_LABEL = { checking: "Checking", up: "Connected", down: "Unreachable" } as const;

/** Spend against the budget as a bar at its true ratio; past the budget the bar is full and the overrun is said in words. */
function SpendMeter({ spent, budget }: { spent: number; budget: number }) {
  const ratio = budget > 0 ? spent / budget : 0;
  const over = spent > budget;
  return (
    <>
      <span className="num">
        {fmtUsd(spent)} <span className="text-muted">of {fmtUsd(budget)}</span>
      </span>
      <span
        role="meter"
        aria-label="API spend against budget"
        aria-valuemin={0}
        aria-valuemax={budget}
        aria-valuenow={Math.min(spent, budget)}
        aria-valuetext={`${fmtUsd(spent)} of ${fmtUsd(budget)}`}
        className="mt-1.5 block h-1.5 overflow-hidden rounded-full bg-active"
      >
        <span
          className={cn("block h-full rounded-full", over ? "bg-unsupported" : "bg-accent")}
          style={{ width: `${Math.min(100, Math.max(0, ratio * 100))}%` }}
        />
      </span>
      {over ? <span className="mt-1 block text-xs text-unsupported-ink">Over budget by {fmtUsd(spent - budget)}</span> : null}
    </>
  );
}

/**
 * What the server is running, read from /api/health and /api/meta: reachability, the generator,
 * whether a deployment bundle is loaded, spend against budget, and whether the Ask view is
 * replaying fixtures. Nothing here is typed in; a value the API does not send is a dash.
 */
export function SystemStatus({ className }: { className?: string }) {
  const { reachability, health, meta, refresh } = useApi();
  const host = API_URL.replace(/^https?:\/\//, "");
  const spent = meta?.spend?.total_usd;
  const budget = meta?.budget_usd;
  const bundle = health?.deployment_bundle;

  return (
    <StatusCard
      title="System"
      className={className}
      action={
        <button
          type="button"
          onClick={refresh}
          disabled={reachability === "checking"}
          aria-label="Check the API again"
          title="Check the API again"
          className="-m-1.5 flex size-8 items-center justify-center rounded-control text-muted hover:bg-active hover:text-ink max-lg:size-11"
        >
          <RefreshGlyph size={14} />
        </button>
      }
    >
      <StatusCard.Row label="API">
        <span className="inline-flex items-center gap-1.5" role="status" title={`${API_LABEL[reachability]} at ${host}`}>
          <StatusDot tone={reachability === "up" ? "ok" : reachability === "down" ? "down" : "idle"} />
          {API_LABEL[reachability]}
        </span>
      </StatusCard.Row>
      <StatusCard.Row label="Generator">
        <span className="block truncate font-mono text-xs leading-[1.125rem]" title={meta?.models?.llm}>
          {meta?.models?.llm ?? "–"}
        </span>
      </StatusCard.Row>
      <StatusCard.Row label="Deployment bundle" stacked={bundle === false}>
        {bundle === undefined ? "–" : bundle ? "Loaded" : "Not built: every answer is withheld"}
      </StatusCard.Row>
      <StatusCard.Row label="Spend" stacked>
        {typeof spent === "number" && typeof budget === "number" ? (
          <SpendMeter spent={spent} budget={budget} />
        ) : (
          <span className="num">{fmtUsd(spent)}</span>
        )}
      </StatusCard.Row>
      {FIXTURES ? (
        <StatusCard.Row label="Ask view" stacked>
          <span className="text-unsupported-ink">Replaying fixtures, not the API</span>
        </StatusCard.Row>
      ) : null}
    </StatusCard>
  );
}

/** The collapsed rail's version: one dot for the API, named for assistive tech and in a tooltip. */
export function RailStatus() {
  const { reachability, refresh } = useApi();
  return (
    <button
      type="button"
      onClick={refresh}
      data-tip={`API ${API_LABEL[reachability].toLowerCase()}`}
      aria-label={`API ${API_LABEL[reachability].toLowerCase()}. Check again`}
      className="flex size-10 items-center justify-center rounded-control hover:bg-active"
    >
      <StatusDot tone={reachability === "up" ? "ok" : reachability === "down" ? "down" : "idle"} />
    </button>
  );
}
