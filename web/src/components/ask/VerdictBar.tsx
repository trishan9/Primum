"use client";

import { useMemo } from "react";

import { cn } from "@/components/ui/cn";
import { verdictShares, verdictTakeaway } from "@/lib/charts/ask";
import { fmtPct } from "@/lib/format";
import type { ClaimVerdict, VerdictLabel } from "@/lib/types";

import { VERDICT_COPY } from "./copy";

const FILL: Record<VerdictLabel, string> = {
  supported: "bg-supported",
  unsupported: "bg-unsupported",
  contradicted: "bg-contradicted",
};

/**
 * The claim verdicts as one bar split by count, in the fixed order supported, unsupported,
 * contradicted. The verdict badges above it are its legend (glyph, word and count), so colour
 * never carries the meaning alone; the takeaway says it in a sentence and the hidden table
 * lists every verdict, zeros included.
 */
export function VerdictBar({ verdicts, className }: { verdicts: readonly ClaimVerdict[]; className?: string }) {
  const shares = useMemo(() => verdictShares(verdicts), [verdicts]);
  if (shares.total === 0) return null;
  const takeaway = verdictTakeaway(shares);

  return (
    <figure className={cn("num", className)}>
      <div aria-hidden="true" className="flex h-2 gap-0.5">
        {shares.rows
          .filter((row) => row.count > 0)
          .map((row) => (
            <span
              key={row.label}
              title={`${VERDICT_COPY[row.label].label}: ${row.count}`}
              className={cn("h-full rounded-[3px]", FILL[row.label])}
              style={{
                flexGrow: row.count,
                flexBasis: 0,
                // A second encoding for colour-blind readers: red and ochre can look alike, so the
                // contradicted segment also carries diagonal stripes.
                backgroundImage:
                  row.label === "contradicted"
                    ? "repeating-linear-gradient(135deg, transparent 0 3px, var(--surface) 3px 4.5px)"
                    : undefined,
              }}
            />
          ))}
      </div>
      <figcaption className="mt-1.5 text-xs text-muted">{takeaway}</figcaption>
      <table className="sr-only">
        <caption>Claims by verdict</caption>
        <thead>
          <tr>
            <th scope="col">Verdict</th>
            <th scope="col">Claims</th>
            <th scope="col">Share</th>
          </tr>
        </thead>
        <tbody>
          {shares.rows.map((row) => (
            <tr key={row.label}>
              <th scope="row">{VERDICT_COPY[row.label].label}</th>
              <td>{row.count}</td>
              <td>{fmtPct(row.share)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
