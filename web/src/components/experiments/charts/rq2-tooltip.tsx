"use client";

import type { ReactNode } from "react";

import type { ChartTheme } from "@/components/charts/theme";

/**
 * The hover tooltip for the hand-drawn RQ2/RQ3 plots, styled by the shared nivo theme so it matches
 * the nivo charts beside them. Positioned inside the plot's relative box, flipped to stay inside it.
 * Hidden from assistive technology: every figure has its data table, which says the same.
 */
export function HoverTip({ kit, x, y, width, children }: { kit: ChartTheme; x: number; y: number; width: number; children: ReactNode }) {
  const flip = x > width / 2;
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute z-10 w-max"
      style={{
        ...(kit.theme.tooltip?.container as object),
        // Never wider than the room on its side of the pointer, so it stays inside the card.
        maxWidth: Math.max(160, Math.min(352, flip ? x - 12 : width - x - 12)),
        left: flip ? undefined : x + 12,
        right: flip ? width - x + 12 : undefined,
        top: Math.max(0, y - 12),
      }}
    >
      {children}
    </div>
  );
}

/** One tooltip line: a muted label and a value. */
export function TipRow({ kit, label, children }: { kit: ChartTheme; label?: string; children: ReactNode }) {
  return (
    <p>
      {label ? <span style={{ color: kit.tokens.muted }}>{label} </span> : null}
      {children}
    </p>
  );
}
