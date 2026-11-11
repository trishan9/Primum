"use client";

import type { ReactNode } from "react";

import { useWidth } from "./useSize";

/**
 * Measures its own width and draws `children(width)` at that width and the given height, so a
 * chart can pick its tick count or margins by the room it has (fewer ticks at 375px). Use the
 * non-responsive nivo component inside (`Bar`, `Line`, not `ResponsiveBar`). Draws nothing until
 * measured; the space is held at full height so nothing shifts.
 */
export function Measured({ height, children }: { height: number; children: (width: number) => ReactNode }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  return (
    <div ref={ref} className="relative min-w-0" style={{ height }}>
      {width > 0 ? children(width) : null}
    </div>
  );
}

/** Tick values for a [0, 1] proportion axis with room for about one label per 72px. */
export function proportionTicks(width: number): number[] {
  if (width < 260) return [0, 0.5, 1];
  if (width < 520) return [0, 0.25, 0.5, 0.75, 1];
  return [0, 0.2, 0.4, 0.6, 0.8, 1];
}
