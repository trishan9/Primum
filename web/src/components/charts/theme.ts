"use client";

import { useMemo } from "react";
import type { PartialTheme } from "@nivo/theming";

import { categorical, diverging, emphasis, entityColors, sequential, statusColor, type StatusName } from "./colors";
import { useChartTokens, type ChartTokens } from "./tokens";
import { useReducedMotion } from "./useReducedMotion";

/** The one nivo theme, built from the design tokens: hairline grid, muted 12px ticks, the app's tooltip. */
export function nivoTheme(t: ChartTokens): PartialTheme {
  const text = { fontFamily: t.fontFamily, fontSize: 12, fill: t.muted, outlineWidth: 0, outlineColor: t.surface, outlineOpacity: 0 };
  return {
    background: "transparent",
    text,
    axis: {
      domain: { line: { stroke: t.axis, strokeWidth: 1 } },
      ticks: { line: { stroke: t.axis, strokeWidth: 1 }, text: { ...text, fontVariantNumeric: "tabular-nums" } },
      legend: { text: { ...text, fill: t.muted, fontWeight: 400 } },
    },
    grid: { line: { stroke: t.grid, strokeWidth: 1 } },
    crosshair: { line: { stroke: t.control, strokeWidth: 1, strokeOpacity: 1, strokeDasharray: "0" } },
    legends: { text: { ...text, fill: t.text }, title: { text: { ...text, fill: t.muted } }, ticks: { text } },
    labels: { text: { ...text, fill: t.text, fontVariantNumeric: "tabular-nums" } },
    markers: { lineColor: t.text, lineStrokeWidth: 1, text: { ...text, fill: t.text, fontWeight: 500 } },
    dots: { text },
    annotations: {
      text: { ...text, fill: t.text, outlineWidth: 2, outlineColor: t.surface, outlineOpacity: 1 },
      link: { stroke: t.muted, strokeWidth: 1, outlineWidth: 2, outlineColor: t.surface, outlineOpacity: 1 },
      outline: { stroke: t.muted, strokeWidth: 1, outlineWidth: 2, outlineColor: t.surface, outlineOpacity: 1 },
      symbol: { fill: t.muted, outlineWidth: 2, outlineColor: t.surface, outlineOpacity: 1 },
    },
    tooltip: {
      container: {
        background: t.surface,
        color: t.text,
        fontFamily: t.fontFamily,
        fontSize: 12,
        lineHeight: 1.45,
        fontVariantNumeric: "tabular-nums",
        borderRadius: 8,
        border: `1px solid ${t.border}`,
        boxShadow: t["shadow-float"] || "none",
        padding: "8px 10px",
      },
      chip: { borderRadius: 2 },
      tableCell: { padding: "2px 6px" },
    },
  };
}

export interface ChartTheme {
  tokens: ChartTokens;
  /** pass to every nivo chart's `theme` */
  theme: PartialTheme;
  /**
   * pass to every nivo chart's `animate`. Always false: nothing performs an entrance (DESIGN.md,
   * motion), and a bar caught mid-growth by a fast scroll misreports its value. Switching a
   * dataset redraws in place, instantly.
   */
  animate: false;
  /** false under prefers-reduced-motion; for CSS transitions a chart may still use on hover */
  motion: boolean;
  colors: {
    categorical: (index: number) => string;
    entity: (domain: readonly string[]) => (key: string) => string;
    emphasis: (focus: string | readonly string[]) => (key: string) => string;
    sequential: (domain: [number, number]) => (value: number) => string;
    diverging: (domain: [number, number], mid?: number) => (value: number) => string;
    status: (status: StatusName) => string;
  };
}

/**
 * Everything a chart needs to match the app: the nivo theme, the colour helpers bound to the
 * current tokens, and whether to animate. Null until the tokens are read on the client; render
 * a `ChartSkeleton` then.
 */
export function useChartTheme(): ChartTheme | null {
  const tokens = useChartTokens();
  const reduced = useReducedMotion();
  return useMemo(() => {
    if (!tokens) return null;
    return {
      tokens,
      theme: nivoTheme(tokens),
      animate: false,
      motion: !reduced,
      colors: {
        categorical: (i) => categorical(tokens, i),
        entity: (domain) => entityColors(tokens, domain),
        emphasis: (focus) => emphasis(tokens, focus),
        sequential: (domain) => sequential(tokens, domain),
        diverging: (domain, mid) => diverging(tokens, domain, mid),
        status: (s) => statusColor(tokens, s),
      },
    };
  }, [tokens, reduced]);
}
