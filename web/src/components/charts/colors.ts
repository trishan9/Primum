import { scaleLinear } from "d3-scale";

import type { ChartTokens } from "./tokens";

/**
 * Colour by the job it does (see DESIGN.md, Tokens):
 * - categorical: identity. Four validated hues in fixed order, then one grey. Assign by entity,
 *   never by rank, so a filter never repaints the survivors.
 * - sequential: magnitude. One hue (the accent), from its soft step to its ink step: pale to dark
 *   in light mode, dark to pale in dark mode, so "more" is always "further from the surface".
 * - diverging: polarity around zero. Orange (series-2) and the accent, neutral raised grey between.
 * - status: supported / unsupported / contradicted / withheld (hatch). Only when the colour means it.
 */

/** The categorical slot for a series index: series-1..4, then series-rest for every further one. */
export function categorical(tokens: ChartTokens, index: number): string {
  if (index === 0) return tokens["series-1"];
  if (index === 1) return tokens["series-2"];
  if (index === 2) return tokens["series-3"];
  if (index === 3) return tokens["series-4"];
  return tokens["series-rest"];
}

/**
 * A colour per entity, fixed by its place in the full domain (pass every entity the chart can
 * show, not the filtered list). Unknown keys get series-rest.
 */
export function entityColors(tokens: ChartTokens, domain: readonly string[]): (key: string) => string {
  const map = new Map(domain.map((key, i) => [key, categorical(tokens, i)]));
  return (key) => map.get(key) ?? tokens["series-rest"];
}

/**
 * Emphasis: one entity in the accent, the rest in series-rest. For "the one that matters among
 * many" (the deployed retriever, the stated confidence), where eight hues would only decorate.
 */
export function emphasis(tokens: ChartTokens, focus: string | readonly string[]): (key: string) => string {
  const set = new Set(typeof focus === "string" ? [focus] : focus);
  return (key) => (set.has(key) ? tokens["series-1"] : tokens["series-rest"]);
}

/** Magnitude on one hue; values outside the domain are clamped. */
export function sequential(tokens: ChartTokens, domain: [number, number]): (value: number) => string {
  return scaleLinear<string>().domain(domain).range([tokens["accent-soft"], tokens["accent-ink"]]).clamp(true);
}

/** Polarity around `mid` (default 0): negative toward series-2, positive toward the accent. */
export function diverging(tokens: ChartTokens, [lo, hi]: [number, number], mid = 0): (value: number) => string {
  const extent = Math.max(Math.abs(lo - mid), Math.abs(hi - mid)) || 1;
  return scaleLinear<string>()
    .domain([mid - extent, mid, mid + extent])
    .range([tokens["series-2"], tokens.raised, tokens["series-1"]])
    .clamp(true);
}

export type StatusName = "supported" | "unsupported" | "contradicted" | "withheld";

/** Status fills. Always pair with a glyph and a word. */
export function statusColor(tokens: ChartTokens, status: StatusName): string {
  return status === "withheld" ? tokens.hatch : tokens[status];
}

/** Ink for a label set inside a filled mark: text on pale fills, surface on dark ones. */
export function inkOn(tokens: ChartTokens, fill: string): string {
  const hex = fill.replace("#", "");
  if (!/^[0-9a-f]{6}$/i.test(hex)) return tokens.text;
  const [r, g, b] = [0, 2, 4].map((i) => Number.parseInt(hex.slice(i, i + 2), 16) / 255);
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const luminance = 0.2126 * lin(r as number) + 0.7152 * lin(g as number) + 0.0722 * lin(b as number);
  // The darkest and lightest inks of the current mode, whichever way round they are.
  const darkInk = tokens.dark ? tokens.surface : tokens.text;
  const lightInk = tokens.dark ? tokens.text : tokens.surface;
  return luminance > 0.4 ? darkInk : lightInk;
}
