"use client";

import { scaleLinear } from "d3-scale";
import { useState, type ReactNode } from "react";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Measured } from "@/components/charts/Measured";
import { useChartTheme, type ChartTheme } from "@/components/charts/theme";

import { FOREST_TOP, forestHeight } from "./rq2-heights";
import { HoverTip } from "./rq2-tooltip";

export type MarkShape = "dot" | "diamond" | "ring" | "square" | "diamond-open";

export interface ForestRow {
  key: string;
  label: string;
  value: number;
  /** interval ends; null draws the point alone (the file has no interval for it) */
  lo: number | null;
  hi: number | null;
  color: string;
  shape: MarkShape;
  /** set the label in the ink colour and weight 600 (the reference row) */
  strong?: boolean;
  /** printed at the right edge: the value as the reader should quote it */
  valueLabel: string;
  tip: ReactNode;
}

export interface ForestReference {
  value: number;
  label: string;
  /** the accent for the reference estimator, axis grey for "no difference" or chance */
  tone: "accent" | "axis";
  dashed?: boolean;
}

const TOP = FOREST_TOP;

/** A mark drawn at (cx, cy), wearing a 2px surface ring so it stays legible where it overlaps its interval. */
export function Mark({ shape, cx, cy, color, surface, size = 4.5 }: { shape: MarkShape; cx: number; cy: number; color: string; surface: string; size?: number }) {
  if (shape === "diamond" || shape === "diamond-open") {
    const s = size + 1.5;
    const d = `M${cx} ${cy - s}L${cx + s} ${cy}L${cx} ${cy + s}L${cx - s} ${cy}Z`;
    if (shape === "diamond-open") return <path d={d} fill={surface} stroke={color} strokeWidth={2} />;
    return <path d={d} fill={color} stroke={surface} strokeWidth={2} paintOrder="stroke" />;
  }
  if (shape === "square") {
    const s = size - 0.25;
    return <rect x={cx - s} y={cy - s} width={2 * s} height={2 * s} rx={1} fill={color} stroke={surface} strokeWidth={2} paintOrder="stroke" />;
  }
  if (shape === "ring") return <circle cx={cx} cy={cy} r={size} fill={surface} stroke={color} strokeWidth={2} />;
  return <circle cx={cx} cy={cy} r={size} fill={color} stroke={surface} strokeWidth={2} paintOrder="stroke" />;
}

/**
 * A horizontal dot-and-interval plot (forest plot): one row per entity, the point estimate as a
 * mark, its interval as a thin line, optional vertical references (zero, chance, the reference
 * estimator). The axis runs over `domain` as given: the caller chooses a domain the data honestly
 * needs (a dot plot need not start at zero; a difference is centred on zero).
 */
export function ForestPlot({
  rows,
  domain,
  references = [],
  tickFormat,
  axisLabel,
  ariaLabel,
  rowHeight = 28,
  ends,
}: {
  rows: readonly ForestRow[];
  domain: [number, number];
  references?: readonly ForestReference[];
  tickFormat: (v: number) => string;
  axisLabel: string;
  ariaLabel: string;
  rowHeight?: number;
  /** words under the two ends of the axis ("lower Brier, better" / "worse") */
  ends?: [string, string];
}) {
  const kit = useChartTheme();
  const height = forestHeight(rows.length, rowHeight, !!ends);
  if (!kit) return <ChartSkeleton height={height} />;
  return (
    <Measured height={height}>
      {(width) => (
        <Plot kit={kit} width={width} height={height} rows={rows} domain={domain} references={references} tickFormat={tickFormat} axisLabel={axisLabel} ariaLabel={ariaLabel} rowHeight={rowHeight} ends={ends} />
      )}
    </Measured>
  );
}

function Plot({
  kit,
  width,
  height,
  rows,
  domain,
  references,
  tickFormat,
  axisLabel,
  ariaLabel,
  rowHeight,
  ends,
}: {
  kit: ChartTheme;
  width: number;
  height: number;
  rows: readonly ForestRow[];
  domain: [number, number];
  references: readonly ForestReference[];
  tickFormat: (v: number) => string;
  axisLabel: string;
  ariaLabel: string;
  rowHeight: number;
  ends?: [string, string];
}) {
  const [hover, setHover] = useState<number | null>(null);
  const { tokens } = kit;
  const labelWidth = Math.min(176, Math.max(104, width * 0.34));
  const valueWidth = 48;
  const x0 = labelWidth + 8;
  const x1 = width - valueWidth - 8;
  const x = scaleLinear().domain(domain).range([x0, x1]);
  const ticks = x.ticks(Math.max(2, Math.floor((x1 - x0) / 64)));
  const plotBottom = TOP + rows.length * rowHeight;
  // Labels are cut to the room they have (about 6.5px a character at 12px); the full name is in <title>.
  const maxChars = Math.max(6, Math.floor(labelWidth / 6.5));
  const clip = (label: string) => (label.length > maxChars ? `${label.slice(0, maxChars - 1)}…` : label);
  const cyOf = (i: number) => TOP + i * rowHeight + rowHeight / 2;
  const hovered = hover !== null ? rows[hover] : undefined;

  return (
    <div className="relative" onMouseLeave={() => setHover(null)}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block overflow-visible">
        {/* grid */}
        {ticks.map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={TOP} y2={plotBottom} stroke={tokens.grid} />
        ))}
        {/* hovered row band */}
        {hovered ? <rect x={0} y={TOP + (hover as number) * rowHeight} width={width} height={rowHeight} rx={6} fill={tokens.raised} /> : null}
        {/* references */}
        {references.map((r) => {
          // "axis" tone references (chance, no difference) are what the rows are read against: muted ink, ≥ 3:1.
          const color = r.tone === "accent" ? tokens.accent : tokens.muted;
          const anchor = x(r.value) > (x0 + x1) / 2 ? "end" : "start";
          return (
            <g key={r.label}>
              <line x1={x(r.value)} x2={x(r.value)} y1={TOP - 4} y2={plotBottom} stroke={color} strokeWidth={1} strokeDasharray={r.dashed ? "4 3" : undefined} />
              <text x={x(r.value) + (anchor === "start" ? 4 : -4)} y={TOP - 8} textAnchor={anchor} fontSize={11} fill={tokens.muted}>
                {r.label}
              </text>
            </g>
          );
        })}
        {/* rows */}
        {rows.map((row, i) => {
          const cy = cyOf(i);
          const hasInterval = row.lo !== null && row.hi !== null;
          return (
            <g key={row.key}>
              <text x={labelWidth} y={cy} dy="0.32em" textAnchor="end" fontSize={12} fill={row.strong ? tokens.text : tokens.muted} fontWeight={row.strong ? 600 : 400}>
                <title>{row.label}</title>
                {clip(row.label)}
              </text>
              {hasInterval ? <line x1={x(row.lo as number)} x2={x(row.hi as number)} y1={cy} y2={cy} stroke={row.color} strokeWidth={1.5} strokeLinecap="round" /> : null}
              <Mark shape={row.shape} cx={x(row.value)} cy={cy} color={row.color} surface={hovered === row ? tokens.raised : tokens.surface} />
              <text x={width} y={cy} dy="0.32em" textAnchor="end" fontSize={12} fill={tokens.text} fontWeight={row.strong ? 600 : 400} style={{ fontVariantNumeric: "tabular-nums" }}>
                {row.valueLabel}
              </text>
              <rect x={0} y={cy - rowHeight / 2} width={width} height={rowHeight} fill="transparent" onMouseEnter={() => setHover(i)} />
            </g>
          );
        })}
        {/* axis */}
        <line x1={x0} x2={x1} y1={plotBottom} y2={plotBottom} stroke={tokens.axis} />
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={plotBottom + 16} textAnchor="middle" fontSize={12} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
            {tickFormat(t)}
          </text>
        ))}
        <text x={(x0 + x1) / 2} y={plotBottom + 34} textAnchor="middle" fontSize={11} fill={tokens.muted}>
          {axisLabel}
        </text>
        {ends ? (
          <>
            <text x={x0} y={plotBottom + 50} fontSize={11} fill={tokens.muted}>
              {ends[0]}
            </text>
            <text x={x1} y={plotBottom + 50} textAnchor="end" fontSize={11} fill={tokens.muted}>
              {ends[1]}
            </text>
          </>
        ) : null}
      </svg>
      {hovered && hover !== null ? (
        <HoverTip kit={kit} x={x(hovered.value)} y={cyOf(hover)} width={width}>
          {hovered.tip}
        </HoverTip>
      ) : null}
    </div>
  );
}
