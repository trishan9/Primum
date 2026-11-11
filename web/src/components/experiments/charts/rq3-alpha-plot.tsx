"use client";

import { scaleLinear, scalePoint } from "d3-scale";
import { line } from "d3-shape";
import { useState, type ReactNode } from "react";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Measured } from "@/components/charts/Measured";
import { useChartTheme, type ChartTheme } from "@/components/charts/theme";
import { fmtAlpha } from "@/lib/format";

import { Mark, type MarkShape } from "./rq2-forest";
import { HoverTip } from "./rq2-tooltip";
import { ALPHA_AXIS, ALPHA_ROW, ALPHA_TOP, alphaPlotHeight } from "./rq3-heights";

export interface AlphaPoint {
  alpha: number;
  value: number;
  lo: number | null;
  hi: number | null;
  /** drawn as an open ring: a value that carries no information (nothing was certified) */
  hollow?: boolean;
}

export interface AlphaSeries {
  key: string;
  color: string;
  shape: MarkShape;
  dashed?: boolean;
  points: readonly AlphaPoint[];
}

export interface AlphaReference {
  /** a constant (δ) or one value per α (the 1 − α target) */
  value: number | ((alpha: number) => number);
  label: string;
}

/**
 * The RQ3 figure: one column per tested α (a point scale, since only those levels exist), each
 * series as marks joined by a thin line, its interval as a whisker, series nudged apart so their
 * whiskers never overlap. Optional references (δ, the 1 − α target), a highlighted α, and text
 * rows under the axis for a quantity that belongs to the same α but not on this y-axis.
 */
export function AlphaPlot(props: {
  alphas: readonly number[];
  series: readonly AlphaSeries[];
  yDomain: [number, number];
  yFormat: (v: number) => string;
  yLabel: string;
  plotHeight: number;
  references?: readonly AlphaReference[];
  highlight?: number | null;
  rows?: readonly { label: string; values: (alpha: number) => string }[];
  tip: (alpha: number) => ReactNode;
  ariaLabel: string;
}) {
  const kit = useChartTheme();
  const height = alphaPlotHeight(props.plotHeight, props.rows?.length ?? 0);
  if (!kit) return <ChartSkeleton height={height} />;
  return <Measured height={height}>{(width) => <Plot kit={kit} width={width} height={height} {...props} />}</Measured>;
}

function Plot({
  kit,
  width,
  height,
  alphas,
  series,
  yDomain,
  yFormat,
  yLabel,
  plotHeight,
  references = [],
  highlight = null,
  rows = [],
  tip,
  ariaLabel,
}: Parameters<typeof AlphaPlot>[0] & { kit: ChartTheme; width: number; height: number }) {
  const { tokens } = kit;
  const [hover, setHover] = useState<number | null>(null);
  const left = rows.length > 0 ? 84 : 52;
  const right = 12;
  const bottom = ALPHA_TOP + plotHeight;
  const x = scalePoint<number>().domain([...alphas]).range([left, width - right]).padding(0.5);
  const step = x.step();
  const y = scaleLinear().domain(yDomain).range([bottom, ALPHA_TOP]).nice();
  const ticks = y.ticks(plotHeight < 200 ? 4 : 5);
  const dodge = (i: number) => (i - (series.length - 1) / 2) * Math.min(10, step / (series.length + 2));
  const xs = (alpha: number) => x(alpha) ?? 0;
  const join = line<{ x: number; y: number }>()
    .x((d) => d.x)
    .y((d) => d.y);
  const hovered = hover !== null ? alphas[hover] : undefined;

  return (
    <div className="relative" onMouseLeave={() => setHover(null)}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block overflow-visible">
        {highlight !== null && alphas.includes(highlight) ? <rect x={xs(highlight) - step / 2} y={ALPHA_TOP} width={step} height={plotHeight} fill={tokens["accent-soft"]} opacity={0.5} /> : null}
        {hovered !== undefined ? <rect x={xs(hovered) - step / 2} y={ALPHA_TOP} width={step} height={plotHeight} fill={tokens.raised} /> : null}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={left} x2={width - right} y1={y(t)} y2={y(t)} stroke={tokens.grid} />
            <text x={left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={12} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
              {yFormat(t)}
            </text>
          </g>
        ))}
        <text transform={`translate(12 ${ALPHA_TOP + plotHeight / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill={tokens.muted}>
          {yLabel}
        </text>

        {series.map((s, i) => {
          const pts = s.points.map((p) => ({ ...p, px: xs(p.alpha) + dodge(i) }));
          return (
            <g key={s.key}>
              <path
                d={join(pts.map((p) => ({ x: p.px, y: y(p.value) }))) ?? undefined}
                fill="none"
                stroke={s.color}
                strokeWidth={1.5}
                strokeOpacity={0.7}
                strokeDasharray={s.dashed ? "5 3" : undefined}
                strokeLinecap={s.dashed ? "butt" : undefined}
              />
              {pts.map((p) =>
                p.lo !== null && p.hi !== null ? <line key={`w${p.alpha}`} x1={p.px} x2={p.px} y1={y(p.lo)} y2={y(p.hi)} stroke={s.color} strokeWidth={1.5} strokeLinecap="round" /> : null,
              )}
              {pts.map((p) => (
                <Mark key={`m${p.alpha}`} shape={p.hollow ? (s.shape === "diamond" || s.shape === "diamond-open" ? "diamond-open" : "ring") : s.shape} cx={p.px} cy={y(p.value)} color={s.color} surface={hovered === p.alpha ? tokens.raised : tokens.surface} />
              ))}
            </g>
          );
        })}

        {/* References (δ, the 1 − α target) after the series, so no series line hides the promise being tested. */}
        {references.map((r) => {
          const pts = typeof r.value === "number" ? [left, width - right].map((px) => ({ x: px, y: y(r.value as number) })) : alphas.map((a) => ({ x: xs(a), y: y((r.value as (a: number) => number)(a)) }));
          const last = pts[pts.length - 1] as { x: number; y: number };
          return (
            <g key={r.label}>
              <path d={join(pts) ?? undefined} fill="none" stroke={tokens.muted} strokeWidth={1.5} strokeDasharray="5 4" />
              <text
                x={typeof r.value === "number" ? width - right : Math.min(width - right, last.x + step / 2 - 2)}
                y={last.y - 6}
                textAnchor="end"
                fontSize={11}
                fill={tokens.muted}
                stroke={tokens.surface}
                strokeWidth={3}
                paintOrder="stroke"
              >
                {r.label}
              </text>
            </g>
          );
        })}

        <line x1={left} x2={width - right} y1={bottom} y2={bottom} stroke={tokens.axis} />
        {alphas.map((a) => (
          <text key={a} x={xs(a)} y={bottom + 16} textAnchor="middle" fontSize={12} fill={a === highlight ? tokens.text : tokens.muted} fontWeight={a === highlight ? 600 : 400} style={{ fontVariantNumeric: "tabular-nums" }}>
            {fmtAlpha(a)}
          </text>
        ))}
        <text x={(left + width - right) / 2} y={bottom + 34} textAnchor="middle" fontSize={11} fill={tokens.muted}>
          Target risk α
        </text>
        {rows.map((row, r) => {
          const ry = bottom + ALPHA_AXIS + 8 + r * ALPHA_ROW + ALPHA_ROW / 2;
          return (
            <g key={row.label}>
              <text x={left - 8} y={ry} dy="0.32em" textAnchor="end" fontSize={11} fill={tokens.muted}>
                {row.label}
              </text>
              {alphas.map((a) => (
                <text key={a} x={xs(a)} y={ry} dy="0.32em" textAnchor="middle" fontSize={12} fill={tokens.text} style={{ fontVariantNumeric: "tabular-nums" }}>
                  {row.values(a)}
                </text>
              ))}
            </g>
          );
        })}
        {alphas.map((a, i) => (
          <rect key={a} x={xs(a) - step / 2} y={ALPHA_TOP} width={step} height={plotHeight} fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      {hovered !== undefined ? (
        <HoverTip kit={kit} x={xs(hovered) + step / 4} y={ALPHA_TOP + 8} width={width}>
          {tip(hovered)}
        </HoverTip>
      ) : null}
    </div>
  );
}
