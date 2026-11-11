"use client";

import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Measured } from "@/components/charts/Measured";
import { useChartTheme, type ChartTheme } from "@/components/charts/theme";
import type { Summary } from "@/lib/charts/common";
import { histogramValues, type ConfidenceHistogram, type HistogramScale } from "@/lib/charts/pubmedqa";
import { fmtInt, fmtPct } from "@/lib/format";

import { CONFIDENCE_HIST_HEIGHT } from "./benchmark-heights";
import { HoverTip, TipRow } from "./rq2-tooltip";

const M = { top: 10, right: 12, bottom: 40, left: 58 };

interface Props {
  hist: ConfidenceHistogram;
  scale: HistogramScale;
  medians: { correct: Summary | null; wrong: Summary | null };
  ariaLabel: string;
}

/**
 * The stated confidence of right answers (above the axis) and wrong ones (below), mirrored on one
 * shared x axis over [0, 1] so the two shapes are compared directly; bars from the axis, both
 * halves on the same magnitude scale. Position says which group a bar is, so both wear one
 * neutral fill and the words "Right" and "Wrong" label the halves. A tick marks each group's
 * median, the figure the takeaway quotes. Bins are 0.1 wide, or one per stated value when the
 * generator used at most 12 (see `confidenceBins`).
 */
export function ConfidenceHistogramChart(props: Props) {
  const kit = useChartTheme();
  if (!kit) return <ChartSkeleton height={CONFIDENCE_HIST_HEIGHT} />;
  return <Measured height={CONFIDENCE_HIST_HEIGHT}>{(width) => <Plot kit={kit} width={width} {...props} />}</Measured>;
}

function Plot({ kit, width, hist, scale, medians, ariaLabel }: Props & { kit: ChartTheme; width: number }) {
  const { tokens } = kit;
  const [hover, setHover] = useState<number | null>(null);
  const values = useMemo(() => histogramValues(hist, scale), [hist, scale]);
  const height = CONFIDENCE_HIST_HEIGHT;
  const mid = M.top + (height - M.top - M.bottom) / 2;
  const half = mid - M.top;
  const x = scaleLinear().domain([0, 1]).range([M.left, width - M.right]);
  const top = Math.max(...values.flatMap((b) => [b.correct, b.wrong]), scale === "count" ? 1 : 0.01);
  const h = scaleLinear().domain([0, top]).range([0, half - 4]).nice(3);
  const yTicks = h.ticks(2).filter((t) => t > 0);
  const fmt = (v: number) => (scale === "count" ? fmtInt(v) : fmtPct(v, 0));

  // Distinct-value bins have no width of their own: give each a bar as wide as the closest gap allows.
  const levels = values.map((b) => b.lo);
  const minGap = hist.distinct && levels.length > 1 ? Math.min(...levels.slice(1).map((v, i) => v - (levels[i] as number))) : 0.1;
  const barWidth = hist.distinct ? Math.max(3, Math.min(18, (x(minGap) - x(0)) * 0.7)) : 0;
  const span = (b: { lo: number; hi: number }) =>
    hist.distinct ? { x0: x(b.lo) - barWidth / 2, x1: x(b.lo) + barWidth / 2 } : { x0: x(b.lo) + 1, x1: x(b.hi) - 1 };
  const hovered = hover !== null ? values[hover] : undefined;
  const fill = tokens["series-rest"];
  const median = (s: Summary | null, sign: 1 | -1) =>
    s ? (
      <g>
        <line x1={x(s.median)} x2={x(s.median)} y1={mid} y2={mid - sign * (half - 2)} stroke={tokens.text} strokeWidth={1.5} />
        <text
          x={x(s.median) + 4}
          y={mid - sign * (half - 10)}
          dy="0.32em"
          fontSize={11}
          fill={tokens.text}
          stroke={tokens.surface}
          strokeWidth={3}
          paintOrder="stroke"
          textAnchor={x(s.median) > width - 110 ? "end" : "start"}
          dx={x(s.median) > width - 110 ? -8 : 0}
        >
          median {s.median.toFixed(2)}
        </text>
      </g>
    ) : null;

  return (
    <div className="relative" onMouseLeave={() => setHover(null)}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel} className="block overflow-visible">
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={width - M.right} y1={mid - h(t)} y2={mid - h(t)} stroke={tokens.grid} />
            <line x1={M.left} x2={width - M.right} y1={mid + h(t)} y2={mid + h(t)} stroke={tokens.grid} />
            <text x={M.left - 8} y={mid - h(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
              {fmt(t)}
            </text>
            <text x={M.left - 8} y={mid + h(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
              {fmt(t)}
            </text>
          </g>
        ))}
        <text x={4} y={M.top + 10} fontSize={12} fontWeight={600} fill={tokens.text}>
          Right
        </text>
        <text x={4} y={M.top + 24} fontSize={11} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
          {fmtInt(hist.nCorrect)}
        </text>
        <text x={4} y={height - M.bottom - 18} fontSize={12} fontWeight={600} fill={tokens.text}>
          Wrong
        </text>
        <text x={4} y={height - M.bottom - 4} fontSize={11} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
          {fmtInt(hist.nWrong)}
        </text>
        {hovered ? <rect x={span(hovered).x0 - 2} y={M.top} width={span(hovered).x1 - span(hovered).x0 + 4} height={2 * half} rx={3} fill={tokens.raised} /> : null}
        {values.map((b, i) => {
          const { x0, x1 } = span(b);
          const w = Math.max(1, x1 - x0);
          const up = h(b.correct);
          const down = h(b.wrong);
          return (
            <g key={i}>
              {b.correct > 0 ? <rect x={x0} y={mid - Math.max(1, up) - 1} width={w} height={Math.max(1, up)} rx={2} fill={fill} /> : null}
              {b.wrong > 0 ? <rect x={x0} y={mid + 1} width={w} height={Math.max(1, down)} rx={2} fill={fill} /> : null}
            </g>
          );
        })}
        <line x1={M.left} x2={width - M.right} y1={mid} y2={mid} stroke={tokens.axis} />
        {median(medians.correct, 1)}
        {median(medians.wrong, -1)}
        {(width >= 420 ? [0, 0.2, 0.4, 0.6, 0.8, 1] : [0, 0.5, 1]).map((t) => (
          <text key={t} x={x(t)} y={height - M.bottom + 16} textAnchor="middle" fontSize={12} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
            {t.toFixed(1)}
          </text>
        ))}
        <text x={(M.left + width - M.right) / 2} y={height - 6} textAnchor="middle" fontSize={11} fill={tokens.muted}>
          Stated confidence
        </text>
        {values.map((b, i) => {
          const { x0, x1 } = span(b);
          return <rect key={`hit${i}`} x={x0 - 2} y={M.top} width={Math.max(6, x1 - x0 + 4)} height={2 * half} fill="transparent" onMouseEnter={() => setHover(i)} />;
        })}
      </svg>
      {hovered ? (
        <HoverTip kit={kit} x={(span(hovered).x0 + span(hovered).x1) / 2} y={M.top + 8} width={width}>
          <TipRow kit={kit}>
            <strong>Stated {hist.distinct ? hovered.lo.toFixed(2) : `${hovered.lo.toFixed(1)} to ${hovered.hi.toFixed(1)}`}</strong>
          </TipRow>
          <TipRow kit={kit} label="Right:">
            {fmt(hovered.correct)}
            {scale === "share" ? " of right answers" : ""}
          </TipRow>
          <TipRow kit={kit} label="Wrong:">
            {fmt(hovered.wrong)}
            {scale === "share" ? " of wrong answers" : ""}
          </TipRow>
        </HoverTip>
      ) : null}
    </div>
  );
}
