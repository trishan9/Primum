"use client";

import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { useMemo, useState } from "react";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Measured, proportionTicks } from "@/components/charts/Measured";
import { useChartTheme, type ChartTheme } from "@/components/charts/theme";
import { estimatorLabel, type ReliabilityBin, type ReliabilitySeries } from "@/lib/charts/rq2";
import { fmtInt, fmtPct } from "@/lib/format";

import { Mark } from "./rq2-forest";
import { RELIABILITY_HEIGHT, RELIABILITY_STRIP } from "./rq2-heights";
import { HoverTip, TipRow } from "./rq2-tooltip";

const M = { top: 10, right: 12, left: 48, axis: 40 };

/**
 * Reliability diagram: each bin at its mean predicted probability (x) and observed share correct
 * (y), against the dashed diagonal of perfect calibration: the stated confidence and one learned
 * model only (other estimators' sparse end bins would only add noise; they are in the table). Under the plot, a strip shows how the
 * predictions are spread over the bins, because a bin with few predictions says little.
 */
export function ReliabilityChart({ series, reference, focus, ariaLabel }: { series: readonly ReliabilitySeries[]; reference: string; focus: string | null; ariaLabel: string }) {
  const kit = useChartTheme();
  if (!kit) return <ChartSkeleton height={RELIABILITY_HEIGHT} />;
  return (
    <Measured height={RELIABILITY_HEIGHT}>
      {(width) => <Plot kit={kit} width={width} series={series} reference={reference} focus={focus} ariaLabel={ariaLabel} />}
    </Measured>
  );
}

function Plot({ kit, width, series, reference, focus, ariaLabel }: { kit: ChartTheme; width: number; series: readonly ReliabilitySeries[]; reference: string; focus: string | null; ariaLabel: string }) {
  const { tokens } = kit;
  const [hover, setHover] = useState<number | null>(null);
  const plotBottom = RELIABILITY_HEIGHT - RELIABILITY_STRIP - M.axis + 24;
  const stripTop = plotBottom + 10;
  const stripBottom = RELIABILITY_HEIGHT - M.axis;
  const x = scaleLinear().domain([0, 1]).range([M.left, width - M.right]);
  const y = scaleLinear().domain([0, 1]).range([plotBottom, M.top]);
  const ticks = proportionTicks(width - M.left - M.right);
  const path = line<ReliabilityBin>()
    .x((b) => x(b.confidence))
    .y((b) => y(b.accuracy));

  const emphasised = useMemo(
    () =>
      [
        { name: reference, color: tokens["series-1"], shape: "dot" as const },
        ...(focus ? [{ name: focus, color: tokens["series-2"], shape: "diamond" as const }] : []),
      ].flatMap((e) => {
        const s = series.find((c) => c.name === e.name);
        return s ? [{ ...e, series: s }] : [];
      }),
    [series, reference, focus, tokens],
  );

  // The bin edges every emphasised series uses (their lo/hi), for the strip and hover columns.
  const edges = useMemo(() => {
    const map = new Map<string, { lo: number; hi: number }>();
    for (const e of emphasised) for (const b of e.series.bins) map.set(`${b.lo}`, { lo: b.lo, hi: b.hi });
    return [...map.values()].sort((a, b) => a.lo - b.lo);
  }, [emphasised]);
  const maxShare = Math.max(...emphasised.flatMap((e) => e.series.bins.map((b) => b.count / e.series.total)), 0.01);
  const stripY = scaleLinear().domain([0, maxShare]).range([stripBottom, stripTop]);
  const hovered = hover !== null ? edges[hover] : undefined;

  return (
    <div className="relative" onMouseLeave={() => setHover(null)}>
      <svg width={width} height={RELIABILITY_HEIGHT} role="img" aria-label={ariaLabel} className="block overflow-visible">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={M.top} y2={plotBottom} stroke={tokens.grid} />
            <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} stroke={tokens.grid} />
            <text x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fontSize={12} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
              {fmtPct(t)}
            </text>
            <text x={x(t)} y={stripBottom + 16} textAnchor="middle" fontSize={12} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
              {fmtPct(t)}
            </text>
          </g>
        ))}
        {hovered ? <rect x={x(hovered.lo)} y={M.top} width={x(hovered.hi) - x(hovered.lo)} height={stripBottom - M.top} fill={tokens.raised} /> : null}
        {/* The yardstick: muted ink (≥ 3:1), dashed. Its label sits in the empty lower right, away from the points. */}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={tokens.muted} strokeDasharray="4 3" />
        <text
          x={x(0.98)}
          y={y(0.08)}
          textAnchor="end"
          fontSize={11}
          fill={tokens.muted}
          stroke={tokens.surface}
          strokeWidth={3}
          paintOrder="stroke"
        >
          dashed diagonal: perfect calibration
        </text>
        {emphasised.map((e) => (
          <g key={e.name}>
            <path d={path(e.series.bins) ?? undefined} fill="none" stroke={e.color} strokeWidth={2} strokeLinejoin="round" />
            {e.series.bins.map((b) => (
              <Mark key={b.lo} shape={e.shape} cx={x(b.confidence)} cy={y(b.accuracy)} color={e.color} surface={tokens.surface} size={3.5} />
            ))}
          </g>
        ))}
        <line x1={M.left} x2={width - M.right} y1={plotBottom} y2={plotBottom} stroke={tokens.axis} />
        <text transform={`translate(12 ${(M.top + plotBottom) / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill={tokens.muted}>
          Observed share correct
        </text>

        {/* the strip: share of each series' predictions per bin, side by side */}
        <text x={M.left - 8} y={(stripTop + stripBottom) / 2} dy="0.32em" textAnchor="end" fontSize={11} fill={tokens.muted}>
          share
        </text>
        {edges.map((edge) => {
          const bandWidth = x(edge.hi) - x(edge.lo);
          const barWidth = Math.max(2, Math.min(10, (bandWidth - 6) / Math.max(1, emphasised.length)));
          const start = x(edge.lo) + (bandWidth - barWidth * emphasised.length - 2 * (emphasised.length - 1)) / 2;
          return emphasised.map((e, i) => {
            const bin = e.series.bins.find((b) => b.lo === edge.lo);
            if (!bin) return null;
            const top = stripY(bin.count / e.series.total);
            return <rect key={`${e.name}-${edge.lo}`} x={start + i * (barWidth + 2)} y={top} width={barWidth} height={Math.max(0, stripBottom - top)} rx={1.5} fill={e.color} />;
          });
        })}
        <line x1={M.left} x2={width - M.right} y1={stripBottom} y2={stripBottom} stroke={tokens.axis} />
        <text x={(M.left + width - M.right) / 2} y={stripBottom + 34} textAnchor="middle" fontSize={11} fill={tokens.muted}>
          Predicted probability correct
        </text>
        {edges.map((edge, i) => (
          <rect key={edge.lo} x={x(edge.lo)} y={M.top} width={x(edge.hi) - x(edge.lo)} height={stripBottom - M.top} fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
      {hovered ? (
        <HoverTip kit={kit} x={x((hovered.lo + hovered.hi) / 2)} y={M.top + 20} width={width}>
          <TipRow kit={kit}>
            Predicted {fmtPct(hovered.lo)} to {fmtPct(hovered.hi)}
          </TipRow>
          {emphasised.map((e) => {
            const bin = e.series.bins.find((b) => b.lo === hovered.lo);
            return (
              <TipRow key={e.name} kit={kit} label={`${estimatorLabel(e.name)}:`}>
                {bin ? (
                  <>
                    says {fmtPct(bin.confidence, 1)}, right {fmtPct(bin.accuracy, 1)} <span style={{ color: tokens.muted }}>({fmtInt(bin.count)} predictions)</span>
                  </>
                ) : (
                  "no predictions"
                )}
              </TipRow>
            );
          })}
        </HoverTip>
      ) : null}
    </div>
  );
}
