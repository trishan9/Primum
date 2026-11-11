"use client";

import { useMemo } from "react";
import { Line, type LineCustomSvgLayerProps } from "@nivo/line";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Measured, proportionTicks } from "@/components/charts/Measured";
import { useChartTheme } from "@/components/charts/theme";
import { estimatorShortName, riskAtFullCoverage, type RiskCoverageSeries } from "@/lib/charts/rq2";
import { fmtPct } from "@/lib/format";

import { RISK_COVERAGE_HEIGHT } from "./rq2-heights";

interface Series {
  id: string;
  data: { x: number; y: number }[];
}

/**
 * Curves drawn by emphasis: context curves first, thin and grey; the two compared curves last,
 * 2px in their own colours, so they sit on top. The colour carries the emphasis (set by `colors`).
 */
function EmphasisLines({ series, lineGenerator }: LineCustomSvgLayerProps<Series>) {
  const kit = useChartTheme();
  if (!kit) return null;
  const rest = kit.tokens["series-rest"];
  const ordered = [...series].sort((a, b) => Number(a.color !== rest) - Number(b.color !== rest));
  return (
    <g>
      {ordered.map((s) => {
        const context = s.color === rest;
        return (
          <path
            key={String(s.id)}
            d={lineGenerator(s.data.map((d) => d.position)) ?? undefined}
            fill="none"
            stroke={s.color}
            strokeWidth={context ? 1.25 : 2}
            strokeOpacity={context ? 0.6 : 1}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        );
      })}
    </g>
  );
}

/**
 * Mean risk-coverage curves (answers released most confident first). The stated confidence and
 * one chosen learned model are drawn in colour; every other estimator stays as grey context. The
 * error rate of answering everything is a horizontal reference.
 */
export function RiskCoverageChart({ series, reference, focus, ariaLabel }: { series: readonly RiskCoverageSeries[]; reference: string; focus: string | null; ariaLabel: string }) {
  const kit = useChartTheme();
  const data = useMemo<Series[]>(() => series.map((s) => ({ id: s.name, data: s.points.map((p) => ({ x: p.coverage, y: p.risk })) })), [series]);
  const full = useMemo(() => riskAtFullCoverage(series), [series]);
  const yMax = useMemo(() => Math.ceil(Math.max(...series.flatMap((s) => s.points.map((p) => p.risk)), 0.05) * 20) / 20, [series]);
  if (!kit) return <ChartSkeleton height={RISK_COVERAGE_HEIGHT} />;
  const { theme, tokens, animate } = kit;
  const colorOf = (id: string) => (id === reference ? tokens["series-1"] : id === focus ? tokens["series-2"] : tokens["series-rest"]);

  return (
    <Measured height={RISK_COVERAGE_HEIGHT}>
      {(width) => {
        const margin = { top: 12, right: 20, bottom: 44, left: 52 };
        const xTicks = proportionTicks(width - margin.left - margin.right);
        return (
          <Line<Series>
            width={width}
            height={RISK_COVERAGE_HEIGHT}
            data={data}
            margin={margin}
            xScale={{ type: "linear", min: 0, max: 1 }}
            yScale={{ type: "linear", min: 0, max: yMax }}
            colors={(s) => colorOf(String(s.id))}
            enablePoints={false}
            enableGridX={false}
            gridYValues={5}
            axisBottom={{ tickValues: xTicks, tickSize: 0, tickPadding: 8, format: (v) => fmtPct(Number(v)), legend: "Coverage: share of questions answered", legendOffset: 34, legendPosition: "middle" }}
            axisLeft={{ tickValues: 5, tickSize: 0, tickPadding: 8, format: (v) => fmtPct(Number(v)), legend: "Share of answers wrong", legendOffset: -44, legendPosition: "middle" }}
            markers={
              full !== null
                ? [
                    {
                      axis: "y",
                      value: full,
                      lineStyle: { stroke: tokens.muted, strokeWidth: 1, strokeDasharray: "4 3" },
                      legend: `answer everything ${fmtPct(full, 1)}`,
                      legendPosition: "top-left",
                      textStyle: { fill: tokens.muted, fontSize: 11 },
                    },
                  ]
                : []
            }
            layers={["grid", "markers", "axes", EmphasisLines, "crosshair", "slices"]}
            enableSlices="x"
            sliceTooltip={({ slice }) => {
              const shown = slice.points.filter((p) => p.seriesId === reference || p.seriesId === focus);
              const x = slice.points[0]?.data.x;
              return (
                <div style={theme.tooltip?.container}>
                  <p style={{ color: tokens.muted }}>Most confident {fmtPct(Number(x))} answered, share wrong</p>
                  {shown.map((p) => (
                    <p key={p.id} className="flex items-center gap-1.5">
                      <span aria-hidden="true" style={{ display: "inline-block", width: 10, height: 2, background: p.seriesColor }} />
                      {estimatorShortName(String(p.seriesId))}
                      {p.seriesId === reference ? " (stated)" : ""} <strong>{fmtPct(Number(p.data.y), 1)}</strong>
                    </p>
                  ))}
                </div>
              );
            }}
            theme={theme}
            animate={animate}
            role="img"
            ariaLabel={ariaLabel}
          />
        );
      }}
    </Measured>
  );
}
