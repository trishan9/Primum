"use client";

import { useMemo } from "react";
import { Line } from "@nivo/line";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Measured } from "@/components/charts/Measured";
import { useChartTheme } from "@/components/charts/theme";
import { paddedDomain, type LossSeries } from "@/lib/charts/rq1";
import { fmtInt } from "@/lib/format";

import { LOSS_CHART_HEIGHT } from "./rq1-heights";

type Series = { id: string; data: { x: number; y: number }[] };

/**
 * The contrastive loss as it was logged, one point per logging step, unsmoothed; the y-axis
 * starts at zero (a loss cannot go below it) and reaches the file's chance level, drawn dashed,
 * so the fall is read against where an untrained ranking would sit.
 */
export function LossChart({ series, ariaLabel }: { series: LossSeries; ariaLabel: string }) {
  const kit = useChartTheme();
  const data = useMemo<Series[]>(() => [{ id: "Training loss", data: series.points.map((p) => ({ x: p.step, y: p.loss })) }], [series]);
  const yMax = useMemo(
    () => paddedDomain([...series.points.map((p) => p.loss), ...(series.chance !== null ? [series.chance] : [])], 0.5, [0, Infinity])[1],
    [series],
  );
  if (!kit) return <ChartSkeleton height={LOSS_CHART_HEIGHT} />;
  const { theme, tokens, animate } = kit;

  return (
    <Measured height={LOSS_CHART_HEIGHT}>
      {(width) => {
        const margin = { top: 12, right: 16, bottom: 44, left: 48 };
        const xTicks = width < 480 ? 4 : 8;
        return (
          <Line<Series>
            width={width}
            height={LOSS_CHART_HEIGHT}
            data={data}
            margin={margin}
            xScale={{ type: "linear", min: 0, max: "auto" }}
            yScale={{ type: "linear", min: 0, max: yMax }}
            curve="linear"
            colors={[tokens["series-1"]]}
            lineWidth={1.5}
            enablePoints={false}
            enableGridX={false}
            gridYValues={5}
            axisBottom={{ tickValues: xTicks, tickSize: 0, tickPadding: 8, format: (v) => fmtInt(Number(v)), legend: "Optimiser step", legendOffset: 36, legendPosition: "middle" }}
            axisLeft={{ tickValues: 5, tickSize: 0, tickPadding: 8, format: (v) => Number(v).toFixed(1) }}
            markers={
              series.chance !== null
                ? [
                    {
                      axis: "y",
                      value: series.chance,
                      lineStyle: { stroke: tokens.text, strokeWidth: 1, strokeDasharray: "5 4" },
                      legend: `chance ${series.chance.toFixed(2)}`,
                      legendPosition: "top-right",
                      textStyle: { fill: tokens.text, fontSize: 12 },
                    },
                  ]
                : []
            }
            enableSlices="x"
            sliceTooltip={({ slice }) => {
              const p = slice.points[0];
              if (!p) return null;
              return (
                <div style={theme.tooltip?.container}>
                  <p style={{ color: tokens.muted }}>Step {fmtInt(Number(p.data.x))}</p>
                  <p>
                    loss <strong>{Number(p.data.y).toFixed(3)}</strong>
                  </p>
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
