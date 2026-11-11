"use client";

import { useMemo } from "react";
import { Bar } from "@nivo/bar";

import { HorizontalBarIntervals, type IntervalDatum } from "@/components/charts/barIntervals";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Measured, proportionTicks } from "@/components/charts/Measured";
import { useChartTheme } from "@/components/charts/theme";
import { fmtInt, fmtPct } from "@/lib/format";
import type { ConditionAccuracy } from "@/lib/charts/pubmedqa";

import { accuracyChartHeight } from "./accuracyChartHeight";

interface Datum extends IntervalDatum {
  label: string;
  accuracy: number;
  n: number;
}

/**
 * Accuracy per evidence condition: one bar per condition from zero, its 95% bootstrap interval as
 * a whisker, the value printed past the whisker. One series, so one colour and no legend.
 * Heavy (nivo): import it with `next/dynamic` and `ssr: false`, as BenchmarkSection does.
 */
export function AccuracyByConditionChart({ rows, ariaLabel }: { rows: readonly ConditionAccuracy[]; ariaLabel: string }) {
  const kit = useChartTheme();
  const data = useMemo<Datum[]>(
    () =>
      rows.flatMap((r) =>
        r.accuracy
          ? [
              {
                label: r.label,
                accuracy: r.accuracy.value,
                lo: r.accuracy.lo ?? r.accuracy.value,
                hi: r.accuracy.hi ?? r.accuracy.value,
                n: r.n ?? 0,
                valueLabel: fmtPct(r.accuracy.value, 1),
              },
            ]
          : [],
      )
        // nivo stacks horizontal bands bottom-up; reverse so the conditions read top-down in method order.
        .reverse(),
    [rows],
  );
  const height = accuracyChartHeight(data.length);
  if (!kit) return <ChartSkeleton height={height} />;
  const { theme, tokens, animate } = kit;

  return (
    <Measured height={height}>
      {(width) => {
        const margin = { top: 4, right: 16, bottom: 28, left: 104 };
        const ticks = proportionTicks(width - margin.left - margin.right);
        return (
          <Bar<Datum>
            width={width}
            height={height}
            data={data}
            keys={["accuracy"]}
            indexBy="label"
            layout="horizontal"
            valueScale={{ type: "linear", min: 0, max: 1 }}
            indexScale={{ type: "band", round: true }}
            margin={margin}
            padding={0.5}
            borderRadius={3}
            colors={() => tokens["series-1"]}
            enableLabel={false}
            enableGridX
            enableGridY={false}
            gridXValues={ticks}
            axisBottom={{ tickValues: ticks, tickSize: 0, tickPadding: 8, format: (v) => fmtPct(Number(v)) }}
            axisLeft={{ tickSize: 0, tickPadding: 10 }}
            layers={["grid", "axes", "bars", HorizontalBarIntervals]}
            theme={theme}
            animate={animate}
            role="img"
            ariaLabel={ariaLabel}
            tooltip={({ data: d }) => (
              <div style={theme.tooltip?.container}>
                <p style={{ color: tokens.muted }}>{d.label}</p>
                <p>
                  <strong>{fmtPct(d.accuracy, 1)}</strong> correct
                </p>
                <p style={{ color: tokens.muted }}>
                  95% bootstrap CI {fmtPct(d.lo, 1)} to {fmtPct(d.hi, 1)}, n = {fmtInt(d.n)}
                </p>
              </div>
            )}
          />
        );
      }}
    </Measured>
  );
}
