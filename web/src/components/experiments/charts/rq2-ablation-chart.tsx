"use client";

import { useMemo } from "react";
import { Bar, type BarCustomLayerProps, type BarDatum } from "@nivo/bar";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Measured } from "@/components/charts/Measured";
import { useChartTheme } from "@/components/charts/theme";
import { ablationGroup, signed, type AblationRow } from "@/lib/charts/rq2";
import { fmtMetric } from "@/lib/format";

import { ablationHeight } from "./rq2-heights";

interface Datum extends BarDatum {
  group: string;
  delta: number;
  auroc: number;
  reference: number;
}

/** The signed change printed past each bar's end, on the side the bar grows toward. */
function EndLabels({ bars, xScale }: BarCustomLayerProps<Datum>) {
  const kit = useChartTheme();
  if (!kit) return null;
  const x = xScale as unknown as (v: number) => number;
  return (
    <g aria-hidden="true">
      {bars.map((bar) => {
        const d = bar.data.data;
        const negative = d.delta < 0;
        const end = negative ? bar.x : bar.x + bar.width;
        const zero = x(0);
        return (
          <text
            key={bar.key}
            x={negative ? Math.min(end, zero) - 6 : Math.max(end, zero) + 6}
            y={bar.y + bar.height / 2}
            dy="0.32em"
            textAnchor={negative ? "end" : "start"}
            fontSize={12}
            fontWeight={600}
            fill={kit.tokens.text}
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {signed(d.delta)}
          </text>
        );
      })}
    </g>
  );
}

/** A solid line at zero: the full model. */
function ZeroLine({ xScale, innerHeight }: BarCustomLayerProps<Datum>) {
  const kit = useChartTheme();
  if (!kit) return null;
  const x = (xScale as unknown as (v: number) => number)(0);
  // The full model is the yardstick every bar is read against, so it wears muted ink (≥ 3:1), not the axis grey.
  return <line x1={x} x2={x} y1={0} y2={innerHeight} stroke={kit.tokens.muted} strokeWidth={1} />;
}

/**
 * Change in mean AUROC when one feature group is dropped from the learned model, as bars from
 * zero (the full model). The file has no interval for these differences of means, so none is
 * drawn: the bars are point differences only. One neutral fill: direction and the signed end
 * label carry the sign, so no colour is spent on it (orange means "learned" on this tab).
 */
export function AblationChart({ rows, reference, ariaLabel }: { rows: readonly AblationRow[]; reference: AblationRow; ariaLabel: string }) {
  const kit = useChartTheme();
  const data = useMemo<Datum[]>(
    () =>
      rows
        .flatMap((r) => (r.deltaFromAll === null ? [] : [{ group: ablationGroup(r.name), delta: r.deltaFromAll, auroc: r.estimate.value, reference: reference.estimate.value }]))
        // nivo draws horizontal bands bottom-up; reverse so the file's order reads top-down.
        .reverse(),
    [rows, reference],
  );
  const height = ablationHeight(data.length);
  const extent = useMemo(() => Math.max(...data.map((d) => Math.abs(d.delta)), 1e-3) * 1.6, [data]);
  if (!kit) return <ChartSkeleton height={height} />;
  const { theme, tokens, animate } = kit;

  return (
    <Measured height={height}>
      {(width) => {
        // Room for the longest group name at 12px (about 7px a character), never more than 45% of the width.
        const longest = Math.max(...data.map((d) => d.group.length), 4);
        const margin = { top: 4, right: 12, bottom: 28, left: Math.min(Math.round(width * 0.45), longest * 7 + 16) };
        const ticks = [-extent / 1.6, 0, extent / 1.6].map((v) => Number(v.toPrecision(2)));
        return (
          <Bar<Datum>
            width={width}
            height={height}
            data={data}
            keys={["delta"]}
            indexBy="group"
            layout="horizontal"
            valueScale={{ type: "linear", min: -extent, max: extent }}
            indexScale={{ type: "band", round: true }}
            margin={margin}
            padding={0.42}
            borderRadius={3}
            colors={() => tokens["series-rest"]}
            enableLabel={false}
            enableGridX
            enableGridY={false}
            gridXValues={ticks}
            axisBottom={{ tickValues: ticks, tickSize: 0, tickPadding: 8, format: (v) => (Number(v) === 0 ? "0" : signed(Number(v), 2)) }}
            axisLeft={{ tickSize: 0, tickPadding: 10 }}
            layers={["grid", "axes", ZeroLine, "bars", EndLabels]}
            theme={theme}
            animate={animate}
            role="img"
            ariaLabel={ariaLabel}
            tooltip={({ data: d }) => (
              <div style={theme.tooltip?.container}>
                <p>
                  <strong>Without {d.group}</strong>
                </p>
                <p>
                  <span style={{ color: tokens.muted }}>Mean AUROC </span>
                  {fmtMetric(d.auroc)} <span style={{ color: tokens.muted }}>vs {fmtMetric(d.reference)} with all features</span>
                </p>
                <p>
                  <span style={{ color: tokens.muted }}>Change </span>
                  {signed(d.delta)} <span style={{ color: tokens.muted }}>(difference of means, no interval)</span>
                </p>
              </div>
            )}
          />
        );
      }}
    </Measured>
  );
}
