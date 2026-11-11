"use client";

import type { BarCustomLayerProps, BarDatum } from "@nivo/bar";

import { useChartTheme } from "./theme";

/**
 * A bar datum that carries its interval. `lo`/`hi` are in the value's units; leave them out (or
 * equal to the value) where the file has no interval and no whisker is drawn. `valueLabel` is the
 * preformatted value printed past the whisker (or the bar end); omit it to print nothing.
 */
export interface IntervalDatum extends BarDatum {
  lo: number;
  hi: number;
  valueLabel: string;
}

const CAP = 5;

function Whisker({ x1, x2, y1, y2, horizontal, color }: { x1: number; x2: number; y1: number; y2: number; horizontal: boolean; color: string }) {
  return horizontal ? (
    <>
      <line x1={x1} x2={x2} y1={y1} y2={y1} stroke={color} strokeWidth={1.5} />
      <line x1={x1} x2={x1} y1={y1 - CAP} y2={y1 + CAP} stroke={color} strokeWidth={1.5} />
      <line x1={x2} x2={x2} y1={y1 - CAP} y2={y1 + CAP} stroke={color} strokeWidth={1.5} />
    </>
  ) : (
    <>
      <line x1={x1} x2={x1} y1={y1} y2={y2} stroke={color} strokeWidth={1.5} />
      <line x1={x1 - CAP} x2={x1 + CAP} y1={y1} y2={y1} stroke={color} strokeWidth={1.5} />
      <line x1={x1 - CAP} x2={x1 + CAP} y1={y2} y2={y2} stroke={color} strokeWidth={1.5} />
    </>
  );
}

/**
 * nivo bar layer for `layout="horizontal"`: a whisker from `lo` to `hi` across each bar and the
 * value label past its right end. Add it after "bars": `layers={["grid", "axes", "bars", HorizontalBarIntervals]}`.
 */
export function HorizontalBarIntervals<D extends IntervalDatum>({ bars, xScale }: BarCustomLayerProps<D>) {
  const kit = useChartTheme();
  if (!kit) return null;
  const x = xScale as unknown as (v: number) => number;
  return (
    <g aria-hidden="true">
      {bars.map((bar) => {
        const d = bar.data.data;
        const cy = bar.y + bar.height / 2;
        const hasInterval = d.hi > d.lo;
        const end = Math.max(bar.x + bar.width, hasInterval ? x(d.hi) : 0);
        return (
          <g key={bar.key}>
            {hasInterval ? <Whisker x1={x(d.lo)} x2={x(d.hi)} y1={cy} y2={cy} horizontal color={kit.tokens.text} /> : null}
            {d.valueLabel ? (
              <text x={end + 8} y={cy} dy="0.32em" fill={kit.tokens.text} fontSize={12} fontWeight={600} style={{ fontVariantNumeric: "tabular-nums" }}>
                {d.valueLabel}
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

/** The same for `layout="vertical"`: whisker up the bar's centre, label above the higher of bar top and `hi`. */
export function VerticalBarIntervals<D extends IntervalDatum>({ bars, yScale }: BarCustomLayerProps<D>) {
  const kit = useChartTheme();
  if (!kit) return null;
  const y = yScale as unknown as (v: number) => number;
  return (
    <g aria-hidden="true">
      {bars.map((bar) => {
        const d = bar.data.data;
        const cx = bar.x + bar.width / 2;
        const hasInterval = d.hi > d.lo;
        const top = Math.min(bar.y, hasInterval ? y(d.hi) : Infinity);
        return (
          <g key={bar.key}>
            {hasInterval ? <Whisker x1={cx} x2={cx} y1={y(d.lo)} y2={y(d.hi)} horizontal={false} color={kit.tokens.text} /> : null}
            {d.valueLabel ? (
              <text x={cx} y={top - 8} textAnchor="middle" fill={kit.tokens.text} fontSize={12} fontWeight={600} style={{ fontVariantNumeric: "tabular-nums" }}>
                {d.valueLabel}
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}
