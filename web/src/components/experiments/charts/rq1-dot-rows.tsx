"use client";

import { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { useChartTheme } from "@/components/charts/theme";
import { useWidth } from "@/components/charts/useSize";

import { Mark, type MarkShape } from "./rq2-forest";
import { dotRowsHeight } from "./rq1-heights";

export interface DotSeries {
  key: string;
  name: string;
  color: string;
  /** the mark, so series are told apart by shape as well as colour (default: dot) */
  shape?: MarkShape;
}

export interface DotMark {
  series: string;
  value: number;
  /** interval ends; null where the file gives none (no whisker is drawn) */
  lo: number | null;
  hi: number | null;
}

export interface DotRow {
  key: string;
  label: string;
  /** set the label in ink and semibold: the deployed retriever */
  emphasis?: boolean;
  marks: DotMark[];
  /** a short preformatted note right of the plot ("p 0.001") */
  aside?: string;
}

interface DotRowsProps {
  rows: readonly DotRow[];
  series: readonly DotSeries[];
  domain: [number, number];
  format: (value: number) => string;
  /** Tick labels, when they should be shorter than `format` ("60%" for "60.0%"). */
  tickFormat?: (value: number) => string;
  /** A vertical reference (zero for differences) drawn in ink, dashed. */
  reference?: { value: number; label: string };
  /** Join each row's lowest and highest mark with a hairline (a dumbbell). */
  connect?: boolean;
  /** What the interval is, for the tooltip ("95% bootstrap CI"). */
  intervalLabel: string;
  ariaLabel: string;
  /** Room on the right for `aside`. */
  asideWidth?: number;
}

const ROW = 34;
const TOP = 8;
const CAP = 4;

/**
 * Rows of point estimates with interval whiskers on one shared axis: one row per retriever, one
 * dot per series. Positions, not lengths, so the axis may start where the data does; the domain
 * is passed in and printed. Arrow keys step through the rows; Escape or Tab leaves.
 */
export function DotRows({ rows, series, domain, format, tickFormat, reference, connect = false, intervalLabel, ariaLabel, asideWidth = 0 }: DotRowsProps) {
  const kit = useChartTheme();
  const [ref, width] = useWidth<HTMLDivElement>();
  const [active, setActive] = useState<number | null>(null);
  const height = dotRowsHeight(rows.length);

  const layout = useMemo(() => {
    const longest = Math.max(0, ...rows.map((r) => r.label.length));
    const labelW = Math.min(Math.max(72, longest * 6.8 + 14), Math.max(96, width * 0.42));
    const left = labelW;
    const right = asideWidth + (asideWidth > 0 ? 12 : 20);
    const plotW = Math.max(40, width - left - right);
    const x = scaleLinear().domain(domain).range([left, left + plotW]);
    const ticks = x.ticks(plotW < 200 ? 3 : plotW < 380 ? 5 : 8);
    return { left, plotW, x, ticks, labelW };
  }, [rows, width, domain, asideWidth]);

  const seriesByKey = useMemo(() => new Map(series.map((s) => [s.key, s])), [series]);

  if (!kit) return <ChartSkeleton height={height} />;
  const { tokens } = kit;
  const { left, plotW, x, ticks } = layout;
  const plotBottom = TOP + rows.length * ROW;
  const activeRow = active !== null ? rows[active] : null;
  const tipY = active !== null ? TOP + active * ROW + ROW : 0;

  return (
    <div ref={ref} className="relative">
      {width > 0 ? (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${ariaLabel} Use the up and down arrow keys to read each row.`}
          tabIndex={0}
          className="block rounded-control outline-offset-4"
          onPointerLeave={() => setActive(null)}
          onBlur={() => setActive(null)}
          onKeyDown={(event) => {
            if (rows.length === 0) return;
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              const step = event.key === "ArrowDown" ? 1 : -1;
              setActive((i) => (i === null ? (step > 0 ? 0 : rows.length - 1) : Math.min(rows.length - 1, Math.max(0, i + step))));
            } else if (event.key === "Escape") setActive(null);
          }}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={TOP} y2={plotBottom} stroke={tokens.grid} />
              <text x={x(t)} y={plotBottom + 18} textAnchor="middle" fill={tokens.muted} fontSize={12} style={{ fontVariantNumeric: "tabular-nums" }}>
                {(tickFormat ?? format)(t)}
              </text>
            </g>
          ))}
          <line x1={left} x2={left + plotW} y1={plotBottom} y2={plotBottom} stroke={tokens.axis} />
          {reference ? (
            <line x1={x(reference.value)} x2={x(reference.value)} y1={TOP - 4} y2={plotBottom} stroke={tokens.muted} strokeDasharray="4 3" />
          ) : null}

          {rows.map((row, i) => {
            const cy = TOP + i * ROW + ROW / 2;
            const on = active === i;
            const values = row.marks.map((m) => m.value);
            return (
              <g key={row.key} onPointerEnter={() => setActive(i)}>
                <rect x={0} y={cy - ROW / 2} width={width} height={ROW} fill={on ? tokens.raised : "transparent"} rx={6} />
                <text
                  x={left - 12}
                  y={cy}
                  dy="0.32em"
                  textAnchor="end"
                  fill={row.emphasis ? tokens.text : tokens.muted}
                  fontSize={12}
                  fontWeight={row.emphasis ? 600 : 400}
                >
                  {row.label}
                </text>
                {connect && values.length > 1 ? (
                  <line x1={x(Math.min(...values))} x2={x(Math.max(...values))} y1={cy} y2={cy} stroke={tokens.control} strokeWidth={1} />
                ) : null}
                {row.marks.map((m) => {
                  const s = seriesByKey.get(m.series);
                  const color = s?.color ?? tokens["series-rest"];
                  const hasCi = m.lo !== null && m.hi !== null && m.hi > m.lo;
                  return (
                    <g key={m.series}>
                      {hasCi ? (
                        <>
                          <line x1={x(m.lo as number)} x2={x(m.hi as number)} y1={cy} y2={cy} stroke={color} strokeWidth={1.5} />
                          <line x1={x(m.lo as number)} x2={x(m.lo as number)} y1={cy - CAP} y2={cy + CAP} stroke={color} strokeWidth={1.5} />
                          <line x1={x(m.hi as number)} x2={x(m.hi as number)} y1={cy - CAP} y2={cy + CAP} stroke={color} strokeWidth={1.5} />
                        </>
                      ) : null}
                      <Mark shape={s?.shape ?? "dot"} cx={x(m.value)} cy={cy} color={color} surface={on ? tokens.raised : tokens.surface} size={on ? 5.5 : 4.5} />
                    </g>
                  );
                })}
                {row.aside ? (
                  <text x={width - 4} y={cy} dy="0.32em" textAnchor="end" fill={tokens.muted} fontSize={12} style={{ fontVariantNumeric: "tabular-nums" }}>
                    {row.aside}
                  </text>
                ) : null}
              </g>
            );
          })}
        </svg>
      ) : (
        <div style={{ height }} />
      )}

      {activeRow && width > 0 ? (
        <div
          role="status"
          className="num pointer-events-none absolute z-10 min-w-40 rounded-control border border-line bg-surface px-2.5 py-2 text-xs shadow-float"
          style={{ top: tipY + 4, left: Math.min(Math.max(8, left), Math.max(8, width - 232)) }}
        >
          <p className="font-medium text-ink">{activeRow.label}</p>
          {activeRow.marks.map((m) => {
            const s = seriesByKey.get(m.series);
            return (
              <p key={m.series} className="mt-1 flex items-center gap-2">
                <svg aria-hidden="true" width="12" height="12" viewBox="0 0 12 12" className="shrink-0">
                  <Mark shape={s?.shape ?? "dot"} cx={6} cy={6} color={s?.color ?? "currentColor"} surface="var(--surface)" size={3.5} />
                </svg>
                {series.length > 1 ? <span className="text-muted">{s?.name}</span> : null}
                <span className="font-semibold text-ink">{format(m.value)}</span>
                {m.lo !== null && m.hi !== null ? (
                  <span className="text-muted">
                    {intervalLabel} {format(m.lo)} to {format(m.hi)}
                  </span>
                ) : null}
              </p>
            );
          })}
          {activeRow.aside ? <p className="mt-1 text-muted">{activeRow.aside}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
