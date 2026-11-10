"use client";

import { useState } from "react";

import { linear, niceDomain, niceTicks, tickFormat } from "./scale";
import { useWidth } from "./useSize";

export interface IntervalSeries {
  name: string;
  color: string;
  /** One entry per category; null where the series has no value. */
  values: ({ value: number; lo: number | null; hi: number | null } | null)[];
}

interface IntervalChartProps {
  categories: string[];
  series: IntervalSeries[];
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
  yFormat?: (value: number) => string;
  /** A level the values are meant to stay under or reach. */
  target?: { y: number; label: string };
  height?: number;
}

const M = { top: 26, right: 14, bottom: 40, left: 46 };
/** Extra room on the right when a target line needs its label beside the plot, clear of the marks. */
const TARGET_GUTTER = 58;

/** Point estimates with intervals, grouped by category: one dot per series, a whisker for its interval. */
export function IntervalChart({ categories, series, xLabel, yLabel, ariaLabel, yFormat, target, height = 264 }: IntervalChartProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [active, setActive] = useState<{ c: number; s: number } | null>(null);

  const flat = series.flatMap((s) => s.values.flatMap((v) => (v ? [v.value, v.lo ?? v.value, v.hi ?? v.value] : [])));
  const max = Math.max(...flat, target?.y ?? 0, 0);
  const yd = niceDomain(0, max * 1.08 || 1, 5);
  const plotW = Math.max(0, width - M.left - M.right - (target ? TARGET_GUTTER : 0));
  const plotH = height - M.top - M.bottom;
  const sy = linear(yd, [M.top + plotH, M.top]);
  const yTicks = niceTicks(yd[0], yd[1], 5);
  const fy = yFormat ?? tickFormat(yTicks);
  const band = categories.length ? plotW / categories.length : 0;
  const offset = Math.min(18, (band * 0.7) / Math.max(1, series.length));
  const cx = (c: number, s: number) => M.left + band * (c + 0.5) + (s - (series.length - 1) / 2) * offset;

  const marks = categories.flatMap((_, c) => series.map((_, s) => ({ c, s }))).filter(({ c, s }) => series[s]?.values[c]);
  const activeValue = active ? series[active.s]?.values[active.c] : null;
  const tipX = active ? cx(active.c, active.s) : 0;
  const flip = tipX > width / 2;

  return (
    <div ref={ref} className="relative">
      {width > 0 ? (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${ariaLabel} Use the left and right arrow keys to read values.`}
          tabIndex={0}
          className="block rounded-control outline-offset-4"
          onPointerLeave={() => setActive(null)}
          onBlur={() => setActive(null)}
          onKeyDown={(event) => {
            if (marks.length === 0) return;
            if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
              event.preventDefault();
              const index = active ? marks.findIndex((m) => m.c === active.c && m.s === active.s) : -1;
              const step = event.key === "ArrowRight" ? 1 : -1;
              const next = Math.min(marks.length - 1, Math.max(0, index === -1 ? (step > 0 ? 0 : marks.length - 1) : index + step));
              setActive(marks[next] ?? null);
            } else if (event.key === "Escape") setActive(null);
          }}
        >
          <text x={0} y={11} className="fill-(--muted) text-[12px]">
            {yLabel}
          </text>
          {yTicks.map((tick) => (
            <g key={tick}>
              <line x1={M.left} x2={M.left + plotW} y1={sy(tick)} y2={sy(tick)} stroke="var(--grid)" strokeWidth="1" />
              <text x={M.left - 8} y={sy(tick)} dy="0.32em" textAnchor="end" className="num fill-(--muted) text-[12px]">
                {fy(tick)}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={M.left + plotW} y1={sy(0)} y2={sy(0)} stroke="var(--axis)" strokeWidth="1" />

          {categories.map((category, c) => (
            <text key={category} x={M.left + band * (c + 0.5)} y={M.top + plotH + 16} textAnchor="middle" className="num fill-(--muted) text-[12px]">
              {category}
            </text>
          ))}
          <text x={M.left + plotW / 2} y={height - 4} textAnchor="middle" className="fill-(--muted) text-[12px]">
            {xLabel}
          </text>

          {target ? (
            <g>
              <line x1={M.left} x2={M.left + plotW} y1={sy(target.y)} y2={sy(target.y)} stroke="var(--text)" strokeWidth="1" strokeDasharray="5 4" />
              <text x={M.left + plotW + 6} y={sy(target.y)} dy="0.32em" className="num fill-(--text) text-[12px] font-medium">
                {target.label}
              </text>
            </g>
          ) : null}

          {marks.map(({ c, s }) => {
            const item = series[s] as IntervalSeries;
            const value = item.values[c];
            if (!value) return null;
            const x = cx(c, s);
            const on = active?.c === c && active?.s === s;
            return (
              <g key={`${c}-${s}`} onPointerEnter={() => setActive({ c, s })}>
                <rect x={x - Math.max(12, offset / 2)} y={M.top} width={Math.max(24, offset)} height={plotH} fill="transparent" />
                {value.lo !== null && value.hi !== null ? (
                  <line x1={x} x2={x} y1={sy(value.lo)} y2={sy(value.hi)} stroke={item.color} strokeWidth={on ? 2.5 : 1.5} strokeLinecap="round" />
                ) : null}
                <circle cx={x} cy={sy(value.value)} r={on ? 5.5 : 4.5} fill={item.color} stroke="var(--surface)" strokeWidth="2" />
              </g>
            );
          })}
        </svg>
      ) : (
        <div style={{ height }} />
      )}

      {active && activeValue && width > 0 ? (
        <div
          role="status"
          className="num pointer-events-none absolute top-6 z-10 min-w-36 rounded-control border border-line bg-surface px-2.5 py-2 text-xs shadow-float"
          style={flip ? { right: width - tipX + 12 } : { left: tipX + 12 }}
        >
          <p className="text-muted">
            {xLabel} <span className="text-ink">{categories[active.c]}</span>
          </p>
          <p className="mt-1 flex items-center gap-2">
            <svg aria-hidden="true" width="10" height="10" viewBox="0 0 10 10" className="shrink-0">
              <circle cx="5" cy="5" r="4" fill={series[active.s]?.color} />
            </svg>
            <span className="font-semibold text-ink">{fy(activeValue.value)}</span>
            <span className="text-muted">{series[active.s]?.name}</span>
          </p>
          {activeValue.lo !== null && activeValue.hi !== null ? (
            <p className="mt-0.5 text-muted">
              interval {fy(activeValue.lo)} to {fy(activeValue.hi)}
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
