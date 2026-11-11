"use client";

import { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { useChartTheme } from "@/components/charts/theme";
import { useWidth } from "@/components/charts/useSize";
import { stripJitter, type PerQuery } from "@/lib/charts/rq1";
import { fmtPct } from "@/lib/format";

import { stripsHeight } from "./rq1-heights";

const ROW = 40;
const TOP = 6;
const ASIDE = 64;

/**
 * Every query above zero as a faint dot, one strip per retriever, over a light band for the
 * middle half and a thin median tick (no outline, so the band never covers the dots). The
 * queries scoring exactly zero, a quarter or more of them, would pile into one invisible dot, so
 * they are drawn as a bar at 0 whose height is their share, and the share is printed on the
 * right: the plot and the column agree. Vertical spread inside a strip is a fixed deterministic
 * jitter and carries no data. The axis is the metric's full [0, 1] range.
 */
export function PerQueryStrips({ rows, deployed, ariaLabel }: { rows: readonly PerQuery[]; deployed: string | null; ariaLabel: string }) {
  const kit = useChartTheme();
  const [ref, width] = useWidth<HTMLDivElement>();
  const [active, setActive] = useState<number | null>(null);
  const height = stripsHeight(rows.length);

  const layout = useMemo(() => {
    const longest = Math.max(0, ...rows.map((r) => r.system.length));
    const left = Math.min(Math.max(72, longest * 6.8 + 14), Math.max(96, width * 0.42));
    const plotW = Math.max(40, width - left - ASIDE - 8);
    // 0 sits 6px in, so the zero bar is not cut by the label column.
    const x = scaleLinear().domain([0, 1]).range([left + 6, left + plotW]);
    return { left, plotW, x, ticks: plotW < 200 ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1] };
  }, [rows, width]);

  if (!kit) return <ChartSkeleton height={height} />;
  const { tokens } = kit;
  const { left, plotW, x, ticks } = layout;
  const plotBottom = TOP + rows.length * ROW;
  const activeRow = active !== null ? rows[active] : null;

  return (
    <div ref={ref} className="relative">
      {width > 0 ? (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${ariaLabel} Use the up and down arrow keys to read each retriever's summary.`}
          tabIndex={0}
          className="block rounded-control outline-offset-4"
          onPointerLeave={() => setActive(null)}
          onBlur={() => setActive(null)}
          onKeyDown={(event) => {
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
                {t.toFixed(t === 0 || t === 1 ? 0 : 2)}
              </text>
            </g>
          ))}
          <text x={width - 4} y={plotBottom + 18} textAnchor="end" fill={tokens.muted} fontSize={12}>
            at zero
          </text>

          {rows.map((row, i) => {
            const cy = TOP + i * ROW + ROW / 2;
            const isDeployed = row.system === deployed;
            const dot = isDeployed ? tokens["series-1"] : tokens["series-rest"];
            const { q1, q3, median } = row.summary;
            const on = active === i;
            return (
              <g key={row.system} onPointerEnter={() => setActive(i)}>
                <rect x={0} y={cy - ROW / 2} width={width} height={ROW} rx={6} fill={on ? tokens.raised : "transparent"} />
                <text x={left - 12} y={cy} dy="0.32em" textAnchor="end" fontSize={12} fill={isDeployed ? tokens.text : tokens.muted} fontWeight={isDeployed ? 600 : 400}>
                  {row.system}
                </text>
                <rect x={x(q1)} y={cy - 10} width={Math.max(1, x(q3) - x(q1))} height={20} rx={3} fill={tokens.active} />
                <g fill={dot} fillOpacity={0.4}>
                  {row.values.map((v, j) =>
                    v === 0 ? null : <circle key={j} cx={x(v)} cy={cy + stripJitter(j) * (ROW / 2 - 7)} r={2} />,
                  )}
                </g>
                {row.zeroShare > 0 ? (
                  <rect
                    x={x(0) - 3}
                    y={cy + (ROW / 2 - 4) - Math.max(1, row.zeroShare * (ROW - 8))}
                    width={6}
                    height={Math.max(1, row.zeroShare * (ROW - 8))}
                    rx={1.5}
                    fill={dot}
                  />
                ) : null}
                <line x1={x(median)} x2={x(median)} y1={cy - 12} y2={cy + 12} stroke={tokens.text} strokeWidth={1.5} />
                <text x={width - 4} y={cy} dy="0.32em" textAnchor="end" fill={tokens.text} fontSize={12} style={{ fontVariantNumeric: "tabular-nums" }}>
                  {fmtPct(row.zeroShare, 0)}
                </text>
              </g>
            );
          })}
          <line x1={left} x2={left + plotW} y1={plotBottom} y2={plotBottom} stroke={tokens.axis} />
        </svg>
      ) : (
        <div style={{ height }} />
      )}
      {activeRow && width > 0 && active !== null ? (
        <div
          role="status"
          className="num pointer-events-none absolute z-10 min-w-44 rounded-control border border-line bg-surface px-2.5 py-2 text-xs shadow-float"
          style={{ top: TOP + (active + 1) * ROW + 4, left: Math.min(Math.max(8, left), Math.max(8, width - 220)) }}
        >
          <p className="font-medium text-ink">{activeRow.system}</p>
          <p className="mt-1 text-muted">
            median <span className="font-semibold text-ink">{activeRow.summary.median.toFixed(3)}</span>, mean {activeRow.summary.mean.toFixed(3)}
          </p>
          <p className="text-muted">
            middle half {activeRow.summary.q1.toFixed(3)} to {activeRow.summary.q3.toFixed(3)}
          </p>
          <p className="text-muted">
            {fmtPct(activeRow.zeroShare, 1)} of {activeRow.summary.n} queries score zero
          </p>
        </div>
      ) : null}
    </div>
  );
}
