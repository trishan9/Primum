"use client";

import { useMemo, useState } from "react";

import { extent, linear, niceDomain, niceTicks, tickFormat } from "./scale";
import { useWidth } from "./useSize";

export interface XYPoint {
  x: number;
  y: number;
}

export interface XYSeries {
  name: string;
  points: XYPoint[];
  color: string;
  dash?: boolean;
  /** Draw a dot at every point: for binned data, where each point is a measurement. */
  markers?: boolean;
}

interface XYChartProps {
  series: XYSeries[];
  xLabel: string;
  yLabel: string;
  ariaLabel: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  height?: number;
  xFormat?: (value: number) => string;
  yFormat?: (value: number) => string;
  /** The line y = x, for charts where it means "perfect". */
  diagonal?: string;
  hLines?: { y: number; label: string }[];
  vLines?: { x: number; label: string }[];
}

const M = { top: 26, right: 14, bottom: 40, left: 46 };

/**
 * A line chart drawn to the house rules: one y-axis, hairline grid, 2px lines, a crosshair that
 * snaps to the nearest x and reads out every series, and arrow-key access to the same readout.
 * Dashes are reserved for reference lines (a threshold, a target), never for the grid.
 */
export function XYChart({
  series,
  xLabel,
  yLabel,
  ariaLabel,
  xDomain,
  yDomain,
  height = 264,
  xFormat,
  yFormat,
  diagonal,
  hLines = [],
  vLines = [],
}: XYChartProps) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [cursor, setCursor] = useState<number | null>(null);

  const xs = useMemo(() => {
    const all = new Set<number>();
    for (const s of series) for (const p of s.points) if (Number.isFinite(p.x) && Number.isFinite(p.y)) all.add(p.x);
    return [...all].sort((a, b) => a - b);
  }, [series]);

  const allY = series.flatMap((s) => s.points.map((p) => p.y)).concat(hLines.map((h) => h.y));
  const xd = xDomain ?? niceDomain(...extent(xs));
  const yd = yDomain ?? niceDomain(Math.min(0, extent(allY)[0]), extent(allY)[1]);

  const plotW = Math.max(0, width - M.left - M.right - (hLines.length ? 58 : 0));
  const plotH = height - M.top - M.bottom;
  const sx = linear(xd, [M.left, M.left + plotW]);
  const sy = linear(yd, [M.top + plotH, M.top]);
  const xTicks = niceTicks(xd[0], xd[1], plotW < 360 ? 4 : 6);
  const yTicks = niceTicks(yd[0], yd[1], 5);
  const fx = xFormat ?? tickFormat(xTicks);
  const fy = yFormat ?? tickFormat(yTicks);

  const cursorX = cursor === null ? null : (xs[cursor] ?? null);
  const tolerance = (xd[1] - xd[0]) / 40;
  const readout =
    cursorX === null
      ? []
      : series.map((s) => {
          let best: XYPoint | null = null;
          for (const p of s.points) {
            if (!Number.isFinite(p.y)) continue;
            if (!best || Math.abs(p.x - cursorX) < Math.abs(best.x - cursorX)) best = p;
          }
          return { series: s, point: best && Math.abs(best.x - cursorX) <= tolerance + 1e-12 ? best : null };
        });

  const nearestIndex = (clientX: number, target: SVGSVGElement): number | null => {
    if (xs.length === 0) return null;
    const px = clientX - target.getBoundingClientRect().left;
    let best = 0;
    for (let i = 1; i < xs.length; i++) {
      if (Math.abs(sx(xs[i] as number) - px) < Math.abs(sx(xs[best] as number) - px)) best = i;
    }
    return best;
  };

  const tipLeft = cursorX === null ? 0 : sx(cursorX);
  const flip = tipLeft > width / 2;

  return (
    <div ref={ref} className="relative">
      {width > 0 ? (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={`${ariaLabel} Use the left and right arrow keys to read values.`}
          tabIndex={0}
          className="block touch-pan-y rounded-control outline-offset-4"
          onPointerMove={(event) => setCursor(nearestIndex(event.clientX, event.currentTarget))}
          onPointerLeave={() => setCursor(null)}
          onBlur={() => setCursor(null)}
          onKeyDown={(event) => {
            if (xs.length === 0) return;
            if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
              event.preventDefault();
              const step = event.key === "ArrowRight" ? 1 : -1;
              setCursor((current) => Math.min(xs.length - 1, Math.max(0, (current ?? (step > 0 ? -1 : xs.length)) + step)));
            } else if (event.key === "Escape") setCursor(null);
          }}
        >
          <text x={0} y={11} className="fill-(--muted) text-[12px]">
            {yLabel}
          </text>

          {yTicks.map((tick) => (
            <g key={`y${tick}`}>
              <line x1={M.left} x2={M.left + plotW} y1={sy(tick)} y2={sy(tick)} stroke="var(--grid)" strokeWidth="1" />
              <text x={M.left - 8} y={sy(tick)} dy="0.32em" textAnchor="end" className="num fill-(--muted) text-[12px]">
                {fy(tick)}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={M.left + plotW} y1={sy(yd[0])} y2={sy(yd[0])} stroke="var(--axis)" strokeWidth="1" />

          {xTicks.map((tick, i) => (
            <text
              key={`x${tick}`}
              x={sx(tick)}
              y={M.top + plotH + 16}
              textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}
              className="num fill-(--muted) text-[12px]"
            >
              {fx(tick)}
            </text>
          ))}
          <text x={M.left + plotW / 2} y={height - 4} textAnchor="middle" className="fill-(--muted) text-[12px]">
            {xLabel}
          </text>

          {diagonal ? (
            <g>
              <line
                x1={sx(Math.max(xd[0], yd[0]))}
                y1={sy(Math.max(xd[0], yd[0]))}
                x2={sx(Math.min(xd[1], yd[1]))}
                y2={sy(Math.min(xd[1], yd[1]))}
                stroke="var(--axis)"
                strokeWidth="1"
                strokeDasharray="4 4"
              />
              <text
                x={sx(Math.min(xd[1], yd[1])) - 6}
                y={sy(Math.min(xd[1], yd[1])) + 14}
                textAnchor="end"
                className="fill-(--muted) text-[12px]"
              >
                {diagonal}
              </text>
            </g>
          ) : null}

          {hLines.map((h) => (
            <g key={`h${h.y}${h.label}`}>
              <line x1={M.left} x2={M.left + plotW} y1={sy(h.y)} y2={sy(h.y)} stroke="var(--text)" strokeWidth="1" strokeDasharray="5 4" />
              <text x={M.left + plotW + 6} y={sy(h.y)} dy="0.32em" className="num fill-(--text) text-[12px] font-medium">
                {h.label}
              </text>
            </g>
          ))}
          {vLines.map((v) => (
            <g key={`v${v.x}${v.label}`}>
              <line x1={sx(v.x)} x2={sx(v.x)} y1={M.top} y2={M.top + plotH} stroke="var(--text)" strokeWidth="1" strokeDasharray="5 4" />
              <text
                x={sx(v.x) + (sx(v.x) > M.left + plotW * 0.7 ? -5 : 5)}
                y={M.top + 10}
                textAnchor={sx(v.x) > M.left + plotW * 0.7 ? "end" : "start"}
                className="num fill-(--text) text-[12px] font-medium"
              >
                {v.label}
              </text>
            </g>
          ))}

          {series.map((s) => {
            const points = s.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
            const d = points.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join(" ");
            return (
              <g key={s.name}>
                <path
                  d={d}
                  fill="none"
                  stroke={s.color}
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  strokeDasharray={s.dash ? "5 4" : undefined}
                />
                {s.markers
                  ? points.map((p) => (
                      <circle key={`${p.x}`} cx={sx(p.x)} cy={sy(p.y)} r="4" fill={s.color} stroke="var(--surface)" strokeWidth="2" />
                    ))
                  : null}
              </g>
            );
          })}

          {cursorX !== null ? (
            <g pointerEvents="none">
              <line x1={sx(cursorX)} x2={sx(cursorX)} y1={M.top} y2={M.top + plotH} stroke="var(--control)" strokeWidth="1" />
              {readout.map(({ series: s, point }) =>
                point ? (
                  <circle key={s.name} cx={sx(point.x)} cy={sy(point.y)} r="4.5" fill={s.color} stroke="var(--surface)" strokeWidth="2" />
                ) : null,
              )}
            </g>
          ) : null}
        </svg>
      ) : (
        <div style={{ height }} />
      )}

      {cursorX !== null && width > 0 ? (
        <div
          role="status"
          className="num pointer-events-none absolute top-6 z-10 min-w-36 rounded-control border border-line bg-surface px-2.5 py-2 text-xs shadow-float"
          style={flip ? { right: width - tipLeft + 10 } : { left: tipLeft + 10 }}
        >
          <p className="text-muted">
            {xLabel} <span className="text-ink">{fx(cursorX)}</span>
          </p>
          <ul className="mt-1 space-y-0.5">
            {readout.map(({ series: s, point }) => (
              <li key={s.name} className="flex items-center gap-2">
                <svg aria-hidden="true" width="12" height="8" viewBox="0 0 12 8" className="shrink-0">
                  <path d="M0 4h12" stroke={s.color} strokeWidth="2" strokeLinecap="round" />
                </svg>
                <span className="font-semibold text-ink">{point ? fy(point.y) : "–"}</span>
                <span className="text-muted">{s.name}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
