"use client";

import { useId, useMemo } from "react";
import { arc } from "d3-shape";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { useChartTheme } from "@/components/charts/theme";
import { cn } from "@/components/ui/cn";
import { CheckGlyph } from "@/components/ui/glyphs";
import { gaugeAngle, gaugeReading, gaugeTakeaway } from "@/lib/charts/ask";
import { fmtAlpha, fmtProb } from "@/lib/format";

interface ConfidenceGaugeProps {
  /** Calibrated probability that the answer is correct (the confidence event's `p_correct`). */
  p: number | null;
  /** Certified threshold for this alpha; null when none certifies it. */
  threshold: number | null;
  alpha: number;
  /** The server's decision (`answered`). */
  answered: boolean;
  className?: string;
}

// Geometry in SVG units; the SVG scales to its box, text stays legible down to 200px wide.
const W = 240;
const R = 96;
const BAND = 18;
const CX = W / 2;
const CY = R + 22;
const H = CY + 12;
const TICKS = Array.from({ length: 11 }, (_, i) => i / 10);

function polar(value: number, radius: number): [number, number] {
  const a = gaugeAngle(value);
  return [CX + radius * Math.sin(a), CY - radius * Math.cos(a)];
}

/**
 * The gate as a half-circle: the band left of the certified threshold is hatched (the system
 * withholds any answer whose pointer lands there), the band right of it is the cleared zone, and
 * the needle is the calibrated confidence. The readings beside it carry every number in words, so
 * neither the hatch nor the colour is needed to read it.
 */
export function ConfidenceGauge({ p, threshold, alpha, answered, className }: ConfidenceGaugeProps) {
  const chart = useChartTheme();
  const patternId = `${useId().replace(/:/g, "")}-hatch`;
  const reading = useMemo(() => gaugeReading(p, threshold, answered), [p, threshold, answered]);
  const takeaway = gaugeTakeaway(reading, alpha);

  const paths = useMemo(() => {
    const band = arc<{ start: number; end: number }>()
      .innerRadius(R - BAND)
      .outerRadius(R)
      .cornerRadius(3)
      .startAngle((d) => gaugeAngle(d.start))
      .endAngle((d) => gaugeAngle(d.end));
    const gate = reading.gate;
    return {
      withheld: gate === null ? band({ start: 0, end: 1 }) : gate > 0 ? band({ start: 0, end: gate }) : null,
      cleared: gate === null || gate >= 1 ? null : band({ start: gate, end: 1 }),
    };
  }, [reading.gate]);

  if (!chart) return <ChartSkeleton height={150} label="Loading confidence gauge" className={className} />;
  const t = chart.tokens;
  const needleColour = reading.released ? t.accent : t.text;
  const needleTurn = reading.pointer === null ? null : (gaugeAngle(reading.pointer) * 180) / Math.PI;
  const gate = reading.gate;
  const gateIn = gate === null ? null : polar(gate, R - BAND - 5);
  const gateOut = gate === null ? null : polar(gate, R + 5);
  const gateLabel = gate === null ? null : polar(gate, R + 14);

  return (
    <figure className={cn("num flex flex-wrap items-center gap-x-6 gap-y-3", className)}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full max-w-[15rem] shrink-0 overflow-visible"
        aria-hidden="true"
      >
        <defs>
          <pattern id={patternId} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill={t.raised} />
            <path d="M0 0v6" stroke={t.hatch} strokeWidth="1.6" />
          </pattern>
        </defs>
        <g transform={`translate(${CX},${CY})`}>
          {paths.withheld ? <path d={paths.withheld} fill={`url(#${patternId})`} stroke={t.border} strokeWidth="1" /> : null}
          {paths.cleared ? <path d={paths.cleared} fill={t["accent-soft"]} stroke={t.border} strokeWidth="1" /> : null}
        </g>

        {TICKS.map((tick, i) => {
          const [x1, y1] = polar(tick, R + 3);
          const [x2, y2] = polar(tick, R + (i % 5 === 0 ? 8 : 6));
          return <line key={tick} x1={x1} y1={y1} x2={x2} y2={y2} stroke={t.control} strokeWidth="1" />;
        })}
        <text x={CX - R + BAND / 2} y={CY + 13} textAnchor="middle" fontSize="11" fill={t.muted}>
          0
        </text>
        {gate === null || Math.abs(gate - 0.5) > 0.12 ? (
          <text x={CX} y={CY - R - 11} textAnchor="middle" fontSize="11" fill={t.muted}>
            0.5
          </text>
        ) : null}
        <text x={CX + R - BAND / 2} y={CY + 13} textAnchor="middle" fontSize="11" fill={t.muted}>
          1
        </text>

        {gate !== null && gateLabel ? (
          <text
            x={gateLabel[0]}
            y={gateLabel[1]}
            textAnchor={gate < 0.4 ? "end" : gate > 0.6 ? "start" : "middle"}
            dominantBaseline="middle"
            fontSize="11"
            fontWeight="600"
            fill={t.text}
          >
            {fmtProb(gate)}
          </text>
        ) : null}
        {gateIn && gateOut ? (
          <line
            x1={gateIn[0]}
            y1={gateIn[1]}
            x2={gateOut[0]}
            y2={gateOut[1]}
            stroke={t.text}
            strokeWidth="2"
            strokeDasharray="3 2"
          />
        ) : null}

        {needleTurn !== null ? (
          <g
            style={{ transform: `translate(${CX}px, ${CY}px) rotate(${needleTurn}deg)`, transition: "transform 700ms var(--ease-out)" }}
          >
            <line x1="0" y1="-6" x2="0" y2={-(R + 2)} stroke={t.surface} strokeWidth="5" strokeLinecap="round" />
            <line x1="0" y1="-6" x2="0" y2={-(R + 2)} stroke={needleColour} strokeWidth="2.5" strokeLinecap="round" />
          </g>
        ) : null}
        <circle cx={CX} cy={CY} r="6" fill={needleColour} stroke={t.surface} strokeWidth="2" />
      </svg>

      <dl className="grid min-w-0 flex-[1_1_12rem] grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-1.5 text-[0.8125rem]">
        <dt className="text-muted">P(correct)</dt>
        <dd className="text-[1.375rem] font-semibold leading-tight text-ink">{fmtProb(reading.pointer)}</dd>
        <dt className="text-muted">Threshold</dt>
        <dd className="text-ink">
          {gate === null ? (
            <span className="text-muted">none certifies α = {fmtAlpha(alpha)}</span>
          ) : (
            <>
              <span className="font-medium">{fmtProb(gate)}</span>
              <span className="text-muted"> at α = {fmtAlpha(alpha)}</span>
            </>
          )}
        </dd>
        {reading.margin !== null ? (
          <>
            <dt className="text-muted">{reading.margin >= 0 ? "Clears it by" : "Short by"}</dt>
            <dd className="text-ink">{fmtProb(Math.abs(reading.margin))}</dd>
          </>
        ) : null}
        <dt className="text-muted">Decision</dt>
        <dd className="flex items-center gap-1.5 font-medium text-ink">
          {reading.released ? (
            <CheckGlyph size={13} className="text-accent" />
          ) : (
            <span aria-hidden="true" className="hatch inline-block size-3 rounded-[3px]" />
          )}
          {reading.released ? "Released" : "Withheld"}
        </dd>
      </dl>
      <figcaption className="sr-only">{takeaway}</figcaption>
    </figure>
  );
}
