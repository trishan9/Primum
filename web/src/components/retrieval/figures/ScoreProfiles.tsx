"use client";

import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";

import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { useChartTheme } from "@/components/charts/theme";
import type { ChartTokens } from "@/components/charts/tokens";
import { useWidth } from "@/components/charts/useSize";
import { fmtScore } from "@/lib/format";
import type { ScoreProfile } from "@/lib/charts/retrieval";

import { scoreKind } from "../retrievers";
import { SCORE_PANEL_HEIGHT } from "./sizes";

interface ScoreProfilesProps {
  profiles: readonly ScoreProfile[];
  k: number;
  activeId: string | null;
  pinnedId: string | null;
  onHover: (id: string | null) => void;
  onPin: (id: string) => void;
}

/**
 * Small multiples, one per retriever: its score at each rank of its own top k. Each panel has its
 * own vertical scale, fitted to that list's lowest and highest score, because BM25, cosine,
 * reciprocal-rank fusion and cross-encoder scores are not on one scale; read a panel's shape, not
 * its height against another. A dashed rule marks the list's largest step between neighbours.
 */
export function ScoreProfiles({ profiles, k, activeId, pinnedId, onHover, onPin }: ScoreProfilesProps) {
  const kit = useChartTheme();
  if (!kit) return <ChartSkeleton height={SCORE_PANEL_HEIGHT * Math.ceil(profiles.length / 2)} />;
  const highlight = activeId ?? pinnedId;
  return (
    <ul className="grid grid-cols-1 gap-x-6 gap-y-4 min-[420px]:grid-cols-2 xl:grid-cols-4">
      {profiles.map((profile) => (
        <li key={profile.retriever} className="min-w-0">
          <Panel profile={profile} k={k} tokens={kit.tokens} highlight={highlight} onHover={onHover} onPin={onPin} />
        </li>
      ))}
    </ul>
  );
}

const MARGIN = { top: 8, right: 10, bottom: 22, left: 44 };

function Panel({
  profile,
  k,
  tokens,
  highlight,
  onHover,
  onPin,
}: {
  profile: ScoreProfile;
  k: number;
  tokens: ChartTokens;
  highlight: string | null;
  onHover: (id: string | null) => void;
  onPin: (id: string) => void;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const height = SCORE_PANEL_HEIGHT - 28;
  const { points, min, max, steepest } = profile;
  // Fitted to the list's own range (not from zero): these are dots on a line, not bars, and the
  // axis says where it starts. A list of equal scores gets a band around its one value.
  const pad = max > min ? (max - min) * 0.08 : Math.abs(max) * 0.05 || 0.5;
  const y = scaleLinear()
    .domain([min - pad, max + pad])
    .range([height - MARGIN.bottom, MARGIN.top]);
  const x = scaleLinear()
    .domain([1, Math.max(2, k)])
    .range([MARGIN.left + 4, Math.max(MARGIN.left + 8, width - MARGIN.right - 4)]);
  const yTicks = max > min ? [min, max] : [max];
  const xTicks = k <= 5 ? Array.from({ length: k }, (_, i) => i + 1) : [1, ...Array.from({ length: Math.floor(k / 5) }, (_, i) => (i + 1) * 5)].filter((t) => t <= k);
  const trace = line<{ rank: number; score: number }>()
    .x((d) => x(d.rank))
    .y((d) => y(d.score));
  const lit = points.find((p) => p.passageId === highlight);

  return (
    <div>
      <p className="flex items-baseline justify-between gap-2 text-xs">
        <span className="truncate font-mono text-ink">{profile.retriever}</span>
        <span className="shrink-0 text-muted">{scoreKind(profile.retriever)} score</span>
      </p>
      <div ref={ref} className="relative mt-1 min-w-0" style={{ height }}>
        {width > 0 ? (
          <svg width={width} height={height} aria-hidden="true" className="block select-none" style={{ fontFamily: tokens.fontFamily }}>
            {yTicks.map((t) => (
              <g key={t}>
                <line x1={MARGIN.left} x2={width - MARGIN.right} y1={y(t)} y2={y(t)} stroke={tokens.grid} />
                <text x={MARGIN.left - 6} y={y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
                  {fmtScore(t)}
                </text>
              </g>
            ))}
            <line x1={MARGIN.left} x2={MARGIN.left} y1={MARGIN.top} y2={height - MARGIN.bottom} stroke={tokens.axis} />
            {xTicks.map((t) => (
              <text key={t} x={x(t)} y={height - 6} textAnchor="middle" fontSize={11} fill={tokens.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
                {t}
              </text>
            ))}
            {steepest ? (
              <line
                x1={x(steepest.afterRank + 0.5)}
                x2={x(steepest.afterRank + 0.5)}
                y1={MARGIN.top}
                y2={height - MARGIN.bottom}
                stroke={tokens.muted}
                strokeDasharray="3 3"
              />
            ) : null}
            <path d={trace(points) ?? undefined} fill="none" stroke={tokens["series-rest"]} strokeWidth={1.5} />
            {points.map((p) => {
              const on = p.passageId === highlight;
              return (
                <g
                  key={p.rank}
                  className="cursor-pointer"
                  onPointerEnter={() => onHover(p.passageId)}
                  onPointerLeave={() => onHover(null)}
                  onClick={() => onPin(p.passageId)}
                >
                  <circle cx={x(p.rank)} cy={y(p.score)} r={9} fill="transparent" />
                  <circle
                    cx={x(p.rank)}
                    cy={y(p.score)}
                    r={on ? 5 : 3.5}
                    fill={on ? tokens.accent : tokens["series-rest"]}
                    stroke={tokens.surface}
                    strokeWidth={on ? 2 : 1}
                  />
                </g>
              );
            })}
            {lit ? (
              <text
                x={x(lit.rank) + 72 > width ? x(lit.rank) - 8 : x(lit.rank) + 8}
                y={Math.max(y(lit.score) - 8, MARGIN.top + 8)}
                textAnchor={x(lit.rank) + 72 > width ? "end" : "start"}
                fontSize={11}
                fontWeight={500}
                fill={tokens.text}
                stroke={tokens.surface}
                strokeWidth={3}
                paintOrder="stroke"
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                #{lit.rank} · {fmtScore(lit.score)}
              </text>
            ) : null}
          </svg>
        ) : null}
      </div>
    </div>
  );
}
