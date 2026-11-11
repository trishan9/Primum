"use client";

import { useMemo } from "react";
import { scaleLinear } from "d3-scale";

import { ChartCard } from "@/components/charts/ChartCard";
import { stageWaterfall, waterfallTakeaway } from "@/lib/charts/ask";
import { fmtMs, fmtPct } from "@/lib/format";
import type { StageName } from "@/lib/types";

import { STAGE_COPY } from "./copy";

const label = (stage: string) => STAGE_COPY[stage as StageName]?.label ?? stage;

/**
 * Where the run's time went: one row per stage, each bar starting where the previous stage
 * ended, on one time axis from zero to the run's total. Durations are printed, because the
 * fastest stages are a sliver of a pixel next to generation.
 */
export function StageWaterfall({ timings, source, className }: { timings: Readonly<Record<string, number>>; source: string; className?: string }) {
  const waterfall = useMemo(() => stageWaterfall(timings), [timings]);
  const ticks = useMemo(() => {
    const x = scaleLinear().domain([0, waterfall.total]).nice(4);
    const [, end] = x.domain() as [number, number];
    return { end, values: x.ticks(4) };
  }, [waterfall.total]);
  if (waterfall.spans.length < 2 || waterfall.total <= 0) return null;
  const pct = (ms: number) => `${(ms / ticks.end) * 100}%`;

  return (
    <ChartCard className={className}>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h4">Where the time went</ChartCard.Title>
          <ChartCard.Takeaway>{waterfallTakeaway(waterfall, label)}</ChartCard.Takeaway>
        </ChartCard.Heading>
      </ChartCard.Header>
      <div className="num px-4 pb-3 pt-3 sm:px-5" aria-hidden="true">
        <div className="grid grid-cols-[6.5rem_minmax(0,1fr)_3.75rem] items-center gap-x-3 gap-y-1.5 text-xs max-sm:grid-cols-[5.25rem_minmax(0,1fr)_3.5rem]">
          {waterfall.spans.map((span) => (
            <div key={span.stage} className="contents">
              <span className="truncate text-muted">{label(span.stage)}</span>
              <span className="relative h-3.5 rounded-[3px] bg-raised">
                <span
                  className="absolute inset-y-0 rounded-[3px] bg-accent"
                  style={{ left: pct(span.start), width: `max(2px, ${pct(span.ms)})` }}
                />
              </span>
              <span className="text-right text-ink">{fmtMs(span.ms)}</span>
            </div>
          ))}
          <span />
          <span className="relative h-4 border-t border-control">
            {ticks.values.map((v, i) => (
              <span
                key={v}
                className={`absolute top-1 whitespace-nowrap text-faint ${i === 0 ? "" : v === ticks.end ? "-translate-x-full" : "-translate-x-1/2 max-sm:hidden"}`}
                style={{ left: pct(v) }}
              >
                {v === 0 ? "0" : fmtMs(v)}
              </span>
            ))}
          </span>
          <span />
        </div>
      </div>
      <ChartCard.Data
        columns={[
          { key: "stage", label: "Stage" },
          { key: "start", label: "Starts at", numeric: true },
          { key: "ms", label: "Duration", numeric: true },
          { key: "share", label: "Share", numeric: true },
        ]}
        rows={waterfall.spans.map((s) => ({ stage: label(s.stage), start: fmtMs(s.start), ms: fmtMs(s.ms), share: fmtPct(s.share) }))}
      >
        <ChartCard.Note>Stages run one after another</ChartCard.Note>
        <ChartCard.Source>{source}</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
}
