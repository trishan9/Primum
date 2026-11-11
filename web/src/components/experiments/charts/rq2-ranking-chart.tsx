"use client";

import { useMemo } from "react";

import { useChartTheme } from "@/components/charts/theme";
import { estimatorShortName, type RankedEstimator } from "@/lib/charts/rq2";
import { fmtMetric } from "@/lib/format";

import { ForestPlot, type ForestRow, type MarkShape } from "./rq2-forest";
import { RANKING_ROW } from "./rq2-heights";
import { TipRow } from "./rq2-tooltip";

const SHAPE: Record<RankedEstimator["kind"], MarkShape> = { stated: "dot", learned: "diamond", signal: "ring", other: "ring" };
const KIND_WORD: Record<RankedEstimator["kind"], string> = { stated: "the model's stated confidence", learned: "learned combination", signal: "single signal", other: "estimator" };

/**
 * Every estimator's AUROC under the main calibrator, best mean first: stated confidence in the
 * accent (the reference, with its mean as a dashed line), learned combinations as orange diamonds,
 * single signals as grey rings. Whiskers are the split range the file reports, named by the caller.
 */
export function RankingChart({ ranking, intervalLabel, ariaLabel }: { ranking: readonly RankedEstimator[]; intervalLabel: string; ariaLabel: string }) {
  const kit = useChartTheme();
  const rows = useMemo<ForestRow[]>(() => {
    if (!kit) return [];
    const { tokens } = kit;
    const color = { stated: tokens["series-1"], learned: tokens["series-2"], signal: tokens["series-rest"], other: tokens["series-rest"] };
    return ranking.map((r) => ({
      key: r.name,
      label: r.kind === "stated" ? `${estimatorShortName(r.name)} (stated)` : estimatorShortName(r.name),
      value: r.estimate.value,
      lo: r.estimate.lo,
      hi: r.estimate.hi,
      color: color[r.kind],
      shape: SHAPE[r.kind],
      strong: r.kind === "stated",
      valueLabel: fmtMetric(r.estimate.value),
      tip: (
        <>
          <TipRow kit={kit}>
            <strong>{r.name}</strong>
          </TipRow>
          <TipRow kit={kit} label="Kind">{KIND_WORD[r.kind]}</TipRow>
          <TipRow kit={kit} label="Mean AUROC">{fmtMetric(r.estimate.value)}</TipRow>
          {r.estimate.lo !== null && r.estimate.hi !== null ? (
            <TipRow kit={kit} label={intervalLabel}>
              {fmtMetric(r.estimate.lo)} to {fmtMetric(r.estimate.hi)}
            </TipRow>
          ) : null}
        </>
      ),
    }));
  }, [kit, ranking, intervalLabel]);

  const domain = useMemo<[number, number]>(() => {
    const lows = ranking.map((r) => r.estimate.lo ?? r.estimate.value);
    const highs = ranking.map((r) => r.estimate.hi ?? r.estimate.value);
    // Chance (0.5) is always on the axis, so a signal near it reads as "no better than a coin".
    const lo = Math.floor(Math.min(0.5, ...lows) * 20) / 20;
    const hi = Math.min(1, Math.ceil(Math.max(...highs) * 20) / 20);
    return [lo, hi];
  }, [ranking]);

  const stated = ranking.find((r) => r.kind === "stated");
  return (
    <ForestPlot
      rows={rows}
      domain={domain}
      rowHeight={RANKING_ROW}
      references={[
        { value: 0.5, label: "chance", tone: "axis", dashed: true },
        ...(stated ? [{ value: stated.estimate.value, label: "stated", tone: "accent" as const, dashed: true }] : []),
      ]}
      tickFormat={(v) => v.toFixed(2)}
      axisLabel="AUROC, higher is better"
      ariaLabel={ariaLabel}
    />
  );
}
