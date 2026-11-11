"use client";

import { useMemo } from "react";

import { useChartTheme } from "@/components/charts/theme";
import type { ComparisonRow } from "@/lib/charts/rq2";
import { fmtP, fmtPct } from "@/lib/format";

import { ForestPlot, type ForestRow } from "./rq2-forest";
import { BRIER_ROW } from "./rq2-heights";
import { TipRow } from "./rq2-tooltip";

const fmtDiff = (v: number) => {
  const text = Math.abs(v).toFixed(4);
  return Number(text) === 0 ? text : `${v > 0 ? "+" : "−"}${text}`;
};

/** What a difference interval says against zero; lower Brier is better. */
function reading(row: ComparisonRow): string {
  const b = row.brier;
  if (!b || b.lo === null || b.hi === null) return "no interval reported";
  if (b.hi < 0) return "better: interval entirely below zero";
  if (b.lo > 0) return "worse: interval entirely above zero";
  return "no clear difference: interval spans zero";
}

/**
 * Each learned model's Brier loss minus the stated confidence's, with the file's bootstrap CI,
 * on an axis centred on zero. A model is better only if its whole interval sits left of zero.
 */
export function BrierDiffChart({ rows, reference, intervalLabel, ariaLabel }: { rows: readonly ComparisonRow[]; reference: string; intervalLabel: string; ariaLabel: string }) {
  const kit = useChartTheme();
  const drawn = useMemo(() => rows.filter((r) => r.brier), [rows]);
  const forest = useMemo<ForestRow[]>(() => {
    if (!kit) return [];
    return drawn.map((r) => {
      const b = r.brier as NonNullable<ComparisonRow["brier"]>;
      return {
        key: r.model,
        label: r.model,
        value: b.value,
        lo: b.lo,
        hi: b.hi,
        color: kit.tokens["series-2"],
        shape: "diamond" as const,
        valueLabel: fmtDiff(b.value),
        tip: (
          <>
            <TipRow kit={kit}>
              <strong>{r.model}</strong> minus {reference}
            </TipRow>
            <TipRow kit={kit} label="Brier difference">{fmtDiff(b.value)}</TipRow>
            {b.lo !== null && b.hi !== null ? (
              <TipRow kit={kit} label={intervalLabel}>
                {fmtDiff(b.lo)} to {fmtDiff(b.hi)}
              </TipRow>
            ) : null}
            <TipRow kit={kit} label="p">{fmtP(r.brierP)}</TipRow>
            {r.aurocBetterShare !== null ? <TipRow kit={kit} label="Splits with higher AUROC">{fmtPct(r.aurocBetterShare)}</TipRow> : null}
            <TipRow kit={kit}>{reading(r)}</TipRow>
          </>
        ),
      };
    });
  }, [kit, drawn, reference, intervalLabel]);

  // Symmetric about zero, so "left of zero" and "right of zero" get the same room.
  const extent = useMemo(() => {
    const m = Math.max(...drawn.flatMap((r) => [Math.abs(r.brier?.lo ?? 0), Math.abs(r.brier?.hi ?? 0), Math.abs(r.brier?.value ?? 0)]), 1e-4);
    return m * 1.15;
  }, [drawn]);

  return (
    <ForestPlot
      rows={forest}
      domain={[-extent, extent]}
      rowHeight={BRIER_ROW}
      references={[{ value: 0, label: "no difference", tone: "axis" }]}
      tickFormat={(v) => (v === 0 ? "0" : fmtDiff(v).replace(/0+$/, ""))}
      axisLabel="Brier difference, learned minus stated"
      ends={["← lower, better", "worse →"]}
      ariaLabel={ariaLabel}
    />
  );
}
