"use client";

import { useMemo } from "react";

import { ChartCard } from "@/components/charts/ChartCard";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { useChartTheme } from "@/components/charts/theme";
import { cn } from "@/components/ui/cn";
import { scoreMatrix, scoreMatrixTakeaway } from "@/lib/charts/ask";
import { fmtScore } from "@/lib/format";
import type { Hit } from "@/lib/types";

/**
 * Each retrieved passage's score at every ranking stage, as a compact matrix of bars. Every
 * column has its own zero-based scale (BM25, a cosine and a reciprocal-rank sum are not on one
 * scale), so bars compare passages within a column, never across. The final stage, which set
 * the order, is the accent; the stages that fed it are grey. A dash is a passage that stage did
 * not return (for BM25 and dense: no rank, the backend still reports a score); its bar is faint.
 */
export function ScoreProfile({ hits, retriever, cited, className }: { hits: readonly Hit[]; retriever: string | null; cited: readonly number[]; className?: string }) {
  const chart = useChartTheme();
  const matrix = useMemo(() => scoreMatrix(hits), [hits]);
  if (matrix.rows.length === 0 || matrix.columns.length === 0) return null;
  const finalKey = matrix.columns.at(-1)!.key;
  const signed = matrix.columns.some((c) => c.min < 0);
  const height = 24 + matrix.rows.length * 30;

  return (
    <ChartCard className={className}>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">Score profile</ChartCard.Title>
          <ChartCard.Takeaway>{scoreMatrixTakeaway(matrix)}</ChartCard.Takeaway>
        </ChartCard.Heading>
      </ChartCard.Header>
      <div className="px-4 pb-3 pt-3 sm:px-5">
        {!chart ? (
          <ChartSkeleton height={height} label="Loading score profile" />
        ) : (
          <div
            aria-hidden="true"
            className="num grid items-center gap-x-3 gap-y-1.5 text-xs"
            style={{ gridTemplateColumns: `1.5rem repeat(${matrix.columns.length}, minmax(0, 1fr))` }}
          >
            <span />
            {matrix.columns.map((c) => (
              <span key={c.key} className={cn("truncate", c.key === finalKey ? "font-medium text-ink" : "text-muted")}>
                {c.label}
              </span>
            ))}
            {matrix.rows.map((row) => (
              <div key={row.passageId} className="contents">
                <span
                  className={cn(
                    "flex size-5 items-center justify-center rounded-full text-[0.6875rem] font-semibold",
                    cited.includes(row.index) ? "bg-accent text-on-accent" : "bg-raised text-ink ring-1 ring-inset ring-line",
                  )}
                >
                  {row.index}
                </span>
                {row.cells.map((cell) => (
                  <span key={cell.key} className="min-w-0">
                    <span className="flex items-baseline justify-between gap-1 leading-4">
                      <span className={cell.inList ? "text-ink" : "text-faint"}>{cell.inList && cell.value !== null ? fmtScore(cell.value) : "–"}</span>
                      {cell.rank !== null ? <span className="text-faint">#{cell.rank}</span> : null}
                    </span>
                    <span className="relative mt-0.5 block h-1.5 rounded-[2px] bg-raised">
                      {cell.length !== null && cell.start !== null && cell.inList ? (
                        <span
                          className="absolute inset-y-0 block rounded-[2px]"
                          style={{
                            left: `${cell.start * 100}%`,
                            width: `max(2px, ${cell.length * 100}%)`,
                            background: chart.colors.emphasis(finalKey)(cell.key),
                          }}
                        />
                      ) : null}
                    </span>
                  </span>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
      <ChartCard.Data
        columns={[{ key: "passage", label: "Passage" }, ...matrix.columns.map((c) => ({ key: c.key, label: c.label, numeric: true }))]}
        rows={matrix.rows.map((row) => ({
          passage: String(row.index),
          ...Object.fromEntries(
            row.cells.map((cell) => [
              cell.key,
              cell.value === null
                ? "not returned"
                : `${fmtScore(cell.value)}${cell.rank !== null ? ` (#${cell.rank})` : ""}${cell.inList ? "" : " (not in this list)"}`,
            ]),
          ),
        }))}
      >
        <ChartCard.Note>{signed ? "Own scale per column; bars run from 0 either way" : "Own scale per column, from 0"}; – not in that list</ChartCard.Note>
        {retriever ? <ChartCard.Source>{retriever}</ChartCard.Source> : null}
      </ChartCard.Data>
    </ChartCard>
  );
}
