"use client";

import { inkOn } from "@/components/charts/colors";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { useChartTheme } from "@/components/charts/theme";
import { fmtProb } from "@/lib/format";
import type { Overlap } from "@/lib/charts/retrieval";

/**
 * How many of their top k each pair of retrievers shares: a lower-triangle heatmap built as a real
 * table, so a screen reader reads it by row and column headers. One hue, pale to dark, for 0 to k
 * shared; the count is printed in every cell, so the colour never carries the number alone. The
 * diagonal (a list against itself) and the mirrored upper half are left out.
 */
export function OverlapMatrix({ pairs, columns, k }: { pairs: readonly Overlap[]; columns: readonly string[]; k: number }) {
  const kit = useChartTheme();
  if (!kit) return <ChartSkeleton height={44 * columns.length + 60} />;
  const fill = kit.colors.sequential([0, k]);
  const lookup = new Map(pairs.map((p) => [`${p.a}\u0000${p.b}`, p]));
  const rows = columns.slice(1);
  const cols = columns.slice(0, -1);

  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto overscroll-x-contain">
        <table className="num mx-auto border-separate border-spacing-[3px] text-[0.8125rem]">
          <thead>
            <tr>
              <td />
              {cols.map((c) => (
                <th key={c} scope="col" className="h-28 w-11 px-0 pb-1 align-bottom font-normal">
                  <span className="inline-block max-h-28 truncate font-mono text-xs text-ink [writing-mode:vertical-rl] rotate-180">{c}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={r}>
                <th scope="row" className="max-w-36 truncate pr-2 text-right font-mono text-xs font-normal text-ink">
                  {r}
                </th>
                {cols.map((c, ci) => {
                  if (ci > ri) return <td key={c} aria-hidden="true" />;
                  const pair = lookup.get(`${c}\u0000${r}`) ?? lookup.get(`${r}\u0000${c}`);
                  if (!pair) return <td key={c} />;
                  const bg = fill(pair.shared);
                  return (
                    <td
                      key={c}
                      title={`${pair.a} and ${pair.b} share ${pair.shared} of their top ${k} (Jaccard ${fmtProb(pair.jaccard)})`}
                      className="h-10 w-11 rounded-[4px] text-center font-medium"
                      style={{ background: bg, color: inkOn(kit.tokens, bg), boxShadow: `inset 0 0 0 1px ${kit.tokens.rule}` }}
                    >
                      {pair.shared}
                      <span className="sr-only"> of {k}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-center gap-2 text-xs text-muted" aria-hidden="true">
        <span className="num">0</span>
        <span className="h-2 w-28 rounded-full" style={{ background: `linear-gradient(to right, ${fill(0)}, ${fill(k)})` }} />
        <span className="num">{k} shared</span>
      </div>
    </div>
  );
}
