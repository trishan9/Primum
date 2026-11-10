"use client";

import { useId, useState } from "react";

import { extent, linear, niceDomain, niceTicks, tickFormat } from "@/components/charts/scale";
import { useWidth } from "@/components/charts/useSize";
import { fmtInt, fmtMetric, fmtP } from "@/lib/format";
import { isProbabilityLike, metricDirection, metricLabel, type MetricTable, type Stat } from "@/lib/results";

interface MetricTableBlockProps {
  table: MetricTable;
  title: string;
  /** What a row is: "Retriever", "Estimator". */
  rowLabel: string;
  /** A row whose value is drawn as a dashed reference line, usually the baseline. */
  reference?: string;
}

const ROW = 46;
const PAD = 20;

/** A count (n_failed, pool_size, triage.urgent) is an exact integer with no interval, not an estimate. */
function isCount(key: string, cell: Stat): boolean {
  if (cell.lo !== null || !Number.isInteger(cell.value)) return false;
  return Math.abs(cell.value) > 1 || /^n(_|$)|count|triage\.|pool_size/i.test(key);
}

/**
 * A results table laid out as a forest plot: one row per system, every metric in the file with
 * its interval, and beside them the chosen metric drawn as a point and whisker on a shared axis.
 * This is the form the clinical literature uses to compare effects, used here to compare systems.
 */
export function MetricTableBlock({ table, title, rowLabel, reference }: MetricTableBlockProps) {
  const id = useId();
  const [plotRef, plotW] = useWidth<HTMLDivElement>();
  const plottable = table.metrics.filter((m) => !isProbabilityLike(m));
  // Plot a metric with a known good direction first, not a count such as `n`.
  const defaultMetric = plottable.find((m) => metricDirection(m) !== null) ?? plottable[0] ?? "";
  const [selected, setMetric] = useState(defaultMetric);
  // The table can be re-read with other columns; fall back rather than plot a column that is gone.
  const metric = plottable.includes(selected) ? selected : defaultMetric;

  const stats = table.rows.map((row) => row.cells[metric]).filter((s): s is Stat => s !== undefined);
  const direction = metricDirection(metric);
  const hasIntervals = stats.some((s) => s.lo !== null);

  const values = stats.flatMap((s) => [s.value, s.lo ?? s.value, s.hi ?? s.value]);
  const [lo, hi] = extent(values);
  const pad = (hi - lo) * 0.1 || Math.abs(hi) * 0.05 || 0.05;
  const domain = niceDomain(lo - pad, hi + pad, 4);
  const sx = linear(domain, [PAD, Math.max(PAD + 1, plotW - PAD)]);
  const ticks = niceTicks(domain[0], domain[1], plotW < 260 ? 3 : 5);
  const fmt = tickFormat(ticks);

  let best: string | null = null;
  if (direction && stats.length > 1) {
    let bestValue = direction === "higher" ? -Infinity : Infinity;
    for (const row of table.rows) {
      const stat = row.cells[metric];
      if (!stat) continue;
      if (direction === "higher" ? stat.value > bestValue : stat.value < bestValue) {
        bestValue = stat.value;
        best = row.name;
      }
    }
  }
  const referenceStat = reference ? table.rows.find((r) => r.name === reference)?.cells[metric] : undefined;

  return (
    <section aria-labelledby={`${id}-title`} className="min-w-0 rounded-card border border-line bg-surface p-4 sm:p-5">
      <h3 id={`${id}-title`} className="text-[0.9375rem] font-semibold leading-snug text-ink">
        {title}
      </h3>
      <p className="mt-1 max-w-[82ch] text-sm leading-relaxed text-muted">
        {metric ? (
          <>
            The plot shows {metricLabel(metric)}: a point for each estimate
            {hasIntervals ? " and a bar for its reported interval" : "; the file reports no intervals"}.{" "}
            {direction ? `${direction === "higher" ? "Higher" : "Lower"} is better, and the best row is marked in blue. ` : ""}
            {referenceStat ? `The dashed line marks ${reference}. ` : ""}
            {plottable.length > 1 ? "Select another column heading to plot that metric." : ""}
          </>
        ) : (
          "The file reports no metric that can be plotted."
        )}
      </p>

      <div className="mt-3 relative overflow-x-auto" tabIndex={0} role="group" aria-labelledby={`${id}-title`}>
        <table className="num w-full text-sm" style={{ minWidth: 160 + 240 + table.metrics.length * 104 }}>
          <thead>
            <tr className="align-bottom text-[0.8125rem] text-muted">
              <th scope="col" className="sticky left-0 z-10 w-40 bg-surface pb-2 pr-3 text-left font-normal">
                {rowLabel}
              </th>
              <th scope="col" className="pb-2 text-left font-normal">
                <div ref={plotRef} className="w-full min-w-60 px-5">
                  {metric ? metricLabel(metric) : ""}
                </div>
              </th>
              {table.metrics.map((m) => (
                <th key={m} scope="col" className="w-26 whitespace-nowrap pb-1 pl-3 text-right font-normal">
                  {plottable.includes(m) ? (
                    <button
                      type="button"
                      aria-pressed={m === metric}
                      onClick={() => setMetric(m)}
                      className={`-mr-1 min-h-11 rounded-control px-1 underline-offset-4 sm:min-h-7 ${
                        m === metric ? "font-semibold text-ink underline decoration-2" : "underline decoration-control hover:text-ink hover:decoration-ink"
                      }`}
                    >
                      {metricLabel(m)}
                    </button>
                  ) : (
                    metricLabel(m)
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row) => {
              const stat = row.cells[metric];
              const isBest = row.name === best;
              const color = isBest ? "var(--accent)" : "var(--text)";
              return (
                <tr key={row.name} className="border-t border-rule">
                  <th scope="row" className={`sticky left-0 z-10 max-w-40 truncate bg-surface pr-3 text-left text-ink ${isBest ? "font-semibold" : "font-medium"}`} title={row.name}>
                    {row.name}
                  </th>
                  <td className="p-0">
                    {plotW > 0 ? (
                      <svg width={plotW} height={ROW} aria-hidden="true" className="block">
                        {ticks.map((tick) => (
                          <line key={tick} x1={sx(tick)} x2={sx(tick)} y1={0} y2={ROW} stroke="var(--grid)" strokeWidth="1" />
                        ))}
                        {referenceStat ? (
                          <line x1={sx(referenceStat.value)} x2={sx(referenceStat.value)} y1={0} y2={ROW} stroke="var(--text)" strokeWidth="1" strokeDasharray="4 4" />
                        ) : null}
                        {stat ? (
                          <>
                            {stat.lo !== null && stat.hi !== null ? (
                              <line x1={sx(stat.lo)} x2={sx(stat.hi)} y1={ROW / 2} y2={ROW / 2} stroke={color} strokeWidth="1.5" strokeLinecap="round" />
                            ) : null}
                            <circle cx={sx(stat.value)} cy={ROW / 2} r="4.5" fill={color} stroke="var(--surface)" strokeWidth="2" />
                          </>
                        ) : null}
                      </svg>
                    ) : (
                      <div style={{ height: ROW }} />
                    )}
                  </td>
                  {table.metrics.map((m) => {
                    const cell = row.cells[m];
                    return (
                      <td key={m} className="whitespace-nowrap pl-3 text-right text-ink">
                        {cell ? (
                          <>
                            <span className={m === metric && isBest ? "font-semibold" : ""}>
                              {isProbabilityLike(m) ? fmtP(cell.value) : isCount(m, cell) ? fmtInt(cell.value) : fmtMetric(cell.value)}
                            </span>
                            {cell.lo !== null && cell.hi !== null ? (
                              <span className="block text-[0.8125rem] leading-tight text-muted">
                                {fmtMetric(cell.lo)} to {fmtMetric(cell.hi)}
                              </span>
                            ) : null}
                          </>
                        ) : (
                          <span className="text-muted">–</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr className="border-t border-line">
              <td />
              <td className="p-0">
                {plotW > 0 && metric ? (
                  <svg width={plotW} height={22} aria-hidden="true" className="block">
                    {ticks.map((tick) => (
                      <text key={tick} x={sx(tick)} y={16} textAnchor="middle" className="num fill-(--muted) text-[12px]">
                        {fmt(tick)}
                      </text>
                    ))}
                  </svg>
                ) : null}
              </td>
              <td colSpan={table.metrics.length} />
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}
