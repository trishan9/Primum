"use client";

import { scaleLinear } from "d3-scale";
import { useId, useMemo, useState, type ReactNode } from "react";

import { useWidth } from "@/components/charts/useSize";
import { Card } from "@/components/ui/card";
import { cn } from "@/components/ui/cn";
import { Disclosure } from "@/components/ui/disclosure";
import { Picker } from "@/components/ui/picker";
import { ScrollArea } from "@/components/ui/scroll-area";
import { clearLeader, constantColumns } from "@/lib/charts/common";
import { fmtInt, fmtMetric, fmtP } from "@/lib/format";
import { isProbabilityLike, metricDirection, metricLabel, type MetricTable, type Stat } from "@/lib/results";

interface MetricTableBlockProps {
  table: MetricTable;
  title: string;
  /** What a row is: "Retriever", "Estimator". */
  rowLabel: string;
  /**
   * The reference row (the baseline, the stated confidence): its mark is the accent, its label
   * medium weight, and its value is drawn as a dashed line on every row. Never the "best" row.
   */
  reference?: string;
  /** A control on the right of the header, e.g. a switch between variants of the table. */
  actions?: ReactNode;
  /** Where the numbers come from, printed in the footer. */
  source?: ReactNode;
  /** Draw the chosen column as a point and interval beside the numbers (default true). */
  plot?: boolean;
  /** The column to plot first, when the table has it. */
  defaultMetric?: string;
  /**
   * `card` (default) stands on its own; `disclosure` is the "All metrics" table under the chart
   * it backs, closed until asked for.
   */
  variant?: "card" | "disclosure";
}

const ROW = 40;
const PAD = 14;
const RIGHT_PAD = 26;

/** A count (n_failed, pool_size, triage.urgent) is an exact integer with no interval, not an estimate. */
function isCount(key: string, cell: Stat): boolean {
  if (cell.lo !== null || !Number.isInteger(cell.value)) return false;
  return Math.abs(cell.value) > 1 || /^n(_|$)|count|triage\.|pool_size/i.test(key);
}

/** Column names for the file's keys: `ndcg@10_diff` reads "nDCG@10 diff", `ndcg@10_p_holm` reads "nDCG@10 p, Holm". */
function columnLabel(key: string): string {
  const diff = key.match(/^(.+?)_diff$/);
  if (diff) return `${metricLabel(diff[1] as string)} diff`;
  const p = key.match(/^(.+?)_p(_holm)?$/);
  if (p) return p[2] ? `${metricLabel(p[1] as string)} p, Holm` : `${metricLabel(p[1] as string)} p`;
  return metricLabel(key);
}

function formatCell(key: string, cell: Stat): string {
  return isProbabilityLike(key) ? fmtP(cell.value) : isCount(key, cell) ? fmtInt(cell.value) : fmtMetric(cell.value);
}

/** The x scale of the plotted column: padded around the data and every interval, then niced. */
function useAxis(stats: readonly Stat[], width: number) {
  return useMemo(() => {
    const values = stats.flatMap((s) => [s.value, s.lo ?? s.value, s.hi ?? s.value]);
    const lo = values.length ? Math.min(...values) : 0;
    const hi = values.length ? Math.max(...values) : 1;
    const pad = (hi - lo) * 0.1 || Math.abs(hi) * 0.05 || 0.05;
    const x = scaleLinear()
      .domain([lo - pad, hi + pad])
      .range([PAD, Math.max(PAD + 1, width - RIGHT_PAD)])
      .nice(4);
    const count = width < 260 ? 3 : 5;
    return { x, ticks: x.ticks(count), format: x.tickFormat(count) };
  }, [stats, width]);
}

type Axis = ReturnType<typeof useAxis>;

/** One row's point and interval on the shared axis, with zero and the reference as lines. */
function RowPlot({ axis, width, stat, isReference, referenceValue }: { axis: Axis; width: number; stat: Stat | undefined; isReference: boolean; referenceValue: number | null }) {
  const { x, ticks } = axis;
  const [d0, d1] = x.domain() as [number, number];
  const color = isReference ? "var(--accent)" : "var(--text)";
  return (
    <svg width={width} height={ROW} aria-hidden="true" className="block">
      {ticks.map((tick) => (
        <line key={tick} x1={x(tick)} x2={x(tick)} y1={0} y2={ROW} stroke="var(--grid)" strokeWidth="1" />
      ))}
      {d0 < 0 && d1 > 0 ? <line x1={x(0)} x2={x(0)} y1={0} y2={ROW} stroke="var(--muted)" strokeWidth="1" /> : null}
      {referenceValue !== null ? <line x1={x(referenceValue)} x2={x(referenceValue)} y1={0} y2={ROW} stroke="var(--muted)" strokeWidth="1" strokeDasharray="4 3" /> : null}
      {stat ? (
        <>
          {stat.lo !== null && stat.hi !== null ? (
            <line x1={x(stat.lo)} x2={x(stat.hi)} y1={ROW / 2} y2={ROW / 2} stroke={color} strokeWidth={1.5} strokeLinecap="round" opacity={isReference ? 1 : 0.6} />
          ) : null}
          <circle cx={x(stat.value)} cy={ROW / 2} r={4.5} fill={color} stroke="var(--surface)" strokeWidth="2" paintOrder="stroke" />
        </>
      ) : null}
    </svg>
  );
}

/** The tick labels under the plot column; the end ticks are anchored inward so they never clip. */
function AxisTicks({ axis, width }: { axis: Axis; width: number }) {
  const { x, ticks, format } = axis;
  return (
    <svg width={width} height={24} aria-hidden="true" className="block">
      {ticks.map((tick, i) => {
        const px = x(tick);
        const anchor = i === ticks.length - 1 && px > width - 24 ? "end" : i === 0 && px < 24 ? "start" : "middle";
        return (
          <text key={tick} x={px} y={16} textAnchor={anchor} className="num fill-(--muted) text-[12px]">
            {format(tick)}
          </text>
        );
      })}
    </svg>
  );
}

/**
 * A results table: one row per system, every metric in the file with its interval and, unless
 * `plot` is off, the chosen column drawn as a point and interval on a shared axis (a forest plot,
 * the form the clinical literature uses to compare effects). No row is crowned: a row is marked
 * "highest mean" only when its interval clears the runner-up's, and then in ink, never the accent.
 * Columns that are the same in every row are dropped and named in the footer. Below 640px the
 * table becomes a list: each name on its own line, the plot full width, the value beside it.
 */
export function MetricTableBlock({ table: raw, title, rowLabel, reference, actions, source, plot = true, defaultMetric: preferred, variant = "card" }: MetricTableBlockProps) {
  const id = useId();
  const [plotRef, plotW] = useWidth<HTMLDivElement>();
  const [listRef, listW] = useWidth<HTMLUListElement>();

  const { table, dropped } = useMemo(() => {
    const constant = constantColumns(raw);
    return { table: { ...raw, metrics: raw.metrics.filter((m) => !constant.includes(m)) }, dropped: constant };
  }, [raw]);

  const plottable = table.metrics.filter((m) => !isProbabilityLike(m));
  // Plot the asked-for column, else one with a known good direction, not a count such as `n`.
  const defaultMetric = (preferred && plottable.includes(preferred) ? preferred : null) ?? plottable.find((m) => metricDirection(m) !== null) ?? plottable[0] ?? "";
  const [selected, setMetric] = useState(defaultMetric);
  // The table can be re-read with other columns (another variant); fall back rather than plot a column that is gone.
  const metric = plottable.includes(selected) ? selected : defaultMetric;
  const showPlot = plot && metric !== "";

  const stats = useMemo(() => table.rows.map((row) => row.cells[metric]).filter((s): s is Stat => s !== undefined), [table, metric]);
  const direction = metricDirection(metric);
  const hasIntervals = stats.some((s) => s.lo !== null);
  const tableAxis = useAxis(stats, plotW);
  const listAxis = useAxis(stats, Math.max(0, listW - 112));
  const [d0, d1] = tableAxis.x.domain() as [number, number];

  const leader = useMemo(() => {
    if (!direction) return null;
    const rows = table.rows.flatMap((r) => (r.cells[metric] ? [{ name: r.name, estimate: r.cells[metric] as Stat }] : []));
    return clearLeader(rows, direction)?.name ?? null;
  }, [table, metric, direction]);
  const referenceStat = reference ? table.rows.find((r) => r.name === reference)?.cells[metric] : undefined;
  const referenceValue = referenceStat?.value ?? null;

  const summary = (
    <p className="mt-1 max-w-[76ch] text-[0.8125rem] leading-[1.35rem] text-muted">
      {!metric ? (
        "The file reports no metric that can be plotted."
      ) : showPlot ? (
        <>
          Plotting <span className="font-medium text-ink">{columnLabel(metric)}</span>
          {hasIntervals ? " with its reported interval" : "; the file reports no intervals"}.{" "}
          {direction ? `${direction === "higher" ? "Higher" : "Lower"} is better. ` : ""}
          {leader ? `${leader} is marked: its interval clears the next row's. ` : direction && stats.length > 1 && hasIntervals ? "No row's interval clears the next one's, so none is marked. " : ""}
          {referenceStat ? `The accent row and dashed line are ${reference}. ` : ""}
          {d0 < 0 && d1 > 0 ? "The solid line is no difference. " : ""}
          {plottable.length > 1 ? "Pick a column to plot it." : ""}
        </>
      ) : (
        <>Every column the file reports, with its interval{referenceStat ? `; ${reference} is the reference row` : ""}.</>
      )}
    </p>
  );

  const nameCell = (name: string) => {
    const isRef = name === reference;
    return (
      <span className="flex min-w-0 items-center gap-2">
        <span className={cn("min-w-0 break-words", isRef && "font-medium")} title={name}>
          {name}
        </span>
        {name === leader ? <span className="shrink-0 rounded-chip border border-line px-1.5 py-px text-[0.6875rem] text-ink">highest mean</span> : null}
      </span>
    );
  };

  const body = (
    <>
      {/* ≥ 640px: the table, scrolling sideways when the columns do not fit. */}
      <ScrollArea labelledBy={`${id}-title`} className="mt-3 hidden sm:block">
        <table
          className="num w-full border-collapse text-sm [--name-w:13rem]"
          style={{ minWidth: `calc(var(--name-w) + ${showPlot ? 240 : 0}px + ${table.metrics.length * 104}px)` }}
        >
          <thead>
            <tr className="align-bottom text-xs text-muted">
              <th scope="col" className="sticky left-0 z-10 w-(--name-w) bg-surface pb-2 pl-5 pr-3 text-left font-normal">
                {rowLabel}
              </th>
              {showPlot ? (
                <th scope="col" className="pb-2 text-left font-normal">
                  <div ref={plotRef} className="w-full min-w-60 px-4">
                    {columnLabel(metric)}
                  </div>
                </th>
              ) : null}
              {table.metrics.map((m) => (
                <th key={m} scope="col" className="w-26 whitespace-nowrap pb-1 pl-2 text-right font-normal last:pr-4">
                  {showPlot && plottable.includes(m) ? (
                    <button
                      type="button"
                      aria-pressed={m === metric}
                      onClick={() => setMetric(m)}
                      className={cn(
                        "min-h-7 rounded-chip px-2 transition-colors duration-(--dur-fast)",
                        m === metric ? "bg-accent-soft font-medium text-accent-ink" : "text-muted hover:bg-raised hover:text-ink",
                      )}
                    >
                      {columnLabel(m)}
                    </button>
                  ) : (
                    <span className="px-2">{columnLabel(m)}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row) => (
              <tr key={row.name} className="group/row border-t border-rule">
                <th
                  scope="row"
                  className="sticky left-0 z-10 max-w-(--name-w) bg-surface py-1.5 pl-5 pr-3 text-left font-normal text-ink transition-colors duration-(--dur-fast) group-hover/row:bg-raised"
                >
                  {nameCell(row.name)}
                </th>
                {showPlot ? (
                  <td className="p-0 transition-colors duration-(--dur-fast) group-hover/row:bg-raised">
                    {plotW > 0 ? (
                      <RowPlot axis={tableAxis} width={plotW} stat={row.cells[metric]} isReference={row.name === reference} referenceValue={referenceValue} />
                    ) : (
                      <div style={{ height: ROW }} />
                    )}
                  </td>
                ) : null}
                {table.metrics.map((m) => {
                  const cell = row.cells[m];
                  return (
                    <td
                      key={m}
                      className={cn(
                        "whitespace-nowrap py-1.5 pl-2 pr-2 text-right text-ink transition-colors duration-(--dur-fast) group-hover/row:bg-raised last:pr-6",
                        showPlot && m === metric && "bg-raised/50",
                      )}
                    >
                      {cell ? (
                        <>
                          <span>{formatCell(m, cell)}</span>
                          {cell.lo !== null && cell.hi !== null ? (
                            <span className="block text-xs leading-tight text-muted">
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
            ))}
          </tbody>
          {showPlot ? (
            <tfoot>
              <tr className="border-t border-line">
                <td className="sticky left-0 bg-surface" />
                <td className="p-0">{plotW > 0 ? <AxisTicks axis={tableAxis} width={plotW} /> : null}</td>
                <td colSpan={table.metrics.length} />
              </tr>
            </tfoot>
          ) : null}
        </table>
      </ScrollArea>

      {/* < 640px: a list. The name gets its own line; below it the plot and the value. */}
      <div className="mt-3 sm:hidden">
        {showPlot && plottable.length > 1 ? (
          <div className="px-4 pb-2">
            <Picker legend="Column to plot" showLabel value={metric} options={plottable.map((m) => ({ value: m, label: columnLabel(m) }))} onChange={setMetric} segmentedFrom="never" />
          </div>
        ) : null}
        <ul ref={listRef} className="num border-t border-rule px-4 text-sm">
          {table.rows.map((row) => {
            const stat = row.cells[metric];
            return (
              <li key={row.name} className="border-b border-rule py-2 last:border-b-0">
                <div className="text-ink">{nameCell(row.name)}</div>
                {showPlot ? (
                  <div className="mt-0.5 flex items-center gap-3">
                    <div className="min-w-0 flex-1">
                      {listW > 0 ? <RowPlot axis={listAxis} width={Math.max(0, listW - 112)} stat={stat} isReference={row.name === reference} referenceValue={referenceValue} /> : null}
                    </div>
                    <div className="w-25 shrink-0 text-right">
                      {stat ? (
                        <>
                          <span className="text-ink">{formatCell(metric, stat)}</span>
                          {stat.lo !== null && stat.hi !== null ? (
                            <span className="block text-xs leading-tight text-muted">
                              {fmtMetric(stat.lo)} to {fmtMetric(stat.hi)}
                            </span>
                          ) : null}
                        </>
                      ) : (
                        <span className="text-muted">–</span>
                      )}
                    </div>
                  </div>
                ) : (
                  <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                    {table.metrics.map((m) => {
                      const cell = row.cells[m];
                      return (
                        <div key={m} className="flex min-w-0 justify-between gap-2">
                          <dt className="truncate text-muted">{columnLabel(m)}</dt>
                          <dd className="text-ink">{cell ? formatCell(m, cell) : "–"}</dd>
                        </div>
                      );
                    })}
                  </dl>
                )}
              </li>
            );
          })}
        </ul>
        {showPlot && listW > 0 ? (
          <div className="px-4">
            <AxisTicks axis={listAxis} width={Math.max(0, listW - 112)} />
          </div>
        ) : null}
      </div>

      <footer className="flex min-h-10 flex-wrap items-center gap-x-3 gap-y-1 border-t border-rule px-4 py-2 text-xs text-faint sm:px-5">
        <span className="num">
          {fmtInt(table.rows.length)} {rowLabel.toLowerCase()}
          {table.rows.length === 1 ? "" : "s"}, {fmtInt(table.metrics.length)} column{table.metrics.length === 1 ? "" : "s"}
        </span>
        {dropped.length > 0 ? (
          <span className="min-w-0 text-muted">
            Not shown, the same in every row: {dropped.map((m) => `${columnLabel(m)} = ${formatCell(m, raw.rows[0]!.cells[m]!)}`).join(", ")}
          </span>
        ) : null}
        {source ? <span className="ml-auto min-w-0 truncate font-mono">{source}</span> : null}
      </footer>
    </>
  );

  if (variant === "disclosure") {
    return (
      <Disclosure variant="framed">
        <Disclosure.Summary meta={`${fmtInt(table.rows.length)} rows`}>
          <span id={`${id}-title`}>All metrics: {title.charAt(0).toLowerCase() + title.slice(1)}</span>
        </Disclosure.Summary>
        <div className="border-t border-rule">
          {actions || summary ? (
            <div className="flex flex-col gap-x-4 gap-y-2 px-4 pt-3 sm:flex-row sm:items-start sm:justify-between sm:px-5">
              <div className="min-w-0 flex-1">{summary}</div>
              {actions ? <div className="flex min-w-0 max-w-full flex-wrap items-center gap-2">{actions}</div> : null}
            </div>
          ) : null}
          {body}
        </div>
      </Disclosure>
    );
  }

  return (
    <Card aria-labelledby={`${id}-title`}>
      <header className="flex flex-col gap-x-4 gap-y-3 px-4 pt-4 sm:flex-row sm:items-start sm:justify-between sm:px-5">
        <div className="min-w-0 flex-1">
          <h3 id={`${id}-title`} className="text-[0.9375rem] font-semibold leading-6 text-ink">
            {title}
          </h3>
          {summary}
        </div>
        {actions ? <div className="flex min-w-0 max-w-full flex-wrap items-center gap-2">{actions}</div> : null}
      </header>
      {body}
    </Card>
  );
}
