"use client";

import { createContext, use, useId, useMemo, type HTMLAttributes, type ReactNode } from "react";

import { Card } from "@/components/ui/card";
import { cn } from "@/components/ui/cn";
import { ChevronGlyph } from "@/components/ui/glyphs";

/* ---------- Legend ---------- */

export interface LegendItem {
  name: string;
  color: string;
  /**
   * How the series is drawn, so the swatch is the mark it explains. `dot` alone is a point with
   * its whisker; `dot`, `diamond`, `square` and `ring` with `joined` sit on the line that joins
   * the series' points.
   */
  mark: "line" | "dot" | "dash" | "bar" | "hatch" | "diamond" | "diamond-open" | "square" | "ring";
  /** draw the joining line behind the marker (series drawn as marks joined by a line) */
  joined?: boolean;
  /** the joining line is dashed */
  dashed?: boolean;
  /** a context series: thin and translucent, as drawn */
  faint?: boolean;
}

function Marker({ mark, color }: { mark: LegendItem["mark"]; color: string }) {
  if (mark === "diamond") return <path d="M8 1.5L12.5 6L8 10.5L3.5 6Z" fill={color} />;
  if (mark === "diamond-open") return <path d="M8 2L12 6L8 10L4 6Z" fill="var(--surface)" stroke={color} strokeWidth="1.5" />;
  if (mark === "square") return <rect x="4.25" y="2.25" width="7.5" height="7.5" rx="1" fill={color} />;
  if (mark === "ring") return <circle cx="8" cy="6" r="3.5" fill="var(--surface)" stroke={color} strokeWidth="1.75" />;
  return <circle cx="8" cy="6" r="4" fill={color} />;
}

function Swatch({ item }: { item: LegendItem }) {
  const { mark, color } = item;
  if (mark === "line" || mark === "dash") {
    return (
      <path
        d="M1 6h14"
        stroke={color}
        strokeWidth={item.faint ? 1.25 : 2}
        strokeOpacity={item.faint ? 0.6 : 1}
        strokeLinecap="round"
        strokeDasharray={mark === "dash" || item.dashed ? "4 3" : undefined}
      />
    );
  }
  if (mark === "bar") return <rect x="3" y="1" width="10" height="10" rx="2" fill={color} />;
  if (mark === "hatch") {
    return (
      <>
        <rect x="3" y="1" width="10" height="10" rx="2" fill="none" stroke={color} strokeWidth="1" />
        <path d="M3 8l4-4M6 11l7-7M10 11l3-3" stroke={color} strokeWidth="1" />
      </>
    );
  }
  if (item.joined) {
    return (
      <>
        <path d="M1 6h14" stroke={color} strokeWidth="1.5" strokeOpacity="0.7" strokeDasharray={item.dashed ? "3 2" : undefined} />
        <Marker mark={mark} color={color} />
      </>
    );
  }
  if (mark === "dot") {
    return (
      <>
        <path d="M8 2v8" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="8" cy="6" r="3.5" fill={color} />
      </>
    );
  }
  return <Marker mark={mark} color={color} />;
}

/** The swatch is the series' own mark, scaled down; one series needs no legend (the title names it). Text stays in ink. */
export function Legend({ items, className }: { items: readonly LegendItem[]; className?: string }) {
  if (items.length < 2) return null;
  return (
    <ul className={cn("flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted", className)}>
      {items.map((item) => (
        <li key={item.name} className="flex items-center gap-1.5">
          <svg aria-hidden="true" width="16" height="12" viewBox="0 0 16 12" className="shrink-0">
            <Swatch item={item} />
          </svg>
          <span className="text-ink">{item.name}</span>
        </li>
      ))}
    </ul>
  );
}

/* ---------- The card ---------- */

interface Ids {
  title: string;
  takeaway: string;
  toggle: string;
}
const IdsContext = createContext<Ids | null>(null);

function useIds(): Ids {
  const ids = use(IdsContext);
  if (!ids) throw new Error("ChartCard parts must be used inside <ChartCard>.");
  return ids;
}

/**
 * A chart as a card: a `<figure>` named by its title and described by its takeaway. Compose it:
 *
 *   <ChartCard>
 *     <ChartCard.Header>
 *       <ChartCard.Heading>
 *         <ChartCard.Title>Accuracy by evidence condition</ChartCard.Title>
 *         <ChartCard.Takeaway>{sentence built from the data}</ChartCard.Takeaway>
 *       </ChartCard.Heading>
 *       <ChartCard.Actions>{a Picker}</ChartCard.Actions>
 *       <ChartCard.Legend items={…} />
 *     </ChartCard.Header>
 *     <ChartCard.Body height={260}>{the chart}</ChartCard.Body>
 *     <ChartCard.Data columns={…} rows={…}>
 *       <ChartCard.Note>Whiskers: 95% bootstrap CI over questions</ChartCard.Note>
 *       <ChartCard.Source>results/pubmedqa_runs.json</ChartCard.Source>
 *     </ChartCard.Data>
 *   </ChartCard>
 *
 * A chart with no table (rare: every figure should have one) ends in `ChartCard.Footer` holding
 * the same `Note` and `Source`.
 */
function ChartCardRoot({ className, children, ...rest }: HTMLAttributes<HTMLElement> & { children: ReactNode }) {
  const id = useId();
  const ids = useMemo(() => ({ title: `${id}-title`, takeaway: `${id}-takeaway`, toggle: `${id}-toggle` }), [id]);
  return (
    <IdsContext value={ids}>
      <Card as="figure" aria-labelledby={ids.title} aria-describedby={ids.takeaway} className={cn("flex flex-col", className)} {...rest}>
        {children}
      </Card>
    </IdsContext>
  );
}

function ChartCardHeader({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-4 pt-4 sm:px-5", className)}>{children}</div>;
}

function ChartCardHeading({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("min-w-0 flex-[1_1_18rem]", className)}>{children}</div>;
}

/** The question the figure answers, or what it shows. A `<p>` by default; pass `as="h3"` inside a sectioned page. */
function ChartCardTitle({ as: Tag = "p", className, children }: { as?: "p" | "h2" | "h3" | "h4"; className?: string; children: ReactNode }) {
  const { title } = useIds();
  return (
    <Tag id={title} className={cn("text-[0.9375rem] font-semibold leading-6 text-ink", className)}>
      {children}
    </Tag>
  );
}

/**
 * The one-line reading of the figure, templated from the computed data ("Retrieval lifts accuracy
 * from 52.8% to 69.7%"), never typed in. It is the figure's accessible description.
 */
function ChartCardTakeaway({ className, children }: { className?: string; children: ReactNode }) {
  const { takeaway } = useIds();
  return (
    <p id={takeaway} className={cn("mt-1 max-w-[72ch] text-[0.8125rem] leading-[1.35rem] text-muted", className)}>
      {children}
    </p>
  );
}

function ChartCardActions({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("flex shrink-0 flex-wrap items-center gap-2", className)}>{children}</div>;
}

function ChartCardLegend({ items, className }: { items: readonly LegendItem[]; className?: string }) {
  return <Legend items={items} className={cn("w-full", className)} />;
}

/**
 * The plot area, at a fixed height so responsive nivo charts can measure it and the card never
 * jumps. `height` is the whole figure including axis labels.
 */
function ChartCardBody({ height, className, children }: { height?: number; className?: string; children: ReactNode }) {
  return (
    <div className={cn("min-w-0 flex-1 px-3 pb-3 pt-4 sm:px-4", className)}>
      <div className="relative min-w-0" style={height ? { height } : undefined}>
        {children}
      </div>
    </div>
  );
}

function ChartCardFooter({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("flex min-h-10 flex-wrap items-center gap-x-4 gap-y-1 border-t border-rule px-4 py-2 text-xs text-muted sm:px-5", className)}>{children}</div>;
}

/** What the marks and intervals are: "Whiskers: 95% bootstrap CI over questions · n = 1,000". */
function ChartCardNote({ className, children }: { className?: string; children: ReactNode }) {
  return <span className={cn("min-w-0 text-muted", className)}>{children}</span>;
}

/** The results file the numbers were read from. */
function ChartCardSource({ className, children }: { className?: string; children: ReactNode }) {
  return <span className={cn("ml-auto min-w-0 truncate font-mono text-faint", className)}>{children}</span>;
}

export interface DataColumn {
  key: string;
  label: string;
  numeric?: boolean;
}

/**
 * The figure's text alternative: its data as a table behind a "View data" toggle, as the card's
 * last row. Pass the rows already formatted (the same formatters the chart uses), so the table
 * says exactly what is drawn. Children sit on the right of the toggle row: put the card's
 * `Note` and `Source` there instead of a separate `Footer`, so the card ends in one row.
 */
function ChartCardData({
  columns,
  rows,
  caption,
  className,
  children,
}: {
  columns: readonly DataColumn[];
  rows: readonly Record<string, string>[];
  caption?: string;
  className?: string;
  children?: ReactNode;
}) {
  const { title, toggle } = useIds();
  if (rows.length === 0) return null;
  return (
    <details className={cn("group/data border-t border-rule", className)}>
      {/* Named by the toggle alone ("View data, 8 rows"), not by the note and source beside it. */}
      <summary aria-labelledby={toggle} className="flex min-h-11 cursor-pointer list-none flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2 text-xs text-muted sm:min-h-10 sm:px-5 [&::-webkit-details-marker]:hidden">
        <span id={toggle} className="flex shrink-0 items-center gap-1.5 whitespace-nowrap font-medium hover:text-ink">
          <ChevronGlyph size={14} className="-rotate-90 transition-transform duration-(--dur) ease-(--ease-out) group-open/data:rotate-0" />
          View data
          <span className="num font-normal text-faint">
            <span className="sr-only">, </span>
            {rows.length} {rows.length === 1 ? "row" : "rows"}
          </span>
        </span>
        {children}
      </summary>
      <div className="max-h-80 overflow-auto border-t border-rule px-4 sm:px-5" tabIndex={0} role="region" aria-labelledby={title}>
        <table className="num w-full text-[0.8125rem]">
          {caption ? <caption className="py-2 text-left text-xs text-muted">{caption}</caption> : null}
          <thead>
            <tr className="text-xs text-muted">
              {columns.map((column) => (
                <th key={column.key} scope="col" className={cn("sticky top-0 bg-surface py-2 pr-4 font-normal last:pr-0", column.numeric ? "text-right" : "text-left")}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-t border-rule">
                {columns.map((column) => (
                  <td key={column.key} className={cn("whitespace-nowrap py-1.5 pr-4 text-ink last:pr-0", column.numeric ? "text-right" : "text-left")}>
                    {row[column.key] ?? "–"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export const ChartCard = Object.assign(ChartCardRoot, {
  Header: ChartCardHeader,
  Heading: ChartCardHeading,
  Title: ChartCardTitle,
  Takeaway: ChartCardTakeaway,
  Actions: ChartCardActions,
  Legend: ChartCardLegend,
  Body: ChartCardBody,
  Footer: ChartCardFooter,
  Note: ChartCardNote,
  Source: ChartCardSource,
  Data: ChartCardData,
});
