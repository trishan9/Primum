import type { ReactNode } from "react";

import { ChevronGlyph } from "@/components/ui/glyphs";

export interface LegendItem {
  name: string;
  color: string;
  /** How the series is drawn, so the swatch is the mark it explains. */
  mark: "line" | "dot" | "dash";
}

export function Legend({ items }: { items: LegendItem[] }) {
  if (items.length < 2) return null;
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[0.8125rem] text-muted">
      {items.map((item) => (
        <li key={item.name} className="flex items-center gap-1.5">
          <svg aria-hidden="true" width="16" height="10" viewBox="0 0 16 10" className="shrink-0">
            {item.mark === "dot" ? (
              <circle cx="8" cy="5" r="4" fill={item.color} />
            ) : (
              <path
                d="M0 5h16"
                stroke={item.color}
                strokeWidth="2"
                strokeLinecap="round"
                strokeDasharray={item.mark === "dash" ? "4 3" : undefined}
              />
            )}
          </svg>
          <span>{item.name}</span>
        </li>
      ))}
    </ul>
  );
}

export interface TableColumn {
  key: string;
  label: string;
  numeric?: boolean;
}

interface ChartFrameProps {
  /** The question this figure answers, in words. */
  title: string;
  note?: ReactNode;
  legend?: LegendItem[];
  children: ReactNode;
  /** The same data as a table: the accessible twin of the figure. */
  table?: { columns: TableColumn[]; rows: Record<string, string>[] };
  className?: string;
}

export function ChartFrame({ title, note, legend, children, table, className }: ChartFrameProps) {
  return (
    <figure className={`min-w-0 rounded-card border border-line bg-surface p-4 sm:p-5 ${className ?? ""}`}>
      <figcaption>
        <p className="text-[0.9375rem] font-semibold leading-snug text-ink">{title}</p>
        {note ? <div className="mt-1 max-w-[70ch] text-sm leading-relaxed text-muted">{note}</div> : null}
        {legend && legend.length > 1 ? (
          <div className="mt-2.5">
            <Legend items={legend} />
          </div>
        ) : null}
      </figcaption>
      <div className="mt-3">{children}</div>
      {table && table.rows.length > 0 ? (
        <details className="group mt-3 border-t border-rule pt-2">
          <summary className="flex min-h-11 items-center gap-1.5 text-[0.8125rem] font-medium text-ink sm:min-h-8">
            <ChevronGlyph size={14} className="-rotate-90 text-muted transition-transform duration-150 group-open:rotate-0" />
            Data table ({table.rows.length} {table.rows.length === 1 ? "row" : "rows"})
          </summary>
          <div className="mt-1 max-h-80 relative overflow-auto" tabIndex={0} role="group" aria-label={`${title}: data table`}>
            <table className="num w-full text-[0.8125rem]">
              <thead>
                <tr className="text-xs text-muted">
                  {table.columns.map((column) => (
                    <th
                      key={column.key}
                      scope="col"
                      className={`sticky top-0 bg-surface py-1.5 pr-4 font-normal last:pr-0 ${column.numeric ? "text-right" : "text-left"}`}
                    >
                      {column.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, i) => (
                  <tr key={i} className="border-t border-rule">
                    {table.columns.map((column) => (
                      <td
                        key={column.key}
                        className={`whitespace-nowrap py-1.5 pr-4 text-ink last:pr-0 ${column.numeric ? "text-right" : "text-left"}`}
                      >
                        {row[column.key] ?? "–"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </figure>
  );
}
