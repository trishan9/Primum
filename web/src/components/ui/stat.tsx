import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "./cn";

const COLUMNS = {
  2: "sm:grid-cols-2",
  3: "sm:grid-cols-3",
  4: "sm:grid-cols-2 lg:grid-cols-4",
  5: "sm:grid-cols-3 lg:grid-cols-5",
  6: "sm:grid-cols-3 lg:grid-cols-6",
} as const;

/* Below 480px a cell is a row: label and note on the left, the value right-aligned beside them. */
const NARROW_ROW = "max-[30rem]:grid max-[30rem]:grid-cols-[minmax(0,1fr)_auto] max-[30rem]:gap-x-4";
const NARROW_VALUE =
  "max-[30rem]:col-start-2 max-[30rem]:row-span-3 max-[30rem]:row-start-1 max-[30rem]:mt-0 max-[30rem]:self-center max-[30rem]:text-right";

/**
 * One container split by hairlines into equal cells: the way a set of headline figures is shown.
 * Choose `columns` from the number of stats so no cell is orphaned. Between 480px and `sm` it is two
 * columns, an odd last cell taking the full width; below 480px each stat is one row.
 */
export function StatGroup({
  columns = 4,
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLDListElement> & { columns?: keyof typeof COLUMNS }) {
  return (
    <dl
      className={cn(
        "grid grid-cols-1 gap-px overflow-hidden rounded-card border border-line bg-line min-[30rem]:grid-cols-2",
        "min-[30rem]:[&>*:last-child:nth-child(odd)]:col-span-2 sm:[&>*:last-child:nth-child(odd)]:col-span-1",
        COLUMNS[columns],
        className,
      )}
      {...rest}
    >
      {children}
    </dl>
  );
}

/**
 *   <Stat>
 *     <Stat.Label>Coverage</Stat.Label>
 *     <Stat.Value unit="%">{fmt(value)}</Stat.Value>
 *     <Stat.Note>at α = 0.10, n = …</Stat.Note>
 *   </Stat>
 *
 * Inside a StatGroup it is a cell; alone it sits bare on a card. The value always comes from data.
 */
function StatRoot({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("flex min-w-0 flex-col bg-surface px-4 py-3.5 sm:px-5", NARROW_ROW, className)} {...rest}>
      {children}
    </div>
  );
}

function StatLabel({ className, children }: { className?: string; children: ReactNode }) {
  return <dt className={cn("text-[0.8125rem] font-medium text-muted", className)}>{children}</dt>;
}

function StatValue({
  unit,
  lead = false,
  className,
  children,
}: {
  /** A smaller grey suffix: "ms", "%", "of 1,000". */
  unit?: ReactNode;
  /** The screen's answer figure: one size up. Use once per screen. */
  lead?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <dd
      className={cn(
        "num mt-1 font-normal tracking-[-0.01em] text-ink",
        lead ? "text-[2rem] leading-10 lg:text-[2.5rem] lg:leading-[3rem]" : "text-kpi",
        NARROW_VALUE,
        className,
      )}
    >
      {children}
      {unit ? <span className="ml-1 text-sm font-normal tracking-normal text-muted">{unit}</span> : null}
    </dd>
  );
}

function StatNote({ className, children }: { className?: string; children: ReactNode }) {
  return <dd className={cn("mt-1 text-xs leading-4 text-muted", className)}>{children}</dd>;
}

export const Stat = Object.assign(StatRoot, { Label: StatLabel, Value: StatValue, Note: StatNote });

export { StatLabel, StatNote, StatValue };
