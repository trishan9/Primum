import { cn } from "@/components/ui/cn";

/**
 * Where a chart will be, at the chart's real height so nothing shifts when it arrives: a flat
 * well with three faint gridlines. Use it as the `loading` of a `next/dynamic` chart and while
 * `useChartTheme()` is still null. Pass `height="100%"` to fill a `ChartCard.Body` that already
 * has the chart's height, so a lazy chart's skeleton never guesses a row count. Announces itself
 * once to screen readers.
 */
export function ChartSkeleton({ height = 260, label = "Loading chart", className }: { height?: number | string; label?: string; className?: string }) {
  return (
    <div role="status" className={cn("relative overflow-hidden rounded-control bg-raised", className)} style={{ height }}>
      <span className="sr-only">{label}</span>
      <span aria-hidden="true" className="absolute inset-0 flex animate-[skeleton-pulse_1.6s_ease-in-out_infinite] flex-col justify-evenly px-4">
        <span className="h-px bg-(--grid)" />
        <span className="h-px bg-(--grid)" />
        <span className="h-px bg-(--grid)" />
      </span>
    </div>
  );
}
