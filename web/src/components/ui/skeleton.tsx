import { cn } from "./cn";

/**
 * A flat bar where a value will land, at the value's real size so nothing shifts when it does.
 * Hidden from assistive tech: the region it sits in should carry aria-busy instead.
 */
export function Skeleton({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn("block rounded-chip bg-active animate-[skeleton-pulse_1.6s_ease-in-out_infinite]", className)}
    />
  );
}

/** Lines of text still loading: full width, the last one shorter. */
export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <span aria-hidden="true" className={cn("flex flex-col gap-2", className)}>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} className={cn("h-3.5", i === lines - 1 ? "w-3/5" : "w-full")} />
      ))}
    </span>
  );
}
