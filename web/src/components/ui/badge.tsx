import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "./cn";

export type BadgeTone = "neutral" | "accent" | "supported" | "unsupported" | "contradicted" | "emergency" | "outline";

const TONE: Record<BadgeTone, string> = {
  neutral: "bg-active text-ink",
  accent: "bg-accent-soft text-accent-ink",
  supported: "bg-supported-bg text-supported-ink",
  unsupported: "bg-unsupported-bg text-unsupported-ink",
  contradicted: "bg-contradicted-bg text-contradicted-ink",
  emergency: "bg-emergency-bg text-emergency-ink",
  outline: "border border-line text-muted",
};

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  children: ReactNode;
}

/**
 * A short state beside a value or inside a row: a pale tint with dark text of the same hue.
 * Status colour always travels with a word (and usually a glyph), never alone. Never above a heading.
 */
export function Badge({ tone = "neutral", className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex min-h-[1.375rem] flex-none items-center gap-1 whitespace-nowrap rounded-chip px-2 text-xs font-medium leading-none",
        TONE[tone],
        className,
      )}
      {...rest}
    >
      {children}
    </span>
  );
}

/** A 8px dot for a live state (API reachable, stage running). Decorative: pair it with a word. */
export function StatusDot({ tone, className }: { tone: "ok" | "warn" | "down" | "idle" | "busy"; className?: string }) {
  const style = {
    ok: "bg-supported",
    warn: "bg-unsupported",
    down: "border-[1.5px] border-contradicted",
    idle: "border-[1.5px] border-control",
    busy: "bg-accent animate-[skeleton-pulse_1.4s_ease-in-out_infinite]",
  }[tone];
  return <span aria-hidden="true" className={cn("inline-block size-2 shrink-0 rounded-full", style, className)} />;
}
