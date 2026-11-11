import type { ReactNode } from "react";

/** Inline code: a shell command or identifier, the one place monospace belongs.
 *  `wrap` lets a long value (a file path) break anywhere instead of overflowing a narrow screen. */
export function Code({ children, wrap = false }: { children: ReactNode; wrap?: boolean }) {
  return (
    <code
      className={`${wrap ? "[overflow-wrap:anywhere]" : "whitespace-nowrap"} rounded-chip border border-line bg-raised px-1.5 py-px font-mono text-[0.8125rem] text-ink`}
    >
      {children}
    </code>
  );
}
