import type { ReactNode } from "react";

/** A key the reader can press, as in "Enter to ask". */
export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex min-w-5 items-center justify-center rounded-[4px] border border-line bg-surface px-1 font-sans text-[0.6875rem] font-medium leading-4 text-muted shadow-control">
      {children}
    </kbd>
  );
}
