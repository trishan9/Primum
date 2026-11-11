import type { ReactNode } from "react";

import { cn } from "./cn";

/**
 * A block with nothing to show yet. It keeps its frame and says, in one or two sentences,
 * why it is empty and what fills it. Compose it:
 *
 *   <EmptyState>
 *     <EmptyState.Title>No results file for RQ3</EmptyState.Title>
 *     <EmptyState.Description>Run <Code>primum exp rq3</Code> to write it.</EmptyState.Description>
 *     <EmptyState.Actions><Button>Check again</Button></EmptyState.Actions>
 *     <EmptyState.Aside>What will appear here…</EmptyState.Aside>
 *   </EmptyState>
 *
 * With an Aside it spans the canvas in two columns, so no half of the frame is left blank.
 */
function EmptyStateRoot({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div
      className={cn(
        "grid gap-x-10 gap-y-4 rounded-card border border-dashed border-frame px-4 py-5 sm:px-5",
        "has-[>[data-empty-aside]]:lg:grid-cols-2",
        className,
      )}
    >
      {children}
    </div>
  );
}

function EmptyStateMain({ children }: { children: ReactNode }) {
  return <div className="min-w-0">{children}</div>;
}

function EmptyStateTitle({ children }: { children: ReactNode }) {
  return <p className="text-sm font-medium text-ink">{children}</p>;
}

function EmptyStateDescription({ children }: { children: ReactNode }) {
  return <div className="mt-1 max-w-[62ch] text-sm leading-relaxed text-muted">{children}</div>;
}

function EmptyStateActions({ children }: { children: ReactNode }) {
  return <div className="mt-3 flex flex-wrap gap-2">{children}</div>;
}

function EmptyStateAside({ children }: { children: ReactNode }) {
  return (
    <div data-empty-aside="" className="min-w-0 max-w-[62ch] text-sm leading-relaxed text-muted">
      {children}
    </div>
  );
}

export const EmptyState = Object.assign(EmptyStateRoot, {
  Main: EmptyStateMain,
  Title: EmptyStateTitle,
  Description: EmptyStateDescription,
  Actions: EmptyStateActions,
  Aside: EmptyStateAside,
});

/** The earlier one-call form, kept for the views that use it. */
export function EmptyBlock({
  title,
  children,
  action,
  aside,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <EmptyState className={aside ? undefined : "max-w-[60rem]"}>
      <EmptyStateMain>
        <EmptyStateTitle>{title}</EmptyStateTitle>
        {children ? <EmptyStateDescription>{children}</EmptyStateDescription> : null}
        {action ? <EmptyStateActions>{action}</EmptyStateActions> : null}
      </EmptyStateMain>
      {aside ? <EmptyStateAside>{aside}</EmptyStateAside> : null}
    </EmptyState>
  );
}
