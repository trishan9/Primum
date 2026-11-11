import type { ReactNode } from "react";

import { cn } from "./cn";

/**
 * The title row of a view: a noun h1 at 22px, one grey sentence about the data, real actions on
 * the right. Compose it:
 *
 *   <PageHeader>
 *     <PageHeader.Heading>
 *       <PageHeader.Title>Experiments</PageHeader.Title>
 *       <PageHeader.Description>Every number here is read from the results files.</PageHeader.Description>
 *     </PageHeader.Heading>
 *     <PageHeader.Actions>…</PageHeader.Actions>
 *   </PageHeader>
 */
function PageHeaderRoot({ className, children }: { className?: string; children: ReactNode }) {
  return <header className={cn("flex flex-wrap items-end justify-between gap-x-6 gap-y-3", className)}>{children}</header>;
}

function PageHeaderHeading({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("min-w-0 max-w-[72ch] flex-[1_1_28rem]", className)}>{children}</div>;
}

function PageHeaderTitle({ id, className, children }: { id?: string; className?: string; children: ReactNode }) {
  return (
    <h1 id={id} className={cn("text-title font-semibold tracking-[-0.012em] text-ink", className)}>
      {children}
    </h1>
  );
}

function PageHeaderDescription({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("mt-1.5 text-sm leading-relaxed text-muted sm:text-[0.9375rem]", className)}>{children}</div>;
}

function PageHeaderActions({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("flex flex-wrap items-center gap-2", className)}>{children}</div>;
}

export const PageHeader = Object.assign(PageHeaderRoot, {
  Heading: PageHeaderHeading,
  Title: PageHeaderTitle,
  Description: PageHeaderDescription,
  Actions: PageHeaderActions,
});
