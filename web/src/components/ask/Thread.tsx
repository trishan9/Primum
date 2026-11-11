import type { ReactNode } from "react";

import { cn } from "@/components/ui/cn";
import { PrimumMark } from "@/components/ui/glyphs";

/**
 * The conversation, composed from small parts:
 *
 *   <Thread>
 *     <UserMessage meta={…}>question</UserMessage>
 *     <AssistantMessage status={…}>trace, decision, answer, claims</AssistantMessage>
 *   </Thread>
 */
export function Thread({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("flex flex-col gap-7", className)}>{children}</div>;
}

/** The reader's question: a right-aligned bubble in the sans, with the request's settings under it. */
export function UserMessage({ children, meta }: { children: ReactNode; meta?: ReactNode }) {
  return (
    <div className="flex flex-col items-end gap-1.5">
      <h2 className="sr-only">Question</h2>
      <p className="max-w-[min(36rem,92%)] whitespace-pre-wrap break-words rounded-[18px] rounded-br-md bg-raised px-4 py-2.5 text-[0.9688rem] leading-relaxed text-ink">
        {children}
      </p>
      {meta ? <div className="flex flex-wrap justify-end gap-1.5 text-xs text-muted">{meta}</div> : null}
    </div>
  );
}

/** A small fact about the request, under the question: α, mode, benchmark item. */
export function MessageMeta({ children }: { children: ReactNode }) {
  return <span className="num inline-flex h-6 items-center rounded-chip border border-line px-2">{children}</span>;
}

/**
 * The system's turn: no bubble, the mark as its avatar, and a status line beside its name.
 * Everything Primum produced for the question sits inside it, in the order it happened.
 */
export function AssistantMessage({
  status,
  busy,
  children,
}: {
  status: ReactNode;
  busy: boolean;
  children: ReactNode;
}) {
  return (
    <article aria-label="Primum's reply" aria-busy={busy} className="sm:flex sm:gap-4">
      <Avatar className="max-sm:hidden" />
      <div className="min-w-0 flex-1">
        {/* The reply's heading: everything Primum produced sits under it as h3s. */}
        <h2 className="flex min-h-8 flex-wrap items-center gap-x-2 gap-y-0.5 text-sm font-normal">
          {/* On a phone the avatar joins the name row, so the reply keeps the full width. */}
          <Avatar className="sm:hidden" />
          <span className="font-semibold text-ink">Primum</span>
          <span className="sr-only">: </span>
          <span className="text-muted">{status}</span>
        </h2>
        <div className="mt-3 flex flex-col gap-5">{children}</div>
      </div>
    </article>
  );
}

function Avatar({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex size-8 shrink-0 items-center justify-center rounded-full border border-line bg-surface text-ink",
        className,
      )}
    >
      <PrimumMark size={18} />
    </span>
  );
}
