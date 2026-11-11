import type { ReactNode } from "react";

import { cn } from "@/components/ui/cn";
import { ArrowRightGlyph, ClockGlyph, DocumentGlyph, PrimumMark, WarningGlyph } from "@/components/ui/glyphs";

/**
 * What the thread shows before the first question, composed from parts:
 *
 *   <EmptyThread>
 *     <EmptyThread.Intro title=…>one sentence</EmptyThread.Intro>
 *     <EmptyThread.Section title="Try a question">…</EmptyThread.Section>
 *   </EmptyThread>
 */
function EmptyThreadRoot({ children }: { children: ReactNode }) {
  return <div className="flex flex-col gap-8 pb-6 pt-2 sm:pt-8">{children}</div>;
}

function Intro({ title, children }: { title: string; children: ReactNode }) {
  return (
    <header>
      <h1 className="flex items-center gap-2.5 text-title font-semibold tracking-[-0.012em] text-ink">
        <PrimumMark size={20} />
        {title}
      </h1>
      <p className="mt-2 max-w-[60ch] text-[0.9375rem] leading-relaxed text-muted">{children}</p>
    </header>
  );
}

function Section({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section aria-label={title}>
      <div className="mb-3 flex items-baseline justify-between gap-4">
        <h2 className="text-sm font-semibold text-ink">{title}</h2>
        {aside ? <div className="text-xs text-muted">{aside}</div> : null}
      </div>
      {children}
    </section>
  );
}

export const EmptyThread = Object.assign(EmptyThreadRoot, { Intro, Section });

const PROMPT =
  "group flex h-full min-h-16 w-full items-start gap-3 rounded-card border border-line bg-surface px-4 py-3.5 text-left transition-[border-color,background-color] duration-(--dur-fast) hover:border-control hover:bg-raised/40";

/** A suggested question: one tap puts it in the composer. A note, when present, sits at the foot of the card. */
export function PromptButton({
  question,
  note,
  onSelect,
  className,
}: {
  question: string;
  note?: string;
  onSelect: () => void;
  className?: string;
}) {
  return (
    <button type="button" onClick={onSelect} className={cn(PROMPT, className)}>
      <span className="flex min-h-full min-w-0 flex-1 flex-col self-stretch">
        <span className="block text-[0.9375rem] leading-snug text-ink">{question}</span>
        {note ? (
          <span className="mt-auto inline-flex items-center gap-1 pt-1.5 text-xs font-medium text-emergency-ink">
            <WarningGlyph size={12} />
            {note}
          </span>
        ) : null}
      </span>
      <ArrowRightGlyph
        size={14}
        className="mt-1 text-faint transition-[color,transform] duration-(--dur-fast) group-hover:translate-x-0.5 group-hover:text-ink motion-reduce:transform-none"
      />
    </button>
  );
}

/** Opens the PubMedQA picker: the one route to a question the guarantee covers. */
export function BenchmarkButton({ open, onClick }: { open: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={open}
      className={cn(PROMPT, "items-center border-dashed border-frame bg-transparent sm:col-span-2")}
    >
      <DocumentGlyph size={16} className="text-accent-ink" />
      <span className="min-w-0 flex-1">
        <span className="block text-[0.9375rem] font-medium leading-snug text-ink">
          Pick a PubMedQA benchmark question
        </span>
        <span className="mt-0.5 block text-meta leading-snug text-muted">
          The certified guarantee applies only to these held-out questions, answered by the same generator.
        </span>
      </span>
    </button>
  );
}

/** A stored reading from this server, reopened without a new model call. */
export function RecentReading({
  question,
  verdict,
  onOpen,
}: {
  question: string;
  verdict: string;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="flex min-h-11 w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-raised/60 sm:px-5"
    >
      <ClockGlyph size={14} className="text-faint" />
      <span className="min-w-0 flex-1 truncate text-sm text-ink">{question}</span>
      <span className="shrink-0 text-xs text-muted">{verdict}</span>
    </button>
  );
}
