"use client";

import type { FormEvent, ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/components/ui/cn";
import { CheckGlyph, CrossGlyph, SearchGlyph } from "@/components/ui/glyphs";
import { Kbd } from "@/components/ui/kbd";

import { familyOf } from "./retrievers";

/**
 * The playground's query bar, a card in three bands. Compose it:
 *
 *   <QueryBar onSubmit={run}>
 *     <QueryBar.Input id value onChange busy describedBy />
 *     <QueryBar.Options>
 *       <RetrieverChips … />
 *       <QueryBar.Settings>…selects…</QueryBar.Settings>
 *     </QueryBar.Options>
 *     <QueryBar.Note>…</QueryBar.Note>
 *   </QueryBar>
 */
function QueryBarRoot({ onSubmit, children }: { onSubmit: () => void; children: ReactNode }) {
  return (
    <form
      role="search"
      aria-label="Compare retrievers"
      noValidate
      onSubmit={(event: FormEvent) => {
        event.preventDefault();
        onSubmit();
      }}
      className="overflow-hidden rounded-card border border-control bg-surface transition-[border-color,box-shadow] duration-[var(--dur)] has-[input[type=search]:focus-visible]:border-accent has-[input[type=search]:focus-visible]:ring-3 has-[input[type=search]:focus-visible]:ring-accent/15"
    >
      {children}
    </form>
  );
}

function QueryBarInput({
  id,
  value,
  onChange,
  busy,
  describedBy,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  busy: boolean;
  describedBy?: string;
}) {
  return (
    <div className="flex items-center gap-2 py-2 pl-4 pr-2 sm:gap-3 sm:py-2.5 sm:pl-5 sm:pr-2.5">
      <SearchGlyph size={18} className="text-muted" />
      <label htmlFor={id} className="sr-only">
        Query
      </label>
      <input
        id={id}
        type="search"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Search passages"
        autoComplete="off"
        spellCheck={false}
        enterKeyHint="search"
        aria-describedby={describedBy}
        className="h-11 min-w-0 flex-1 bg-transparent text-[1.0625rem] text-ink placeholder:text-faint focus:outline-none sm:text-lg [&::-webkit-search-cancel-button]:hidden"
      />
      {value ? (
        <button
          type="button"
          onClick={() => onChange("")}
          aria-label="Clear the query"
          title="Clear the query"
          className="inline-flex size-11 shrink-0 items-center justify-center rounded-control text-muted transition-colors duration-[var(--dur-fast)] hover:bg-raised hover:text-ink sm:size-9"
        >
          <CrossGlyph size={14} />
        </button>
      ) : null}
      <span className="hidden items-center gap-1 pr-1 text-xs text-faint md:flex" aria-hidden="true">
        <Kbd>Enter</Kbd>
      </span>
      <Button type="submit" variant="primary" className="min-w-24 sm:min-w-28" disabled={busy}>
        {busy ? "Comparing" : "Compare"}
      </Button>
    </div>
  );
}

/** The band of settings under the input, one tone step down. */
function QueryBarOptions({ children }: { children: ReactNode }) {
  return (
    <div className="flex flex-col gap-4 border-t border-rule bg-ground px-4 py-3.5 sm:px-5 lg:flex-row lg:flex-wrap lg:items-end lg:justify-between lg:gap-x-8">
      {children}
    </div>
  );
}

/** Small labelled selects, kept on one line where there is room. */
function QueryBarSettings({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-end gap-3 sm:flex-nowrap">{children}</div>;
}

function QueryBarSelect({
  id,
  label,
  value,
  onChange,
  children,
  className,
}: {
  id: string;
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <label htmlFor={id} className="mb-1.5 block text-meta font-medium text-muted">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="num h-11 w-full min-w-0 rounded-control border border-control bg-surface px-2.5 text-sm text-ink transition-colors duration-[var(--dur-fast)] hover:border-ink sm:h-9"
      >
        {children}
      </select>
    </div>
  );
}

/** A footnote across the bottom of the bar: the concept-dictionary note, a validation problem. */
function QueryBarNote({ tone = "muted", id, children }: { tone?: "muted" | "problem"; id?: string; children: ReactNode }) {
  return (
    <div
      id={id}
      role={tone === "problem" ? "alert" : undefined}
      className={cn(
        "flex items-start gap-2 border-t border-rule px-4 py-2.5 text-meta sm:px-5",
        tone === "problem" ? "bg-contradicted-bg font-medium text-contradicted-ink" : "bg-ground text-muted",
      )}
    >
      {children}
    </div>
  );
}

export const QueryBar = Object.assign(QueryBarRoot, {
  Input: QueryBarInput,
  Options: QueryBarOptions,
  Settings: QueryBarSettings,
  Select: QueryBarSelect,
  Note: QueryBarNote,
});

/**
 * Retrievers as toggle chips (`aria-pressed`). Columns follow the order the chips are listed in,
 * which is the server's order, so a comparison always reads lexical, dense, fused, left to right.
 */
export function RetrieverChips({
  available,
  selected,
  isOffered,
  onToggle,
}: {
  available: string[];
  selected: string[];
  isOffered: (name: string) => boolean;
  onToggle: (name: string) => void;
}) {
  const count = available.filter((name) => selected.includes(name) && isOffered(name)).length;
  return (
    <fieldset className="min-w-0 lg:flex-[1_1_30rem]">
      <legend className="mb-1.5 flex w-full items-baseline gap-2 text-meta font-medium text-muted">
        Retrievers
        <span className="num font-normal text-faint">{count} selected, shown left to right</span>
      </legend>
      {/* The chips wrap at every width: a scrolling row with a hidden scrollbar reads as cut off. */}
      <div className="flex flex-wrap gap-1.5">
        {available.map((name) => {
          const offered = isOffered(name);
          const on = selected.includes(name) && offered;
          return (
            <button
              key={name}
              type="button"
              aria-pressed={on}
              disabled={!offered}
              onClick={() => onToggle(name)}
              title={offered ? `${familyOf(name)} retriever` : "Needs the concept dictionary, which this server has not built"}
              className={cn(
                "inline-flex min-h-11 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-chip border px-2.5 font-mono text-[0.8125rem] transition-[background-color,border-color,color] duration-[var(--dur-fast)] sm:min-h-8",
                // Selected is ink on the active tone: the accent is kept for the results the chips produce.
                on
                  ? "border-control bg-active text-ink"
                  : "border-line bg-surface text-muted hover:border-control hover:text-ink",
                "disabled:cursor-not-allowed disabled:border-dashed disabled:bg-transparent disabled:text-faint disabled:line-through",
              )}
            >
              <span
                aria-hidden="true"
                className={cn(
                  "inline-flex size-3.5 items-center justify-center rounded-[4px] border",
                  on ? "border-primary bg-primary text-on-primary" : "border-control",
                )}
              >
                {on ? <CheckGlyph size={10} /> : null}
              </span>
              {name}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
