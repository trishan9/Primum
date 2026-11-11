"use client";

import { useId, useState } from "react";

import { useApi } from "@/components/shell/ApiProvider";
import { cn } from "@/components/ui/cn";
import {
  ArrowUpGlyph,
  ChevronGlyph,
  CrossGlyph,
  DocumentGlyph,
  SearchGlyph,
  SlidersGlyph,
  StopGlyph,
} from "@/components/ui/glyphs";
import { Kbd } from "@/components/ui/kbd";
import { Segmented } from "@/components/ui/segmented";
import { fmtAlpha, fmtPct } from "@/lib/format";
import type { BenchmarkQuestion } from "@/lib/types";

const MIN = 3;
const MAX = 2000;

export interface ComposerValue {
  question: string;
  alpha: number;
  retriever: string;
  benchmark: BenchmarkQuestion | null;
}

interface ComposerProps {
  value: ComposerValue;
  /** Receives only the fields that changed. */
  onChange: (patch: Partial<ComposerValue>) => void;
  onSubmit: () => void;
  onStop: () => void;
  running: boolean;
  alphaOptions: readonly number[];
  retrieverOptions: readonly string[];
  /** the retriever the server uses when none is chosen, once `/api/meta` has said */
  defaultRetriever?: string;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  /** In a thread the reply is the context, so the placeholder invites a follow-up question. */
  compact?: boolean;
}

/**
 * The composer, docked under the thread as in any chat tool: the question, then a toolbar with
 * the two settings that change the answer (risk level α, retriever) and the send button. On a
 * phone the settings fold behind one button, so the dock is a single row and the thread keeps
 * the screen.
 */
export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  running,
  alphaOptions,
  retrieverOptions,
  defaultRetriever,
  textareaRef: textarea,
  compact = false,
}: ComposerProps) {
  const id = useId();
  const { health } = useApi();
  const [touched, setTouched] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const length = value.question.trim().length;
  const problem =
    length < MIN
      ? `Write a question of at least ${MIN} characters.`
      : length > MAX
        ? `Keep the question under ${MAX.toLocaleString("en-US")} characters.`
        : null;

  const submit = () => {
    setTouched(true);
    if (problem) {
      textarea.current?.focus();
      return;
    }
    setTouched(false);
    setSettingsOpen(false);
    onSubmit();
  };

  const retrieverName = value.retriever || defaultRetriever || "server default";

  return (
    <div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        aria-label="Ask a question"
        noValidate
        className={cn(
          "flex flex-wrap items-end rounded-[1.25rem] border border-control bg-surface shadow-dock",
          "transition-[border-color,box-shadow] duration-(--dur)",
          // One focus treatment: an accent edge and a soft ring inside the radius, never a second outline.
          "has-[textarea:focus-visible]:border-accent has-[textarea:focus-visible]:ring-3 has-[textarea:focus-visible]:ring-accent/15",
        )}
      >
        {value.benchmark ? (
          <div className="flex basis-full flex-wrap items-center gap-2 px-3 pt-3">
            <span className="num inline-flex min-h-7 items-center gap-1.5 rounded-chip bg-accent-soft pl-2 pr-1 text-xs font-medium text-accent-ink">
              <DocumentGlyph size={13} />
              PubMedQA <span className="font-mono">{value.benchmark.qid}</span>, benchmark mode
              <button
                type="button"
                onClick={() => onChange({ benchmark: null })}
                aria-label="Ask it as an open question instead"
                title="Ask it as an open question instead"
                className="ml-0.5 inline-flex size-7 items-center justify-center rounded-[4px] hover:bg-surface/60 max-sm:size-9"
              >
                <CrossGlyph size={12} />
              </button>
            </span>
          </div>
        ) : null}

        <label htmlFor={`${id}-q`} className="sr-only">
          Your question
        </label>
        <textarea
          id={`${id}-q`}
          ref={textarea}
          rows={1}
          value={value.question}
          maxLength={MAX + 200}
          onChange={(event) => onChange({ question: event.target.value, benchmark: null })}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              if (!running) submit();
            }
          }}
          placeholder={compact ? "Ask another question" : "Ask a clinical question"}
          aria-invalid={touched && problem ? true : undefined}
          aria-describedby={`${id}-help${touched && problem ? ` ${id}-error` : ""}`}
          className="block max-h-52 min-h-[3.25rem] min-w-0 basis-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-base leading-relaxed text-ink [field-sizing:content] placeholder:text-faint focus:outline-none max-sm:grow max-sm:basis-0 max-sm:pb-3.5 max-sm:pr-1"
        />

        <div
          className={cn(
            "flex basis-full items-end gap-2 px-2.5 pb-2.5 pt-1.5",
            settingsOpen ? "max-sm:pt-0" : "max-sm:basis-auto max-sm:pb-1.5 max-sm:pl-0 max-sm:pr-1.5",
          )}
        >
          <div
            id={`${id}-settings`}
            className={cn("flex min-w-0 flex-1 flex-wrap items-center gap-x-3 gap-y-1", !settingsOpen && "max-sm:hidden")}
          >
            <div className="flex shrink-0 items-center gap-2">
              <span aria-hidden="true" className="pl-1 text-xs font-medium text-muted">
                Risk α
              </span>
              <Segmented
                legend="Risk level α"
                hideLegend
                name={`${id}-alpha`}
                value={fmtAlpha(value.alpha)}
                options={alphaOptions.map((a) => ({ value: fmtAlpha(a), label: fmtAlpha(a) }))}
                onChange={(next) => onChange({ alpha: Number(next) })}
                disabled={running}
              />
            </div>

            {/* The select fills the whole chip, so the tap target is the chip's full 44px. */}
            <label
              className={cn(
                "relative inline-flex shrink-0 items-center rounded-control text-muted hover:bg-raised hover:text-ink",
                "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent",
                running && "pointer-events-none opacity-70",
              )}
            >
              <SearchGlyph size={14} className="pointer-events-none absolute left-2" />
              <span className="sr-only">Retriever</span>
              <select
                value={value.retriever}
                disabled={running}
                onChange={(event) => onChange({ retriever: event.target.value })}
                className="min-h-11 max-w-[13rem] cursor-pointer appearance-none truncate bg-transparent pl-7 pr-6 text-meta font-medium text-ink focus:outline-none sm:min-h-9"
              >
                <option value="">{defaultRetriever ? `Default (${defaultRetriever})` : "Server default"}</option>
                {retrieverOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              <ChevronGlyph size={12} className="pointer-events-none absolute right-2 text-muted" />
            </label>
          </div>

          <div className="ml-auto flex shrink-0 items-center gap-1.5">
            <button
              type="button"
              onClick={() => setSettingsOpen((open) => !open)}
              aria-expanded={settingsOpen}
              aria-controls={`${id}-settings`}
              aria-label={`Settings: risk α ${fmtAlpha(value.alpha)}, retriever ${retrieverName}`}
              className={cn(
                "num inline-flex h-11 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-muted hover:bg-raised hover:text-ink sm:hidden",
                settingsOpen && "bg-active text-ink",
              )}
            >
              <SlidersGlyph size={15} />
              <span aria-hidden="true">α {fmtAlpha(value.alpha)}</span>
            </button>
            {running ? (
              <button
                type="button"
                onClick={onStop}
                className="inline-flex min-h-11 items-center gap-1.5 rounded-full border border-control bg-surface px-3.5 text-sm font-medium text-ink hover:bg-raised sm:min-h-9"
              >
                <StopGlyph size={13} />
                Stop
              </button>
            ) : null}
            <button
              type="submit"
              disabled={running}
              aria-label={running ? "Working on the question" : "Ask"}
              title="Ask (Enter)"
              className="inline-flex size-11 items-center justify-center rounded-full bg-primary text-on-primary transition-opacity duration-(--dur-fast) hover:opacity-90 active:opacity-80 disabled:bg-active disabled:text-faint sm:size-9"
            >
              <ArrowUpGlyph size={17} />
            </button>
          </div>
        </div>
      </form>

      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 px-2 text-xs leading-relaxed text-muted">
        <p id={`${id}-help`} className="min-w-0 flex-1 basis-80 max-sm:sr-only">
          {health?.deployment_bundle === false
            ? "No certified threshold exists yet, so answers are shown unscored."
            : `At α = ${fmtAlpha(value.alpha)}, an answer is released only above a threshold certified on PubMedQA benchmark questions to keep errors among released answers at or below ${fmtPct(value.alpha)}; for your own question it is an operating point, not a guarantee.`}
        </p>
        {touched && problem ? (
          <p id={`${id}-error`} role="alert" className="font-medium text-contradicted-ink">
            {problem}
          </p>
        ) : (
          <p className="hidden items-center gap-1 sm:flex" aria-hidden="true">
            <Kbd>Enter</Kbd> to ask, <Kbd>Shift</Kbd> <Kbd>Enter</Kbd> for a new line
          </p>
        )}
      </div>
    </div>
  );
}
