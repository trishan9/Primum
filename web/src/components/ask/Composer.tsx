"use client";

import { useId, useState } from "react";

import { Button, Segmented } from "@/components/ui/controls";
import { StopGlyph } from "@/components/ui/glyphs";
import { useApi } from "@/components/shell/ApiProvider";
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
  onChange: (value: ComposerValue) => void;
  onSubmit: () => void;
  onStop: () => void;
  running: boolean;
  alphaOptions: readonly number[];
  retrieverOptions: readonly string[];
  /** the retriever the server uses when none is chosen, once `/api/meta` has said */
  defaultRetriever?: string;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
}

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
}: ComposerProps) {
  const id = useId();
  const { health } = useApi();
  const [touched, setTouched] = useState(false);

  const length = value.question.trim().length;
  const problem =
    length < MIN ? `Write a question of at least ${MIN} characters.` : length > MAX ? `Keep the question under ${MAX.toLocaleString("en-US")} characters.` : null;

  const submit = () => {
    setTouched(true);
    if (problem) {
      textarea.current?.focus();
      return;
    }
    onSubmit();
  };

  return (
    <div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className="rounded-card border border-control bg-surface outline-offset-2 outline-accent focus-within:border-ink has-[textarea:focus-visible]:outline-2"
        noValidate
      >
        <label htmlFor={`${id}-q`} className="sr-only">
          Your question
        </label>
        <textarea
          id={`${id}-q`}
          ref={textarea}
          rows={2}
          value={value.question}
          maxLength={MAX + 200}
          onChange={(event) => onChange({ ...value, question: event.target.value, benchmark: null })}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              if (!running) submit();
            }
          }}
          placeholder="Ask a clinical question"
          aria-invalid={touched && problem ? true : undefined}
          aria-describedby={`${id}-help${touched && problem ? ` ${id}-error` : ""}`}
          className="block w-full resize-none rounded-t-card bg-transparent px-4 pb-2 pt-3.5 font-serif text-[1.25rem] leading-snug text-ink placeholder:text-faint focus:outline-none sm:px-5"
        />

        {value.benchmark ? (
          <p className="num px-4 pb-1 text-xs text-muted sm:px-5">
            PubMedQA item <span className="font-mono text-ink">{value.benchmark.qid}</span>, benchmark mode.{" "}
            <button
              type="button"
              onClick={() => onChange({ ...value, benchmark: null })}
              className="text-ink underline decoration-control underline-offset-2 hover:decoration-ink"
            >
              Ask it as an open question instead
            </button>
          </p>
        ) : null}

        <div className="flex flex-wrap items-end gap-x-5 gap-y-3 border-t border-rule px-4 py-3 sm:px-5">
          <Segmented
            legend="Risk level α"
            name={`${id}-alpha`}
            value={fmtAlpha(value.alpha)}
            options={alphaOptions.map((a) => ({ value: fmtAlpha(a), label: fmtAlpha(a) }))}
            onChange={(next) => onChange({ ...value, alpha: Number(next) })}
            disabled={running}
          />

          <div>
            <label htmlFor={`${id}-retriever`} className="mb-1.5 block text-[0.8125rem] font-medium text-ink">
              Retriever
            </label>
            <select
              id={`${id}-retriever`}
              value={value.retriever}
              disabled={running}
              onChange={(event) => onChange({ ...value, retriever: event.target.value })}
              className="h-11 rounded-control border border-control bg-surface px-2.5 text-sm text-ink disabled:opacity-70 sm:h-9"
            >
              <option value="">{defaultRetriever ? `Server default (${defaultRetriever})` : "Server default"}</option>
              {retrieverOptions.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          <div className="ml-auto flex items-center gap-3">
            {running ? (
              <Button onClick={onStop}>
                <StopGlyph size={14} />
                Stop
              </Button>
            ) : null}
            <Button type="submit" variant="primary" disabled={running} className="min-w-24">
              {running ? "Working" : "Ask"}
            </Button>
          </div>
        </div>
      </form>

      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 px-1 text-xs leading-relaxed text-muted">
        <p id={`${id}-help`}>
          {health?.deployment_bundle === false
            ? "No certified threshold exists yet, so answers are shown unscored."
            : `At α = ${fmtAlpha(value.alpha)}, an answer is released only above a threshold certified on PubMedQA benchmark questions to keep errors among released answers at or below ${fmtPct(value.alpha)}; for your own question it is an operating point, not a guarantee.`}
        </p>
        {touched && problem ? (
          <p id={`${id}-error`} role="alert" className="font-medium text-contradicted">
            {problem}
          </p>
        ) : null}
      </div>
    </div>
  );
}
