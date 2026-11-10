"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/controls";
import { ApiError, getBenchmarkQuestions } from "@/lib/api";
import type { BenchmarkQuestion } from "@/lib/types";

const PAGE = 8;

interface BenchmarkPickerProps {
  onPick: (item: BenchmarkQuestion) => void;
  onClose: () => void;
}

/** A page of PubMedQA questions at a time, fetched when the picker opens. Gold labels are never requested. */
export function BenchmarkPicker({ onPick, onClose }: BenchmarkPickerProps) {
  const [items, setItems] = useState<BenchmarkQuestion[]>([]);
  const [offset, setOffset] = useState(0);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [problem, setProblem] = useState<string>("");
  const [exhausted, setExhausted] = useState(false);
  const headingRef = useRef<HTMLHeadingElement>(null);

  const load = useCallback(async (from: number) => {
    setStatus("loading");
    try {
      const page = await getBenchmarkQuestions({ limit: PAGE, offset: from });
      setItems((previous) => (from === 0 ? page : [...previous, ...page]));
      setExhausted(page.length < PAGE);
      setOffset(from + page.length);
      setStatus("ready");
    } catch (error) {
      setProblem(
        error instanceof ApiError && error.kind === "unreachable"
          ? "The API is not answering, so the benchmark questions could not be loaded."
          : error instanceof ApiError && error.kind === "not_found"
            ? "The server has no PubMedQA questions yet. Build them with `primum data build pubmedqa`."
            : error instanceof Error
              ? error.message
              : "The benchmark questions could not be loaded.",
      );
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    const start = window.setTimeout(() => {
      headingRef.current?.focus();
      void load(0);
    }, 0);
    return () => window.clearTimeout(start);
  }, [load]);

  // Escape closes the picker wherever focus is, including after a list update moved it.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="mt-3 rounded-card border border-line bg-surface p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 ref={headingRef} tabIndex={-1} className="text-[0.9375rem] font-semibold text-ink outline-none">
            PubMedQA benchmark questions
          </h2>
          <p className="mt-0.5 max-w-[62ch] text-[0.8125rem] leading-relaxed text-muted">
            Expert-labelled yes, no or maybe research questions. Picking one runs the pipeline in benchmark mode,
            which adds a decision and a conformal prediction set.
          </p>
        </div>
        <Button variant="quiet" onClick={onClose}>
          Close
        </Button>
      </div>

      {status === "error" ? (
        <div className="mt-3 rounded-control border border-dashed border-(--frame) px-3 py-3 text-sm text-ink" role="alert">
          <p>{problem}</p>
          <Button className="mt-2" onClick={() => void load(offset)}>
            Try again
          </Button>
        </div>
      ) : null}

      {items.length > 0 ? (
        <ul className="mt-3 divide-y divide-rule border-y border-rule">
          {items.map((item) => (
            <li key={item.qid}>
              <button
                type="button"
                onClick={() => onPick(item)}
                className="flex w-full items-baseline gap-3 rounded-control px-1 py-2.5 text-left hover:bg-ground"
              >
                <span className="num w-20 shrink-0 font-mono text-xs text-muted">{item.qid}</span>
                <span className="font-serif text-[1rem] leading-snug text-ink">{item.question}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}

      {status === "loading" && items.length === 0 ? (
        <p className="mt-3 text-sm text-muted" aria-busy="true">
          Loading questions.
        </p>
      ) : null}

      {status === "ready" && items.length === 0 ? (
        <p className="mt-3 text-sm text-muted">The server returned no benchmark questions.</p>
      ) : null}

      {status !== "error" && items.length > 0 && !exhausted ? (
        <Button className="mt-3" onClick={() => void load(offset)} disabled={status === "loading"}>
          {status === "loading" ? "Loading" : `Show ${PAGE} more`}
        </Button>
      ) : null}
    </div>
  );
}
