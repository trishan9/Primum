"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ApiError, getBenchmarkQuestions } from "@/lib/api";
import type { BenchmarkQuestion } from "@/lib/types";

const PAGE = 8;

interface BenchmarkPickerProps {
  onPick: (item: BenchmarkQuestion) => void;
  onClose: () => void;
}

type PageResult = { page: BenchmarkQuestion[] } | { problem: string };

/** Fetches one page and turns a failure into the sentence the picker shows. Never throws. */
async function requestPage(from: number, signal: AbortSignal): Promise<PageResult> {
  try {
    return { page: await getBenchmarkQuestions({ limit: PAGE, offset: from }, signal) };
  } catch (error) {
    return {
      problem:
        error instanceof ApiError && error.kind === "unreachable"
          ? "The API is not answering, so the benchmark questions could not be loaded."
          : error instanceof ApiError && error.kind === "not_found"
            ? "The server has no PubMedQA questions yet. Build them with `primum data build pubmedqa`."
            : error instanceof Error
              ? error.message
              : "The benchmark questions could not be loaded.",
    };
  }
}

/** A page of PubMedQA questions at a time, fetched when the picker opens. Gold labels are never requested. */
export function BenchmarkPicker({ onPick, onClose }: BenchmarkPickerProps) {
  const [items, setItems] = useState<BenchmarkQuestion[]>([]);
  const [offset, setOffset] = useState(0);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [problem, setProblem] = useState<string>("");
  const [exhausted, setExhausted] = useState(false);
  const inFlight = useRef<AbortController | null>(null);

  // Applies one fetched page (or its failure). A request superseded or cancelled is ignored.
  const apply = useCallback((from: number, result: PageResult, signal: AbortSignal) => {
    if (signal.aborted) return;
    if ("problem" in result) {
      setProblem(result.problem);
      setStatus("error");
      return;
    }
    setItems((previous) => (from === 0 ? result.page : [...previous, ...result.page]));
    setExhausted(result.page.length < PAGE);
    setOffset(from + result.page.length);
    setStatus("ready");
  }, []);

  const request = (from: number) => {
    inFlight.current?.abort();
    const abort = new AbortController();
    inFlight.current = abort;
    return { signal: abort.signal, result: requestPage(from, abort.signal) };
  };

  const loadMore = (from: number) => {
    setStatus("loading");
    const { signal, result } = request(from);
    void result.then((r) => apply(from, r, signal));
  };

  useEffect(() => {
    const abort = new AbortController();
    inFlight.current = abort;
    void requestPage(0, abort.signal).then((r) => apply(0, r, abort.signal));
    return () => inFlight.current?.abort();
  }, [apply]);

  // Escape closes the picker wherever focus is, including after a list update moved it.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <Card as="div" className="p-4 sm:p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 tabIndex={-1} autoFocus className="text-[0.9375rem] font-semibold text-ink outline-none">
            PubMedQA benchmark questions
          </h2>
          <p className="mt-0.5 max-w-[62ch] text-[0.8125rem] leading-relaxed text-muted">
            Expert-labelled yes, no or maybe research questions. Picking one runs the pipeline in benchmark mode, which
            adds a decision and a conformal prediction set.
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose}>
          Close
        </Button>
      </div>

      {status === "error" ? (
        <div
          className="mt-3 rounded-control border border-dashed border-(--frame) px-3 py-3 text-sm text-ink"
          role="alert"
        >
          <p>{problem}</p>
          <Button className="mt-2" onClick={() => loadMore(offset)}>
            Try again
          </Button>
        </div>
      ) : null}

      {items.length > 0 ? (
        <ul className="-mx-2 mt-3 divide-y divide-rule">
          {items.map((item) => (
            <li key={item.qid}>
              <button
                type="button"
                onClick={() => onPick(item)}
                className="flex min-h-11 w-full items-baseline gap-3 rounded-control px-2 py-2.5 text-left hover:bg-raised"
              >
                <span className="num w-20 shrink-0 font-mono text-xs text-muted">{item.qid}</span>
                <span className="text-sm leading-snug text-ink">{item.question}</span>
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
        <Button size="sm" className="mt-3" onClick={() => loadMore(offset)} disabled={status === "loading"}>
          {status === "loading" ? "Loading" : `Show ${PAGE} more`}
        </Button>
      ) : null}
    </Card>
  );
}
