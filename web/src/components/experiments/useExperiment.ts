"use client";

import { useEffect, useState } from "react";

import { useApi } from "@/components/shell/ApiProvider";
import { ApiError, getExperiment, MAIN_RUN, type ExperimentName } from "@/lib/api";

export type Loaded<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  /** 404: the results file does not exist because the experiment has not been run. */
  | { status: "missing" }
  /** The API itself is not answering. */
  | { status: "down" }
  | { status: "error"; message: string };

type Arg = string | number;

/**
 * Loads one resource, and loads it again when the API comes back after being unreachable.
 * The loader's inputs are passed as data (`args`), so the request is keyed by exactly what it
 * depends on: a loader cannot close over a value the key does not include. Pass a stable,
 * module-level `fetcher`.
 */
export function useLoaded<A extends readonly Arg[], T>(
  fetcher: (signal: AbortSignal, ...args: A) => Promise<T>,
  args: A,
): Loaded<T> {
  const { reachability } = useApi();
  const [state, setState] = useState<{ key: string; value: Loaded<T> }>({ key: "", value: { status: "loading" } });
  const key = JSON.stringify(args);
  const attempt = `${key}:${reachability === "up" ? "up" : "other"}`;

  useEffect(() => {
    const abort = new AbortController();
    fetcher(abort.signal, ...(JSON.parse(key) as A))
      .then((data) => {
        if (!abort.signal.aborted) setState({ key: attempt, value: { status: "ready", data } });
      })
      .catch((error: unknown) => {
        if (abort.signal.aborted) return;
        const value: Loaded<T> =
          error instanceof ApiError && error.kind === "not_found"
            ? { status: "missing" }
            : error instanceof ApiError && error.kind === "unreachable"
              ? { status: "down" }
              : { status: "error", message: error instanceof Error ? error.message : "The request failed." };
        setState({ key: attempt, value });
      });
    return () => abort.abort();
  }, [fetcher, key, attempt]);

  return state.key === attempt ? state.value : { status: "loading" };
}

const fetchExperiment = (signal: AbortSignal, name: ExperimentName, run: string) => getExperiment(name, signal, run);

/** One results file of a run; without `run`, the main run. */
export function useExperiment(name: ExperimentName, run: string = MAIN_RUN): Loaded<unknown> {
  return useLoaded(fetchExperiment, [name, run] as const);
}
