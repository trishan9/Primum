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

/** Loads one resource, and loads it again when the API comes back after being unreachable. */
export function useLoaded<T>(load: (signal: AbortSignal) => Promise<T>, key: string): Loaded<T> {
  const { reachability } = useApi();
  const [state, setState] = useState<{ key: string; value: Loaded<T> }>({ key: "", value: { status: "loading" } });
  const attempt = `${key}:${reachability === "up" ? "up" : "other"}`;

  useEffect(() => {
    const abort = new AbortController();
    load(abort.signal)
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
    // `load` is recreated every render by callers; `attempt` is what identifies a new request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt]);

  return state.key === attempt ? state.value : { status: "loading" };
}

/** One results file of a run; without `run`, the main run. */
export function useExperiment(name: ExperimentName, run: string = MAIN_RUN): Loaded<unknown> {
  return useLoaded((signal) => getExperiment(name, signal, run), `${run}/${name}`);
}
