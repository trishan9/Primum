"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, getHealth, getMeta } from "@/lib/api";
import type { Health, Meta } from "@/lib/types";

export type ApiReachability = "checking" | "up" | "down";

interface ApiContextValue {
  reachability: ApiReachability;
  health: Health | null;
  meta: Meta | null;
  /** Why the last check failed, when it did. */
  problem: string | null;
  refresh: () => void;
}

const ApiContext = createContext<ApiContextValue>({
  reachability: "checking",
  health: null,
  meta: null,
  problem: null,
  refresh: () => {},
});

const POLL_MS = 20_000;

/**
 * One shared view of whether the API is there and what it offers. Every page reads it,
 * so "the backend is down" is said once, the same way, everywhere.
 */
export function ApiProvider({ children }: { children: React.ReactNode }) {
  const [reachability, setReachability] = useState<ApiReachability>("checking");
  const [health, setHealth] = useState<Health | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const check = useCallback(async () => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    try {
      const nextHealth = await getHealth(controller.signal);
      if (controller.signal.aborted) return;
      setHealth(nextHealth);
      setReachability("up");
      setProblem(null);
      // Meta is useful but not required: a server without it still counts as reachable.
      try {
        const nextMeta = await getMeta(controller.signal);
        if (!controller.signal.aborted) setMeta(nextMeta);
      } catch {
        /* keep the previous meta */
      }
    } catch (error) {
      if (controller.signal.aborted) return;
      setReachability("down");
      setHealth(null);
      setProblem(error instanceof ApiError ? error.message : "The health check failed.");
    }
  }, []);

  useEffect(() => {
    const first = window.setTimeout(() => void check(), 0);
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void check();
    }, POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") void check();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
      inFlight.current?.abort();
    };
  }, [check]);

  const value = useMemo<ApiContextValue>(
    () => ({ reachability, health, meta, problem, refresh: () => void check() }),
    [reachability, health, meta, problem, check],
  );

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
}

export function useApi(): ApiContextValue {
  return useContext(ApiContext);
}
