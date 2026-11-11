"use client";

import { useMemo, useSyncExternalStore } from "react";

/**
 * The design tokens a chart draws with, resolved to concrete colours. nivo interpolates and
 * darkens colours (react-spring, d3-color), which `var(--x)` cannot survive, so the values are
 * read from the root element's computed style and read again whenever the theme changes
 * (the `data-theme` stamp or the OS colour scheme).
 */
const NAMES = [
  "surface",
  "raised",
  "active",
  "ground",
  "border",
  "rule",
  "control",
  "text",
  "muted",
  "faint",
  "grid",
  "axis",
  "accent",
  "accent-soft",
  "accent-ink",
  "series-1",
  "series-2",
  "series-3",
  "series-4",
  "series-rest",
  "hatch",
  "supported",
  "supported-bg",
  "unsupported",
  "unsupported-bg",
  "contradicted",
  "contradicted-bg",
  "shadow-float",
] as const;

export type TokenName = (typeof NAMES)[number];

export type ChartTokens = Record<TokenName, string> & {
  /** The body's resolved font stack (IBM Plex Sans), for SVG text. */
  fontFamily: string;
  dark: boolean;
};

/*
 * One store for every chart on the page: a single cached snapshot, one MutationObserver and one
 * matchMedia listener, set up when the first chart subscribes and torn down after the last
 * leaves. getSnapshot is called on every render of every subscriber, so it must be cheap: it
 * returns the cache, which is cleared only when the theme stamp or the OS scheme changes.
 */
let cached: string | null = null;
const listeners = new Set<() => void>();
let teardown: (() => void) | null = null;

function invalidate() {
  cached = null;
  for (const listener of listeners) listener();
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange);
  if (!teardown) {
    const observer = new MutationObserver(invalidate);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", invalidate);
    teardown = () => {
      observer.disconnect();
      media.removeEventListener("change", invalidate);
    };
  }
  return () => {
    listeners.delete(onChange);
    if (listeners.size === 0 && teardown) {
      teardown();
      teardown = null;
      cached = null;
    }
  };
}

/** A string snapshot, so React compares it by value and re-renders only when a token changed. */
function computeSnapshot(): string {
  const style = getComputedStyle(document.documentElement);
  const values: Record<string, string | boolean> = {};
  for (const name of NAMES) values[name] = style.getPropertyValue(`--${name}`).trim();
  values.fontFamily = getComputedStyle(document.body).fontFamily;
  const stamped = document.documentElement.dataset.theme;
  values.dark = stamped === "dark" || (stamped !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  return JSON.stringify(values);
}

const snapshot = (): string => (cached ??= computeSnapshot());

const serverSnapshot = () => "";

/** Parse once per snapshot string, so every chart shares one tokens object per theme. */
let lastRaw = "";
let lastParsed: ChartTokens | null = null;
function parsed(raw: string): ChartTokens {
  if (raw !== lastRaw || !lastParsed) {
    lastRaw = raw;
    lastParsed = JSON.parse(raw) as ChartTokens;
  }
  return lastParsed;
}

/**
 * The resolved tokens, or null on the server and before hydration (draw a skeleton then). The
 * value is memoised on the snapshot string, so its identity changes only with the theme.
 */
export function useChartTokens(): ChartTokens | null {
  const raw = useSyncExternalStore(subscribe, snapshot, serverSnapshot);
  return useMemo(() => (raw ? parsed(raw) : null), [raw]);
}
