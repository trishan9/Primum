"use client";

import { useSyncExternalStore } from "react";

import { MoonGlyph, SunGlyph } from "@/components/ui/glyphs";

type Theme = "light" | "dark";

const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener("change", listener);
  return () => {
    listeners.delete(listener);
    media.removeEventListener("change", listener);
  };
}

function currentTheme(): Theme {
  const stamped = document.documentElement.dataset.theme;
  if (stamped === "light" || stamped === "dark") return stamped;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

/** Follows the system until the reader chooses; the choice is remembered on this device. */
export function ThemeToggle() {
  const theme = useSyncExternalStore<Theme | null>(subscribe, currentTheme, () => null);

  const toggle = () => {
    const next: Theme = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("primum-theme", next);
    } catch {
      /* private mode: the choice lasts for this page only */
    }
    listeners.forEach((listener) => listener());
  };

  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme ? `Switch to ${next} theme` : "Switch theme"}
      title={theme ? `Switch to ${next} theme` : "Switch theme"}
      className="flex size-11 items-center justify-center rounded-control text-muted transition-colors duration-150 hover:bg-raised hover:text-ink sm:size-9"
    >
      {theme === "dark" ? <SunGlyph size={18} /> : <MoonGlyph size={18} />}
    </button>
  );
}
