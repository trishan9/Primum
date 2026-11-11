"use client";

import { useSyncExternalStore } from "react";

/**
 * Whether the wide-screen sidebar is collapsed to a rail. The truth is `data-sidebar` on <html>,
 * stamped before first paint from localStorage, so CSS lays the shell out and nothing shifts on
 * hydration; this store only lets controls read and change it.
 */
const KEY = "primum-sidebar";
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function read(): boolean {
  return document.documentElement.dataset.sidebar === "collapsed";
}

export function useSidebarCollapsed(): [collapsed: boolean | null, toggle: () => void] {
  const collapsed = useSyncExternalStore<boolean | null>(subscribe, read, () => null);
  const toggle = () => {
    const next = !read();
    if (next) document.documentElement.dataset.sidebar = "collapsed";
    else delete document.documentElement.dataset.sidebar;
    try {
      localStorage.setItem(KEY, next ? "collapsed" : "open");
    } catch {
      /* private mode: the choice lasts for this page only */
    }
    listeners.forEach((listener) => listener());
  };
  return [collapsed, toggle];
}
