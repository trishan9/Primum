"use client";

import { useEffect, type RefObject } from "react";

/**
 * Publishes the docked composer's height as `--composer-h` on the root element. The page's
 * `scroll-padding-bottom` reads it, so a focused control or a scrolled-to card always comes to rest
 * above the composer instead of under it (WCAG 2.4.11).
 */
export function useDockHeight(ref: RefObject<HTMLElement | null>) {
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const root = document.documentElement;
    const publish = () => root.style.setProperty("--composer-h", `${Math.ceil(element.getBoundingClientRect().height)}px`);
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(element);
    return () => {
      observer.disconnect();
      root.style.removeProperty("--composer-h");
    };
  }, [ref]);
}
