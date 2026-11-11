/**
 * Scroll behaviour that honours the reader's motion setting. An explicit `behavior: "smooth"` in
 * script overrides the CSS reduced-motion rule, so every programmatic scroll asks here first.
 */
export function scrollBehavior(): ScrollBehavior {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return "auto";
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}
